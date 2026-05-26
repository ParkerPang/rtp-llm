import functools
import logging
from typing import Any, Optional

from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W

# Methods that should be guarded against silently dropping input_embeddings.
# Disaggregate models implement forward_micro_batch instead of forward, so both
# must be wrapped to cover every entry point a request can land on.
_GUARDED_FORWARD_METHODS = ("forward", "forward_micro_batch")


def _inputs_carry_embeddings(inputs: Any) -> bool:
    """True if any PyModelInputs in `inputs` has a non-empty input_embeddings."""
    if inputs is None:
        return False
    # forward() gets a single PyModelInputs; forward_micro_batch() gets a list.
    candidates = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
    for inp in candidates:
        embs = getattr(inp, "input_embeddings", None)
        if embs is not None and len(embs) > 0:
            return True
    return False


def _wrap_with_embeddings_guard(method, method_name):
    @functools.wraps(method)
    def wrapper(self, inputs, *args, **kwargs):
        if not type(self).supports_input_embeddings and _inputs_carry_embeddings(
            inputs
        ):
            raise NotImplementedError(
                f"{type(self).__name__}.{method_name}() received input_embeddings "
                f"but the model does not opt in (supports_input_embeddings=False). "
                f"Either set supports_input_embeddings=True on the class and call "
                f"GptModelBase.apply_input_embeddings() in the forward path, or "
                f"reject the request upstream."
            )
        return method(self, inputs, *args, **kwargs)

    return wrapper


class GptModelBase(nn.Module):
    # Subclasses that consume inputs.input_embeddings (by calling
    # apply_input_embeddings() in their forward path) must override this to
    # True. The default False makes any new subclass that forgets to plumb
    # embeddings through fail loudly at the entry — the __init_subclass__
    # wrapper below rejects such requests with NotImplementedError instead of
    # silently dropping user-supplied data.
    supports_input_embeddings: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method_name in _GUARDED_FORWARD_METHODS:
            method = cls.__dict__.get(method_name)
            if method is None or getattr(method, "__embeddings_guarded__", False):
                continue
            wrapped = _wrap_with_embeddings_guard(method, method_name)
            wrapped.__embeddings_guarded__ = True  # type: ignore[attr-defined]
            setattr(cls, method_name, wrapped)

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = len(self.kv_cache.kv_cache_base_by_layer)
            layer0_shape = (
                self.kv_cache.kv_cache_base_by_layer[0].shape
                if num_layers > 0
                and self.kv_cache.kv_cache_base_by_layer[0] is not None
                else None
            )
            num_scale_layers = len(self.kv_cache.kv_scale_base_by_layer)
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shape={layer0_shape}, "
                f"num_scale_layers={num_scale_layers}, "
            )
        return True

    ## for cuda graph attn kernel params' fill
    def fill_params(
        self,
        sequence_lengths: Tensor,
        input_lengths: Tensor,
        kv_cache_block_id_host: Tensor,
        replay_batch_size: int,
        capture_batch_size: int,
        seq_size_per_block: int,
    ):
        assert capture_batch_size in self.params_dict
        params_ptr = self.params_dict[capture_batch_size]
        assert params_ptr is not None
        params_ptr.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )
        return fmha_impl

    @staticmethod
    def apply_input_embeddings(inputs_embeds: Tensor, inputs: PyModelInputs) -> Tensor:
        if inputs.input_embeddings is not None and len(inputs.input_embeddings) > 0:
            embs = inputs.input_embeddings
            locs = inputs.input_embeddings_locs
            token_num = inputs_embeds.size(0)
            hidden_size = inputs_embeds.size(1)
            if locs is None or locs.numel() != len(embs):
                raise ValueError(
                    f"input_embeddings_locs.numel():{0 if locs is None else locs.numel()} "
                    f"!= input_embeddings count:{len(embs)}"
                )
            for i, emb in enumerate(embs):
                if emb.dim() != 2:
                    raise ValueError(
                        f"input_embeddings[{i}] must be 2-D, got dim={emb.dim()}"
                    )
                if emb.size(1) != hidden_size:
                    raise ValueError(
                        f"input_embeddings[{i}].size(1):{emb.size(1)} != hidden_size:{hidden_size}"
                    )
                loc = locs[i].item()
                if loc < 0:
                    raise ValueError(f"input_embeddings_locs[{i}]:{loc} must be >= 0")
                if loc + emb.size(0) > token_num:
                    raise ValueError(
                        f"input_embeddings_locs[{i}]:{loc} + rows:{emb.size(0)} > token_num:{token_num}"
                    )
                inputs_embeds[loc : loc + emb.size(0)] = emb.to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
        return inputs_embeds

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
