"""ROCm-specific activation function implementations."""

import os
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase

_ENABLE_TRITON_SILU_AND_MUL_ENV = "ENABLE_TRITON_SILU_AND_MUL"

_WARMED_TRITON_CONFIGS: set[tuple[str, torch.dtype, int]] = set()


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() in {"1", "true"}


_ENABLE_TRITON_SILU_AND_MUL = _env_enabled(_ENABLE_TRITON_SILU_AND_MUL_ENV)


@triton.jit
def _silu_and_mul_kernel(
    output_ptr,
    input_ptr,
    n_cols: tl.constexpr,
    input_row_stride,
    output_row_stride,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < n_cols

    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride

    gate = tl.load(input_row + offsets, mask=mask).to(tl.float32)
    up = tl.load(input_row + n_cols + offsets, mask=mask).to(tl.float32)
    output = gate * tl.sigmoid(gate) * up

    tl.store(output_row + offsets, output, mask=mask)


def _silu_and_mul_torch(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return (
        F.silu(gate.to(torch.float32, non_blocking=True))
        * up.to(torch.float32, non_blocking=True)
    ).to(gate_up.dtype, non_blocking=True)


def _silu_and_mul_triton(gate_up: torch.Tensor) -> torch.Tensor:
    d = gate_up.shape[-1] // 2
    output = torch.empty(
        gate_up.shape[:-1] + (d,), dtype=gate_up.dtype, device=gate_up.device
    )
    gate_up_2d = gate_up.reshape(-1, gate_up.shape[-1])
    output_2d = output.reshape(-1, d)
    block_size = triton.next_power_of_2(d)
    _silu_and_mul_kernel[(gate_up_2d.shape[0],)](
        output_2d,
        gate_up_2d,
        d,
        gate_up_2d.stride(0),
        output_2d.stride(0),
        block_size,
    )
    return output


def _warmup_silu_and_mul_triton(
    hidden_size: Optional[int],
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> None:
    if hidden_size is None or dtype is None or device is None:
        return
    device = torch.device(device)
    if device.type != "cuda":
        return

    config = (str(device), dtype, hidden_size)
    if config in _WARMED_TRITON_CONFIGS:
        return

    gate_up = torch.empty((1, 2 * hidden_size), dtype=dtype, device=device)
    _silu_and_mul_triton(gate_up)
    _WARMED_TRITON_CONFIGS.add(config)


class FusedSiluAndMul(SiluAndMulBase):
    """
    ROCm implementation of silu_and_mul.

    Currently fallback to triton impl as a workaround for the precision issue of
    aiter.silu_and_mul

    Seting ENABLE_TRITON_SILU_AND_MUL=0 to fallback to the torch ref impl
    """

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if _ENABLE_TRITON_SILU_AND_MUL:
            _warmup_silu_and_mul_triton(hidden_size, dtype, device)

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication.

        Args:
            gate_up: Input tensor with concatenated gate and up projections

        Output: result of silu and mul
        """

        if _ENABLE_TRITON_SILU_AND_MUL:
            return _silu_and_mul_triton(gate_up)
        else:
            return _silu_and_mul_torch(gate_up)
