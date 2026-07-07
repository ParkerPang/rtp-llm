"""SM120 production-registry coverage for FP8 PER_BLOCK PureTP MoE."""

import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_fp8_grouped_gemm_executor import (
    Sm120Fp8GroupedGemmExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterFp8PerBlock,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_per_block import (
    CudaSm120Fp8GroupedGemmNoDPStrategy,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.ops import MoeConfig, ParallelismConfig


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm12x(),
    "SM120 MoE strategy coverage requires consumer Blackwell",
)
class TestSM120Fp8PerBlockStrategies(unittest.TestCase):

    def setUp(self):
        model_config = ModelConfig()
        model_config.quant_config = init_quant_config("FP8_PER_BLOCK")
        model_config.data_type = "bf16"
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 1
        parallelism_config.tp_size = 2
        parallelism_config.dp_size = 1
        moe_config = MoeConfig()
        moe_config.moe_strategy = "auto"
        moe_config.use_all_gather = True
        self.config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            enable_cuda_graph=False,
        )
        self.registry = FusedMoeFactory.get_registry()

    def test_production_registry_selects_sm120_grouped_gemm(self):
        candidates = [
            strategy
            for strategy in self.registry.list_strategies()
            if strategy.can_handle(self.config)
        ]
        self.assertEqual(len(candidates), 1)
        self.assertIsInstance(candidates[0], CudaSm120Fp8GroupedGemmNoDPStrategy)

    def test_selected_strategy_uses_pure_tp_and_triton_executor(self):
        strategy = self.registry.get_strategy(self.config)
        attributes = strategy.get_attributes()
        self.assertIs(attributes.router_class, PureTpRouterFp8PerBlock)
        self.assertIs(attributes.executor_class, Sm120Fp8GroupedGemmExecutor)

    def test_explicit_strategy_selects_sm120_grouped_gemm(self):
        self.config.moe_strategy = "fp8_per_block_sm120_grouped"
        self.assertIsInstance(
            self.registry.get_strategy(self.config),
            CudaSm120Fp8GroupedGemmNoDPStrategy,
        )

    def test_cuda_graph_is_rejected(self):
        self.config.enable_cuda_graph = True
        candidates = [
            strategy
            for strategy in self.registry.list_strategies()
            if strategy.can_handle(self.config)
        ]
        self.assertFalse(
            any(
                isinstance(strategy, CudaSm120Fp8GroupedGemmNoDPStrategy)
                for strategy in candidates
            )
        )
        with self.assertRaisesRegex(ValueError, "No suitable MOE strategy"):
            self.registry.get_strategy(self.config)


if __name__ == "__main__":
    unittest.main()
