"""SM120 production-registry coverage for unsupported FP8 PER_BLOCK MoE."""

import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.ops import MoeConfig, ParallelismConfig


@unittest.skipUnless(
    torch.cuda.is_available() and is_sm12x(),
    "SM120 MoE rejection coverage requires consumer Blackwell",
)
class TestSM120Fp8PerBlockStrategies(unittest.TestCase):

    def setUp(self):
        model_config = ModelConfig()
        model_config.quant_config = init_quant_config("FP8_PER_BLOCK")
        model_config.data_type = "bf16"
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 1
        parallelism_config.tp_size = 1
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

    def test_production_registry_has_no_sm120_fp8_per_block_candidate(self):
        candidates = [
            strategy
            for strategy in self.registry.list_strategies()
            if strategy.can_handle(self.config)
        ]
        self.assertEqual(candidates, [])

    def test_production_registry_fails_with_actionable_error(self):
        with self.assertRaisesRegex(
            ValueError, "SM12x FP8_PER_BLOCK MoE is not supported yet"
        ):
            self.registry.get_strategy(self.config)


if __name__ == "__main__":
    unittest.main()
