import unittest
from unittest import mock

import torch

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.models_py.modules.factory.linear.linear_base import LinearBase


class SM120FactoryDiagnosticTest(unittest.TestCase):

    def test_constructor_rejects_missing_weight_scales(self):
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "requires weight_scales"):
            CudaFp8VllmBlockwiseLinear(weight=weight, weight_scales=None)

    def test_restore_non_square_blockwise_layout(self):
        weight = torch.arange(384 * 256).reshape(384, 256)
        weight_scales = torch.arange(3 * 2).reshape(3, 2)

        restored = LinearBase._restore_blockwise_weight_layout(weight, weight_scales)

        restored_weight, restored_scales, K, N, scale_K, scale_N = restored
        self.assertEqual((N, K), restored_weight.shape)
        self.assertEqual((scale_N, scale_K), restored_scales.shape)
        self.assertEqual((K, N, scale_K, scale_N), (384, 256, 3, 2))
        torch.testing.assert_close(restored_weight.flatten(), weight.flatten())
        torch.testing.assert_close(restored_scales.flatten(), weight_scales.flatten())

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._has_cutlass_scaled_mm_blockwise_sm120_fp8",
        return_value=False,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear.is_sm12x",
        return_value=True,
    )
    def test_missing_binding_reports_rebuild_action(self, _is_sm12x, _has_binding):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 1), dtype=torch.float32)
        with mock.patch.object(
            LinearFactory, "_strategies", [CudaFp8VllmBlockwiseLinear]
        ):
            with self.assertRaisesRegex(
                ValueError, r"rebuild on x86 with --config=cuda12_9"
            ):
                LinearFactory.create_linear(
                    weight=weight,
                    bias=None,
                    weight_scales=weight_scales,
                    quant_config=quant_config,
                )

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._has_cutlass_scaled_mm_blockwise_sm120_fp8",
        return_value=True,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear.is_sm12x",
        return_value=True,
    )
    def test_unaligned_shape_reports_factory_contract(self, _is_sm12x, _has_binding):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((320, 256), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((3, 2), dtype=torch.float32)
        with mock.patch.object(
            LinearFactory, "_strategies", [CudaFp8VllmBlockwiseLinear]
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"requires K and N to be multiples of 128, got K=320 and N=256",
            ):
                LinearFactory.create_linear(
                    weight=weight,
                    bias=None,
                    weight_scales=weight_scales,
                    quant_config=quant_config,
                )

    def test_factory_reports_all_actionable_rejections(self):
        class FirstStrategy:
            can_handle = mock.Mock(return_value=False)
            explain_rejection = mock.Mock(return_value="first reason")

        class SecondStrategy:
            can_handle = mock.Mock(return_value=False)
            explain_rejection = mock.Mock(return_value="second reason")

        with mock.patch.object(
            LinearFactory, "_strategies", [FirstStrategy, SecondStrategy]
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"FirstStrategy: first reason; SecondStrategy: second reason; "
                r"configuration: weight.dtype=torch.float32, has_scales=False",
            ):
                LinearFactory.create_linear(
                    weight=torch.empty((1, 1)),
                    bias=None,
                    weight_scales=None,
                    quant_config=None,
                )

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear.has_deep_gemm",
        return_value=False,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear.supports_deep_gemm",
        return_value=False,
    )
    def test_deepgemm_unavailable_is_rejected_before_construction(
        self, _supports_deep_gemm, _has_deep_gemm
    ):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 1), dtype=torch.float32)

        self.assertFalse(
            CudaFp8GEMMLinear.can_handle(quant_config, weight, weight_scales)
        )
        self.assertRegex(
            CudaFp8GEMMLinear.explain_rejection(quant_config, weight, weight_scales),
            "not installed",
        )


if __name__ == "__main__":
    unittest.main()
