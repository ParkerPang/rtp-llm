"""Linear strategy base class

Defines the unified interface for all Linear strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.ops import HWKernelConfig


class LinearBase(nn.Module, ABC):
    """Linear strategy base class

    Each strategy is both a strategy checker and a Linear implementation.
    It inherits from nn.Module and implements forward() directly.
    """

    @classmethod
    @abstractmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Determine whether this strategy can handle the given configuration

        Args:
            quant_config: Quantization configuration (required)
            weight: Weight tensor
            weight_scales: Weight scales tensor (None for non-FP8)
            weight_scale_2: Second weight scale tensor (for NVFP4, can be None)
            input_scale: Input scale tensor (for NVFP4, can be None)

        Returns:
            Whether this configuration can be handled
        """
        pass

    @classmethod
    def explain_rejection(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> Optional[str]:
        """Return an actionable rejection reason, or ``None`` for a non-match.

        Strategy selection must remain side-effect free: ``can_handle`` only
        returns a boolean, while deterministic diagnostics belong here.
        """
        return None

    @abstractmethod
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        """Initialize the Linear module with weights

        Args:
            weight: Weight tensor
            weight_scales: Weight scales tensor
            input_scales: Input scales tensor
            bias: Bias tensor
            quant_config: Quantization configuration (required)
            weight_scale_2: Second weight scale tensor (for FP4, can be None)
        """
        super().__init__()

    def maybe_cache_quant_scale(self, max_len: int) -> None:
        """For quantized linear gemm input (fp8, fp4, etc),
        further quant scale calculation is not needed and can be constructed by simply filling ones.
        This method is used to cache the quant scale with given max length.

        Args:
            max_len: max input length to cache.
        """
        pass

    @staticmethod
    def _restore_blockwise_weight_layout(
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        block_size: int = 128,
        mismatch_label: str = "Weight scale dim mismatch:",
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """Restore loader tensors from logical (K,N) to physical (N,K).

        FP8 blockwise loaders expose the contiguous physical ``(N, K)`` data
        through logical tensor shapes ``(K, N)`` and ``(scale_K, scale_N)``.
        Reshaping restores the physical views without transposing elements.
        """
        if weight.dim() != 2 or weight_scales.dim() != 2:
            raise ValueError(
                "Blockwise weight and scales must both be 2D tensors, got "
                f"{weight.dim()}D and {weight_scales.dim()}D"
            )
        if not weight.is_contiguous():
            raise ValueError(
                "Blockwise weight must be contiguous before restoring its "
                "physical (N, K) layout"
            )
        if not weight_scales.is_contiguous():
            raise ValueError(
                "Blockwise weight scales must be contiguous before restoring "
                "their physical (scale_N, scale_K) layout"
            )
        K, N = weight.shape
        scale_K, scale_N = weight_scales.shape
        if (N + block_size - 1) // block_size != scale_N or (
            K + block_size - 1
        ) // block_size != scale_K:
            raise ValueError(
                f"{mismatch_label} N: {N}, scale_N: {scale_N}, "
                f"K: {K}, scale_K: {scale_K} "
                f"(expected ceil_div by {block_size})"
            )
        return (
            weight.reshape(N, K),
            weight_scales.reshape(scale_N, scale_K),
            K,
            N,
            scale_K,
            scale_N,
        )

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            input: Input tensor

        Returns:
            Output tensor
        """
        pass

    def forward_with_bias_gelu(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass followed by GELU.

        Backends with a fused GEMM+bias+GELU epilogue can override this method.
        The default implementation preserves existing device behavior.
        """
        return F.gelu(self.forward(input))

    def __repr__(self) -> str:
        """Return string representation of the strategy"""
        return f"{self.__class__.__name__}"
