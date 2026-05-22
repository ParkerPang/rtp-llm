"""Common activation functions that are architecture-independent base operations."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class SiluAndMulBase(torch.nn.Module):
    """Base class for silu_and_mul operation."""

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

    @abstractmethod
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        pass
