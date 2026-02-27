"""JumpReLU activation — learnable thresholds and Moreau bandwidth (§3).

Forward/backward logic lives in the fused Triton kernel (src/kernels/jumprelu_kernel.py).
This module holds the learnable parameters only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class JumpReLU(nn.Module):
    """JumpReLU parameter container: log-thresholds and Moreau bandwidth."""

    def __init__(self, F: int) -> None:
        super().__init__()
        self.log_threshold = nn.Parameter(torch.zeros(F))
        self.register_buffer("gamma", torch.ones(F))

    @property
    def threshold(self) -> Tensor:
        """Per-feature thresholds: exp(log_threshold)."""
        return self.log_threshold.exp()
