"""JumpReLU with Rectangle STE and per-feature adaptive bandwidth (§3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _HeavisideRectangleSTE(torch.autograd.Function):
    """Heaviside forward with rectangle-STE backward for threshold gradients."""

    @staticmethod
    def forward(ctx, u: Tensor) -> Tensor:
        ctx.save_for_backward(u)
        return (u > 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (u,) = ctx.saved_tensors
        return grad_output * (u.abs() < 1).float()


class JumpReLU(nn.Module):
    """JumpReLU with detached hard gate and STE L0 surrogate."""

    def __init__(self, F: int) -> None:
        super().__init__()
        self.F = F
        self.log_threshold = nn.Parameter(torch.zeros(F))
        self.register_buffer("epsilon", torch.full((F,), 0.01))

    @property
    def threshold(self) -> Tensor:
        """Current thresholds θ = exp(log_threshold)."""
        return self.log_threshold.exp()

    def forward(
        self, pre_act: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return gated activations, gate mask, and STE L0 surrogate."""
        theta = self.threshold

        gate_mask = (pre_act > theta).detach().float()
        z = pre_act * gate_mask

        u = (pre_act - theta) / self.epsilon
        l0_probs = _HeavisideRectangleSTE.apply(u)

        return z, gate_mask, l0_probs
