"""JumpReLU with Rectangle STE and per-feature adaptive bandwidth (§3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _HeavisideRectangleSTE(torch.autograd.Function):
    """Forward: Heaviside H(u > 0). Backward: Rectangle kernel 𝟙(|u| < 1).

    Used ONLY for the L0 loss computation. The actual JumpReLU gate uses
    detached hard gating — gradients to θ flow exclusively through L0.
    """

    @staticmethod
    def forward(ctx, u: Tensor) -> Tensor:
        ctx.save_for_backward(u)
        return (u > 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (u,) = ctx.saved_tensors
        # Rectangle kernel: gradient passthrough where |u| < 1
        return grad_output * (u.abs() < 1).float()


class JumpReLU(nn.Module):
    """JumpReLU activation with per-feature learnable thresholds (§3.1).

    Two distinct operations per forward pass:
    1. Gate (for z): z = pre_act · (pre_act > θ).detach()
       Hard binary gate. No STE needed — the indicator is detached.
    2. L0 proxy (for loss): l0 = H_STE((pre_act - θ) / ε)
       Rectangle STE routes gradients to θ through the L0 loss.

    Parameters:
        log_threshold: [F] learnable log-thresholds (θ = exp(log_threshold))

    Buffers:
        epsilon: [F] per-feature adaptive bandwidth (set during initialization)
    """

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
        """Apply JumpReLU gating.

        Args:
            pre_act: [B, F] encoder pre-activations (W_enc @ x̃ + b_enc).

        Returns:
            z: [B, F] gated activations (hard gate, no STE).
            gate_mask: [B, F] binary mask (detached).
            l0_probs: [B, F] STE-smoothed Heaviside for L0 gradient routing.
        """
        theta = self.threshold  # [F]

        # Hard gate (detached — no gradient through indicator)
        gate_mask = (pre_act > theta).detach().float()
        z = pre_act * gate_mask

        # L0 proxy with Rectangle STE for threshold gradients
        # u = (pre_act - θ) / ε — normalized distance from threshold
        u = (pre_act - theta) / self.epsilon
        l0_probs = _HeavisideRectangleSTE.apply(u)

        return z, gate_mask, l0_probs

