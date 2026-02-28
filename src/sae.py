"""Stratified Sparse Autoencoder: V anchored + (F-V) free decoder columns.

JumpReLU gating with learnable log-thresholds and Moreau bandwidth (γ).
Forward/backward logic lives in the fused Triton kernel (src/kernel.py).
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.kernel import FusedJumpReLUFunction


class StratifiedSAE(nn.Module):
    """Sparse autoencoder with anchored and free decoder strata."""

    def __init__(self, d: int, F: int, V: int) -> None:
        super().__init__()
        self.d = d
        self.F = F
        self.V = V
        self.F_free = F - V

        self.W_enc = nn.Parameter(torch.empty(F, d))
        self.b_enc = nn.Parameter(torch.zeros(F))

        self.W_dec_A = nn.Parameter(torch.empty(d, V))
        self.W_dec_B = nn.Parameter(torch.empty(d, self.F_free))

        self.b_dec = nn.Parameter(torch.zeros(d))

        self.log_threshold = nn.Parameter(torch.zeros(F))
        self.register_buffer("gamma", torch.ones(F))

        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec_A)
        nn.init.xavier_uniform_(self.W_dec_B)

    def decode(self, z: Tensor) -> Tensor:
        """Decode to original space."""
        z_A = z[:, : self.V]
        z_B = z[:, self.V :]
        return z_A @ self.W_dec_A.T + z_B @ self.W_dec_B.T + self.b_dec

    def forward(
        self, x_tilde: Tensor, lambda_disc: float = 0.0
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused encode/decode via Triton kernel with optional discretization correction."""
        pre_act = x_tilde @ self.W_enc.T + self.b_enc
        z, gate_mask, l0_probs, disc_raw = FusedJumpReLUFunction.apply(
            pre_act, self.log_threshold, self.gamma, lambda_disc
        )
        x_hat = self.decode(z)
        return x_hat, z, gate_mask, l0_probs, disc_raw

    @torch.no_grad()
    def recalibrate_gamma(self, pre_act: Tensor, c_epsilon: float) -> None:
        """Recalibrate Moreau bandwidth from current pre-activation statistics."""
        q75 = torch.quantile(pre_act, 0.75, dim=0)
        q25 = torch.quantile(pre_act, 0.25, dim=0)
        iqr = q75 - q25
        self.gamma.copy_((c_epsilon * iqr).pow(2) / 2.0)

    @torch.no_grad()
    def normalize_free_decoder(self) -> None:
        """Project free decoder columns to unit norm. Called after each optimizer step."""
        norms = self.W_dec_B.norm(dim=0, keepdim=True)
        self.W_dec_B.div_(norms)
