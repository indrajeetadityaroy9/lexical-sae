"""Stratified Sparse Autoencoder: V anchored + (F-V) free decoder columns (§2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.kernels.jumprelu_kernel import FusedJumpReLUFunction
from src.model.jumprelu import JumpReLU


class StratifiedSAE(nn.Module):
    """SAE with stratified decoder architecture.

    Decoder is partitioned into two strata:
    - Stratum A (Anchored): V features initialized to W_vocab (unembedding).
      Subject to drift constraint C2.
    - Stratum B (Free): F-V features, random unit-norm columns.
      Subject to orthogonality constraint C3.

    Encoder is a single [F, d] matrix (not partitioned). Initialization
    uses matched-filter (§2.3) for anchored rows and Gram-Schmidt for free rows.

    Uses split decode (no concatenation) to avoid 1GB memcpy per step.
    """

    def __init__(self, d: int, F: int, V: int) -> None:
        super().__init__()
        self.d = d
        self.F = F
        self.V = V
        self.F_free = F - V

        # Encoder: [F, d]
        self.W_enc = nn.Parameter(torch.empty(F, d))
        self.b_enc = nn.Parameter(torch.zeros(F))

        # Stratified decoder
        self.W_dec_A = nn.Parameter(torch.empty(d, V))  # Anchored
        self.W_dec_B = nn.Parameter(torch.empty(d, self.F_free))  # Free

        # Decoder bias
        self.b_dec = nn.Parameter(torch.zeros(d))

        # JumpReLU activation
        self.jumprelu = JumpReLU(F)

        # Default initialization (overridden by initialization.py)
        self._default_init()

    def _default_init(self) -> None:
        """Default Xavier initialization (overridden by initialize_sae)."""
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec_A)
        nn.init.xavier_uniform_(self.W_dec_B)

    def encode(self, x_tilde: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode whitened activations.

        Args:
            x_tilde: [B, d] whitened activations.

        Returns:
            z: [B, F] sparse codes.
            gate_mask: [B, F] binary activation mask.
            l0_probs: [B, F] STE-smoothed L0 for gradient routing.
        """
        pre_act = x_tilde @ self.W_enc.T + self.b_enc  # [B, F]
        return self.jumprelu(pre_act)

    def decode(self, z: Tensor) -> Tensor:
        """Decode to original space using split matmul (no concat).

        Args:
            z: [B, F] sparse codes.

        Returns:
            x_hat: [B, d] reconstruction in original (unwhitened) space.
        """
        z_A = z[:, : self.V]  # [B, V]
        z_B = z[:, self.V :]  # [B, F-V]
        return z_A @ self.W_dec_A.T + z_B @ self.W_dec_B.T + self.b_dec

    def forward(
        self, x_tilde: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass: encode → decode.

        Args:
            x_tilde: [B, d] whitened activations.

        Returns:
            x_hat: [B, d] reconstruction in original space.
            z: [B, F] sparse codes.
            gate_mask: [B, F] binary activation mask.
            l0_probs: [B, F] STE-smoothed L0.
        """
        z, gate_mask, l0_probs = self.encode(x_tilde)
        x_hat = self.decode(z)
        return x_hat, z, gate_mask, l0_probs

    def forward_fused(
        self, x_tilde: Tensor, lambda_disc: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Fused forward: encode + decode with discretization in a single JumpReLU pass.

        Uses the Triton fused kernel (gate + L0 + disc in one pass).

        Args:
            x_tilde: [B, d] whitened activations.
            lambda_disc: Current discretization penalty weight.

        Returns:
            x_hat: [B, d] reconstruction in original space.
            z: [B, F] sparse codes.
            gate_mask: [B, F] binary activation mask.
            l0_probs: [B, F] STE-smoothed L0.
            disc_raw: [B, F] per-element discretization correction.
        """
        pre_act = x_tilde @ self.W_enc.T + self.b_enc  # [B, F]
        z, gate_mask, l0_probs, disc_raw = FusedJumpReLUFunction.apply(
            pre_act, self.jumprelu.log_threshold, self.jumprelu.epsilon, lambda_disc
        )
        x_hat = self.decode(z)
        return x_hat, z, gate_mask, l0_probs, disc_raw

    @torch.no_grad()
    def normalize_free_decoder(self) -> None:
        """Project free decoder columns to unit norm. Called after each optimizer step."""
        norms = self.W_dec_B.norm(dim=0, keepdim=True)
        self.W_dec_B.div_(norms)

