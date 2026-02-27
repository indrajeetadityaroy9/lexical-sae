"""Constraint and augmented-Lagrangian terms."""

from __future__ import annotations

import torch
from torch import Tensor

from src.whitening.whitener import SoftZCAWhitener


def al_cole_psi(g: Tensor, y: Tensor) -> Tensor:
    """AL-CoLe smooth penalty: Ψ(g, y) = (max(0, 2g + y)^2 - y^2) / 4."""
    inner = 2.0 * g + y
    return (torch.clamp(inner, min=0.0).pow(2) - y.pow(2)) / 4.0


def compute_augmented_lagrangian(
    l0_corr: Tensor,
    v_fast: Tensor,
    lambdas: Tensor,
    rhos: Tensor,
) -> Tensor:
    """Compute l0_corr + sum_i rho_i * Ψ(v_fast_i, lambda_i / rho_i)."""
    y = lambdas / rhos
    psi_values = al_cole_psi(v_fast, y)
    constraint_penalty = (rhos * psi_values).sum()
    return l0_corr + constraint_penalty


def compute_faithfulness_violation(
    x: Tensor,
    x_hat: Tensor,
    whitener: SoftZCAWhitener,
    tau_faith: float,
) -> Tensor:
    """Phase-1 faithfulness in the whitened metric."""
    diff = x - x_hat
    mahal_sq = whitener.compute_mahalanobis_sq(diff)
    return mahal_sq.mean() - tau_faith


def compute_faithfulness_violation_phase2(
    x: Tensor,
    x_hat: Tensor,
    kl_div: Tensor,
    tau_faith: float,
    kl_running_mean: Tensor,
) -> Tensor:
    """Phase-2 faithfulness blend of normalized MSE and KL."""
    mse = (x - x_hat).pow(2).sum(dim=1).mean()
    mse_normalized = (mse - tau_faith) / tau_faith
    kl_normalized = kl_div / kl_running_mean
    return 0.5 * mse_normalized + 0.5 * kl_normalized


def compute_drift_violation(
    W_dec_A: Tensor,
    W_vocab: Tensor,
    tau_drift: float,
) -> Tensor:
    """Anchored decoder drift: ||W_dec_A - W_vocab||_F^2 - tau_drift."""
    return (W_dec_A - W_vocab).pow(2).sum() - tau_drift


def compute_frame_energy(W_dec_B: Tensor, d: int) -> Tensor:
    """Frame energy: deviation of free decoder from tight frame structure.

    R_frame = || W_B W_B^T / trace(W_B W_B^T) - (1/d) I_d ||_F^2

    Equals zero when W_dec_B forms a tight frame (all eigenvalues of the
    normalized frame operator equal 1/d). From spectral superposition theory
    (arXiv:2602.02224, Thm 3).
    """
    frame_op = W_dec_B @ W_dec_B.T  # [d, d]
    frame_op_normalized = frame_op / frame_op.diagonal().sum()
    target = torch.eye(d, device=W_dec_B.device) / d
    return (frame_op_normalized - target).pow(2).sum()


def compute_orthogonality_violation(
    z: Tensor,
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    tau_ortho: float,
) -> Tensor:
    """Differentiable co-activation orthogonality violation (replaces Triton kernel).

    Computes mean pairwise cos² between active decoder columns, fully differentiable
    through z (gate activations) and W_dec columns.
    """
    W_dec = torch.cat([W_dec_A, W_dec_B], dim=1)  # [d, F]
    col_norms = W_dec.norm(dim=0, keepdim=True)
    W_hat = W_dec / col_norms  # [d, F]

    # Gram matrix of normalized columns: G[i,j] = cos(angle(d_i, d_j))
    G = W_hat.T @ W_hat  # [F, F]
    G_sq = G.pow(2)

    # Weight by co-activation: soft gates from z
    active = (z > 0).float()  # [B, F]
    # Pairwise co-activation counts: C[i,j] = sum_b active[b,i] * active[b,j]
    C = active.T @ active  # [F, F]

    # Subtract diagonal (self-pairs) instead of allocating F×F identity mask
    weighted = G_sq * C
    numerator = weighted.sum() - weighted.diagonal().sum()
    denominator = C.sum() - C.diagonal().sum()

    return numerator / denominator - tau_ortho
