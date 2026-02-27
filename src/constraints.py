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


def compute_orthogonality_violation(
    z: Tensor,
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    tau_ortho: float,
) -> Tensor:
    """Co-activation orthogonality violation from Triton kernel output."""
    from src.kernels.ortho_kernel import compute_ortho_triton

    return compute_ortho_triton(z, W_dec_A, W_dec_B, tau_ortho)
