"""Constraint evaluation for the augmented Lagrangian (§3.2, §3.3, §6.1)."""

from __future__ import annotations

import torch
from torch import Tensor

from src.whitening.whitener import SoftZCAWhitener


# --- Penalty function (§3.3) ---


def al_cole_psi(g: Tensor, y: Tensor) -> Tensor:
    """AL-CoLe smooth penalty Ψ(g, y) = (max(0, 2g + y)² - y²) / 4.

    Properties:
    - Ψ(g, y) ≤ 0 when g ≤ 0 (inactive constraints contribute non-positively).
    - ∇_g Ψ = max(0, 2g+y) / 2 (everywhere continuously differentiable).
    - Ψ(0, y) = 0 when y > 0 (smooth transition at boundary).

    Eliminates the kink at g=0 in the standard quadratic penalty ρ/2·max(0,g)²,
    which is critical for clean ESO signals and uncorrupted Adam momentum.

    Args:
        g: Constraint violation (EMA-filtered ṽ_fast, not raw).
        y: λ_i / ρ_i (scaled dual variable).

    Returns:
        Penalty value (scalar or same shape as inputs).
    """
    inner = 2.0 * g + y
    return (torch.clamp(inner, min=0.0).pow(2) - y.pow(2)) / 4.0


# --- Lagrangian assembly (§3.3, §6.1) ---


def compute_augmented_lagrangian(
    l0_corr: Tensor,
    v_fast: Tensor,
    lambdas: Tensor,
    rhos: Tensor,
) -> Tensor:
    """Assemble the SPALF augmented Lagrangian.

    𝓛(θ, λ, ρ) = ℓ₀^corr(θ) + Σᵢ ρᵢ · Ψ(ṽ_fast_i, λᵢ/ρᵢ)

    IMPORTANT: Takes EMA-filtered violations (ṽ_fast), NOT raw per-batch violations.
    Data flow: raw v_i → DualRateEMA.update() → ṽ_fast → this function.

    Args:
        l0_corr: Scalar discretization-corrected sparsity objective.
        v_fast: [3] EMA-filtered constraint violations (faith, drift, ortho).
        lambdas: [3] dual variables from ADRC controller.
        rhos: [3] adaptive penalty coefficients from CAPU.

    Returns:
        Scalar augmented Lagrangian loss for backward pass.
    """
    y = lambdas / rhos  # [3] scaled dual variables
    psi_values = al_cole_psi(v_fast, y)  # [3]
    constraint_penalty = (rhos * psi_values).sum()
    return l0_corr + constraint_penalty


# --- Violation computations (§3.2) ---


def compute_faithfulness_violation(
    x: Tensor,
    x_hat: Tensor,
    whitener: SoftZCAWhitener,
    tau_faith: float,
) -> Tensor:
    """C1 — Faithfulness: whitened-space MSE (§3.2).

    g_faith = (1/N) Σ ‖x̃ - W_white(x̂ - μ)‖² - τ_faith
            = (1/N) Σ ‖W_white(x - x̂)‖² - τ_faith
            = (1/N) Σ (x - x̂)^T Σ_α^{-1} (x - x̂) - τ_faith

    Uses cached precision matrix to avoid materializing whitened reconstruction.

    Args:
        x: [B, d] raw activations (original space).
        x_hat: [B, d] SAE reconstruction (original space).
        whitener: Soft-ZCA whitener with cached precision matrix.
        tau_faith: Faithfulness threshold = (1 - R²_target) * d.

    Returns:
        Scalar violation (positive = constraint violated).
    """
    diff = x - x_hat  # [B, d]
    mahal_sq = whitener.compute_mahalanobis_sq(diff)  # [B]
    return mahal_sq.mean() - tau_faith


def compute_faithfulness_violation_phase2(
    x: Tensor,
    x_hat: Tensor,
    kl_div: Tensor,
    tau_faith: float,
    kl_running_mean: Tensor,
) -> Tensor:
    """Phase 2 faithfulness: scale-normalized MSE/KL blend (§6.3).

    v_faith^(2) = 0.5 · (‖x - x̂‖² - τ_faith) / τ_faith + 0.5 · D_KL / D̄_KL

    Note: MSE is in ORIGINAL space (not whitened), unlike Phase 1.

    Args:
        x: [B, d] raw activations.
        x_hat: [B, d] SAE reconstruction.
        kl_div: Scalar KL(p_orig ‖ p_patched).
        tau_faith: Faithfulness threshold.
        kl_running_mean: Running mean of KL divergence (for normalization).

    Returns:
        Scalar violation.
    """
    mse = (x - x_hat).pow(2).sum(dim=1).mean()
    mse_normalized = (mse - tau_faith) / tau_faith
    kl_normalized = kl_div / kl_running_mean
    return 0.5 * mse_normalized + 0.5 * kl_normalized


def compute_drift_violation(
    W_dec_A: Tensor,
    W_vocab: Tensor,
    tau_drift: float,
) -> Tensor:
    """C2 — Vocabulary Drift: Frobenius norm in original space (§3.2).

    g_drift = ‖W_dec^(A) - W_vocab‖²_F - τ_drift

    Args:
        W_dec_A: [d, V] anchored decoder columns.
        W_vocab: [d, V] vocabulary/unembedding matrix.
        tau_drift: Drift threshold = δ_drift² · ‖W_vocab‖²_F.

    Returns:
        Scalar violation.
    """
    return (W_dec_A - W_vocab).pow(2).sum() - tau_drift


def compute_orthogonality_violation(
    z: Tensor,
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    tau_ortho: float,
) -> Tensor:
    """C3 — Co-Activation Orthogonality: pairwise cos² of active decoder columns (§3.2).

    g_ortho = (1/|B|) Σ_b [ (1/(|A_b|²-|A_b|)) Σ_{i≠j ∈ A_b} cos²(w_i, w_j) ] - τ_ortho

    where A_b = {j : z_{b,j} > 0} spans both strata.

    Uses Triton-accelerated kernel for fused per-sample pairwise computation.

    Args:
        z: [B, F] sparse codes (used to determine active sets).
        W_dec_A: [d, V] anchored decoder.
        W_dec_B: [d, F-V] free decoder.
        tau_ortho: Orthogonality threshold = c_ortho / d.

    Returns:
        Scalar violation.
    """
    from src.kernels.ortho_kernel import compute_ortho_triton

    return compute_ortho_triton(z, W_dec_A, W_dec_B, tau_ortho)
