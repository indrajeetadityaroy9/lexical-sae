"""Constraint violation functions: faithfulness, drift, orthogonality."""

import torch
from torch import Tensor

from src.whitening.whitener import SoftZCAWhitener


def compute_faithfulness_violation(
    x: Tensor,
    x_hat: Tensor,
    whitener: SoftZCAWhitener,
    tau_faith: float,
) -> Tensor:
    """Whitened-metric faithfulness violation: E[||x - x_hat||_M^2] - tau."""
    diff = x - x_hat
    mahal_sq = whitener.compute_mahalanobis_sq(diff)
    return mahal_sq.mean() - tau_faith


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
    gamma: Tensor,
) -> Tensor:
    """Differentiable co-activation orthogonality violation."""
    W_dec = torch.cat([W_dec_A, W_dec_B], dim=1)
    col_norms = W_dec.norm(dim=0, keepdim=True)
    W_hat = W_dec / col_norms

    G = W_hat.T @ W_hat
    G_sq = G.pow(2)

    # Masking inactive features avoids injecting orthogonality gradients through dead units.
    temperature = (2.0 * gamma.mean()).sqrt()
    active = torch.sigmoid(z / temperature) * (z > 0).float()

    C = active.T @ active

    weighted = G_sq * C
    numerator = weighted.sum() - weighted.diagonal().sum()
    denominator = C.sum() - C.diagonal().sum()

    return numerator / denominator - tau_ortho
