"""SPALF evaluation suite."""

import torch
from torch import Tensor

from src.evaluation.downstream_loss import evaluate_downstream_loss
from src.evaluation.sparsity_frontier import compute_sparsity_frontier


@torch.no_grad()
def drift_fidelity(
    W_dec_A: Tensor,
    W_vocab: Tensor,
) -> dict[str, float]:
    """Compute cosine similarity between anchored decoder and vocabulary columns.

    Measures how well anchored features preserve their vocabulary alignment.

    Args:
        W_dec_A: [d, V] anchored decoder columns.
        W_vocab: [d, V] original vocabulary/unembedding matrix.

    Returns:
        Dict with mean, min, max cosine similarity, and fraction > 0.99.
    """
    # Column-wise cosine similarity
    A_norm = W_dec_A / W_dec_A.norm(dim=0, keepdim=True)
    V_norm = W_vocab / W_vocab.norm(dim=0, keepdim=True)
    cos_sim = (A_norm * V_norm).sum(dim=0)  # [V]

    return {
        "mean": cos_sim.mean().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
        "frac_above_099": (cos_sim > 0.99).float().mean().item(),
        "frac_above_095": (cos_sim > 0.95).float().mean().item(),
    }


@torch.no_grad()
def feature_absorption_rate(
    W_dec_A: Tensor,
    W_dec_B: Tensor,
    W_vocab: Tensor,
    top_k: int = 10,
) -> dict[str, float]:
    """Estimate feature absorption rate (per arXiv:2409.14507).

    Checks if free decoder columns have high cosine similarity to vocabulary
    directions, indicating absorption of vocabulary features into free stratum.

    Args:
        W_dec_A: [d, V] anchored decoder.
        W_dec_B: [d, F-V] free decoder.
        W_vocab: [d, V] vocabulary matrix.
        top_k: Number of top alignments to report.

    Returns:
        Dict with absorption statistics.
    """
    # Normalize all column sets
    B_norm = W_dec_B / W_dec_B.norm(dim=0, keepdim=True)
    V_norm = W_vocab / W_vocab.norm(dim=0, keepdim=True)

    # Max cosine similarity of each free feature to any vocabulary direction
    # [F-V, V] = B_norm^T @ V_norm
    cos_matrix = B_norm.T @ V_norm  # [F-V, V]
    max_cos_per_free = cos_matrix.abs().max(dim=1).values  # [F-V]

    # Absorption: free features that are very aligned with vocabulary
    n_absorbed_099 = (max_cos_per_free > 0.99).sum().item()
    n_absorbed_095 = (max_cos_per_free > 0.95).sum().item()
    n_absorbed_090 = (max_cos_per_free > 0.90).sum().item()
    n_free = W_dec_B.shape[1]

    return {
        "n_free": n_free,
        "n_absorbed_099": n_absorbed_099,
        "n_absorbed_095": n_absorbed_095,
        "n_absorbed_090": n_absorbed_090,
        "absorption_rate_095": n_absorbed_095 / n_free,
        "max_alignment": max_cos_per_free.max().item(),
        "mean_alignment": max_cos_per_free.mean().item(),
    }


__all__ = [
    "evaluate_downstream_loss",
    "compute_sparsity_frontier",
    "drift_fidelity",
    "feature_absorption_rate",
]
