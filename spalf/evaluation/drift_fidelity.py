"""Drift fidelity: cosine similarity between anchored decoder and vocabulary."""

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def drift_fidelity(
    W_dec_A: Tensor,
    W_vocab: Tensor,
) -> dict[str, float]:
    """Compute column-wise cosine similarity between anchored decoder and vocabulary."""
    cos_sim = F.cosine_similarity(W_dec_A, W_vocab, dim=0)

    return {
        "mean": cos_sim.mean().item(),
        "min": cos_sim.min().item(),
        "max": cos_sim.max().item(),
        "frac_above_099": (cos_sim > 0.99).float().mean().item(),
        "frac_above_095": (cos_sim > 0.95).float().mean().item(),
    }
