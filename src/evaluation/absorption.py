"""Feature absorption: alignment of free decoder features with vocabulary directions."""

import torch
from torch import Tensor


@torch.no_grad()
def feature_absorption_rate(
    W_dec_B: Tensor,
    W_vocab: Tensor,
) -> dict[str, float]:
    """Estimate absorption of vocabulary directions into free decoder features."""
    B_norm = W_dec_B / W_dec_B.norm(dim=0, keepdim=True)
    V_norm = W_vocab / W_vocab.norm(dim=0, keepdim=True)

    cos_matrix = B_norm.T @ V_norm
    max_cos_per_free = cos_matrix.abs().max(dim=1).values

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
