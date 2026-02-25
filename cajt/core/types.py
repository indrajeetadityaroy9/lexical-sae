"""CircuitState and unified masking for CIS.

CircuitState wraps the forward pass outputs as a NamedTuple (torch.compile
compatible without pytree registration). circuit_mask() provides a single
temperature-parameterized masking function used by both training (soft) and
evaluation (hard).
"""

from typing import NamedTuple

import torch


class CircuitState(NamedTuple):
    """Produced by the forward pass, consumed by losses and metrics."""

    logits: torch.Tensor  # [B, C]
    sparse_vector: torch.Tensor  # [B, V]
    W_eff: torch.Tensor  # [B, C, V]
    b_eff: torch.Tensor  # [B, C]


def circuit_mask(
    attribution: torch.Tensor,
    circuit_fraction: float,
    temperature: float,
) -> torch.Tensor:
    """Unified soft/hard circuit mask via temperature-parameterized sigmoid.

    Training uses temperature ~10 (soft, differentiable).
    Evaluation uses temperature ~1e6 (hard, effectively binary).
    Both use the SAME circuit_fraction, ensuring consistent circuit boundaries.

    Args:
        attribution: [B, V] absolute attribution magnitudes for the target class.
        circuit_fraction: fraction of dimensions to retain (e.g. 0.1 = top 10%).
        temperature: sigmoid temperature. Higher = sharper mask.

    Returns:
        [B, V] mask in [0, 1].
    """
    n = attribution.shape[-1]
    k = max(1, int(circuit_fraction * n))
    kth_index = max(1, n - k)
    threshold = torch.kthvalue(attribution, kth_index, dim=-1).values
    return torch.sigmoid(temperature * (attribution - threshold.unsqueeze(-1)))


def circuit_mask_by_mass(
    attribution: torch.Tensor,
    mass_fraction: float = 0.9,
) -> torch.Tensor:
    """Select minimum features capturing `mass_fraction` of total attribution mass.

    Data-adaptive replacement for fixed circuit_fraction. Sorts attributions
    descending, selects until cumulative sum >= mass_fraction * total.
    Sparse models get small circuits, dense models get larger ones.

    Args:
        attribution: [B, V] attribution magnitudes (will take abs internally).
        mass_fraction: fraction of total mass to capture (e.g. 0.9 = 90%).

    Returns:
        [B, V] binary mask.
    """
    abs_attr = attribution.abs()
    total_mass = abs_attr.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    sorted_attr, sorted_idx = abs_attr.sort(dim=-1, descending=True)
    cumulative = sorted_attr.cumsum(dim=-1) / total_mass
    # Include all features up to and including the one that crosses the threshold
    include_mask = cumulative <= mass_fraction
    # Shift right by 1 to include the crossing feature itself
    include_mask = torch.cat([
        torch.ones_like(include_mask[..., :1]),
        include_mask[..., :-1],
    ], dim=-1)
    # Scatter back to original indices
    mask = torch.zeros_like(abs_attr)
    mask.scatter_(-1, sorted_idx, include_mask.float())
    return mask
