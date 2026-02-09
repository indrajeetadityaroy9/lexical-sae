"""SequenceCircuitState and masking for per-position sparse sequence models.

Parallel to circuits/core.py (CircuitState) but for [B, L, V] representations.
W_eff/b_eff are omitted to avoid materializing [B, L, C, V] during training;
they are computed on-demand for selected positions only.
"""

from typing import NamedTuple

import torch

from splade.circuits.core import circuit_mask


class SequenceCircuitState(NamedTuple):
    """Produced by LexicalSAE forward pass (sequence_labeling mode)."""

    token_logits: torch.Tensor    # [B, L, C]
    sparse_sequence: torch.Tensor  # [B, L, V]
    attention_mask: torch.Tensor   # [B, L]


def sequence_circuit_mask(
    attribution_flat: torch.Tensor,
    circuit_fraction: float,
    temperature: float,
) -> torch.Tensor:
    """Apply circuit_mask to flattened [N, V] attribution tensors.

    Thin wrapper around core.circuit_mask for use with gathered valid positions.

    Args:
        attribution_flat: [N, V] absolute attribution magnitudes.
        circuit_fraction: fraction of dims to retain (e.g. 0.1 = top 10%).
        temperature: sigmoid temperature (higher = sharper).

    Returns:
        [N, V] mask in [0, 1].
    """
    return circuit_mask(attribution_flat, circuit_fraction, temperature)
