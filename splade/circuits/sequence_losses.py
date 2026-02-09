"""Token-level circuit losses for sequence labelling (NER).

All losses use reduction='mean' over N (total valid tokens), NOT B (batch
size), keeping magnitudes comparable to classification (~0.5) and preventing
GECO lambda destabilization.

The _gather_valid_positions bridge flattens [B,L,V] → [N,V] so existing
compute_attribution_tensor and circuit_mask work unchanged.
"""

import math
from typing import Callable

import torch
import torch.nn.functional as F

from splade.circuits.core import circuit_mask
from splade.data.ner_loader import IGNORE_INDEX
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.training.constants import CIRCUIT_TEMPERATURE
from splade.utils.cuda import DEVICE

# Centroid EMA: 20-step effective window → decay ≈ 0.905
_CENTROID_EMA_WINDOW = 20
_CENTROID_EMA_DECAY = 1.0 - 2.0 / (_CENTROID_EMA_WINDOW + 1)


def _gather_valid_positions(
    sparse_sequence: torch.Tensor,
    token_labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten [B,L,V] to [N,V], filtering padding and IGNORE_INDEX positions.

    This is the critical bridge between the sequence model and existing
    [N,V]-based attribution/masking functions.

    Args:
        sparse_sequence: [B, L, V] per-position sparse representations.
        token_labels: [B, L] per-token labels (IGNORE_INDEX for skip).
        attention_mask: [B, L] attention mask (0 for padding).

    Returns:
        sparse_flat: [N, V] valid position sparse vectors.
        labels_flat: [N] valid position labels.
    """
    # Valid = attended AND not IGNORE_INDEX
    valid = attention_mask.bool() & (token_labels != IGNORE_INDEX)
    sparse_flat = sparse_sequence[valid]   # [N, V]
    labels_flat = token_labels[valid]      # [N]
    return sparse_flat, labels_flat


class TokenAttributionCentroidTracker:
    """EMA centroids per NER tag for separation loss.

    Handles the "empty class" edge case: if a tag has no instances in
    the current batch, its centroid is not updated and its EMA state
    is preserved from previous batches.
    """

    def __init__(self, num_tags: int, vocab_size: int):
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.momentum = _CENTROID_EMA_DECAY
        self.centroids = torch.zeros(num_tags, vocab_size, device=DEVICE)
        self._initialized = torch.zeros(num_tags, dtype=torch.bool, device=DEVICE)

    @torch.no_grad()
    def update(
        self,
        sparse_flat: torch.Tensor,
        W_eff: torch.Tensor,
        labels_flat: torch.Tensor,
    ) -> None:
        """Update centroids for tags present in this batch.

        Tags absent from the batch are silently skipped (no NaN risk).

        Args:
            sparse_flat: [N, V] valid position sparse vectors.
            W_eff: [N, C, V] effective weight matrix.
            labels_flat: [N] per-position labels.
        """
        for c in range(self.num_tags):
            mask = labels_flat == c
            if not mask.any():
                continue
            class_sparse = sparse_flat[mask]
            class_W_eff = W_eff[mask]
            class_labels = torch.full(
                (class_sparse.shape[0],), c, device=sparse_flat.device,
            )
            attr = (
                compute_attribution_tensor(class_sparse, class_W_eff, class_labels)
                .abs()
                .mean(dim=0)
            )
            if self._initialized[c]:
                self.centroids[c].lerp_(attr, 1.0 - self.momentum)
            else:
                self.centroids[c].copy_(attr)
                self._initialized[c] = True

    def get_normalized_centroids(self) -> torch.Tensor:
        """Return L2-normalized centroids for cosine similarity."""
        norms = self.centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.centroids / norms


def compute_token_completeness_loss(
    sparse_flat: torch.Tensor,
    W_eff: torch.Tensor,
    labels_flat: torch.Tensor,
    classifier_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    circuit_fraction: float = 0.1,
    temperature: float = CIRCUIT_TEMPERATURE,
) -> torch.Tensor:
    """Token-level completeness: circuit-masked predictions must match full.

    Returns scalar averaged over N valid tokens.

    Args:
        sparse_flat: [N, V] valid position sparse vectors.
        W_eff: [N, C, V] effective weight matrix.
        labels_flat: [N] labels.
        classifier_logits_fn: fn(sparse [N,V]) -> logits [N,C].
        circuit_fraction: fraction of dims to retain.
        temperature: circuit mask temperature.
    """
    attr_magnitude = compute_attribution_tensor(sparse_flat, W_eff, labels_flat).abs()
    soft_mask = circuit_mask(attr_magnitude, circuit_fraction, temperature)
    masked_sparse = sparse_flat * soft_mask
    masked_logits = classifier_logits_fn(masked_sparse)
    return F.cross_entropy(masked_logits, labels_flat)


def compute_token_separation_loss(
    centroid_tracker: TokenAttributionCentroidTracker,
) -> torch.Tensor:
    """Cosine similarity between initialized tag centroids.

    Only computes similarity between centroids that have been initialized
    (i.e., have seen at least one batch with that tag). Returns 0.0 if
    fewer than 2 centroids are initialized.

    Returns scalar averaged over centroid pairs.
    """
    initialized_mask = centroid_tracker._initialized
    num_init = int(initialized_mask.sum().item())
    if num_init < 2:
        return torch.tensor(0.0, device=DEVICE)

    # Select only initialized centroids
    centroids = centroid_tracker.get_normalized_centroids()
    active_centroids = centroids[initialized_mask]  # [K, V]

    sim_matrix = active_centroids @ active_centroids.T
    n = active_centroids.shape[0]
    mask = torch.triu(torch.ones(n, n, device=DEVICE, dtype=torch.bool), diagonal=1)
    return sim_matrix[mask].mean()


def compute_token_sharpness_loss(
    sparse_flat: torch.Tensor,
    W_eff: torch.Tensor,
    labels_flat: torch.Tensor,
) -> torch.Tensor:
    """L1 (FLOPS) sparsity loss with L0 proxy.

    Replaces Hoyer sparsity with constant-gradient L1 pressure on
    the sparse activations directly. L1 gradient is +/-1 regardless
    of magnitude, providing steady sparsity pressure that avoids the
    vanishing-gradient deadlock of Hoyer (L1/L2 ratio).

    The L0 proxy (sigmoid approximation) encourages small activations
    to snap to exactly zero.

    Both terms are normalized by vocab_size to keep the loss ~O(0.1),
    comparable to completeness and separation losses.

    Args:
        sparse_flat: [N, V] valid position sparse vectors.
        W_eff: [N, C, V] unused (kept for API compatibility).
        labels_flat: [N] unused (kept for API compatibility).
    """
    V = sparse_flat.shape[-1]
    # L1: constant-gradient pressure on activation magnitudes
    l1_loss = sparse_flat.sum(dim=-1).mean() / V
    # L0 proxy: smooth count of non-zeros, drives small values to zero
    l0_proxy = torch.sigmoid(sparse_flat * 10.0).sum(dim=-1).mean() / V
    return l1_loss + 0.1 * l0_proxy
