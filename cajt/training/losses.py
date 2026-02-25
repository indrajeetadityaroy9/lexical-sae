"""Unified CIS circuit losses — completeness, separation, sparsity, feature frequency.

All losses consume W_eff from the forward pass via the shared DLA function
(mechanistic/attribution.py:compute_attribution_tensor) and use the unified
circuit_mask() from circuits/core.py.

All EMA windows are derived from steps_per_epoch via eta = 2/(W+1),
matching the GECO controller's adaptive timescale pattern.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from cajt.core.types import circuit_mask
from cajt.core.attribution import compute_attribution_tensor
from cajt.core.constants import CIRCUIT_TEMPERATURE
from cajt.runtime import DEVICE

_UNDER_ACTIVE_FRACTION = 0.1

class AttributionCentroidTracker(nn.Module):
    """Maintains EMA of per-class mean absolute attribution vectors.

    Used by the separation loss to encourage distinct per-class circuits
    without requiring all classes to appear in every mini-batch.

    EMA decay is derived from steps_per_epoch: decay = 1 - 2/(W+1),
    the same adaptive timescale used by GECO.

    The learned log_margin parameter replaces the fixed margin=0.1
    in the contrastive separation loss.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        steps_per_epoch: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.momentum = 1.0 - 2.0 / (steps_per_epoch + 1)
        _INIT_MARGIN = 0.1
        self.log_margin = nn.Parameter(torch.tensor(math.log(_INIT_MARGIN)))
        # Non-parameter state (not updated by optimizer)
        self.register_buffer("centroids", torch.zeros(num_classes, vocab_size))
        self.register_buffer("_initialized", torch.zeros(num_classes, dtype=torch.bool))

    @property
    def margin(self) -> torch.Tensor:
        """Learned contrastive margin (always positive via exp)."""
        return self.log_margin.exp()

    @torch.no_grad()
    def update(
        self,
        sparse_vector: torch.Tensor,
        W_eff: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue
            class_sparse = sparse_vector[mask]
            class_W_eff = W_eff[mask]
            class_labels = torch.full(
                (class_sparse.shape[0],), c, device=sparse_vector.device,
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
        """Return L2-normalized centroids for cosine similarity computation."""
        norms = self.centroids.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.centroids / norms


class LossNormalizer:
    """Normalizes a loss by its running mean so all losses contribute at unit scale.

    EMA decay is derived from steps_per_epoch: decay = 1 - 2/(W+1),
    adapting the smoothing window to dataset size. This replaces manual
    scaling constants with an adaptive mechanism that automatically
    adjusts to any dataset, sparsity target, or threshold initialization.
    """

    def __init__(self, steps_per_epoch: int):
        self._ema: torch.Tensor | None = None
        self._decay = 1.0 - 2.0 / (steps_per_epoch + 1)

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            val = loss.detach()
            if self._ema is None:
                self._ema = val
            else:
                self._ema = self._ema * self._decay + val * (1.0 - self._decay)
        return loss / self._ema.clamp(min=1e-8)


class FeatureFrequencyTracker:
    """Tracks per-feature activation frequency via EMA.

    Maintains continuous frequency estimates and penalizes features whose
    activation frequency falls below a target derived from sparsity_target.

    EMA decay is derived from steps_per_epoch: decay = 1 - 2/(W+1),
    adapting the smoothing window to dataset size.
    """

    def __init__(self, num_features: int, target_sparsity: float, steps_per_epoch: int):
        self.num_features = num_features
        self.target_freq = target_sparsity
        self.freq_ema = torch.zeros(num_features, device=DEVICE)
        self._decay = 1.0 - 2.0 / (steps_per_epoch + 1)
        self._ramp_steps = steps_per_epoch  # smooth warmup over 1 epoch
        self._step = 0

    @property
    def warmup_weight(self) -> float:
        """Smooth linear ramp from 0→1 over first epoch."""
        return min(1.0, self._step / self._ramp_steps) if self._ramp_steps > 0 else 1.0

    @torch.no_grad()
    def update(self, gate_mask: torch.Tensor) -> None:
        """Update frequency estimates from gate mask.

        Args:
            gate_mask: [..., V] binary gate tensor from JumpReLU.
        """
        flat = gate_mask.reshape(-1, gate_mask.shape[-1])
        batch_freq = flat.float().mean(dim=0)
        self.freq_ema = self.freq_ema * self._decay + batch_freq * (1.0 - self._decay)
        self._step += 1


def compute_completeness_loss(
    sparse_vector: torch.Tensor,
    W_eff: torch.Tensor,
    labels: torch.Tensor,
    classifier_forward_fn: Callable[[torch.Tensor], torch.Tensor],
    full_logits: torch.Tensor,
    circuit_fraction: float = 0.1,
    temperature: float = CIRCUIT_TEMPERATURE,
) -> torch.Tensor:
    """KL-divergence circuit completeness: masked circuit should preserve full distribution.

    Uses circuit_mask() from types.py with the same fraction and temperature
    parameters used throughout the system. Computes KL(full || masked) to
    measure information loss from circuit extraction.
    """
    attr_magnitude = compute_attribution_tensor(sparse_vector, W_eff, labels).abs()
    soft_mask = circuit_mask(attr_magnitude, circuit_fraction, temperature)
    masked_sparse = sparse_vector * soft_mask
    masked_logits = classifier_forward_fn(masked_sparse)

    full_probs = F.softmax(full_logits.detach(), dim=-1)
    masked_log_probs = F.log_softmax(masked_logits, dim=-1)
    return F.kl_div(masked_log_probs, full_probs, reduction="batchmean")


def _centroid_cosine_loss(
    centroid_tracker: AttributionCentroidTracker,
) -> torch.Tensor:
    """Mean pairwise cosine similarity between class centroids (early training)."""
    initialized_mask = centroid_tracker._initialized
    num_init = int(initialized_mask.sum().item())
    if num_init < 2:
        return torch.tensor(0.0, device=DEVICE)

    centroids = centroid_tracker.get_normalized_centroids()
    active_centroids = centroids[initialized_mask]

    sim_matrix = active_centroids @ active_centroids.T
    n = active_centroids.shape[0]
    mask = torch.triu(torch.ones(n, n, device=DEVICE, dtype=torch.bool), diagonal=1)
    return sim_matrix[mask].mean()


def compute_separation_loss(
    centroid_tracker: AttributionCentroidTracker,
    sparse_vector: torch.Tensor | None = None,
    W_eff: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contrastive separation with hard negative mining and learned margin.

    Each sample's attribution should be closer to its class centroid than to
    the nearest other-class centroid, with a learned margin. Falls back to
    centroid-only cosine loss if per-sample data is not provided or fewer
    than 2 centroids are initialized.
    """
    initialized_mask = centroid_tracker._initialized
    num_init = int(initialized_mask.sum().item())
    if num_init < 2 or sparse_vector is None or W_eff is None or labels is None:
        return _centroid_cosine_loss(centroid_tracker)

    attr = compute_attribution_tensor(sparse_vector, W_eff, labels).abs()
    attr_norm = F.normalize(attr, dim=-1)

    centroids_norm = centroid_tracker.get_normalized_centroids()
    sim = attr_norm @ centroids_norm.T  # [B, C]

    pos_sim = sim.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]

    # Hard negative: highest similarity to any OTHER class
    neg_mask = torch.ones_like(sim, dtype=torch.bool)
    neg_mask.scatter_(1, labels.unsqueeze(1), False)
    neg_sim = sim.masked_fill(~neg_mask, -1e9).max(dim=1).values  # [B]

    return F.relu(neg_sim - pos_sim + centroid_tracker.margin).mean()


def compute_gate_sparsity_loss(l0_probs: torch.Tensor) -> torch.Tensor:
    """Differentiable L0 penalty: mean P(z_j > θ_j) over all dimensions.

    With JumpReLU, l0_probs = σ((z - θ) / ε) provides a smooth approximation
    to the expected number of active features, with gradients flowing to θ
    via the sigmoid STE. Minimizing this directly encourages the model to
    raise thresholds θ_j, closing gates on low-importance features.
    """
    return l0_probs.mean()


def compute_frequency_penalty(
    freq_tracker: FeatureFrequencyTracker,
    activation_module: torch.nn.Module,
) -> torch.Tensor:
    """Feature frequency penalty for under-active feature resurrection.

    Penalizes high log_threshold values for features whose activation frequency
    falls below _UNDER_ACTIVE_FRACTION of the target sparsity, pushing thresholds
    down to make reactivation easier. Applies a smooth linear warmup ramp over
    the first epoch (derived from steps_per_epoch) instead of a hard cutoff.

    Over-active features are already handled by the L0 sparsity loss.

    Args:
        freq_tracker: Tracker with per-feature frequency estimates.
        activation_module: JumpReLUGate module with log_threshold parameter.
    """
    weight = freq_tracker.warmup_weight
    if weight == 0.0:
        return torch.tensor(0.0, device=DEVICE)

    under_active = (freq_tracker.freq_ema < freq_tracker.target_freq * _UNDER_ACTIVE_FRACTION).detach()
    if not under_active.any():
        return torch.tensor(0.0, device=DEVICE)

    log_thresholds = activation_module.log_threshold  # [V]
    return weight * log_thresholds[under_active].mean()
