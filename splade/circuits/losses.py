"""Unified CIS circuit losses — completeness, separation, sparsity, feature frequency.

All losses consume W_eff from the forward pass via the shared DLA function
(mechanistic/attribution.py:compute_attribution_tensor) and use the unified
circuit_mask() from circuits/core.py.

Centroid EMA uses a 20-step window (decay ≈ 0.905), derived from semantics
rather than arbitrary tuning.
"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from splade.circuits.core import circuit_mask
from splade.mechanistic.attribution import compute_attribution_tensor
from splade.training.constants import CIRCUIT_TEMPERATURE
from splade.utils.cuda import DEVICE

# Centroid EMA: 20-step effective window → decay = 1 - 2/(20+1) ≈ 0.905
_CENTROID_EMA_WINDOW = 20
_CENTROID_EMA_DECAY = 1.0 - 2.0 / (_CENTROID_EMA_WINDOW + 1)


class AttributionCentroidTracker(nn.Module):
    """Maintains EMA of per-class mean absolute attribution vectors.

    Used by the separation loss to encourage distinct per-class circuits
    without requiring all classes to appear in every mini-batch.

    EMA decay is derived from a 20-step effective window (≈ 0.905),
    making the semantics explicit rather than tuning an opaque constant.

    The learned log_margin parameter replaces the fixed margin=0.1
    in the contrastive separation loss.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.momentum = _CENTROID_EMA_DECAY
        # log_margin: learned contrastive margin, init ≈ 0.1 (exp(-2.3) ≈ 0.1)
        self.log_margin = nn.Parameter(torch.tensor(-2.3))
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

    Uses EMA tracking with the same decay as FeatureFrequencyTracker (0.99,
    ~100-step effective window). This replaces manual scaling constants
    (like the former /32 in frequency penalty) with an adaptive mechanism
    that automatically adjusts to any dataset, sparsity target, or
    threshold initialization.
    """

    def __init__(self, decay: float = 0.99):
        self._ema: torch.Tensor | None = None
        self._decay = decay

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

    Replaces DeadFeatureTracker. Instead of binary dead/alive detection
    with magic thresholds, maintains continuous frequency estimates and
    penalizes features whose activation frequency falls below a target
    derived from sparsity_target.

    The EMA decay of 0.99 gives a ~100-step effective window, providing
    the same temporal smoothing as the old DEAD_FEATURE_WINDOW=100 but
    without requiring a separate constant.
    """

    def __init__(self, num_features: int, target_sparsity: float):
        self.num_features = num_features
        self.target_freq = target_sparsity
        self.freq_ema = torch.zeros(num_features, device=DEVICE)
        self._decay = 0.99  # ~100-step effective window
        self._step = 0

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
    circuit_fraction: float = 0.1,
    temperature: float = CIRCUIT_TEMPERATURE,
    full_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL-divergence circuit completeness: masked circuit should preserve full distribution.

    Uses circuit_mask() from core.py with the same fraction and temperature
    parameters used throughout the system. Computes KL(full || masked) to
    measure information loss from circuit extraction.

    Args:
        full_logits: Pre-computed full-model logits. If None, recomputes them.
    """
    attr_magnitude = compute_attribution_tensor(sparse_vector, W_eff, labels).abs()
    soft_mask = circuit_mask(attr_magnitude, circuit_fraction, temperature)
    masked_sparse = sparse_vector * soft_mask
    masked_logits = classifier_forward_fn(masked_sparse)

    if full_logits is None:
        full_logits = classifier_forward_fn(sparse_vector)
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
    falls below 10% of the target sparsity, pushing thresholds down to make
    reactivation easier. The threshold for "under-active" is derived from
    sparsity_target rather than a magic constant.

    Over-active features are already handled by the L0 sparsity loss.

    Args:
        freq_tracker: Tracker with per-feature frequency estimates.
        activation_module: JumpReLUGate module with log_threshold parameter.
    """
    if freq_tracker._step < 50:
        return torch.tensor(0.0, device=DEVICE)

    under_active = (freq_tracker.freq_ema < freq_tracker.target_freq * 0.1).detach()
    if not under_active.any():
        return torch.tensor(0.0, device=DEVICE)

    log_thresholds = activation_module.log_threshold  # [V]
    return log_thresholds[under_active].mean()
