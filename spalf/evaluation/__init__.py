"""SPALF evaluation suite."""

from spalf.evaluation.absorption import feature_absorption_rate
from spalf.evaluation.dead_latents import count_dead_latents
from spalf.evaluation.downstream_loss import evaluate_downstream_loss
from spalf.evaluation.drift_fidelity import drift_fidelity
from spalf.evaluation.sparsity_frontier import compute_sparsity_frontier

__all__ = [
    "evaluate_downstream_loss",
    "compute_sparsity_frontier",
    "drift_fidelity",
    "feature_absorption_rate",
    "count_dead_latents",
]
