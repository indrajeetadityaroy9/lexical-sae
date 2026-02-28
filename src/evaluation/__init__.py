"""SPALF evaluation suite."""

from src.evaluation.absorption import feature_absorption_rate
from src.evaluation.downstream_loss import evaluate_downstream_loss
from src.evaluation.drift_fidelity import drift_fidelity
from src.evaluation.sparsity_frontier import compute_sparsity_frontier

__all__ = [
    "evaluate_downstream_loss",
    "compute_sparsity_frontier",
    "drift_fidelity",
    "feature_absorption_rate",
]
