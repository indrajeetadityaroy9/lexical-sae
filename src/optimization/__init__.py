"""Augmented Lagrangian optimization: constraints, penalties, and dual control."""

from src.optimization.capu import MonotoneCAPU
from src.optimization.constraints import (
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_orthogonality_violation,
)
from src.optimization.discretization import DiscretizationSchedule
from src.optimization.dual_updater import DualUpdater
from src.optimization.ema_filter import DualRateEMA
from src.optimization.lagrangian import compute_augmented_lagrangian

__all__ = [
    "compute_augmented_lagrangian",
    "compute_faithfulness_violation",
    "compute_drift_violation",
    "compute_orthogonality_violation",
    "DualUpdater",
    "MonotoneCAPU",
    "DualRateEMA",
    "DiscretizationSchedule",
]
