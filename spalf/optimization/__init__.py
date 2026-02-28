"""Augmented Lagrangian optimization: penalties and dual control."""

from spalf.optimization.capu import MonotoneCAPU
from spalf.optimization.discretization import DiscretizationSchedule
from spalf.optimization.dual_updater import NuPIDualUpdater
from spalf.optimization.ema_filter import DualRateEMA
from spalf.optimization.lagrangian import compute_augmented_lagrangian

__all__ = [
    "compute_augmented_lagrangian",
    "NuPIDualUpdater",
    "MonotoneCAPU",
    "DualRateEMA",
    "DiscretizationSchedule",
]
