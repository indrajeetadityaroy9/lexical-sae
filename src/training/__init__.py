"""
Training module for SPLADE models.

Provides training infrastructure including:
- Knowledge distillation from larger models
- Training loops and callbacks
- Loss functions for sparse learning
"""

from .distillation import (
    DistillationConfig,
    DistillationLoss,
    DistillationTrainer,
    create_teacher_labels,
)

__all__ = [
    "DistillationConfig",
    "DistillationLoss",
    "DistillationTrainer",
    "create_teacher_labels",
]
