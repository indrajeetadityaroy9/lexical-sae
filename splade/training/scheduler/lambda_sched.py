"""Adaptive lambda schedules for regularization."""

import math
import torch

class SatLinearSchedule:
    """Sparsity-Accelerated Training (SAT) Schedule: Dense -> Sparse.
    
    Linearly increases lambda (regularization strength) to gradually force sparsity.
    """
    def __init__(self, warmup_steps: int, total_steps: int, final_lambda: float = 0.1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.final_lambda = final_lambda
        self._step = 0
        self._current_sparsity = 0.0

    def compute_lambda(self, activations: torch.Tensor) -> float:
        with torch.no_grad():
            self._current_sparsity = (
                (activations.abs() < 1e-6).float().mean().item()
            )

        current_step = self._step
        self._step += 1
        if current_step < self.warmup_steps:
            return 0.0  # Dense phase

        progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.final_lambda * min(1.0, progress)