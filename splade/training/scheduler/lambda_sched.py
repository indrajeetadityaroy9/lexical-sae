import torch

from splade.training.constants import LAMBDA_FINAL


class SatLambdaSchedule:
    def __init__(self, warmup_steps: int, total_steps: int):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        self._current_sparsity = 0.0
        self._sparsity_sum = 0.0
        self._sparsity_count = 0

    def compute_lambda(self, activations: torch.Tensor) -> float:
        with torch.inference_mode():
            self._sparsity_sum += (activations.abs() < 1e-6).float().mean()
            self._sparsity_count += 1

        current_step = self._step
        self._step += 1
        if current_step < self.warmup_steps:
            return 0.0

        progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return LAMBDA_FINAL * (progress ** 2)

    def sync_sparsity(self) -> None:
        if self._sparsity_count > 0:
            if isinstance(self._sparsity_sum, torch.Tensor):
                self._current_sparsity = (self._sparsity_sum / self._sparsity_count).item()
            else:
                self._current_sparsity = self._sparsity_sum / self._sparsity_count
        self._sparsity_sum = 0.0
        self._sparsity_count = 0
