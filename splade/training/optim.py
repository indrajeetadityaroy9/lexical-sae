"""Optimization schedules and stabilization utilities."""

import math
import torch
from transformers import AutoConfig

def _compute_base_lr(model_name: str) -> float:
    config = AutoConfig.from_pretrained(model_name)
    return 2e-5 * (768 / config.hidden_size)

def _adaptive_gradient_clip(
    model: torch.nn.Module,
    clip_factor: float = 0.01,
    eps: float = 1e-3,
    skip_params: set | None = None,
) -> None:
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if skip_params is not None and any(parameter is p for p in skip_params):
            continue
        if parameter.data.ndim >= 2:
            # Unit-wise: per output-unit (row) norms — NFNet (arXiv:2102.06171)
            p_flat = parameter.data.reshape(parameter.data.shape[0], -1)
            g_flat = parameter.grad.data.reshape(parameter.grad.data.shape[0], -1)
            p_norms = p_flat.norm(2, dim=1)
            g_norms = g_flat.norm(2, dim=1)
            max_norms = torch.clamp(p_norms, min=eps) * clip_factor
            clip_mask = g_norms > max_norms
            if clip_mask.any():
                scale = max_norms / (g_norms + 1e-8)
                scale = torch.where(clip_mask, scale, torch.ones_like(scale))
                parameter.grad.data.mul_(
                    scale.view(parameter.data.shape[0], *([1] * (parameter.data.ndim - 1)))
                )
        else:
            # 1D (biases, LayerNorm): full-parameter norm
            p_norm = parameter.data.norm(2)
            g_norm = parameter.grad.data.norm(2)
            max_norm = torch.clamp(p_norm, min=eps) * clip_factor
            if g_norm > max_norm:
                parameter.grad.data.mul_(max_norm / (g_norm + 1e-8))

class _LRScheduler:
    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self) -> float:
        current_step = self._step
        self._step += 1
        if current_step < self.warmup_steps:
            return self.base_lr * (current_step / max(self.warmup_steps, 1))
        progress = (current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

class _EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict() # Use shallow dict copy
            return False
        self.counter += 1
        return self.counter >= self.patience

def _compute_warmup_steps(total_steps: int) -> int:
    return max(1, min(int(math.sqrt(total_steps) * 2), total_steps // 3))