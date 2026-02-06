"""Training optimizers, schedulers, and gradient clipping."""

import copy
import math

import torch.nn as nn
from transformers import AutoConfig

from src.models.components import _EPS


def _compute_base_lr(model_name: str) -> float:
    """Derive learning rate from model width via muP scaling (Yang et al., arXiv:2203.03466).

    Optimal AdamW LR scales as O(1/width). Calibrated so DistilBERT-base (768) -> 2e-5.
    """
    config = AutoConfig.from_pretrained(model_name)
    return 2e-5 * (768 / config.hidden_size)


def _adaptive_gradient_clip(model: nn.Module, clip_factor: float = 0.01, eps: float = 1e-3) -> None:
    """Adaptive Gradient Clipping (Brock et al., arXiv:2102.06171 'NFNets').

    Clips gradients per-parameter when ||grad|| / (||param|| + eps) > clip_factor.
    The factor 0.01 is a universal constant from the paper, not task-specific.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        p_norm = p.data.norm(2)
        g_norm = p.grad.data.norm(2)
        max_norm = p_norm * clip_factor + eps
        if g_norm > max_norm:
            p.grad.data.mul_(max_norm / (g_norm + 1e-8))


class _LRScheduler:
    """Cosine LR schedule with linear warmup."""

    def __init__(self, base_lr: float, total_steps: int, warmup_steps: int):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self) -> float:
        s = self._step
        self._step += 1
        if s < self.warmup_steps:
            return self.base_lr * (s / max(self.warmup_steps, 1))
        progress = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class _AdaptiveLambdaSchedule:
    """Loss-ratio-balanced lambda for DF-FLOPS regularization.

    Maintains lambda such that reg_loss ~ target_ratio * cls_loss in magnitude.
    Eliminates the task-specific lambda_init/lambda_peak parameters.
    Inspired by GradNorm (Chen et al., arXiv:1711.02257).
    """

    def __init__(self, warmup_steps: int, target_ratio: float = 0.5, ema_decay: float = 0.99):
        self.target_ratio = target_ratio
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._lambda = 1.0
        self._ema_cls: float | None = None
        self._ema_reg: float | None = None
        self._current_sparsity = 0.0

    def compute_lambda(
        self, activations,
        cls_loss_val: float = 0.0, reg_loss_val: float = 0.0,
    ) -> float:
        import torch
        with torch.no_grad():
            self._current_sparsity = (activations.abs() < _EPS[activations.dtype]['div']).float().mean().item()

        # Quadratic warmup phase
        if self._step < self.warmup_steps:
            self._step += 1
            return self._lambda * (self._step / self.warmup_steps) ** 2

        # After warmup: balance loss magnitudes via EMA
        if self._ema_cls is None:
            self._ema_cls = cls_loss_val
            self._ema_reg = reg_loss_val
        else:
            self._ema_cls = self.ema_decay * self._ema_cls + (1 - self.ema_decay) * cls_loss_val
            self._ema_reg = self.ema_decay * self._ema_reg + (1 - self.ema_decay) * reg_loss_val

        if self._ema_reg is not None and self._ema_reg > 1e-12:
            self._lambda = (self.target_ratio * self._ema_cls) / self._ema_reg

        self._step += 1
        return max(0.01, min(self._lambda, 1e4))  # safety clamp


class _EarlyStopping:
    """Early stopping with patience and best-model checkpointing.

    Standard practice per Prechelt (1998) and transformer fine-tuning
    (Dodge et al., arXiv:2002.06305).
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


class _DistillWeightAdapter:
    """Adaptive distillation weight via loss-ratio balancing.

    Maintains weight such that kl_loss * weight ~ target_ratio * cls_loss.
    Based on uncertainty-weighted losses (Kendall et al., arXiv:1705.07115).
    """

    def __init__(self, target_ratio: float = 0.3, ema_decay: float = 0.95):
        self.target_ratio = target_ratio
        self.ema_decay = ema_decay
        self._ema_cls: float | None = None
        self._ema_kl: float | None = None
        self._weight = 0.1

    def compute_weight(self, cls_loss_val: float, kl_loss_val: float) -> float:
        if self._ema_cls is None:
            self._ema_cls = cls_loss_val
            self._ema_kl = kl_loss_val
        else:
            self._ema_cls = self.ema_decay * self._ema_cls + (1 - self.ema_decay) * cls_loss_val
            self._ema_kl = self.ema_decay * self._ema_kl + (1 - self.ema_decay) * kl_loss_val

        if self._ema_kl is not None and self._ema_kl > 1e-12:
            self._weight = (self.target_ratio * self._ema_cls) / self._ema_kl

        return max(0.01, min(self._weight, 2.0))


class _GateTemperatureSchedule:
    """Exponential annealing for Gumbel-Sigmoid temperature (arXiv:2404.03323).

    Anneals from tau_0 to tau_min over total_steps using exponential decay.
    """

    def __init__(self, tau_0: float = 5.0, tau_min: float = 0.1, total_steps: int = 1000):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.total_steps = max(total_steps, 1)
        self._rate = math.log(tau_0 / max(tau_min, 1e-8)) / self.total_steps
        self._step = 0

    def step(self) -> float:
        tau = self.tau_0 * math.exp(-self._rate * self._step)
        tau = max(tau, self.tau_min)
        self._step += 1
        return tau


class _ContrastiveLambdaSchedule:
    """Loss-ratio-balanced lambda for contrastive loss.

    Maintains lambda such that con_loss * lambda ~ target_ratio * cls_loss.
    Follows the _AdaptiveLambdaSchedule / _DistillWeightAdapter pattern.
    """

    def __init__(self, warmup_steps: int, target_ratio: float = 0.2, ema_decay: float = 0.99):
        self.target_ratio = target_ratio
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._lambda = 0.1
        self._ema_cls: float | None = None
        self._ema_con: float | None = None

    def compute_lambda(self, cls_loss_val: float, con_loss_val: float) -> float:
        if self._step < self.warmup_steps:
            self._step += 1
            return self._lambda * (self._step / self.warmup_steps) ** 2

        if self._ema_cls is None:
            self._ema_cls = cls_loss_val
            self._ema_con = con_loss_val
        else:
            self._ema_cls = self.ema_decay * self._ema_cls + (1 - self.ema_decay) * cls_loss_val
            self._ema_con = self.ema_decay * self._ema_con + (1 - self.ema_decay) * con_loss_val

        if self._ema_con is not None and self._ema_con > 1e-12:
            self._lambda = (self.target_ratio * self._ema_cls) / self._ema_con

        self._step += 1
        return max(0.01, min(self._lambda, 1e4))


def _compute_warmup_steps(total_steps: int) -> int:
    """Sqrt-proportional warmup (Liu et al., arXiv:2404.19628 'Warmup Revisited').

    Scales naturally with training duration instead of using a fixed fraction.
    """
    return max(1, min(int(math.sqrt(total_steps) * 2), total_steps // 3))
