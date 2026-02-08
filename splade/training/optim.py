"""Optimization schedules, LR calibration, and gradient stabilization."""

import copy
import math

import numpy as np
import torch
from transformers import AutoConfig

from splade.training.constants import (
    AGC_CLIP_FACTOR,
    AGC_EPS,
    LR_FIND_DIVERGE_FACTOR,
    LR_FIND_END,
    LR_FIND_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from splade.utils.cuda import COMPUTE_DTYPE


def find_lr(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_labels: int,
    device: torch.device,
) -> float:
    """Smith's LR range test (arXiv:1506.01186): find optimal LR empirically.

    Exponentially sweeps LR from 1e-7 to LR_FIND_END over LR_FIND_STEPS steps,
    recording classification loss at each step. Returns the LR at the steepest
    descent of the smoothed loss curve.

    Saves and restores model state — zero side effects on training.
    Falls back to 3e-5 if the test fails (e.g., loss never decreases).
    """
    _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    saved_state = copy.deepcopy(_orig.state_dict())

    temp_optimizer = torch.optim.AdamW(_orig.parameters(), lr=LR_FIND_END, fused=True)
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss()
    )

    start_lr = 1e-7
    lr_mult = (LR_FIND_END / start_lr) ** (1.0 / LR_FIND_STEPS)
    current_lr = start_lr
    best_loss = float("inf")
    lrs: list[float] = []
    losses: list[float] = []

    data_iter = iter(train_loader)
    _orig.train()

    for _ in range(LR_FIND_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch_ids, batch_mask, batch_labels = (b.to(device, non_blocking=True) for b in batch)

        for g in temp_optimizer.param_groups:
            g["lr"] = current_lr

        temp_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            logits, _ = model(batch_ids, batch_mask)
            loss = (
                criterion(logits.squeeze(-1), batch_labels)
                if num_labels == 1
                else criterion(logits, batch_labels.view(-1))
            )
        loss.backward()
        temp_optimizer.step()

        loss_val = loss.item()
        lrs.append(current_lr)
        losses.append(loss_val)
        best_loss = min(best_loss, loss_val)

        if loss_val > LR_FIND_DIVERGE_FACTOR * best_loss and len(losses) > 10:
            break

        current_lr *= lr_mult

    # Restore original weights — zero side effects
    _orig.load_state_dict(saved_state)

    # Find steepest descent: minimum gradient of smoothed loss curve
    if len(losses) < 10:
        return 3e-5  # Fallback

    window = min(10, len(losses) // 3)
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode="valid")
    gradients = np.gradient(smoothed)
    best_idx = int(np.argmin(gradients))
    offset = window // 2
    found_lr = lrs[best_idx + offset]

    # Bound to validated BERT fine-tuning range
    found_lr = max(1e-6, min(1e-3, found_lr))
    return found_lr


def _infer_batch_size(model_name: str, max_length: int) -> int:
    """Infer training batch size from model dimensions.

    Reference point: DistilBERT (H=768) + max_length=128 fits batch=32 on H100 80GB.
    Scales inversely with model_dim * seq_length product.
    """
    config = AutoConfig.from_pretrained(model_name)
    mem_ratio = (768 * 128) / (config.hidden_size * max_length)
    power = min(6, max(3, int(math.log2(max(1, 32 * mem_ratio)))))
    return 2 ** power


def _build_param_groups(model: torch.nn.Module, base_lr: float) -> list[dict]:
    """Separate weight matrices (with decay) from bias/LayerNorm (no decay).

    Standard practice for transformer fine-tuning: regularizing bias and
    normalization parameters hurts performance (Loshchilov & Hutter, 2019).
    """
    _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in _orig.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
    ]


def _adaptive_gradient_clip(
    model: torch.nn.Module,
    skip_params: set | None = None,
) -> None:
    """NFNet Adaptive Gradient Clipping (arXiv:2102.06171, Section 4.1)."""
    skip_ids = frozenset(id(p) for p in skip_params) if skip_params else frozenset()
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in skip_ids:
            continue
        if parameter.data.ndim >= 2:
            p_flat = parameter.data.reshape(parameter.data.shape[0], -1)
            g_flat = parameter.grad.data.reshape(parameter.grad.data.shape[0], -1)
            p_norms = p_flat.norm(2, dim=1)
            g_norms = g_flat.norm(2, dim=1)
            max_norms = torch.clamp(p_norms, min=AGC_EPS) * AGC_CLIP_FACTOR
            clip_mask = g_norms > max_norms
            if clip_mask.any():
                scale = max_norms / (g_norms + 1e-8)
                scale = torch.where(clip_mask, scale, torch.ones_like(scale))
                parameter.grad.data.mul_(
                    scale.view(parameter.data.shape[0], *([1] * (parameter.data.ndim - 1)))
                )
        else:
            p_norm = parameter.data.norm(2)
            g_norm = parameter.grad.data.norm(2)
            max_norm = torch.clamp(p_norm, min=AGC_EPS) * AGC_CLIP_FACTOR
            parameter.grad.data.mul_(torch.clamp(max_norm / (g_norm + 1e-8), max=1.0))


class _LRScheduler:
    """Linear warmup followed by cosine annealing to zero."""

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


def _compute_warmup_steps(total_steps: int) -> int:
    """BERT-validated warmup: 6% of total training steps (arXiv:1810.04805 §5.1)."""
    return max(1, int(WARMUP_RATIO * total_steps))
