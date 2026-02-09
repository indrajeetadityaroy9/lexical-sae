import copy
import itertools
import math

import numpy as np
import torch
from transformers import AutoConfig

from splade.training.constants import (LR_FIND_DIVERGE_FACTOR, LR_FIND_END,
                                       LR_FIND_STEPS, WEIGHT_DECAY)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def find_lr(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_labels: int,
) -> float:
    _orig = unwrap_compiled(model)
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

    data_iter = itertools.cycle(train_loader)
    _orig.train()

    for _ in range(LR_FIND_STEPS):
        batch = next(data_iter)

        batch_ids, batch_mask, batch_labels = (b.to(DEVICE, non_blocking=True) for b in batch)

        for g in temp_optimizer.param_groups:
            g["lr"] = current_lr

        temp_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = model(batch_ids, batch_mask)
            logits = _orig.classify(sparse_seq, batch_mask).logits
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

    _orig.load_state_dict(saved_state)

    if len(losses) < 10:
        return 3e-5

    window = min(10, len(losses) // 3)
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode="valid")
    gradients = np.gradient(smoothed)
    best_idx = int(np.argmin(gradients))
    offset = window // 2
    found_lr = lrs[best_idx + offset]

    found_lr = max(1e-6, min(1e-3, found_lr))
    return found_lr


def _infer_batch_size(model_name: str, max_length: int) -> int:
    config = AutoConfig.from_pretrained(model_name)
    mem_ratio = (768 * 128) / (config.hidden_size * max_length)
    # Base batch size for ~24GB reference GPU (22GB usable)
    base_power = min(6, max(3, int(math.log2(max(1, 32 * mem_ratio)))))
    base_bs = 2 ** base_power
    # Scale by available GPU memory relative to 22GB usable reference
    scale_factor = 1.0
    if torch.cuda.is_available():
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usable_mem = max(1.0, total_mem - 2.0)
            scale_factor = usable_mem / 22.0
        except Exception:
            scale_factor = 1.0
    scaled_bs = int(base_bs * scale_factor)
    # Round down to power of 2, cap at 512
    power = min(9, int(math.log2(max(1, scaled_bs))))
    return 2 ** power


def _build_param_groups(model: torch.nn.Module, base_lr: float) -> list[dict]:
    _orig = unwrap_compiled(model)
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in _orig.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
    ]


def _build_param_groups_with_gate_boost(
    model: torch.nn.Module,
    base_lr: float,
    gate_lr_multiplier: float = 5.0,
) -> list[dict]:
    """Build param groups with boosted LR for DReLU sparsity gates.

    Creates three groups:
      1. Weight-decayed params (matrices, excluding gates)
      2. Non-decayed params (biases, norms, excluding gates)
      3. Gate params (activation.theta) with boosted LR, no decay

    The gate boost gives the optimizer more "kinetic energy" to push
    DReLU thresholds upward, overcoming the CE gradient inertia that
    otherwise traps the model in a dense local minimum.
    """
    _orig = unwrap_compiled(model)
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []

    gate_identifiers = ["thresholds", "theta", "gate"]
    boosted_names: list[str] = []

    for name, param in _orig.named_parameters():
        if not param.requires_grad:
            continue
        if any(ident in name for ident in gate_identifiers):
            gate_params.append(param)
            boosted_names.append(name)
        elif param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    print(f"DEBUG: Boosting LR for sparsity gates: {boosted_names}")
    if not boosted_names:
        raise ValueError(
            f"No sparsity gates found matching {gate_identifiers}. "
            "Check splade/models/layers/activation.py for actual param name."
        )

    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": WEIGHT_DECAY, "_lr_multiplier": 1.0},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0, "_lr_multiplier": 1.0},
        {"params": gate_params, "lr": base_lr * gate_lr_multiplier, "weight_decay": 0.0, "_lr_multiplier": gate_lr_multiplier},
    ]


def _gradient_centralization(
    model: torch.nn.Module,
    skip_params: set | None = None,
) -> None:
    """Parameter-free gradient conditioning via centralization.

    Subtracts the mean from each gradient tensor (for weight matrices only),
    constraining gradients to the hyperplane of zero-mean vectors. This
    improves the Lipschitz smoothness of the loss landscape without any
    tunable hyperparameters.

    Reference: Yong et al., "Gradient Centralization" (arXiv:2004.01461).
    """
    skip_ids = frozenset(id(p) for p in skip_params) if skip_params else frozenset()
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) in skip_ids:
            continue
        if parameter.data.ndim >= 2:
            # Centralize: subtract mean across all dims except the output dim
            parameter.grad.data -= parameter.grad.data.mean(
                dim=tuple(range(1, parameter.grad.data.ndim)),
                keepdim=True,
            )


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
