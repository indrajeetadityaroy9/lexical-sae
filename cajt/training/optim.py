import math

import torch
from schedulefree import AdamWScheduleFree
from transformers import AutoConfig

def build_optimizer(
    model: torch.nn.Module,
    base_lr: float,
) -> AdamWScheduleFree:
    """Build Schedule-Free AdamW optimizer (Defazio & Mishchenko, arXiv:2405.15682).

    Subsumes LR scheduling, warmup, and model EMA into a single optimizer via
    Primal Averaging. Call optimizer.train() during training steps and
    optimizer.eval() before validation/inference to switch to averaged params.

    Args:
        model: The model to optimize.
        base_lr: Base learning rate (default 3e-4, the standard AdamW default).
    """
    param_groups = _build_param_groups(model, base_lr)
    return AdamWScheduleFree(param_groups, warmup_steps=0)


_REF_HIDDEN_SIZE = 768
_REF_SEQ_LENGTH = 128
_REF_BASE_BATCH = 32
_MIN_BATCH_POWER = 3   # 2^3 = 8
_MAX_BATCH_POWER = 6   # 2^6 = 64
_GPU_RESERVED_GB = 2.0
_REF_USABLE_GPU_GB = 22.0
_MAX_SCALED_POWER = 9  # 2^9 = 512


def _infer_batch_size(model_name: str, max_length: int) -> int:
    config = AutoConfig.from_pretrained(model_name)
    mem_ratio = (_REF_HIDDEN_SIZE * _REF_SEQ_LENGTH) / (config.hidden_size * max_length)
    base_power = min(_MAX_BATCH_POWER, max(_MIN_BATCH_POWER, int(math.log2(max(1, _REF_BASE_BATCH * mem_ratio)))))
    base_bs = 2 ** base_power

    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    usable_mem = max(1.0, total_mem - _GPU_RESERVED_GB)
    scale_factor = usable_mem / _REF_USABLE_GPU_GB
    scaled_bs = int(base_bs * scale_factor)
    power = min(_MAX_SCALED_POWER, int(math.log2(max(1, scaled_bs))))
    return 2 ** power


def _build_param_groups(model: torch.nn.Module, base_lr: float) -> list[dict]:
    """Build param groups with derived weight decay.

    Weight decay is derived as 0.1 * base_lr (Loshchilov-Hutter scaling),
    eliminating the need for a separate WEIGHT_DECAY constant.
    """
    weight_decay = 0.1 * base_lr
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "LayerNorm" in name or "layer_norm" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
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
