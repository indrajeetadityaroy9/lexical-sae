"""Training loop with validated recipe: LR range test, EMA, label smoothing, per-param decay."""

import copy
import os

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from splade.training.constants import (
    EMA_DECAY,
    EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING,
    MAX_EPOCHS,
)
from splade.training.losses import DFFlopsRegFunction, DocumentFrequencyTracker
from splade.training.optim import (
    _adaptive_gradient_clip,
    _build_param_groups,
    _compute_warmup_steps,
    _infer_batch_size,
    _LRScheduler,
    find_lr,
)
from splade.training.scheduler.lambda_sched import SatLambdaSchedule
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE
from splade.data.loader import infer_max_length

_NUM_WORKERS = min(os.cpu_count() or 4, 8)
_PREFETCH_FACTOR = 2


class ModelEMA:
    """Exponential Moving Average of model parameters (Polyak averaging).

    Maintains shadow copies of all trainable parameters, updated each optimizer
    step. EMA weights smooth out training noise and produce more stable,
    generalizable models for evaluation and inference.

    References:
        - Polyak & Juditsky (1992): Acceleration of stochastic approximation
        - arXiv:2411.18704: EMA decay 0.999 validated for BERT-scale models
    """

    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, param in _orig.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * param."""
        _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, param in _orig.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Swap model weights with EMA shadow weights. Call restore() to undo."""
        _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
        self._backup.clear()
        for name, param in _orig.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original (non-EMA) weights after apply_shadow()."""
        _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
        for name, param in _orig.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a copy of shadow weights for checkpointing."""
        return {k: v.clone() for k, v in self.shadow.items()}


def _validate_texts(texts: list[str], name: str = "X") -> None:
    if not texts:
        raise ValueError(f"{name} must be non-empty")
    if not all(isinstance(text, str) for text in texts):
        raise TypeError(f"{name} must contain strings")


def tokenize_batch(texts: list[str], tokenizer, max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def train_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    labels: list[int],
    model_name: str,
    num_labels: int,
    val_texts: list[str] | None = None,
    val_labels: list[int] | None = None,
) -> None:
    """Train SPLADE model with validated canonical recipe.

    Training recipe (all auto-derived, no manual tuning):
      - LR: Found via Smith's LR range test at training start
      - Schedule: Linear warmup (6%) + cosine annealing
      - Weight decay: 0.01 on weight matrices, 0.0 on bias/LayerNorm
      - Label smoothing: 0.1 for multi-class classification
      - EMA: Decay 0.999, used for validation and final model
      - Early stopping: Patience 5 on EMA-validated loss
      - Max epochs: 20 (ceiling; early stopping handles termination)
    """
    _validate_texts(texts)
    if len(texts) != len(labels):
        raise ValueError("X and y must have the same length")

    # --- Auto-infer data-dependent parameters ---
    max_length = infer_max_length(texts, tokenizer)
    batch_size = _infer_batch_size(model_name, max_length)

    print(f"Auto-inferred: max_length={max_length}, batch_size={batch_size}")

    # --- Prepare data ---
    label_tensor = torch.tensor(
        labels,
        dtype=torch.float32 if num_labels == 1 else torch.long,
    )
    encoding = tokenize_batch(texts, tokenizer, max_length)
    worker_kwargs = {}
    if _NUM_WORKERS > 0:
        worker_kwargs = dict(
            num_workers=_NUM_WORKERS,
            prefetch_factor=_PREFETCH_FACTOR,
            persistent_workers=True,
        )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"], label_tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        **worker_kwargs,
    )

    # --- LR range test: empirically find optimal learning rate ---
    optimal_lr = find_lr(model, loader, num_labels, DEVICE)
    print(f"LR range test -> optimal_lr={optimal_lr:.2e}")

    # --- Optimizer with per-parameter weight decay ---
    param_groups = _build_param_groups(model, optimal_lr)
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    # --- Schedules ---
    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = _compute_warmup_steps(total_steps)

    lr_scheduler = _LRScheduler(optimal_lr, total_steps, warmup_steps)
    lambda_schedule = SatLambdaSchedule(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # --- Model internals ---
    _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    vocab_size = _orig.padded_vocab_size if hasattr(_orig, "padded_vocab_size") else _orig.vocab_size
    df_tracker = DocumentFrequencyTracker(vocab_size=vocab_size, device=DEVICE)
    classifier_params = set(_orig.classifier.parameters())

    # --- Loss with label smoothing ---
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    )

    # --- EMA for stable evaluation ---
    ema = ModelEMA(model, decay=EMA_DECAY)

    # --- Pre-tokenize validation data ---
    val_ids_gpu = None
    val_mask_gpu = None
    val_label_gpu = None
    if val_texts is not None and val_labels is not None:
        val_encoding = tokenize_batch(val_texts, tokenizer, max_length)
        val_label_gpu = torch.tensor(
            val_labels,
            dtype=torch.float32 if num_labels == 1 else torch.long,
        ).to(DEVICE)
        val_ids_gpu = val_encoding["input_ids"].to(DEVICE)
        val_mask_gpu = val_encoding["attention_mask"].to(DEVICE)

    # --- Early stopping state (tracks EMA validation loss) ---
    best_val_loss = float("inf")
    patience_counter = 0
    best_ema_state: dict[str, torch.Tensor] | None = None

    # === Training loop ===
    model.train()
    for epoch_index in range(MAX_EPOCHS):
        df_tracker.soft_reset()
        total_loss = 0.0
        batch_count = 0

        for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch_index + 1}/{MAX_EPOCHS}"):
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            learning_rate = lr_scheduler.step()
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                logits, sparse = model(batch_ids, batch_mask)
                classification_loss = (
                    criterion(logits.squeeze(-1), batch_labels)
                    if num_labels == 1
                    else criterion(logits, batch_labels.view(-1))
                )

                df_tracker.update(sparse)
                df_weights = df_tracker.get_weights()
                regularization_loss = DFFlopsRegFunction.apply(sparse, df_weights)

                regularization_weight = lambda_schedule.compute_lambda(sparse)
                loss = classification_loss + regularization_weight * regularization_loss

            loss.backward()
            _adaptive_gradient_clip(model, skip_params=classifier_params)
            optimizer.step()

            # EMA update after each optimizer step
            ema.update(model)

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count
        lambda_schedule.sync_sparsity()
        sparsity = lambda_schedule._current_sparsity
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}, Sparsity: {sparsity:.2%}"

        stats = df_tracker.get_stats()
        epoch_msg += f", Top-1 DF: {stats['top1_df_pct']:.1f}%"

        # --- Validation with EMA weights ---
        if val_ids_gpu is not None:
            ema.apply_shadow(model)
            model.eval()
            with torch.inference_mode():
                with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                    val_logits, _ = model(val_ids_gpu, val_mask_gpu)
                    val_loss = (
                        criterion(val_logits.squeeze(-1), val_label_gpu)
                        if num_labels == 1
                        else criterion(val_logits, val_label_gpu.view(-1))
                    )
            val_loss_val = val_loss.item()
            epoch_msg += f", Val Loss (EMA): {val_loss_val:.4f}"

            ema.restore(model)
            model.train()

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                patience_counter = 0
                best_ema_state = ema.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:
                epoch_msg += f" [Early stop: no improvement for {EARLY_STOP_PATIENCE} epochs]"
                print(epoch_msg)
                break

        print(epoch_msg)

    # === Apply best EMA weights as final model ===
    _orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if best_ema_state is not None:
        for name, param in _orig_model.named_parameters():
            if name in best_ema_state:
                param.data.copy_(best_ema_state[name])
        print(f"Applied best EMA weights (val loss: {best_val_loss:.4f})")
    else:
        # No validation set — apply final EMA weights
        ema.apply_shadow(model)
        print("Applied final EMA weights")

    model.eval()
