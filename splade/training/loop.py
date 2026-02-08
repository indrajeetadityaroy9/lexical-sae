import copy
import os

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from splade.data.loader import infer_max_length
from splade.training.constants import (EARLY_STOP_PATIENCE, EMA_DECAY,
                                       LABEL_SMOOTHING, MAX_EPOCHS)
from splade.training.losses import DFFlopsRegFunction, DocumentFrequencyTracker
from splade.training.optim import (_adaptive_gradient_clip,
                                   _build_param_groups, _compute_warmup_steps,
                                   _infer_batch_size, _LRScheduler, find_lr)
from splade.training.scheduler.lambda_sched import SatLambdaSchedule
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE

_NUM_WORKERS = min(os.cpu_count() or 4, 8)
_PREFETCH_FACTOR = 2


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        _orig = model._orig_mod
        for name, param in _orig.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        _orig = model._orig_mod
        for name, param in _orig.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        _orig = model._orig_mod
        self._backup.clear()
        for name, param in _orig.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        _orig = model._orig_mod
        for name, param in _orig.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def train_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    labels: list[int],
    model_name: str,
    num_labels: int,
    val_texts: list[str],
    val_labels: list[int],
) -> None:
    max_length = infer_max_length(texts, tokenizer)
    batch_size = _infer_batch_size(model_name, max_length)

    print(f"Auto-inferred: max_length={max_length}, batch_size={batch_size}")

    label_tensor = torch.tensor(
        labels,
        dtype=torch.float32 if num_labels == 1 else torch.long,
    )
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
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

    optimal_lr = find_lr(model, loader, num_labels, DEVICE)
    print(f"LR range test -> optimal_lr={optimal_lr:.2e}")

    param_groups = _build_param_groups(model, optimal_lr)
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = _compute_warmup_steps(total_steps)

    lr_scheduler = _LRScheduler(optimal_lr, total_steps, warmup_steps)
    lambda_schedule = SatLambdaSchedule(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    _orig = model._orig_mod
    vocab_size = _orig.padded_vocab_size
    df_tracker = DocumentFrequencyTracker(vocab_size=vocab_size, device=DEVICE)
    classifier_params = set(_orig.classifier.parameters())

    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    )

    ema = ModelEMA(model, decay=EMA_DECAY)

    val_encoding = tokenizer(
        val_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    val_label_gpu = torch.tensor(
        val_labels,
        dtype=torch.float32 if num_labels == 1 else torch.long,
    ).to(DEVICE)
    val_ids_gpu = val_encoding["input_ids"].to(DEVICE)
    val_mask_gpu = val_encoding["attention_mask"].to(DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_ema_state: dict[str, torch.Tensor] | None = None

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

            ema.update(model)

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count
        lambda_schedule.sync_sparsity()
        sparsity = lambda_schedule._current_sparsity
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}, Sparsity: {sparsity:.2%}"

        stats = df_tracker.get_stats()
        epoch_msg += f", Top-1 DF: {stats['top1_df_pct']:.1f}%"

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

    _orig_model = model._orig_mod
    for name, param in _orig_model.named_parameters():
        if name in best_ema_state:
            param.data.copy_(best_ema_state[name])
    print(f"Applied best EMA weights (val loss: {best_val_loss:.4f})")

    model.eval()
