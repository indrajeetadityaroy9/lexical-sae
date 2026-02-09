import os

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from splade.data.loader import infer_max_length
from splade.circuits.geco import GECOController
from splade.circuits.losses import (
    AttributionCentroidTracker,
    compute_completeness_loss,
    compute_separation_loss,
    compute_sharpness_loss,
)
from splade.training.constants import (
    EARLY_STOP_PATIENCE,
    EMA_DECAY,
    LABEL_SMOOTHING,
    MAX_EPOCHS,
    WARMUP_RATIO,
)
from splade.training.optim import (_gradient_centralization,
                                   _build_param_groups,
                                   _infer_batch_size, _LRScheduler, find_lr)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled

_NUM_WORKERS = min(os.cpu_count() or 4, 16)
_PREFETCH_FACTOR = 4


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        _orig = unwrap_compiled(model)
        for name, param in _orig.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        _orig = unwrap_compiled(model)
        for name, param in _orig.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        _orig = unwrap_compiled(model)
        self._backup.clear()
        for name, param in _orig.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        _orig = unwrap_compiled(model)
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
    max_length: int | None = None,
    batch_size: int | None = None,
    target_accuracy: float | None = None,
    sparsity_target: float = 0.1,
    warmup_fraction: float = 0.2,
) -> "AttributionCentroidTracker":
    if max_length is None:
        max_length = infer_max_length(texts, tokenizer, model_name=model_name)
    if batch_size is None:
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
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"], label_tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=_NUM_WORKERS,
        prefetch_factor=_PREFETCH_FACTOR,
        persistent_workers=True,
    )

    optimal_lr = find_lr(model, loader, num_labels)
    print(f"LR range test -> optimal_lr={optimal_lr:.2e}")

    param_groups = _build_param_groups(model, optimal_lr)
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    lr_scheduler = _LRScheduler(optimal_lr, total_steps, warmup_steps)

    _orig = unwrap_compiled(model)
    vocab_size = _orig.vocab_size
    classifier_params = set(_orig.classifier_parameters())

    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    )

    ema = ModelEMA(model, decay=EMA_DECAY)

    # CIS circuit losses with GECO adaptive weighting
    centroid_tracker = AttributionCentroidTracker(
        num_classes=num_labels, vocab_size=vocab_size,
    )
    geco = GECOController(steps_per_epoch=steps_per_epoch)
    if target_accuracy is not None:
        geco.tau_ce = target_accuracy
    circuit_delay_steps = int(warmup_fraction * total_steps)
    patience_reset_step = int(warmup_fraction * total_steps)
    patience_was_reset = False
    global_step = 0
    warmup_finalized = False

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
        total_loss = torch.zeros(1, device=DEVICE)
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
                sparse_seq = model(batch_ids, batch_mask)
                logits, sparse, W_eff, _ = _orig.classify(sparse_seq, batch_mask)
                classification_loss = (
                    criterion(logits.squeeze(-1), batch_labels)
                    if num_labels == 1
                    else criterion(logits, batch_labels.view(-1))
                )

                global_step += 1

                # Reset patience at fixed point for both baseline and CIS
                if not patience_was_reset and global_step >= patience_reset_step:
                    patience_was_reset = True
                    patience_counter = 0

                if global_step < circuit_delay_steps:
                    # Warmup: CE only, record CE for GECO tau
                    loss = classification_loss
                    if target_accuracy is None:
                        geco.record_warmup_ce(classification_loss.detach().item())
                else:
                    if not warmup_finalized:
                        if target_accuracy is None:
                            tau = geco.finalize_warmup()
                            print(f"GECO: tau_ce={tau:.4f}")
                        else:
                            print(f"GECO: tau_ce={target_accuracy:.4f} (from config)")
                        warmup_finalized = True

                    cc_loss = compute_completeness_loss(
                        sparse, W_eff, batch_labels.view(-1),
                        _orig.classifier_logits_only,
                        circuit_fraction=sparsity_target,
                    )
                    sep_loss = compute_separation_loss(centroid_tracker)
                    sharp_loss = compute_sharpness_loss(
                        sparse, W_eff, batch_labels.view(-1),
                    )

                    circuit_objective = cc_loss + sep_loss + sharp_loss
                    loss = geco.compute_loss(classification_loss, circuit_objective)

                    centroid_tracker.update(
                        sparse.detach(), W_eff.detach(),
                        batch_labels.view(-1),
                    )

            loss.backward()
            _gradient_centralization(model, skip_params=classifier_params)
            optimizer.step()

            ema.update(model)

            total_loss += loss.detach()
            batch_count += 1

        average_loss = total_loss.item() / batch_count
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}"

        if warmup_finalized:
            epoch_msg += f", GECO lambda={geco.lambda_ce:.4f}"
            if abs(geco._log_lambda) >= 4.9:
                print(
                    f"\n  [WARNING] GECO lambda pinned at {geco.lambda_ce:.2f} "
                    f"(log_lambda={geco._log_lambda:.2f}, "
                    f"constraint_ema={geco._ema_constraint:.4f}). "
                    f"Model is {'sacrificing interpretability for accuracy' if geco._log_lambda > 0 else 'over-constraining accuracy'}."
                )
        else:
            epoch_msg += ", GECO: warming up"

        ema.apply_shadow(model)
        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                val_sparse_seq = model(val_ids_gpu, val_mask_gpu)
                val_logits = _orig.classify(val_sparse_seq, val_mask_gpu).logits
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

    _orig_model = unwrap_compiled(model)
    for name, param in _orig_model.named_parameters():
        if name in best_ema_state:
            param.data.copy_(best_ema_state[name])
    print(f"Applied best EMA weights (val loss: {best_val_loss:.4f})")

    model.eval()

    return centroid_tracker
