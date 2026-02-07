"""Training loop logic."""

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from splade.config.schema import TrainingConfig, ModelConfig, DataConfig
from splade.training.losses import DFFlopsRegFunction, DocumentFrequencyTracker
from splade.training.optim import (
    _adaptive_gradient_clip,
    _compute_base_lr,
    _compute_warmup_steps,
    _EarlyStopping,
    _LRScheduler,
)
from splade.training.scheduler.lambda_sched import SatLinearSchedule
from splade.utils.cuda import AUTOCAST_DEVICE_TYPE, AUTOCAST_ENABLED, COMPUTE_DTYPE, DEVICE
from splade.models.splade import SpladeModel

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
    training_config: TrainingConfig,
    model_config: ModelConfig,
    data_config: DataConfig,
    val_texts: list[str] | None = None,
    val_labels: list[int] | None = None,
) -> None:
    _validate_texts(texts)
    if len(texts) != len(labels):
        raise ValueError("X and y must have the same length")

    epoch_count = training_config.max_epochs
    batch_size = training_config.batch_size
    num_labels = data_config.num_labels
    
    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * epoch_count
    
    # Base LR logic
    if training_config.base_lr is None:
        base_lr = _compute_base_lr(model_config.name)
    else:
        base_lr = training_config.base_lr
    
    if training_config.warmup_steps is None:
        warmup_steps = _compute_warmup_steps(total_steps)
    else:
        warmup_steps = training_config.warmup_steps

    lr_scheduler = _LRScheduler(base_lr, total_steps, warmup_steps)
    
    # SOTA Schedule: SAT Linear
    lambda_schedule = SatLinearSchedule(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        final_lambda=training_config.target_lambda_ratio 
    )
    
    # We need vocab_size. If compiled model, it might be hidden. 
    if hasattr(model, "_orig_mod"):
        vocab_size = model._orig_mod.vocab_size
    else:
        vocab_size = model.vocab_size

    use_df_flops = model_config.regularization == "df_flops"
    df_tracker = DocumentFrequencyTracker(vocab_size=vocab_size, device=DEVICE) if use_df_flops else None
    early_stopping = _EarlyStopping(patience=training_config.patience)

    # NFNet (arXiv:2102.06171 §4.1): skip final linear layer from AGC
    _orig = model._orig_mod if hasattr(model, "_orig_mod") else model
    classifier_params = set(_orig.classifier.parameters())

    label_tensor = torch.tensor(
        labels,
        dtype=torch.float32 if num_labels == 1 else torch.long,
    )
    encoding = tokenize_batch(texts, tokenizer, data_config.max_length)
    worker_kwargs = {}
    if training_config.num_workers > 0:
        worker_kwargs = dict(
            num_workers=training_config.num_workers,
            prefetch_factor=training_config.prefetch_factor,
            persistent_workers=True,
        )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"], label_tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        **worker_kwargs,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss()
    )
    use_scaler = AUTOCAST_ENABLED and COMPUTE_DTYPE == torch.float16
    scaler = torch.amp.GradScaler(AUTOCAST_DEVICE_TYPE, enabled=use_scaler)

    model.train()
    for epoch_index in range(epoch_count):
        if df_tracker is not None:
            df_tracker.soft_reset(momentum=0.9)
        total_loss = 0.0
        batch_count = 0
        for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch_index + 1}/{epoch_count}"):
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            learning_rate = lr_scheduler.step()
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(AUTOCAST_DEVICE_TYPE, dtype=COMPUTE_DTYPE, enabled=AUTOCAST_ENABLED):
                logits, sparse = model(batch_ids, batch_mask)
                classification_loss = (
                    criterion(logits.squeeze(-1), batch_labels)
                    if num_labels == 1
                    else criterion(logits, batch_labels.view(-1))
                )
                
                if use_df_flops:
                    df_tracker.update(sparse)
                    df_weights = df_tracker.get_weights(
                        alpha=training_config.df_alpha,
                        beta=training_config.df_beta,
                    )
                else:
                    df_weights = torch.ones(vocab_size, device=DEVICE)
                regularization_loss = DFFlopsRegFunction.apply(sparse, df_weights)
                
                regularization_weight = lambda_schedule.compute_lambda(sparse)
                loss = classification_loss + regularization_weight * regularization_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _adaptive_gradient_clip(model, clip_factor=training_config.clip_factor, skip_params=classifier_params)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / batch_count
        sparsity = lambda_schedule._current_sparsity
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}, Sparsity: {sparsity:.2%}"

        if df_tracker is not None:
            stats = df_tracker.get_stats()
            epoch_msg += f", Top-1 DF: {stats['top1_df_pct']:.1f}%"

        # Validation-based early stopping when validation data is provided
        if val_texts is not None and val_labels is not None:
            model.eval()
            val_encoding = tokenize_batch(val_texts, tokenizer, data_config.max_length)
            val_label_tensor = torch.tensor(
                val_labels,
                dtype=torch.float32 if num_labels == 1 else torch.long,
            ).to(DEVICE)
            val_ids = val_encoding["input_ids"].to(DEVICE)
            val_mask = val_encoding["attention_mask"].to(DEVICE)
            with torch.inference_mode():
                with torch.amp.autocast(AUTOCAST_DEVICE_TYPE, dtype=COMPUTE_DTYPE, enabled=AUTOCAST_ENABLED):
                    val_logits, _ = model(val_ids, val_mask)
                    val_loss = (
                        criterion(val_logits.squeeze(-1), val_label_tensor)
                        if num_labels == 1
                        else criterion(val_logits, val_label_tensor.view(-1))
                    )
            stop_loss = val_loss.item()
            epoch_msg += f", Val Loss: {stop_loss:.4f}"
            model.train()
        else:
            stop_loss = average_loss

        print(epoch_msg)

        if early_stopping.step(stop_loss, model):
            model.load_state_dict(early_stopping.best_state)
            print(f"Early stopping at epoch {epoch_index + 1}")
            break

    model.eval()
