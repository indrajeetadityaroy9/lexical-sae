import os
import random

import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from cajt.data import infer_max_length
from cajt.training.geco import GECOController, _LOG_LAMBDA_WARN
from cajt.training.losses import (
    AttributionCentroidTracker,
    FeatureFrequencyTracker,
    LossNormalizer,
    compute_completeness_loss,
    compute_frequency_penalty,
    compute_gate_sparsity_loss,
    compute_separation_loss,
)
from cajt.training.optim import (_gradient_centralization,
                                   _infer_batch_size, build_optimizer)
from cajt.runtime import autocast, DEVICE


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


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
    sparsity_target: float = 0.1,
    warmup_fraction: float = 0.2,
    learning_rate: float = 3e-4,
    max_epochs: int = 50,
    early_stop_patience: int = 10,
    label_smoothing: float = 0.1,
    num_workers: int | None = None,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> "AttributionCentroidTracker":
    if num_workers is None:
        num_workers = min(os.cpu_count(), 16)

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
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"], label_tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
        generator=g,
    )

    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * max_epochs

    vocab_size = model.vocab_size_expanded
    classifier_params = set(model.classifier_parameters())

    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    )

    # CIS circuit losses with GECO adaptive weighting
    centroid_tracker = AttributionCentroidTracker(
        num_classes=num_labels, vocab_size=vocab_size,
        steps_per_epoch=steps_per_epoch,
    ).to(DEVICE)

    # Schedule-Free AdamW: subsumes LR scheduling, warmup, and model EMA
    # Build optimizer after centroid_tracker so its learned margin is included
    optimizer = build_optimizer(model, base_lr=learning_rate)
    # Add centroid_tracker's learned margin to the optimizer
    optimizer.add_param_group({
        "params": [centroid_tracker.log_margin],
        "lr": learning_rate,
        "weight_decay": 0.0,
    })
    print(f"Schedule-Free AdamW: lr={learning_rate:.2e}")
    freq_tracker = FeatureFrequencyTracker(
        num_features=vocab_size, target_sparsity=sparsity_target,
        steps_per_epoch=steps_per_epoch,
    )
    geco = GECOController(steps_per_epoch=steps_per_epoch)
    norm_cc = LossNormalizer(steps_per_epoch)
    norm_sep = LossNormalizer(steps_per_epoch)
    norm_gate = LossNormalizer(steps_per_epoch)
    norm_freq = LossNormalizer(steps_per_epoch)
    circuit_delay_steps = int(warmup_fraction * total_steps)
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
    best_state: dict[str, torch.Tensor] | None = None

    model.train()
    optimizer.train()
    for epoch_index in range(max_epochs):
        total_loss = torch.zeros(1, device=DEVICE)
        batch_count = 0

        for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch_index + 1}/{max_epochs}"):
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                sparse_seq, gate_mask, l0_probs = model(batch_ids, batch_mask)
                logits, sparse, W_eff, _ = model.classify(sparse_seq, batch_mask)
                classification_loss = (
                    criterion(logits.squeeze(-1), batch_labels)
                    if num_labels == 1
                    else criterion(logits, batch_labels.view(-1))
                )

                freq_tracker.update(gate_mask.detach())

                if global_step < circuit_delay_steps:
                    loss = classification_loss
                    geco.record_warmup_ce(classification_loss.detach().item())
                else:
                    if not warmup_finalized:
                        tau = geco.finalize_warmup()
                        print(f"GECO: tau_ce={tau:.4f}")
                        warmup_finalized = True
                        patience_counter = 0  # reset patience at warmup→CIS transition

                    cc_loss = compute_completeness_loss(
                        sparse, W_eff, batch_labels.view(-1),
                        model.classifier_logits_only,
                        circuit_fraction=sparsity_target,
                        full_logits=logits,
                    )
                    sep_loss = compute_separation_loss(
                        centroid_tracker,
                        sparse_vector=sparse,
                        W_eff=W_eff,
                        labels=batch_labels.view(-1),
                    )
                    gate_loss = compute_gate_sparsity_loss(l0_probs)
                    freq_loss = compute_frequency_penalty(
                        freq_tracker, model.activation,
                    )

                    circuit_objective = norm_cc(cc_loss) + norm_sep(sep_loss) + norm_gate(gate_loss) + norm_freq(freq_loss)
                    loss = geco.compute_loss(classification_loss, circuit_objective)

                    centroid_tracker.update(
                        sparse.detach(), W_eff.detach(),
                        batch_labels.view(-1),
                    )

            global_step += 1

            loss.backward()
            _gradient_centralization(model, skip_params=classifier_params)
            optimizer.step()

            total_loss += loss.detach()
            batch_count += 1

        average_loss = total_loss.item() / batch_count
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}"

        if warmup_finalized:
            epoch_msg += f", GECO lambda={geco.lambda_ce:.4f}"
            under_active_frac = float(
                (freq_tracker.freq_ema < freq_tracker.target_freq * 0.1).float().mean().item()
            ) if freq_tracker.warmup_weight > 0 else 0.0
            if under_active_frac > 0:
                epoch_msg += f", under-active={under_active_frac:.1%}"
            if abs(geco._log_lambda) >= _LOG_LAMBDA_WARN:
                print(
                    f"\n  GECO lambda pinned at {geco.lambda_ce:.2f} "
                    f"(log_lambda={geco._log_lambda:.2f}, "
                    f"constraint_ema={geco._ema_constraint:.4f}). "
                    f"Model is {'sacrificing interpretability for accuracy' if geco._log_lambda > 0 else 'over-constraining accuracy'}."
                )
        else:
            epoch_msg += ", GECO: warming up"

        # CRITICAL: Schedule-Free AdamW ordering contract
        # 1. optimizer.eval()  → switch to averaged params
        # 2. model.eval()      → disable dropout
        # 3. validate + save   → uses averaged params
        # 4. optimizer.train() → switch back to training params
        # 5. model.train()     → re-enable dropout
        # best_state MUST be saved between steps 1-3.
        optimizer.eval()
        model.eval()
        with torch.inference_mode():
            with autocast():
                val_sparse_seq, *_ = model(val_ids_gpu, val_mask_gpu)
                val_logits = model.classify(val_sparse_seq, val_mask_gpu).logits
                val_loss = (
                    criterion(val_logits.squeeze(-1), val_label_gpu)
                    if num_labels == 1
                    else criterion(val_logits, val_label_gpu.view(-1))
                )
        val_loss_val = val_loss.item()
        epoch_msg += f", Val Loss: {val_loss_val:.4f}"

        # Save best averaged params WHILE optimizer.eval() is still active
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            patience_counter = 0
            # Averaged params are currently active (optimizer.eval() above)
            best_state = {
                name: param.data.clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
        else:
            patience_counter += 1

        # Switch back to training params for next epoch
        optimizer.train()
        model.train()

        if patience_counter >= early_stop_patience:
            epoch_msg += f" [Early stop: no improvement for {early_stop_patience} epochs]"
            print(epoch_msg)
            break

        print(epoch_msg)

    # Restore best averaged weights
    optimizer.eval()
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name])
        print(f"Applied best weights (val loss: {best_val_loss:.4f})")

    model.eval()

    return centroid_tracker
