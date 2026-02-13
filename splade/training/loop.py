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
    FeatureFrequencyTracker,
    LossNormalizer,
    compute_completeness_loss,
    compute_frequency_penalty,
    compute_gate_sparsity_loss,
    compute_separation_loss,
)
from splade.training.constants import (
    EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING,
    MAX_EPOCHS,
)
from splade.training.optim import (_gradient_centralization,
                                   _build_param_groups,
                                   _infer_batch_size, build_optimizer)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled

_NUM_WORKERS = min(os.cpu_count() or 4, 16)
_PREFETCH_FACTOR = 4


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

    steps_per_epoch = -(-len(texts) // batch_size)
    total_steps = steps_per_epoch * MAX_EPOCHS

    _orig = unwrap_compiled(model)
    vocab_size = _orig.vocab_size_expanded
    classifier_params = set(_orig.classifier_parameters())

    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if num_labels == 1
        else torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    )

    # CIS circuit losses with GECO adaptive weighting
    centroid_tracker = AttributionCentroidTracker(
        num_classes=num_labels, vocab_size=vocab_size,
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
    )
    geco = GECOController(steps_per_epoch=steps_per_epoch)
    norm_cc = LossNormalizer()
    norm_sep = LossNormalizer()
    norm_gate = LossNormalizer()
    norm_freq = LossNormalizer()
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
    best_state: dict[str, torch.Tensor] | None = None

    model.train()
    optimizer.train()
    for epoch_index in range(MAX_EPOCHS):
        total_loss = torch.zeros(1, device=DEVICE)
        batch_count = 0

        for batch_ids, batch_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch_index + 1}/{MAX_EPOCHS}"):
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                sparse_seq, gate_mask, l0_probs = model(batch_ids, batch_mask)
                logits, sparse, W_eff, _ = _orig.classify(sparse_seq, batch_mask)
                classification_loss = (
                    criterion(logits.squeeze(-1), batch_labels)
                    if num_labels == 1
                    else criterion(logits, batch_labels.view(-1))
                )

                global_step += 1

                # Update feature frequency tracker every step
                freq_tracker.update(gate_mask.detach())

                # Reset patience at fixed point for both baseline and CIS
                if not patience_was_reset and global_step >= patience_reset_step:
                    patience_was_reset = True
                    patience_counter = 0

                if global_step < circuit_delay_steps:
                    # Warmup: CE only, record CE for GECO tau
                    loss = classification_loss
                    geco.record_warmup_ce(classification_loss.detach().item())
                else:
                    if not warmup_finalized:
                        tau = geco.finalize_warmup()
                        print(f"GECO: tau_ce={tau:.4f}")
                        warmup_finalized = True

                    cc_loss = compute_completeness_loss(
                        sparse, W_eff, batch_labels.view(-1),
                        _orig.classifier_logits_only,
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
                        freq_tracker, _orig.activation,
                    )

                    circuit_objective = norm_cc(cc_loss) + norm_sep(sep_loss) + norm_gate(gate_loss) + norm_freq(freq_loss)
                    loss = geco.compute_loss(classification_loss, circuit_objective)

                    centroid_tracker.update(
                        sparse.detach(), W_eff.detach(),
                        batch_labels.view(-1),
                    )

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
            ) if freq_tracker._step >= 50 else 0.0
            if under_active_frac > 0:
                epoch_msg += f", under-active={under_active_frac:.1%}"
            if abs(geco._log_lambda) >= 4.9:
                print(
                    f"\n  GECO lambda pinned at {geco.lambda_ce:.2f} "
                    f"(log_lambda={geco._log_lambda:.2f}, "
                    f"constraint_ema={geco._ema_constraint:.4f}). "
                    f"Model is {'sacrificing interpretability for accuracy' if geco._log_lambda > 0 else 'over-constraining accuracy'}."
                )
        else:
            epoch_msg += ", GECO: warming up"

        # Switch to averaged params for validation
        optimizer.eval()
        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                val_sparse_seq, *_ = model(val_ids_gpu, val_mask_gpu)
                val_logits = _orig.classify(val_sparse_seq, val_mask_gpu).logits
                val_loss = (
                    criterion(val_logits.squeeze(-1), val_label_gpu)
                    if num_labels == 1
                    else criterion(val_logits, val_label_gpu.view(-1))
                )
        val_loss_val = val_loss.item()
        epoch_msg += f", Val Loss: {val_loss_val:.4f}"

        # Switch back to training params
        optimizer.train()
        model.train()

        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            patience_counter = 0
            # Save averaged params (currently active via optimizer.eval() -> optimizer.train())
            best_state = {
                name: param.data.clone()
                for name, param in unwrap_compiled(model).named_parameters()
                if param.requires_grad
            }
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            epoch_msg += f" [Early stop: no improvement for {EARLY_STOP_PATIENCE} epochs]"
            print(epoch_msg)
            break

        print(epoch_msg)

    # Restore best averaged weights
    optimizer.eval()
        _orig_model = unwrap_compiled(model)
        for name, param in _orig_model.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name])
        print(f"Applied best weights (val loss: {best_val_loss:.4f})")

    model.eval()

    return centroid_tracker
