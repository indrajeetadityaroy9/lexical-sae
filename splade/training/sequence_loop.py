"""GECO-integrated training loop for sequence labelling (NER).

Two-phase training identical to loop.py but with token-level CE
(ignore_index=-100) and token-level circuit losses via _gather_valid_positions.

All losses use reduction='mean' over valid tokens, keeping magnitudes
comparable to classification for GECO stability.
"""

import copy
import itertools
import os

import torch
import torch.nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from splade.circuits.geco import GECOController
from splade.circuits.sequence_losses import (
    TokenAttributionCentroidTracker,
    _gather_valid_positions,
    compute_token_completeness_loss,
    compute_token_separation_loss,
    compute_token_sharpness_loss,
)
from splade.data.ner_loader import IGNORE_INDEX
from splade.training.constants import (
    EARLY_STOP_PATIENCE,
    EMA_DECAY,
    LR_FIND_DIVERGE_FACTOR,
    LR_FIND_END,
    LR_FIND_STEPS,
    MAX_EPOCHS,
    WARMUP_RATIO,
)
from splade.training.loop import ModelEMA
from splade.training.optim import (
    _build_param_groups_with_gate_boost,
    _gradient_centralization,
    _LRScheduler,
)
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled

_NUM_WORKERS = min(os.cpu_count() or 4, 16)
_PREFETCH_FACTOR = 4


def _find_lr_ner(
    model: torch.nn.Module,
    train_loader: DataLoader,
    num_labels: int,
) -> float:
    """LR range test adapted for token-level NER loss."""
    _orig = unwrap_compiled(model)
    saved_state = copy.deepcopy(_orig.state_dict())

    temp_optimizer = torch.optim.AdamW(_orig.parameters(), lr=LR_FIND_END, fused=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    start_lr = 1e-7
    lr_mult = (LR_FIND_END / start_lr) ** (1.0 / LR_FIND_STEPS)
    current_lr = start_lr
    best_loss = float("inf")
    lrs: list[float] = []
    losses: list[float] = []

    data_iter = itertools.cycle(train_loader)
    _orig.train()

    import numpy as np

    for _ in range(LR_FIND_STEPS):
        batch = next(data_iter)
        batch_ids, batch_mask, batch_labels = (
            b.to(DEVICE, non_blocking=True) for b in batch
        )

        for g in temp_optimizer.param_groups:
            g["lr"] = current_lr

        temp_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
            sparse_seq = model(batch_ids, batch_mask)
            token_logits = _orig.tag(sparse_seq)
            # token_logits: [B, L, C] -> [B*L, C]
            B, L, C = token_logits.shape
            logits_flat = token_logits.view(B * L, C)
            labels_flat = batch_labels.view(B * L)
            loss = criterion(logits_flat, labels_flat)

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
    return max(1e-6, min(1e-3, found_lr))


def train_sequence_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    token_labels: torch.Tensor,
    model_name: str,
    num_labels: int,
    val_input_ids: torch.Tensor,
    val_attention_masks: torch.Tensor,
    val_token_labels: torch.Tensor,
    batch_size: int,
    target_accuracy: float | None = None,
    sparsity_target: float = 0.1,
    warmup_fraction: float = 0.2,
    gradient_accumulation_steps: int = 1,
) -> TokenAttributionCentroidTracker:
    """Two-phase GECO training for sequence labelling.

    Phase 1 (warmup): CE only, records CE for GECO tau derivation.
    Phase 2 (main): GECO-constrained circuit optimization with token-level losses.

    All CE and circuit losses use reduction='mean' over valid tokens.

    Args:
        model: LexicalSAE (sequence_labeling).
        input_ids: [N_train, L] token IDs.
        attention_masks: [N_train, L] attention masks.
        token_labels: [N_train, L] per-token labels (IGNORE_INDEX for skip).
        model_name: Model name string.
        num_labels: Number of NER tags.
        val_input_ids: [N_val, L] validation token IDs.
        val_attention_masks: [N_val, L] validation attention masks.
        val_token_labels: [N_val, L] validation labels.
        batch_size: Batch size.
        target_accuracy: GECO tau override (None = auto from warmup).
        sparsity_target: Circuit fraction for completeness loss.
        warmup_fraction: Fraction of training for CE-only warmup.
        gradient_accumulation_steps: Accumulate gradients over this many
            micro-batches before each optimizer step.

    Returns:
        TokenAttributionCentroidTracker with trained centroids.
    """
    loader = DataLoader(
        TensorDataset(input_ids, attention_masks, token_labels),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=_NUM_WORKERS,
        prefetch_factor=_PREFETCH_FACTOR,
        persistent_workers=True,
    )

    optimal_lr = _find_lr_ner(model, loader, num_labels)
    print(f"LR range test -> optimal_lr={optimal_lr:.2e}")

    param_groups = _build_param_groups_with_gate_boost(model, optimal_lr)
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    micro_batches_per_epoch = -(-len(input_ids) // batch_size)
    steps_per_epoch = -(-micro_batches_per_epoch // gradient_accumulation_steps)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))

    lr_scheduler = _LRScheduler(optimal_lr, total_steps, warmup_steps)

    _orig = unwrap_compiled(model)
    vocab_size = _orig.vocab_size
    classifier_params = set(_orig.classifier_parameters())

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    ema = ModelEMA(model, decay=EMA_DECAY)

    centroid_tracker = TokenAttributionCentroidTracker(
        num_tags=num_labels, vocab_size=vocab_size,
    )
    geco = GECOController(steps_per_epoch=steps_per_epoch)
    if target_accuracy is not None:
        geco.tau_ce = target_accuracy
    circuit_delay_steps = int(warmup_fraction * total_steps)
    patience_reset_step = int(warmup_fraction * total_steps)
    patience_was_reset = False
    global_step = 0
    warmup_finalized = False

    # Move validation data to GPU
    val_ids_gpu = val_input_ids.to(DEVICE)
    val_mask_gpu = val_attention_masks.to(DEVICE)
    val_labels_gpu = val_token_labels.to(DEVICE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_ema_state: dict[str, torch.Tensor] | None = None

    target_active_dims = sparsity_target * vocab_size

    accum = gradient_accumulation_steps
    model.train()
    for epoch_index in range(MAX_EPOCHS):
        total_loss = torch.zeros(1, device=DEVICE)
        batch_count = 0
        micro_step = 0
        epoch_active_dims_sum = 0.0
        epoch_active_dims_count = 0

        for batch_ids, batch_mask, batch_labels in tqdm(
            loader, desc=f"Epoch {epoch_index + 1}/{MAX_EPOCHS}"
        ):
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            is_accum_boundary = (micro_step % accum == 0)
            is_last_micro = ((micro_step + 1) % accum == 0) or (
                micro_step + 1 == micro_batches_per_epoch
            )

            if is_accum_boundary:
                learning_rate = lr_scheduler.step()
                for pg in optimizer.param_groups:
                    # Preserve relative LR multiplier (e.g. 5x for gate params)
                    multiplier = pg.get("_lr_multiplier", 1.0)
                    pg["lr"] = learning_rate * multiplier
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                sparse_seq = model(batch_ids, batch_mask)
                token_logits = _orig.tag(sparse_seq)
                # Token-level CE with ignore_index=-100
                B, L, C = token_logits.shape
                logits_flat = token_logits.view(B * L, C)
                labels_flat_ce = batch_labels.view(B * L)
                classification_loss = criterion(logits_flat, labels_flat_ce)

                if is_last_micro:
                    global_step += 1

                    if not patience_was_reset and global_step >= patience_reset_step:
                        patience_was_reset = True
                        patience_counter = 0

                if global_step < circuit_delay_steps:
                    loss = classification_loss
                    if target_accuracy is None and is_last_micro:
                        geco.record_warmup_ce(classification_loss.detach().item())
                else:
                    if not warmup_finalized:
                        if target_accuracy is None:
                            tau = geco.finalize_warmup()
                            print(f"GECO: tau_ce={tau:.4f}")
                        else:
                            print(f"GECO: tau_ce={target_accuracy:.4f} (from config)")
                        warmup_finalized = True

                    # Gather valid positions for circuit losses
                    sparse_flat, labels_flat = _gather_valid_positions(
                        sparse_seq, batch_labels, batch_mask,
                    )

                    # Subsample to cap W_eff memory: [N, C, V] with V~50K
                    # Scale with GPU VRAM: 64 baseline for 22GB usable
                    _base_positions = 64
                    try:
                        _gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        _usable_gb = max(1.0, _gpu_gb - 2.0)
                        _MAX_CIRCUIT_POSITIONS = min(512, int(_base_positions * (_usable_gb / 22.0)))
                    except Exception:
                        _MAX_CIRCUIT_POSITIONS = _base_positions
                    if sparse_flat.shape[0] > _MAX_CIRCUIT_POSITIONS:
                        idx = torch.randperm(
                            sparse_flat.shape[0], device=sparse_flat.device,
                        )[:_MAX_CIRCUIT_POSITIONS]
                        sparse_flat = sparse_flat[idx]
                        labels_flat = labels_flat[idx]

                    if sparse_flat.shape[0] > 0:
                        # Track active dims for sparsity-aware GECO
                        with torch.no_grad():
                            batch_active = (sparse_flat > 0).float().sum(-1).mean().item()
                            epoch_active_dims_sum += batch_active
                            epoch_active_dims_count += 1

                        # On-demand W_eff for sampled valid positions only
                        _, W_eff, _ = _orig.classifier_forward(sparse_flat)

                        cc_loss = compute_token_completeness_loss(
                            sparse_flat, W_eff, labels_flat,
                            _orig.classifier_logits_only,
                            circuit_fraction=sparsity_target,
                        )
                        sep_loss = compute_token_separation_loss(centroid_tracker)
                        sharp_loss = compute_token_sharpness_loss(
                            sparse_flat, W_eff, labels_flat,
                        )

                        _SPARSITY_GAIN = 10.0
                        circuit_objective = cc_loss + sep_loss + _SPARSITY_GAIN * sharp_loss
                        active_dims_ratio = batch_active / max(1.0, target_active_dims)
                        loss = geco.compute_loss_sparsity_aware(
                            classification_loss, circuit_objective, active_dims_ratio,
                        )

                        if is_last_micro:
                            centroid_tracker.update(
                                sparse_flat.detach(), W_eff.detach(), labels_flat,
                            )
                    else:
                        loss = classification_loss

                # Scale loss for gradient accumulation
                scaled_loss = loss / accum

            scaled_loss.backward()

            if is_last_micro:
                _gradient_centralization(model, skip_params=classifier_params)
                optimizer.step()
                ema.update(model)

            total_loss += loss.detach()
            batch_count += 1
            micro_step += 1

        average_loss = total_loss.item() / batch_count
        epoch_msg = f"Epoch {epoch_index + 1}: Loss = {average_loss:.4f}"

        if warmup_finalized:
            epoch_msg += f", GECO lambda={geco.lambda_ce:.4f}"
            if hasattr(geco, '_last_dynamic_tau') and geco._last_dynamic_tau is not None:
                epoch_msg += f", dyn_tau={geco._last_dynamic_tau:.4f}"
            if hasattr(geco, '_last_relaxation') and geco._last_relaxation is not None:
                epoch_msg += f", relax={geco._last_relaxation:.2f}"
            if epoch_active_dims_count > 0:
                mean_active = epoch_active_dims_sum / epoch_active_dims_count
                epoch_msg += f", active_dims={mean_active:.0f}"
            if abs(geco._log_lambda) >= 4.9:
                print(
                    f"\n  [WARNING] GECO lambda pinned at {geco.lambda_ce:.2f} "
                    f"(log_lambda={geco._log_lambda:.2f}). "
                )
        else:
            epoch_msg += ", GECO: warming up"

        # Validation with EMA weights
        ema.apply_shadow(model)
        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                val_sparse_seq = model(val_ids_gpu, val_mask_gpu)
                val_token_logits = _orig.tag(val_sparse_seq)
                B_v, L_v, C_v = val_token_logits.shape
                val_logits_flat = val_token_logits.view(B_v * L_v, C_v)
                val_labels_flat = val_labels_gpu.view(B_v * L_v)
                val_loss = criterion(val_logits_flat, val_labels_flat)
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
