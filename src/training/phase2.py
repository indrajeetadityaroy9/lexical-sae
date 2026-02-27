"""Phase 2 training loop: end-to-end causal calibration (section 7.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import SPALFConfig
from src.constants import SLOW_UPDATE_INTERVAL
from src.constraints import (
    compute_augmented_lagrangian,
    compute_drift_violation,
    compute_faithfulness_violation_phase2,
    compute_orthogonality_violation,
)
from src.control.adrc import ADRCController
from src.control.capu import ModifiedCAPU
from src.control.ema_filter import DualRateEMA
from src.data.activation_store import ActivationStore
from src.data.buffer import ActivationBuffer
from src.data.patching import run_patched_forward
from src.model.sae import StratifiedSAE
from src.training.trainer import DiscretizationSchedule
from src.training.logging import MetricsLogger, StepMetrics
from src.whitening.whitener import SoftZCAWhitener


def run_phase2(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    buffer: ActivationBuffer,
    store: ActivationStore,
    config: SPALFConfig,
    tau_faith: float,
    tau_drift: float,
    tau_ortho: float,
    adrc: ADRCController,
    capu: ModifiedCAPU,
    ema: DualRateEMA,
    disc_schedule: DiscretizationSchedule,
    optimizer: torch.optim.Optimizer,
    metrics_logger: MetricsLogger,
    start_step: int,
) -> None:
    """Phase 2 end-to-end causal calibration (section 7.5).

    Replaces Phase 1 faithfulness constraint with MSE/KL blend.
    CAPU rho is frozen to prevent transient penalty spikes from the new KL signal.
    """
    device = W_vocab.device
    total_steps = config.total_tokens // config.batch_size

    capu.freeze()
    print(f"Phase 2: CAPU frozen at rho={capu.rhos.tolist()}")

    kl_running_mean = torch.tensor(1.0, device=device)
    kl_beta = 0.99

    token_iter = store._token_generator()

    print(f"Starting Phase 2 at step {start_step}: remaining {total_steps - start_step} steps")

    for step in range(start_step, total_steps):
        # === Get tokens for KL computation ===
        tokens = next(token_iter).to(device)

        # === KL divergence computation ===
        orig_logits, patched_logits, _, _, _, _ = run_patched_forward(
            store, sae, whitener, tokens
        )
        log_p_orig = F.log_softmax(orig_logits[:, :-1].float(), dim=-1)
        log_p_patched = F.log_softmax(patched_logits[:, :-1].float(), dim=-1)
        kl_div = F.kl_div(log_p_patched, log_p_orig.exp(), reduction="batchmean")

        with torch.no_grad():
            kl_running_mean = kl_beta * kl_running_mean + (1 - kl_beta) * kl_div

        # === Get activations from buffer for SAE training ===
        x = buffer.next_batch(config.batch_size).to(device)
        x_tilde = whitener.forward(x)
        lambda_disc = disc_schedule.get_lambda(step)
        x_hat, z, gate_mask, l0_probs, disc_raw = sae.forward_fused(x_tilde, lambda_disc)

        # === Compute Phase 2 faithfulness (MSE/KL blend) ===
        v_faith = compute_faithfulness_violation_phase2(
            x, x_hat, kl_div, tau_faith, kl_running_mean
        )

        # === Drift and orthogonality unchanged ===
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, tau_drift)
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, tau_ortho
        )
        violations = torch.stack([v_faith, v_drift, v_ortho])

        # === EMA update ===
        ema.update(violations)

        # === L0 loss + discretization ===
        disc_correction = disc_raw.mean()
        l0_loss = l0_probs.mean()
        l0_corr = l0_loss + disc_correction

        # === Augmented Lagrangian ===
        lagrangian = compute_augmented_lagrangian(
            l0_corr=l0_corr,
            v_fast=ema.v_fast,
            lambdas=adrc.lambdas,
            rhos=capu.rhos,
        )

        # === Adam step ===
        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

        # === ADRC dual update ===
        adrc.step(ema.v_fast, ema.v_fast_prev, ema.v_slow)

        # === omega_o update only (CAPU is frozen) ===
        if step % SLOW_UPDATE_INTERVAL == 0 and step > start_step:
            adrc.update_omega(ema.v_fast, ema.v_fast_prev)

        # === Normalize free decoder ===
        sae.normalize_free_decoder()

        # === Logging ===
        with torch.no_grad():
            l0_mean = gate_mask.sum(dim=1).mean().item()
            mse = (x - x_hat).pow(2).sum(dim=1).mean().item()
            x_var = x.pow(2).sum(dim=1).mean().item()
            r_squared = 1.0 - mse / x_var

        metrics = StepMetrics(
            step=step,
            phase=2,
            l0_mean=l0_mean,
            l0_corr=l0_corr.item(),
            lagrangian=lagrangian.item(),
            v_faith=v_faith.item(),
            v_drift=v_drift.item(),
            v_ortho=v_ortho.item(),
            v_fast_faith=ema.v_fast[0].item(),
            v_fast_drift=ema.v_fast[1].item(),
            v_fast_ortho=ema.v_fast[2].item(),
            lambda_faith=adrc.lambdas[0].item(),
            lambda_drift=adrc.lambdas[1].item(),
            lambda_ortho=adrc.lambdas[2].item(),
            rho_faith=capu.rhos[0].item(),
            rho_drift=capu.rhos[1].item(),
            rho_ortho=capu.rhos[2].item(),
            omega_o=adrc.omega_o,
            lambda_disc=lambda_disc,
            disc_correction=disc_correction.item(),
            mse=mse,
            r_squared=r_squared,
            kl_div=kl_div.item(),
        )
        metrics_logger.log_step(metrics)

    print(f"Phase 2 complete at step {total_steps - 1}")
