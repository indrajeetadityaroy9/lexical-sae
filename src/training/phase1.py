"""Phase 1 training loop."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import SPALFConfig
from src.constants import PHASE_TRANSITION_FALLBACK, PHASE_TRANSITION_PATIENCE, SLOW_UPDATE_INTERVAL
from src.constraints import (
    compute_augmented_lagrangian,
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_orthogonality_violation,
)
from src.control.adrc import ADRCController
from src.control.capu import ModifiedCAPU
from src.control.ema_filter import DualRateEMA
from src.data.buffer import ActivationBuffer
from src.model.sae import StratifiedSAE
from src.training.trainer import DiscretizationSchedule
from src.training.logging import MetricsLogger, StepMetrics
from src.whitening.whitener import SoftZCAWhitener


def run_phase1(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    buffer: ActivationBuffer,
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
    start_step: int = 0,
) -> int:
    """Run Phase 1 and return the transition step."""
    device = W_vocab.device
    total_steps = config.total_tokens // config.batch_size
    fallback_step = int(PHASE_TRANSITION_FALLBACK * total_steps)
    consecutive_satisfied = 0

    print(f"Starting Phase 1: total_steps={total_steps}, tau_faith={tau_faith:.4f}")

    for step in range(start_step, total_steps):
        x = buffer.next_batch(config.batch_size).to(device)
        x_tilde = whitener.forward(x)

        lambda_disc = disc_schedule.get_lambda(step)
        x_hat, z, gate_mask, l0_probs, disc_raw = sae.forward_fused(x_tilde, lambda_disc)

        v_faith = compute_faithfulness_violation(x, x_hat, whitener, tau_faith)
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, tau_drift)
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, tau_ortho
        )
        violations = torch.stack([v_faith, v_drift, v_ortho])

        ema.update(violations)

        disc_correction = disc_raw.mean()
        l0_loss = l0_probs.mean()
        l0_corr = l0_loss + disc_correction

        lagrangian = compute_augmented_lagrangian(
            l0_corr=l0_corr,
            v_fast=ema.v_fast,
            lambdas=adrc.lambdas,
            rhos=capu.rhos,
        )

        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

        adrc.step(ema.v_fast, ema.v_fast_prev, ema.v_slow)

        if step % SLOW_UPDATE_INTERVAL == 0 and step > 0:
            capu.step(ema.v_fast)
            adrc.update_omega(ema.v_fast, ema.v_fast_prev)

        sae.normalize_free_decoder()

        with torch.no_grad():
            l0_mean = gate_mask.sum(dim=1).mean().item()
            mse = (x - x_hat).pow(2).sum(dim=1).mean().item()
            x_var = x.pow(2).sum(dim=1).mean().item()
            r_squared = 1.0 - mse / x_var

        metrics = StepMetrics(
            step=step,
            phase=1,
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
        )
        metrics_logger.log_step(metrics)

        all_satisfied = (ema.v_fast < 0).all().item()
        if all_satisfied:
            consecutive_satisfied += 1
        else:
            consecutive_satisfied = 0

        if consecutive_satisfied >= PHASE_TRANSITION_PATIENCE:
            metrics_logger.log_phase_transition(
                step, f"all constraints satisfied for {PHASE_TRANSITION_PATIENCE} consecutive steps"
            )
            return step

        if step >= fallback_step:
            metrics_logger.log_phase_transition(
                step, f"fallback at {PHASE_TRANSITION_FALLBACK*100:.0f}% of training"
            )
            return step

    print("Phase 1 completed all steps without phase transition")
    return total_steps - 1
