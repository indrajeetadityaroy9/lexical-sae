"""Unified training loop for Phase 1 and Phase 2."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import SPALFConfig
from src.constants import C_EPSILON, PHASE_TRANSITION_FALLBACK
from src.constraints import (
    compute_augmented_lagrangian,
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_faithfulness_violation_phase2,
    compute_frame_energy,
    compute_orthogonality_violation,
)
from src.control.adrc import ADRCController
from src.control.capu import ModifiedCAPU
from src.control.ema_filter import DualRateEMA
from src.data.activation_store import ActivationStore
from src.data.buffer import ActivationBuffer
from src.data.patching import run_patched_forward
from src.model.sae import StratifiedSAE
from src.runtime import DEVICE
from src.training.logging import MetricsLogger, StepMetrics
from src.whitening.whitener import SoftZCAWhitener


class DiscretizationSchedule:
    """Phase-triggered discretization weight schedule.

    Ramps linearly from onset_step to T_total. No disc penalty before onset.
    """

    def __init__(self, T_total: int, lambda_max: float = 1.0) -> None:
        self.T_total = T_total
        self.onset_step = T_total  # no disc until set
        self.lambda_max = lambda_max

    def set_onset(self, step: int) -> None:
        """Set the step at which disc ramp begins (typically Phase 2 start)."""
        self.onset_step = step

    def get_lambda(self, step: int) -> float:
        """Return the discretization penalty at step."""
        if step <= self.onset_step:
            return 0.0
        remaining = self.T_total - self.onset_step
        if remaining <= 0:
            return self.lambda_max
        return self.lambda_max * (step - self.onset_step) / remaining


def run_training_loop(
    phase: int,
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
    lambda_frame: float = 0.0,
    slow_update_interval: int = 100,
    phase_transition_patience: int = 100,
    store: ActivationStore | None = None,
) -> int:
    """Run a training phase and return the last step executed.

    Phase 1: whitened-MSE faithfulness, CAPU updates, phase transition logic.
    Phase 2: KL-aware faithfulness, CAPU frozen, runs to completion.
    """
    total_steps = config.total_tokens // config.batch_size

    # Phase-specific initialization
    kl_div_value = 0.0
    if phase == 1:
        fallback_step = int(PHASE_TRANSITION_FALLBACK * total_steps)
        consecutive_satisfied = 0
        print(f"Starting Phase 1: total_steps={total_steps}, tau_faith={tau_faith:.4f}")
    else:
        capu.freeze()
        print(f"Phase 2: CAPU frozen at rho={capu.rhos.tolist()}")
        kl_sum = torch.tensor(0.0, device=DEVICE)
        kl_count = 0
        token_iter = store._token_generator()
        print(f"Starting Phase 2 at step {start_step}: remaining {total_steps - start_step} steps")

    log_interval = metrics_logger.log_interval

    for step in range(start_step, total_steps):
        # Phase 2 preamble: KL computation via patched forward
        if phase == 2:
            tokens = next(token_iter).to(DEVICE)
            orig_logits, patched_logits, _, _, _, _ = run_patched_forward(
                store, sae, whitener, tokens
            )
            log_p_orig = F.log_softmax(orig_logits[:, :-1].float(), dim=-1)
            log_p_patched = F.log_softmax(patched_logits[:, :-1].float(), dim=-1)
            kl_div = F.kl_div(log_p_patched, log_p_orig.exp(), reduction="batchmean")

            with torch.no_grad():
                kl_sum += kl_div.detach()
                kl_count += 1
                kl_running_mean = kl_sum / kl_count

        # Shared: batch, whiten, fused forward (under AMP)
        x = buffer.next_batch(config.batch_size).to(DEVICE)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_tilde = whitener.forward(x)
            lambda_disc = disc_schedule.get_lambda(step)
            x_hat, z, gate_mask, l0_probs, disc_raw = sae(x_tilde, lambda_disc)

            # Faithfulness: phase-specific
            if phase == 1:
                v_faith = compute_faithfulness_violation(x, x_hat, whitener, tau_faith)
            else:
                v_faith = compute_faithfulness_violation_phase2(
                    x, x_hat, kl_div, tau_faith, kl_running_mean
                )

            v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, tau_drift)
            v_ortho = compute_orthogonality_violation(
                z, sae.W_dec_A, sae.W_dec_B, tau_ortho
            )
            violations = torch.stack([v_faith, v_drift, v_ortho])

            ema.update(violations)

            disc_correction = disc_raw.mean()
            l0_loss = l0_probs.mean()
            l0_corr = l0_loss + disc_correction

            # Frame energy regularization (amortized)
            _frame_energy = 0.0
            if step % slow_update_interval == 0 and step > start_step:
                fe = compute_frame_energy(sae.W_dec_B, sae.d)
                l0_corr = l0_corr + lambda_frame * fe
                _frame_energy = fe.item()

            lagrangian = compute_augmented_lagrangian(
                l0_corr=l0_corr,
                v_fast=violations,
                lambdas=adrc.lambdas,
                rhos=capu.rhos,
            )

        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

        adrc.step(ema.v_fast, ema.v_fast_prev, ema.v_slow)

        if step % slow_update_interval == 0 and step > start_step:
            if phase == 1:
                capu.step(ema.v_fast)
            adrc.update_omega(ema.v_fast, ema.v_fast_prev)
            with torch.no_grad():
                pre_act = x_tilde @ sae.W_enc.T + sae.b_enc
                sae.recalibrate_gamma(pre_act, C_EPSILON)

        sae.normalize_free_decoder()

        # Phase 1: transition check (single sync point per step)
        if phase == 1:
            all_satisfied = (ema.v_fast < 0).all().item()
            if all_satisfied:
                consecutive_satisfied += 1
            else:
                consecutive_satisfied = 0

            if consecutive_satisfied >= phase_transition_patience:
                metrics_logger.log_phase_transition(
                    step, f"all constraints satisfied for {phase_transition_patience} consecutive steps"
                )
                return step

            if step >= fallback_step:
                metrics_logger.log_phase_transition(
                    step, f"fallback at {PHASE_TRANSITION_FALLBACK*100:.0f}% of training"
                )
                return step

        # Metrics extraction deferred to log interval (avoids 20+ GPU syncs per step)
        if step % log_interval == 0:
            with torch.no_grad():
                l0_mean = gate_mask.sum(dim=1).mean().item()
                mse = (x - x_hat).pow(2).sum(dim=1).mean().item()
                x_var = x.pow(2).sum(dim=1).mean().item()
                r_squared = 1.0 - mse / x_var
                kl_div_value = kl_div.item() if phase == 2 else 0.0
                frame_energy = _frame_energy

            omega_o = adrc.omega_o
            metrics = StepMetrics(
                step=step,
                phase=phase,
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
                omega_o_faith=omega_o[0].item(),
                omega_o_drift=omega_o[1].item(),
                omega_o_ortho=omega_o[2].item(),
                lambda_disc=lambda_disc,
                disc_correction=disc_correction.item(),
                mse=mse,
                r_squared=r_squared,
                kl_div=kl_div_value,
                frame_energy=frame_energy,
            )
            metrics_logger.log_step(metrics)

    if phase == 1:
        print("Phase 1 completed all steps without phase transition")
    else:
        print(f"Phase 2 complete at step {total_steps - 1}")
    return total_steps - 1
