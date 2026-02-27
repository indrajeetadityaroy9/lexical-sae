"""Top-level SPALF trainer."""

from __future__ import annotations

import torch

from src.config import CalibrationResult, SPALFConfig
from src.constants import (
    BETA_SLOW,
    C_ETA,
    C_FRAME,
    EPS_NUM,
    LAMBDA_DISC_MAX,
    LOG_INTERVAL,
    OMEGA_O_INIT,
    RHO_0,
)
from src.control.adrc import ADRCController
from src.control.capu import ModifiedCAPU
from src.control.ema_filter import DualRateEMA
from src.data.activation_store import ActivationStore
from src.data.buffer import ActivationBuffer
from src.model.sae import StratifiedSAE
from src.runtime import DEVICE, set_seed
from src.training.calibration import initialize_from_calibration, run_calibration
from src.training.logging import MetricsLogger
from src.training.loop import DiscretizationSchedule, run_training_loop
from src.whitening.whitener import SoftZCAWhitener


class SPALFTrainer:
    """Orchestrates calibration, training phases, and checkpointing."""

    def __init__(self, config: SPALFConfig) -> None:
        self.config = config

    def train(self) -> StratifiedSAE:
        """Run full training pipeline."""
        config = self.config
        set_seed(config.seed)

        print(f"Creating activation store for {config.model_name}")
        store = ActivationStore(
            model_name=config.model_name,
            hook_point=config.hook_point,
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
        )

        buffer = ActivationBuffer(store, buffer_size=2**20)

        print("Running calibration phase...")
        cal = run_calibration(config, store)
        print(f"d={cal.d}, F={cal.F}, L0_target={cal.L0_target}, V={cal.V}")

        print("Initializing StratifiedSAE...")
        sae = initialize_from_calibration(cal, store)
        sae = sae.to(DEVICE)
        sae = torch.compile(sae, mode="max-autotune")
        print(f"SAE initialized: d={sae.d}, F={sae.F}, V={sae.V}, F_free={sae.F_free}")

        lambda_frame = C_FRAME / sae.d

        # Derive constants from BETA_SLOW (two-timescale theory)
        beta_fast = 1.0 - 10.0 * (1.0 - BETA_SLOW)
        slow_update_interval = round(1.0 / (1.0 - BETA_SLOW))
        phase_transition_patience = slow_update_interval

        initial_violations = self._measure_initial_violations(
            sae, cal.whitener, cal.W_vocab, buffer, cal, config.batch_size
        )
        print(f"Initial violations: {initial_violations.tolist()}")

        ema = DualRateEMA(
            n_constraints=3,
            beta_fast=beta_fast,
            beta_slow=BETA_SLOW,
        )

        adrc = ADRCController(
            n_constraints=3,
            omega_o_init=OMEGA_O_INIT,
            beta_fast=beta_fast,
        )

        capu = ModifiedCAPU(
            initial_violations=initial_violations,
            c_eta=C_ETA,
            rho_0=RHO_0,
            beta_slow=BETA_SLOW,
            eps_num=EPS_NUM,
        )

        total_steps = config.total_tokens // config.batch_size
        disc_schedule = DiscretizationSchedule(
            T_total=total_steps,
            lambda_max=LAMBDA_DISC_MAX,
        )

        optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(0.9, 0.999))

        metrics_logger = MetricsLogger(log_interval=LOG_INTERVAL)
        metrics_logger.log_calibration(
            cal.whitener.alpha, cal.whitener.effective_rank, cal.V
        )

        print("Starting Phase 1 training...")
        phase1_end_step = run_training_loop(
            phase=1,
            sae=sae,
            whitener=cal.whitener,
            W_vocab=cal.W_vocab,
            buffer=buffer,
            config=config,
            tau_faith=cal.tau_faith,
            tau_drift=cal.tau_drift,
            tau_ortho=cal.tau_ortho,
            adrc=adrc,
            capu=capu,
            ema=ema,
            disc_schedule=disc_schedule,
            optimizer=optimizer,
            metrics_logger=metrics_logger,
            lambda_frame=lambda_frame,
            slow_update_interval=slow_update_interval,
            phase_transition_patience=phase_transition_patience,
        )

        self._save_checkpoint(
            sae, cal, config, adrc, capu, ema, optimizer,
            phase1_end_step, phase=1,
        )

        disc_schedule.set_onset(phase1_end_step)

        print("Starting Phase 2 training...")
        run_training_loop(
            phase=2,
            sae=sae,
            whitener=cal.whitener,
            W_vocab=cal.W_vocab,
            buffer=buffer,
            config=config,
            tau_faith=cal.tau_faith,
            tau_drift=cal.tau_drift,
            tau_ortho=cal.tau_ortho,
            adrc=adrc,
            capu=capu,
            ema=ema,
            disc_schedule=disc_schedule,
            optimizer=optimizer,
            metrics_logger=metrics_logger,
            start_step=phase1_end_step + 1,
            lambda_frame=lambda_frame,
            slow_update_interval=slow_update_interval,
            store=store,
        )

        self._save_checkpoint(
            sae, cal, config, adrc, capu, ema, optimizer,
            total_steps, phase=2,
        )

        print("Training complete.")
        return sae

    @torch.no_grad()
    def _measure_initial_violations(
        self,
        sae: StratifiedSAE,
        whitener: SoftZCAWhitener,
        W_vocab: torch.Tensor,
        buffer: ActivationBuffer,
        cal: CalibrationResult,
        batch_size: int,
    ) -> torch.Tensor:
        """Measure initial constraint violations for CAPU calibration."""
        from src.constraints import (
            compute_drift_violation,
            compute_faithfulness_violation,
            compute_orthogonality_violation,
        )

        x = buffer.next_batch(batch_size).to(DEVICE)
        x_tilde = whitener.forward(x)
        x_hat, z, _, _, _ = sae(x_tilde)

        v_faith = compute_faithfulness_violation(x, x_hat, whitener, cal.tau_faith)
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, cal.tau_drift)
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, cal.tau_ortho
        )

        return torch.stack([v_faith, v_drift, v_ortho]).abs()

    def _save_checkpoint(
        self,
        sae: StratifiedSAE,
        cal: CalibrationResult,
        config: SPALFConfig,
        adrc: ADRCController,
        capu: ModifiedCAPU,
        ema: DualRateEMA,
        optimizer: torch.optim.Optimizer,
        step: int,
        phase: int,
    ) -> None:
        """Save full training checkpoint using safetensors."""
        from src.checkpoint import save_checkpoint

        save_checkpoint(sae, cal, config, adrc, capu, ema, optimizer, step, phase)
