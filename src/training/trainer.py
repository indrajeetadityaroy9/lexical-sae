"""Top-level SPALF trainer: calibration, initialization, Phase 1, Phase 2."""

from __future__ import annotations

import torch

from src.config import CalibrationResult, SPALFConfig
from src.constants import (
    BETA_FAST,
    BETA_SLOW,
    C_ETA,
    EPS_NUM,
    LAMBDA_DISC_MAX,
    LOG_INTERVAL,
    OMEGA_O_INIT,
    RHO_0,
    S_DISC,
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
from src.training.phase1 import run_phase1
from src.training.phase2 import run_phase2
from src.whitening.whitener import SoftZCAWhitener


class DiscretizationSchedule:
    """Scheduled discretization penalty weight (§3.1).

    λ_disc(t) = 0                                    if t/T ≤ s_disc
    λ_disc(t) = λ_max · (t/T - s_disc) / (1 - s_disc)   if t/T > s_disc

    Default: silent for first 80% (s_disc=0.8), then linear ramp to λ_max=1.0.
    Late onset ensures correction doesn't interfere with early feature discovery.
    """

    def __init__(
        self,
        T_total: int,
        s_disc: float = 0.8,
        lambda_max: float = 1.0,
    ) -> None:
        self.T_total = T_total
        self.s_disc = s_disc
        self.lambda_max = lambda_max

    def get_lambda(self, step: int) -> float:
        """Get discretization penalty weight at current step."""
        r = step / self.T_total
        if r <= self.s_disc:
            return 0.0
        return self.lambda_max * (r - self.s_disc) / (1.0 - self.s_disc)


class SPALFTrainer:
    """Orchestrates the full SPALF training pipeline.

    Sequence:
    1. Create ActivationStore + Buffer
    2. Run calibration (covariance, whitener, threshold computation)
    3. Create and initialize StratifiedSAE
    4. Measure initial violations for CAPU calibration
    5. Create control system components + optimizer
    6. Run Phase 1 (constrained sparse optimization)
    7. Run Phase 2 (end-to-end causal calibration)
    8. Save checkpoint
    """

    def __init__(self, config: SPALFConfig) -> None:
        self.config = config
        self.device = DEVICE

    def train(self) -> StratifiedSAE:
        """Run full training pipeline. Returns trained SAE."""
        config = self.config
        set_seed(config.seed)

        # === 1. Create activation store and buffer ===
        print(f"Creating activation store for {config.model_name}")
        store = ActivationStore(
            model_name=config.model_name,
            hook_point=config.hook_point,
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=self.device,
        )

        buffer = ActivationBuffer(store, buffer_size=2**20)

        # === 2. Calibration: covariance, whitener, thresholds ===
        print("Running calibration phase...")
        cal = run_calibration(config, store)
        print(f"d={cal.d}, F={cal.F}, L0_target={cal.L0_target}, V={cal.V}")

        # === 3. Create and initialize SAE ===
        print("Initializing StratifiedSAE...")
        sae = initialize_from_calibration(cal, store)
        sae = sae.to(self.device)
        print(f"SAE initialized: d={sae.d}, F={sae.F}, V={sae.V}, F_free={sae.F_free}")

        # === 4. Measure initial violations for CAPU calibration ===
        initial_violations = self._measure_initial_violations(
            sae, cal.whitener, cal.W_vocab, buffer, cal, config.batch_size
        )
        print(f"Initial violations: {initial_violations.tolist()}")

        # === 5. Create control system components ===
        ema = DualRateEMA(
            n_constraints=3,
            beta_fast=BETA_FAST,
            beta_slow=BETA_SLOW,
            device=self.device,
        )

        adrc = ADRCController(
            n_constraints=3,
            omega_o_init=OMEGA_O_INIT,
            device=self.device,
        )

        capu = ModifiedCAPU(
            initial_violations=initial_violations,
            c_eta=C_ETA,
            rho_0=RHO_0,
            beta_slow=BETA_SLOW,
            eps_num=EPS_NUM,
            device=self.device,
        )

        total_steps = config.total_tokens // config.batch_size
        disc_schedule = DiscretizationSchedule(
            T_total=total_steps,
            s_disc=S_DISC,
            lambda_max=LAMBDA_DISC_MAX,
        )

        optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(0.9, 0.999))

        metrics_logger = MetricsLogger(log_interval=LOG_INTERVAL)
        metrics_logger.log_calibration(
            cal.whitener.alpha, cal.whitener.effective_rank, cal.V
        )

        # === 6. Phase 1 ===
        print("Starting Phase 1 training...")
        phase1_end_step = run_phase1(
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
        )

        self._save_checkpoint(
            sae, cal, config, adrc, capu, ema, optimizer,
            phase1_end_step, phase=1,
        )

        # === 7. Phase 2 ===
        print("Starting Phase 2 training...")
        run_phase2(
            sae=sae,
            whitener=cal.whitener,
            W_vocab=cal.W_vocab,
            buffer=buffer,
            store=store,
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
        )

        # === 8. Save final checkpoint ===
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

        x = buffer.next_batch(batch_size).to(self.device)
        x_tilde = whitener.forward(x)
        x_hat, z, _, _ = sae(x_tilde)

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
