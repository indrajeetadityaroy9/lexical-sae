"""Top-level SPALF trainer with Accelerate-managed checkpointing."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from src.config import CalibrationResult, SPALFConfig
from src.constants import (
    BETA_SLOW,
    EPS_NUM,
    LAMBDA_DISC_MAX,
)
from src.data.activation_store import ActivationStore
from src.data.buffer import ActivationBuffer
from src.optimization.capu import MonotoneCAPU
from src.optimization.discretization import DiscretizationSchedule
from src.optimization.dual_updater import DualUpdater
from src.optimization.ema_filter import DualRateEMA
from src.sae import StratifiedSAE
from src.training.calibration import initialize_from_calibration, run_calibration
from src.training.loop import run_training_loop
from src.whitening.whitener import SoftZCAWhitener

device = torch.device("cuda")


class SPALFTrainer:
    """Orchestrates calibration, training, and checkpointing."""

    def __init__(self, config: SPALFConfig) -> None:
        self.config = config

    def train(self) -> StratifiedSAE:
        """Run full training pipeline."""
        config = self.config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        project_config = ProjectConfiguration(
            project_dir=config.output_dir,
            automatic_checkpoint_naming=True,
            total_limit=3,
        )
        accelerator = Accelerator(
            mixed_precision="no",
            project_configuration=project_config,
        )

        print(
            json.dumps(
                {
                    "event": "train_start",
                    "model_name": config.model_name,
                    "output_dir": config.output_dir,
                    "seed": config.seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        store = ActivationStore(
            model_name=config.model_name,
            hook_point=config.hook_point,
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
        )

        buffer = ActivationBuffer(store, buffer_size=2**20)

        cal = run_calibration(config, store)

        sae = initialize_from_calibration(cal, store)
        sae = sae.to(device)
        sae = torch.compile(sae, mode="max-autotune")

        # R = ceil(log2(F/d)) enforces fast/slow separation for two-timescale updates.
        R = max(math.ceil(math.log2(cal.F / cal.d)), 2)
        beta_fast = 1.0 - R * (1.0 - BETA_SLOW)
        slow_update_interval = round(1.0 / (1.0 - BETA_SLOW))

        initial_violations = self._measure_initial_violations(
            sae, cal.whitener, cal.W_vocab, buffer, cal, config.batch_size
        )
        print(
            json.dumps(
                {
                    "event": "initial_violations",
                    "faith": initial_violations[0].item(),
                    "drift": initial_violations[1].item(),
                    "ortho": initial_violations[2].item(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        rho_0 = 1.0 / (initial_violations.abs().mean().item() + EPS_NUM)

        ema = DualRateEMA(
            n_constraints=3,
            beta_fast=beta_fast,
            beta_slow=BETA_SLOW,
        )

        dual_updater = DualUpdater(n_constraints=3)

        capu = MonotoneCAPU(
            initial_violations=initial_violations,
            rho_0=rho_0,
            beta_slow=BETA_SLOW,
            eps_num=EPS_NUM,
        )

        total_steps = config.total_tokens // config.batch_size
        disc_schedule = DiscretizationSchedule(
            T_total=total_steps,
            lambda_max=LAMBDA_DISC_MAX,
        )

        optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(0.9, 0.999))

        sae, optimizer = accelerator.prepare(sae, optimizer)
        accelerator.register_for_checkpointing(dual_updater, capu, ema, disc_schedule)

        start_step = 0
        if config.resume_from_checkpoint:
            accelerator.load_state(config.resume_from_checkpoint)
            with open(Path(config.resume_from_checkpoint) / "metadata.json") as f:
                start_step = json.load(f)["step"]
            print(
                json.dumps(
                    {
                        "event": "resume",
                        "checkpoint": config.resume_from_checkpoint,
                        "step": start_step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        print(
            json.dumps(
                {
                    "event": "train_config",
                    "alpha": cal.whitener.alpha,
                    "effective_rank": cal.whitener.effective_rank,
                    "V": cal.V,
                    "total_steps": total_steps,
                    "slow_update_interval": slow_update_interval,
                    "rho_0": rho_0,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        run_training_loop(
            sae=sae,
            whitener=cal.whitener,
            W_vocab=cal.W_vocab,
            buffer=buffer,
            config=config,
            tau_faith=cal.tau_faith,
            tau_drift=cal.tau_drift,
            tau_ortho=cal.tau_ortho,
            dual_updater=dual_updater,
            capu=capu,
            ema=ema,
            disc_schedule=disc_schedule,
            optimizer=optimizer,
            slow_update_interval=slow_update_interval,
            cal=cal,
            accelerator=accelerator,
            start_step=start_step,
            store=store,
        )

        print(json.dumps({"event": "train_complete"}, sort_keys=True), flush=True)
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
        from src.optimization.constraints import (
            compute_drift_violation,
            compute_faithfulness_violation,
            compute_orthogonality_violation,
        )

        x = buffer.next_batch(batch_size).to(device)
        x_tilde = whitener.forward(x)
        x_hat, z, _, _, _ = sae(x_tilde)

        v_faith = compute_faithfulness_violation(x, x_hat, whitener, cal.tau_faith)
        v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, cal.tau_drift)
        v_ortho = compute_orthogonality_violation(
            z, sae.W_dec_A, sae.W_dec_B, cal.tau_ortho, sae.gamma
        )

        return torch.stack([v_faith, v_drift, v_ortho]).abs()
