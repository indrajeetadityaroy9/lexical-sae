"""Single continuous training loop with constraint-triggered KL onset."""


import json
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from src.checkpoint import save_calibration_state
from src.config import CalibrationResult, SPALFConfig
from src.constants import C_EPSILON, EPS_NUM
from src.data.activation_store import ActivationStore
from src.data.buffer import ActivationBuffer
from src.data.patching import run_patched_forward
from src.optimization.constraints import (
    compute_drift_violation,
    compute_faithfulness_violation,
    compute_orthogonality_violation,
)
from src.optimization.capu import MonotoneCAPU
from src.optimization.discretization import DiscretizationSchedule
from src.optimization.dual_updater import DualUpdater
from src.optimization.ema_filter import DualRateEMA
from src.optimization.lagrangian import compute_augmented_lagrangian
from src.sae import StratifiedSAE
from src.whitening.whitener import SoftZCAWhitener

device = torch.device("cuda")

_LOG_INTERVAL: int = 100


def run_training_loop(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    buffer: ActivationBuffer,
    config: SPALFConfig,
    tau_faith: float,
    tau_drift: float,
    tau_ortho: float,
    dual_updater: DualUpdater,
    capu: MonotoneCAPU,
    ema: DualRateEMA,
    disc_schedule: DiscretizationSchedule,
    optimizer: torch.optim.Optimizer,
    slow_update_interval: int,
    cal: CalibrationResult,
    accelerator: Accelerator,
    start_step: int = 0,
    store: ActivationStore | None = None,
) -> None:
    """Run training to completion."""
    total_steps = config.total_tokens // config.batch_size
    token_iter = None
    kl_div_value = 0.0

    if disc_schedule.onset_step < total_steps:
        kl_onset_step: int | None = disc_schedule.onset_step
    else:
        kl_onset_step = None

    print(
        json.dumps(
            {
                "event": "train_loop_start",
                "total_steps": total_steps,
                "start_step": start_step,
                "tau_faith": tau_faith,
                "tau_drift": tau_drift,
                "tau_ortho": tau_ortho,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for step in range(start_step, total_steps):
        kl_div = None
        w_kl = 0.0
        if kl_onset_step is not None and store is not None:
            w_kl = (step - kl_onset_step) / max(total_steps - kl_onset_step, 1)
            if token_iter is None:
                token_iter = store._token_generator()
            tokens = next(token_iter).to(device)
            orig_logits, patched_logits, _, _, _, _ = run_patched_forward(
                store, sae, whitener, tokens
            )
            log_p_orig = F.log_softmax(orig_logits[:, :-1].float(), dim=-1)
            log_p_patched = F.log_softmax(patched_logits[:, :-1].float(), dim=-1)
            kl_div = F.kl_div(log_p_patched, log_p_orig.exp(), reduction="batchmean")

        x = buffer.next_batch(config.batch_size).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_tilde = whitener.forward(x)
            lambda_disc = disc_schedule.get_lambda(step)
            x_hat, z, gate_mask, l0_probs, disc_raw = sae(x_tilde, lambda_disc)

            if kl_div is not None and w_kl > 0:
                kl_self_norm = kl_div / (kl_div.detach() + EPS_NUM)
                v_faith_mse = compute_faithfulness_violation(x, x_hat, whitener, tau_faith)
                v_faith = (1.0 - w_kl) * v_faith_mse + w_kl * kl_self_norm
            else:
                v_faith = compute_faithfulness_violation(x, x_hat, whitener, tau_faith)

            v_drift = compute_drift_violation(sae.W_dec_A, W_vocab, tau_drift)
            v_ortho = compute_orthogonality_violation(
                z, sae.W_dec_A, sae.W_dec_B, tau_ortho, sae.gamma
            )
            violations = torch.stack([v_faith, v_drift, v_ortho])

            ema.update(violations)

            disc_correction = disc_raw.mean()
            l0_loss = l0_probs.mean()
            l0_corr = l0_loss + disc_correction

            lagrangian = compute_augmented_lagrangian(
                l0_corr=l0_corr,
                violations=violations,
                lambdas=dual_updater.lambdas,
                rhos=capu.rhos,
            )

        optimizer.zero_grad()
        lagrangian.backward()
        optimizer.step()

        if step % slow_update_interval == 0 and step > 0:
            dual_updater.step(ema.v_slow, capu.rhos)
            capu.step(ema.v_fast)
            with torch.no_grad():
                pre_act = x_tilde @ sae.W_enc.T + sae.b_enc
                sae.recalibrate_gamma(pre_act, C_EPSILON)

        sae.normalize_free_decoder()

        if kl_onset_step is None and store is not None:
            if (ema.v_slow < 0).all().item():
                kl_onset_step = step
                disc_schedule.set_onset(step)
                print(
                    json.dumps(
                        {
                            "event": "kl_onset",
                            "step": step,
                            "v_slow_faith": ema.v_slow[0].item(),
                            "v_slow_drift": ema.v_slow[1].item(),
                            "v_slow_ortho": ema.v_slow[2].item(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if (
            config.checkpoint_interval > 0
            and step > 0
            and step % config.checkpoint_interval == 0
        ):
            ckpt_dir = Path(config.output_dir) / f"checkpoint_step{step}"
            accelerator.save_state(str(ckpt_dir))
            save_calibration_state(ckpt_dir, whitener, W_vocab, cal, config, step)
            print(
                json.dumps(
                    {"event": "checkpoint_saved", "path": str(ckpt_dir), "step": step},
                    sort_keys=True,
                ),
                flush=True,
            )

        if step % _LOG_INTERVAL == 0:
            with torch.no_grad():
                l0_mean = gate_mask.sum(dim=1).mean().item()
                mse = (x - x_hat).pow(2).sum(dim=1).mean().item()
                x_var = x.pow(2).sum(dim=1).mean().item()
                r_squared = 1.0 - mse / x_var
                kl_div_value = kl_div.item() if kl_div is not None else 0.0
                lam = dual_updater.lambdas

            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "step": step,
                        "l0": l0_mean,
                        "loss": lagrangian.item(),
                        "mse": mse,
                        "r2": r_squared,
                        "v_fast_faith": ema.v_fast[0].item(),
                        "v_fast_drift": ema.v_fast[1].item(),
                        "v_fast_ortho": ema.v_fast[2].item(),
                        "lambda_faith": lam[0].item(),
                        "lambda_drift": lam[1].item(),
                        "lambda_ortho": lam[2].item(),
                        "rho_faith": capu.rhos[0].item(),
                        "rho_drift": capu.rhos[1].item(),
                        "rho_ortho": capu.rhos[2].item(),
                        "w_kl": w_kl,
                        "kl": kl_div_value,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_dir = Path(config.output_dir) / f"checkpoint_step{total_steps}"
    accelerator.save_state(str(final_dir))
    save_calibration_state(final_dir, whitener, W_vocab, cal, config, total_steps)
    print(
        json.dumps(
            {
                "event": "train_loop_complete",
                "final_step": total_steps - 1,
                "final_checkpoint": str(final_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
