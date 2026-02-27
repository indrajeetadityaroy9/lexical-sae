"""Checkpoint save/load using safetensors (§ checkpoint migration)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, load_model, save_file, save_model
from torch import Tensor

from src.config import CalibrationResult, SPALFConfig
from src.control.adrc import ADRCController
from src.control.capu import ModifiedCAPU
from src.control.ema_filter import DualRateEMA
from src.model.sae import StratifiedSAE
from src.runtime import DEVICE
from src.whitening.whitener import SoftZCAWhitener


def save_checkpoint(
    sae: StratifiedSAE,
    cal: CalibrationResult,
    config: SPALFConfig,
    adrc: ADRCController,
    capu: ModifiedCAPU,
    ema: DualRateEMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    phase: int,
) -> Path:
    """Save training checkpoint as a directory of safetensors + metadata JSON.

    Structure:
        model.safetensors     — SAE nn.Module weights
        tensors.safetensors   — whitener tensors + W_vocab
        control.safetensors   — ADRC/CAPU/EMA tensors
        optimizer.safetensors — optimizer state tensors
        metadata.json         — scalars, config, calibration
    """
    ckpt_dir = Path(config.output_dir) / f"spalf_phase{phase}_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. SAE weights
    save_model(sae, str(ckpt_dir / "model.safetensors"))

    # 2. Whitener tensors + W_vocab
    w_sd = cal.whitener.state_dict()
    save_file(
        {
            "mean": w_sd["mean"],
            "eigenvalues": w_sd["eigenvalues"],
            "eigenvectors": w_sd["eigenvectors"],
            "W_vocab": cal.W_vocab,
        },
        str(ckpt_dir / "tensors.safetensors"),
    )

    # 3. Control tensors (flat keys)
    control_tensors = {}
    for k, v in adrc.state_dict().items():
        if isinstance(v, Tensor):
            control_tensors[f"adrc.{k}"] = v
        elif isinstance(v, dict):  # ESO nested dict
            for ek, ev in v.items():
                if isinstance(ev, Tensor):
                    control_tensors[f"adrc.eso.{ek}"] = ev
    for k, v in capu.state_dict().items():
        if isinstance(v, Tensor):
            control_tensors[f"capu.{k}"] = v
    for k, v in ema.state_dict().items():
        if isinstance(v, Tensor):
            control_tensors[f"ema.{k}"] = v
    save_file(control_tensors, str(ckpt_dir / "control.safetensors"))

    # 4. Optimizer state tensors
    opt_sd = optimizer.state_dict()
    opt_tensors = {}
    for pid, state in opt_sd["state"].items():
        for k, v in state.items():
            if isinstance(v, Tensor):
                opt_tensors[f"state.{pid}.{k}"] = v
    save_file(opt_tensors, str(ckpt_dir / "optimizer.safetensors"))

    # 5. Metadata JSON (all non-tensor data)
    metadata = {
        "step": step,
        "phase": phase,
        "config": config.to_dict(),
        "calibration": {
            "d": cal.d,
            "V": cal.V,
            "F": cal.F,
            "L0_target": cal.L0_target,
        },
        "whitener": {
            "kappa_target": w_sd["kappa_target"],
            "alpha": w_sd["alpha"],
            "effective_rank": w_sd["effective_rank"],
            "d": w_sd["d"],
        },
        "optimizer_param_groups": opt_sd["param_groups"],
        "optimizer_scalar_state": {
            str(pid): {k: v for k, v in state.items() if not isinstance(v, Tensor)}
            for pid, state in opt_sd["state"].items()
        },
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Checkpoint saved: {ckpt_dir}")
    return ckpt_dir


def load_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[StratifiedSAE, SoftZCAWhitener, Tensor]:
    """Load SAE, whitener, and W_vocab from a safetensors checkpoint directory.

    Returns:
        sae: Trained StratifiedSAE in eval mode.
        whitener: Frozen SoftZCAWhitener.
        W_vocab: [d, V] unembedding matrix.
    """
    ckpt_dir = Path(checkpoint_path)

    # 1. Load metadata
    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    cal_data = metadata["calibration"]
    w_meta = metadata["whitener"]

    # 2. Reconstruct SAE and load weights
    sae = StratifiedSAE(d=cal_data["d"], F=cal_data["F"], V=cal_data["V"])
    load_model(sae, str(ckpt_dir / "model.safetensors"))
    sae.to(DEVICE)
    sae.eval()

    # 3. Load whitener tensors and reconstruct
    tensors = load_file(str(ckpt_dir / "tensors.safetensors"), device=str(DEVICE))
    whitener = SoftZCAWhitener(
        mean=tensors["mean"],
        eigenvalues=tensors["eigenvalues"],
        eigenvectors=tensors["eigenvectors"],
        kappa_target=w_meta["kappa_target"],
    )
    whitener.to(DEVICE)

    W_vocab = tensors["W_vocab"].to(DEVICE)

    print(
        f"Loaded checkpoint: step={metadata['step']}, phase={metadata['phase']}, "
        f"d={cal_data['d']}, F={cal_data['F']}, V={cal_data['V']}"
    )

    return sae, whitener, W_vocab
