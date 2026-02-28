"""Checkpoint utilities for Accelerate-managed training state + frozen calibration artifacts."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, load_model, save_file
from torch import Tensor

from spalf.config import CalibrationResult, SPALFConfig
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener

device = torch.device("cuda")


def save_calibration_state(
    output_dir: str | Path,
    whitener: SoftZCAWhitener,
    W_vocab: Tensor,
    cal: CalibrationResult,
    config: SPALFConfig,
    step: int,
) -> None:
    """Save frozen calibration artifacts alongside an Accelerate checkpoint."""

    ckpt_dir = Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sd = whitener.state_dict()
    sd["W_vocab"] = W_vocab
    save_file(sd, str(ckpt_dir / "calibration.safetensors"))

    metadata = {
        "step": step,
        "config": config.to_dict(),
        "calibration": {
            "d": cal.d,
            "V": cal.V,
            "F": cal.F,
            "L0_target": cal.L0_target,
        },
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[StratifiedSAE, SoftZCAWhitener, Tensor]:
    """Load SAE, whitener, and vocabulary matrix for evaluation."""
    ckpt_dir = Path(checkpoint_path)

    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    cal_data = metadata["calibration"]

    sae = StratifiedSAE(d=cal_data["d"], F=cal_data["F"], V=cal_data["V"])
    load_model(sae, str(ckpt_dir / "model.safetensors"))
    sae.to(device)
    sae.eval()

    cal_state = load_file(
        str(ckpt_dir / "calibration.safetensors"), device=str(device)
    )
    whitener = SoftZCAWhitener.__new__(SoftZCAWhitener)
    whitener.load_state_dict(cal_state)
    whitener.to(device)

    W_vocab = cal_state["W_vocab"].to(device)

    print(
        json.dumps(
            {
                "event": "checkpoint_loaded",
                "step": metadata["step"],
                "d": cal_data["d"],
                "F": cal_data["F"],
                "V": cal_data["V"],
                "path": str(ckpt_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    return sae, whitener, W_vocab
