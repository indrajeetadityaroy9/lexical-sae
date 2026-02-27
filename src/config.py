"""SPALF experiment configuration: minimal research-relevant parameters only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class SPALFConfig:
    """Experiment configuration."""

    model_name: str = "EleutherAI/pythia-1.4b"
    hook_point: str = "blocks.6.hook_resid_post"

    dataset: str = "monology/pile-uncopyrighted"
    total_tokens: int = 1_000_000_000
    batch_size: int = 4096
    seq_len: int = 128

    F: int = 0
    L0_target: int | None = None
    R2_target: float = 0.97
    V_cap: int | None = None
    lr: float = 3e-4

    seed: int = 42
    output_dir: str = "runs/default"

    checkpoint: str = ""
    eval_suites: list[str] = field(
        default_factory=lambda: [
            "downstream_loss",
            "sparsity_frontier",
            "drift_fidelity",
            "feature_absorption",
        ]
    )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_name": self.model_name,
            "hook_point": self.hook_point,
            "dataset": self.dataset,
            "total_tokens": self.total_tokens,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "F": self.F,
            "L0_target": self.L0_target,
            "R2_target": self.R2_target,
            "V_cap": self.V_cap,
            "lr": self.lr,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "checkpoint": self.checkpoint,
            "eval_suites": list(self.eval_suites),
        }

    def save(self, path: str | Path) -> None:
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> SPALFConfig:
        """Load config from YAML. Unknown keys are ignored."""
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class CalibrationResult:
    """Calibration outputs shared across training and checkpointing."""

    whitener: object  # SoftZCAWhitener (avoid circular import)
    W_vocab: Tensor  # [d, V]
    d: int
    V: int
    F: int  # final (after auto-compute from d_model)
    L0_target: int  # final (after auto-compute from F)
    tau_faith: float
    tau_drift: float
    tau_ortho: float
