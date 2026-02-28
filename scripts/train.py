"""SPALF training entrypoint. Usage: spalf-train path/to/config.yaml"""

import argparse
import os

import torch

from src.config import SPALFConfig
from src.training.trainer import SPALFTrainer

assert torch.cuda.is_available(), "SPALF requires CUDA"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPALF: Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = SPALFConfig.load(args.config)
    trainer = SPALFTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
