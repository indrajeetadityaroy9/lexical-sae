"""SPALF training entrypoint. Usage: spalf-train path/to/config.yaml"""

from __future__ import annotations

import argparse

from src.config import SPALFConfig
from src.training.trainer import SPALFTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPALF: Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = SPALFConfig.load(args.config)
    trainer = SPALFTrainer(config)
    trainer.train()

    print(f"Training complete. Outputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
