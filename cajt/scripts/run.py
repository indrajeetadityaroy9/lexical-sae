"""Unified Lexical-SAE experiment runner.

Usage: python -m cajt.scripts.run --config experiments/core/banking77.yaml
"""

import argparse
import json
import os

import yaml
from dataclasses import asdict

from cajt.config import Config, load_config
from cajt.scripts.run_experiment import run as _run_experiment
from cajt.scripts.run_ablation import run as _run_ablation
from cajt.scripts.run_long_context import run as _run_long_context
from cajt.scripts.run_surgery import run as _run_surgery


def initialize_output(config: Config) -> None:
    """Create output directory and save resolved config."""
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


def save_results(config: Config, results: dict, filename: str = "results.json") -> None:
    """Save experiment results with embedded config to JSON."""
    path = os.path.join(config.output_dir, filename)
    output = {"config": asdict(config), "results": results}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {path}")

_RUNNERS = {
    "experiment": _run_experiment,
    "ablation": _run_ablation,
    "long_context": _run_long_context,
    "surgery": _run_surgery,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexical-SAE experiment runner")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    initialize_output(config)
    results = _RUNNERS[config.experiment_type](config)
    save_results(config, results)


if __name__ == "__main__":
    main()
