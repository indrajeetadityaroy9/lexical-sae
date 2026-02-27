"""SPALF evaluation entrypoint. Usage: spalf-eval path/to/config.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.config import SPALFConfig
from src.data.activation_store import ActivationStore
from src.evaluation import (
    evaluate_downstream_loss,
    compute_sparsity_frontier,
    drift_fidelity,
    feature_absorption_rate,
)
from src.runtime import set_seed


def run_eval(config: SPALFConfig) -> dict:
    """Run requested evaluation suites."""
    set_seed(config.seed)

    sae, whitener, W_vocab = load_checkpoint(config.checkpoint)
    suites = config.eval_suites
    results = {}

    needs_store = "downstream_loss" in suites or "sparsity_frontier" in suites
    store = None
    if needs_store:
        print(f"Loading model for downstream evaluation: {config.model_name}")
        store = ActivationStore(
            model_name=config.model_name,
            hook_point=config.hook_point,
            dataset_name=config.dataset,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
        )

    if "downstream_loss" in suites:
        print("Running downstream_loss evaluation...")
        results["downstream_loss"] = evaluate_downstream_loss(
            sae, whitener, store,
        )

    if "sparsity_frontier" in suites:
        print("Running sparsity_frontier evaluation...")
        results["sparsity_frontier"] = compute_sparsity_frontier(
            sae, whitener, store,
        )

    if "drift_fidelity" in suites:
        print("Running drift_fidelity evaluation...")
        results["drift_fidelity"] = drift_fidelity(sae.W_dec_A, W_vocab)

    if "feature_absorption" in suites:
        print("Running feature_absorption evaluation...")
        results["feature_absorption"] = feature_absorption_rate(
            sae.W_dec_A, sae.W_dec_B, W_vocab,
        )

    return results


def write_results(config: SPALFConfig, results: dict) -> None:
    """Write evaluation outputs: metrics.json + config stamp."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config.save(out_dir / "eval_config.yaml")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results written to {metrics_path}")
    for suite_name, suite_results in results.items():
        if isinstance(suite_results, dict):
            summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in suite_results.items())
            print(f"  {suite_name}: {summary}")
        elif isinstance(suite_results, list):
            print(f"  {suite_name}: {len(suite_results)} data points")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPALF evaluation: load checkpoint, run eval suites, write metrics"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    config = SPALFConfig.load(args.config)

    if not config.checkpoint:
        parser.error("Config must specify 'checkpoint' path for evaluation")

    results = run_eval(config)
    write_results(config, results)


if __name__ == "__main__":
    main()
