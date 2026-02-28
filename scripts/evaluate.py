"""SPALF evaluation entrypoint. Usage: spalf-eval path/to/config.yaml"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
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

assert torch.cuda.is_available(), "SPALF requires CUDA"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def run_eval(config: SPALFConfig) -> dict:
    """Run requested evaluation suites."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    sae, whitener, W_vocab = load_checkpoint(config.checkpoint)
    suites = config.eval_suites
    results = {}

    needs_store = "downstream_loss" in suites or "sparsity_frontier" in suites
    store = None
    if needs_store:
        print(
            json.dumps(
                {"event": "eval_store_init", "model_name": config.model_name},
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

    if "downstream_loss" in suites:
        print(json.dumps({"event": "eval_suite_start", "suite": "downstream_loss"}, sort_keys=True), flush=True)
        results["downstream_loss"] = evaluate_downstream_loss(
            sae, whitener, store,
        )

    if "sparsity_frontier" in suites:
        print(json.dumps({"event": "eval_suite_start", "suite": "sparsity_frontier"}, sort_keys=True), flush=True)
        results["sparsity_frontier"] = compute_sparsity_frontier(
            sae, whitener, store,
        )

    if "drift_fidelity" in suites:
        print(json.dumps({"event": "eval_suite_start", "suite": "drift_fidelity"}, sort_keys=True), flush=True)
        results["drift_fidelity"] = drift_fidelity(sae.W_dec_A, W_vocab)

    if "feature_absorption" in suites:
        print(json.dumps({"event": "eval_suite_start", "suite": "feature_absorption"}, sort_keys=True), flush=True)
        results["feature_absorption"] = feature_absorption_rate(
            sae.W_dec_B, W_vocab,
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

    print(
        json.dumps(
            {"event": "eval_results_written", "path": str(metrics_path)},
            sort_keys=True,
        ),
        flush=True,
    )
    for suite_name, suite_results in results.items():
        if isinstance(suite_results, dict):
            print(
                json.dumps(
                    {
                        "event": "eval_suite_summary",
                        "suite": suite_name,
                        "metrics": suite_results,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elif isinstance(suite_results, list):
            print(
                json.dumps(
                    {
                        "event": "eval_suite_summary",
                        "suite": suite_name,
                        "points": len(suite_results),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


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
