"""CIS experiment pipeline: Train -> Mechanistic Evaluation.

Two-phase pipeline:
  Phase 1: CIS Training (DF-FLOPS + circuit losses + GECO)
  Phase 2: Mechanistic Evaluation (DLA verification, circuit extraction, completeness, separation)
"""

import argparse
import json
import os
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.config.schema import Config
from splade.evaluation.mechanistic import (print_mechanistic_results,
                                           run_mechanistic_evaluation)
from splade.pipelines import prepare_mechanistic_inputs, setup_and_train


def _print_cis_config(config: Config) -> None:
    """Print CIS experiment configuration."""
    train_str = "full" if config.data.train_samples <= 0 else str(config.data.train_samples)
    test_str = "full" if config.data.test_samples <= 0 else str(config.data.test_samples)
    print("\n--- Lexical-SAE (Circuit-Integrated SPLADE) ---")
    print(f"  Model:       {config.model.name}")
    print(f"  Dataset:     {config.data.dataset_name} (train={train_str}, test={test_str})")
    print(f"  Seed:        {config.seed}")
    print()


def run_experiment(config: Config) -> dict:
    """Run the full CIS experiment pipeline."""
    _print_cis_config(config)

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    seed = config.seed

    # Phase 1: CIS Training
    print("\n--- PHASE 1: CIS TRAINING ---")
    exp = setup_and_train(config, seed)
    print(f"Accuracy: {exp.accuracy:.4f}")

    # Phase 2: Mechanistic Evaluation
    print("\n--- PHASE 2: MECHANISTIC EVALUATION ---")

    input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
        exp.tokenizer, exp.test_texts, exp.max_length,
    )

    mechanistic_results = run_mechanistic_evaluation(
        exp.model, input_ids_list, attention_mask_list,
        exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
        run_sae_comparison=True,
        centroid_tracker=exp.centroid_tracker,
        texts=exp.test_texts,
        max_length=exp.max_length,
    )

    print_mechanistic_results(mechanistic_results)

    result = {
        "seed": seed,
        "accuracy": exp.accuracy,
        "dla_verification_error": mechanistic_results.dla_verification_error,
        "mean_active_dims": mechanistic_results.mean_active_dims,
        "semantic_fidelity": mechanistic_results.semantic_fidelity,
        "eraser_metrics": mechanistic_results.eraser_metrics,
        "explainer_comparison": mechanistic_results.explainer_comparison,
        "layerwise_attribution": mechanistic_results.layerwise_attribution,
        "sae_comparison": mechanistic_results.sae_comparison,
        "polysemy_scores": mechanistic_results.polysemy_scores,
        "downstream_loss": mechanistic_results.downstream_loss,
        "naopc": mechanistic_results.naopc,
        "feature_absorption": mechanistic_results.feature_absorption,
        "sparse_probing": mechanistic_results.sparse_probing,
        "autointerp": mechanistic_results.autointerp,
        "mib_metrics": mechanistic_results.mib_metrics,
        "circuit_completeness": {
            str(k): v for k, v in mechanistic_results.circuit_completeness.items()
        },
        "circuits": {},
    }

    for class_idx, circuit in mechanistic_results.circuits.items():
        result["circuits"][str(class_idx)] = {
            "token_ids": circuit.token_ids,
            "token_names": circuit.token_names,
            "attribution_scores": circuit.attribution_scores,
            "completeness_score": circuit.completeness_score,
        }

    _save_results(config, result)

    return result


def _save_results(config: Config, result: dict) -> None:
    """Save experiment results to JSON."""
    output_path = os.path.join(config.output_dir, "experiment_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CIS experiment: Train -> Mechanistic Evaluation",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
