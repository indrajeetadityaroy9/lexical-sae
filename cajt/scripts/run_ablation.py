"""CIS ablation study: Baseline vs Full CIS.

Two-variant ablation to demonstrate that circuit training losses
produce measurably better circuits than baseline:

  1. Baseline:  No circuit losses (warmup_fraction=1.1, never exits warmup)
  2. Full CIS:  All circuit losses active
"""

import copy

from cajt.config import Config
from cajt.evaluation.orchestrator import run_mechanistic_evaluation
from cajt.evaluation.results import print_mechanistic_results
from cajt.evaluation.collect import prepare_mechanistic_inputs
from cajt.training.pipeline import setup_and_train


def _run_variant(config: Config, seed: int, name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    exp = setup_and_train(config, seed)

    input_ids_list, attention_mask_list = prepare_mechanistic_inputs(
        exp.tokenizer, exp.test_texts, exp.max_length,
    )

    results = run_mechanistic_evaluation(
        exp.model, input_ids_list, attention_mask_list,
        exp.test_labels, exp.tokenizer, num_classes=exp.num_labels,
        centroid_tracker=exp.centroid_tracker,
        texts=exp.test_texts,
        max_length=exp.max_length,
    )

    completeness_vals = list(results.circuit_completeness.values())
    sf = results.semantic_fidelity
    metrics = {
        "accuracy": exp.accuracy,
        "dla_error": results.dla_verification_error,
        "completeness_mean": sum(completeness_vals) / len(completeness_vals) if completeness_vals else 0.0,
        "cosine_separation": sf.get("cosine_separation", 0.0),
        "jaccard_separation": sf.get("class_separation", 0.0),
    }

    print(f"\n--- {name} Results ---")
    print(f"  Accuracy: {exp.accuracy:.4f}")
    print_mechanistic_results(results)

    return metrics


def run(config: Config) -> dict:
    """Run baseline vs full CIS ablation."""
    seed = config.seed

    print(f"\nAblation: {config.data.dataset_name}, {config.data.train_samples} train, "
          f"{config.data.test_samples} test, seed={seed}")

    # Variant 1: Baseline — no circuit losses (warmup never ends)
    baseline_config = copy.deepcopy(config)
    baseline_config.training.warmup_fraction = 1.1
    baseline = _run_variant(baseline_config, seed, "Baseline (no circuit losses)")

    # Variant 2: Full CIS — all circuit losses active
    full_cis = _run_variant(config, seed, "Full CIS")

    # Comparison table
    print(f"\n{'='*80}")
    print("ABLATION COMPARISON")
    print(f"{'='*80}")
    header = f"{'Variant':<30} {'Accuracy':>10} {'DLA Err':>10} {'Complete':>10} {'Cos Sep':>10} {'Jac Sep':>10}"
    print(header)
    print("-" * 80)
    for name, m in [("Baseline", baseline), ("Full CIS", full_cis)]:
        print(f"{name:<30} {m['accuracy']:>10.4f} {m['dla_error']:>10.6f} "
              f"{m['completeness_mean']:>10.4f} {m['cosine_separation']:>10.4f} {m['jaccard_separation']:>10.4f}")
    print(f"{'='*80}")

    return {"baseline": baseline, "full_cis": full_cis}
