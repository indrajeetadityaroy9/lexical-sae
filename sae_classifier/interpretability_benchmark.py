"""Interpretability benchmark comparing SAE vs baselines.

Evaluates explanation quality using:
1. Perturbation-based faithfulness (comprehensiveness, sufficiency, monotonicity)
2. Intervention-based metrics (feature necessity, feature sufficiency, ablation effect)
3. Human rationale agreement (on HateXplain dataset)
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from sae_classifier.data import load_classification_data, load_hatexplain, rationale_agreement
from sae_classifier.faithfulness import comprehensiveness, sufficiency, monotonicity, aopc
from sae_classifier.models import SAEClassifier, set_seed


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single explainer."""
    name: str
    # Perturbation-based faithfulness
    comprehensiveness: dict[int, float] = field(default_factory=dict)
    sufficiency: dict[int, float] = field(default_factory=dict)
    monotonicity: float = 0.0
    aopc: float = 0.0
    # Intervention-based (SAE only)
    feature_necessity: dict[int, float] = field(default_factory=dict)
    feature_sufficiency: dict[int, float] = field(default_factory=dict)
    avg_ablation_effect: float = 0.0
    # Human agreement
    rationale_precision: float = 0.0
    rationale_recall: float = 0.0
    rationale_f1: float = 0.0
    # Timing
    explanation_time: float = 0.0
    # Classification performance
    accuracy: float = 0.0


def benchmark_sae(
    texts: list[str],
    labels: list[int],
    num_labels: int,
    test_texts: list[str],
    test_labels: list[int],
    human_rationales: list[list[str]] | None = None,
    sae_expansion: int = 8,
    sae_k: int = 32,
    epochs: int = 3,
    device: str = "auto",
) -> BenchmarkResult:
    """Benchmark SAE-based classifier and explanations."""
    print("\n" + "=" * 60)
    print("Benchmarking SAE Classifier")
    print("=" * 60)

    # Train SAE classifier
    clf = SAEClassifier(
        num_labels=num_labels,
        sae_expansion=sae_expansion,
        sae_k=sae_k,
        epochs=epochs,
        device=device,
    )
    clf.fit(texts, labels)

    # Classification accuracy
    accuracy = clf.score(test_texts, test_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    # Generate explanations
    print("Generating explanations...")
    start_time = time.time()
    attributions = []
    for text in tqdm(test_texts, desc="Explaining"):
        # Use explain_tokens for token-level faithfulness metrics
        exp = clf.explain_tokens(text, top_k=20)
        attributions.append(exp)
    explanation_time = (time.time() - start_time) / len(test_texts)

    # Perturbation-based faithfulness
    print("Computing faithfulness metrics...")
    k_values = [1, 5, 10]
    comp = comprehensiveness(clf, test_texts, attributions, k_values)
    suff = sufficiency(clf, test_texts, attributions, k_values)
    mono = monotonicity(clf, test_texts, attributions)
    aopc_score = aopc(clf, test_texts, attributions, k_max=10)

    # Intervention-based metrics (SAE-specific)
    print("Computing intervention metrics...")
    from sae_classifier.interventions import compute_intervention_metrics
    intervention_results = compute_intervention_metrics(clf, test_texts[:100], k_values)

    # Human rationale agreement
    rationale_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if human_rationales is not None:
        rationale_metrics = rationale_agreement(attributions, human_rationales, k=10)

    return BenchmarkResult(
        name="SAE",
        comprehensiveness=comp,
        sufficiency=suff,
        monotonicity=mono,
        aopc=aopc_score,
        feature_necessity=intervention_results["feature_necessity"],
        feature_sufficiency=intervention_results["feature_sufficiency"],
        avg_ablation_effect=intervention_results["avg_ablation_effect"],
        rationale_precision=rationale_metrics["precision"],
        rationale_recall=rationale_metrics["recall"],
        rationale_f1=rationale_metrics["f1"],
        explanation_time=explanation_time,
        accuracy=accuracy,
    )


def benchmark_baseline(
    explainer_class: type,
    name: str,
    texts: list[str],
    labels: list[int],
    num_labels: int,
    test_texts: list[str],
    test_labels: list[int],
    human_rationales: list[list[str]] | None = None,
    epochs: int = 3,
    device: str = "auto",
    **kwargs,
) -> BenchmarkResult:
    """Benchmark a baseline explainer."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {name}")
    print("=" * 60)

    # Train baseline
    explainer = explainer_class(num_labels=num_labels, device=device, **kwargs)
    explainer.fit(texts, labels, epochs=epochs)

    # Classification accuracy
    preds = explainer.predict(test_texts)
    accuracy = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    # Generate explanations
    print("Generating explanations...")
    start_time = time.time()
    attributions = []
    for text in tqdm(test_texts, desc="Explaining"):
        exp = explainer.explain(text, top_k=20)
        attributions.append(exp)
    explanation_time = (time.time() - start_time) / len(test_texts)

    # Perturbation-based faithfulness
    print("Computing faithfulness metrics...")
    k_values = [1, 5, 10]
    comp = comprehensiveness(explainer, test_texts, attributions, k_values)
    suff = sufficiency(explainer, test_texts, attributions, k_values)
    mono = monotonicity(explainer, test_texts, attributions)
    aopc_score = aopc(explainer, test_texts, attributions, k_max=10)

    # Human rationale agreement
    rationale_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if human_rationales is not None:
        rationale_metrics = rationale_agreement(attributions, human_rationales, k=10)

    return BenchmarkResult(
        name=name,
        comprehensiveness=comp,
        sufficiency=suff,
        monotonicity=mono,
        aopc=aopc_score,
        rationale_precision=rationale_metrics["precision"],
        rationale_recall=rationale_metrics["recall"],
        rationale_f1=rationale_metrics["f1"],
        explanation_time=explanation_time,
        accuracy=accuracy,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print comparison table of results."""
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Method':<20} {'Accuracy':<10} {'Comp@5':<10} {'Suff@5':<10} {'Mono':<10} {'AOPC':<10}")
    print("-" * 70)

    for r in results:
        comp5 = r.comprehensiveness.get(5, 0.0)
        suff5 = r.sufficiency.get(5, 0.0)
        print(f"{r.name:<20} {r.accuracy:<10.4f} {comp5:<10.4f} {suff5:<10.4f} {r.monotonicity:<10.4f} {r.aopc:<10.4f}")

    # SAE-specific intervention metrics
    sae_results = [r for r in results if r.name == "SAE"]
    if sae_results:
        print("\n" + "-" * 70)
        print("SAE Intervention Metrics:")
        r = sae_results[0]
        print(f"  Feature Necessity @5: {r.feature_necessity.get(5, 0.0):.4f}")
        print(f"  Feature Sufficiency @5: {r.feature_sufficiency.get(5, 0.0):.4f}")
        print(f"  Avg Ablation Effect: {r.avg_ablation_effect:.4f}")

    # Rationale agreement (if available)
    if any(r.rationale_f1 > 0 for r in results):
        print("\n" + "-" * 70)
        print("Human Rationale Agreement:")
        print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        for r in results:
            if r.rationale_f1 > 0:
                print(f"{r.name:<20} {r.rationale_precision:<12.4f} {r.rationale_recall:<12.4f} {r.rationale_f1:<12.4f}")

    # Timing
    print("\n" + "-" * 70)
    print("Explanation Time (seconds per sample):")
    for r in results:
        print(f"  {r.name}: {r.explanation_time:.4f}s")

    # Winner analysis
    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)

    metrics = [
        ("Comprehensiveness@5", lambda r: r.comprehensiveness.get(5, 0.0), True),
        ("Sufficiency@5", lambda r: r.sufficiency.get(5, 0.0), False),  # Lower is better
        ("Monotonicity", lambda r: r.monotonicity, True),
        ("AOPC", lambda r: r.aopc, True),
    ]

    for name, getter, higher_better in metrics:
        if higher_better:
            winner = max(results, key=getter)
        else:
            winner = min(results, key=getter)
        print(f"  {name}: {winner.name} ({getter(winner):.4f})")


def run_benchmark(
    dataset: str = "sst2",
    train_samples: int = 1000,
    test_samples: int = 200,
    epochs: int = 3,
    seed: int = 42,
    device: str = "auto",
    include_baselines: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run full interpretability benchmark.

    Args:
        dataset: Dataset to use ("sst2", "imdb", "hatexplain")
        train_samples: Number of training samples
        test_samples: Number of test samples
        epochs: Training epochs
        seed: Random seed
        device: Device to use
        include_baselines: List of baselines to include (default: ["Attention"])

    Returns:
        List of BenchmarkResult objects
    """
    set_seed(seed)

    if include_baselines is None:
        include_baselines = ["Attention"]

    # Load data
    print(f"Loading {dataset} dataset...")
    human_rationales = None

    if dataset == "hatexplain":
        texts, labels, human_rationales, num_labels = load_hatexplain(
            split="train", max_samples=train_samples
        )
        test_texts, test_labels, test_rationales, _ = load_hatexplain(
            split="test", max_samples=test_samples
        )
        human_rationales = test_rationales
    elif dataset == "sst2":
        texts, labels, num_labels = load_classification_data(
            "glue", subset="sst2", split="train", max_samples=train_samples,
            text_column="sentence"
        )
        test_texts, test_labels, _ = load_classification_data(
            "glue", subset="sst2", split="validation", max_samples=test_samples,
            text_column="sentence"
        )
    else:
        texts, labels, num_labels = load_classification_data(
            dataset, split="train", max_samples=train_samples
        )
        test_texts, test_labels, _ = load_classification_data(
            dataset, split="test", max_samples=test_samples
        )

    print(f"Train: {len(texts)} samples, Test: {len(test_texts)} samples")
    print(f"Num labels: {num_labels}")

    results = []

    # Benchmark SAE
    sae_result = benchmark_sae(
        texts, labels, num_labels,
        test_texts, test_labels, human_rationales,
        epochs=epochs, device=device,
    )
    results.append(sae_result)

    # Benchmark baselines
    from sae_classifier.baselines import AttentionExplainer

    baseline_classes = {
        "Attention": AttentionExplainer,
    }

    # Add optional baselines
    if "SPLADE" in include_baselines:
        from sae_classifier.baselines import SPLADEExplainer
        baseline_classes["SPLADE"] = SPLADEExplainer

    for name in include_baselines:
        if name in baseline_classes:
            result = benchmark_baseline(
                baseline_classes[name], name,
                texts, labels, num_labels,
                test_texts, test_labels, human_rationales,
                epochs=epochs, device=device,
            )
            results.append(result)

    # Print comparison
    print_results(results)

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run interpretability benchmark")
    parser.add_argument("--dataset", default="sst2", choices=["sst2", "imdb", "hatexplain"])
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baselines", nargs="+", default=["Attention"],
                        help="Baselines to include (Attention, SPLADE)")

    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        include_baselines=args.baselines,
    )


if __name__ == "__main__":
    main()
