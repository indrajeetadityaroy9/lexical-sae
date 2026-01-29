"""Interpretability benchmark: SPLADE vs post-hoc explanation methods.

Usage:
    python -m splade_classifier.interpretability_benchmark --dataset sst2 --test-samples 200
"""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from splade_classifier import SPLADEClassifier, load_classification_data
from splade_classifier.faithfulness import comprehensiveness, sufficiency, monotonicity, aopc
from splade_classifier.data import load_hatexplain, rationale_agreement
from splade_classifier.baselines import (
    AttentionExplainer,
    LIMEExplainer,
    SHAPExplainer,
    IntegratedGradientsExplainer,
)


@dataclass
class BenchmarkResult:
    """Results for a single explainer."""
    name: str
    comprehensiveness: dict[int, float] = field(default_factory=dict)
    sufficiency: dict[int, float] = field(default_factory=dict)
    monotonicity: float = 0.0
    aopc: float = 0.0
    rationale_f1: float | None = None
    explanation_time: float = 0.0


def benchmark_splade(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    num_labels: int,
    epochs: int,
    batch_size: int,
    k_values: list[int],
    human_rationales: list[list[str]] | None = None,
) -> tuple[BenchmarkResult, SPLADEClassifier]:
    """Benchmark SPLADE classifier with inherent explanations."""
    print("\n" + "="*60)
    print("SPLADE Classifier (Inherent Explanations)")
    print("="*60)

    clf = SPLADEClassifier(num_labels=num_labels, batch_size=batch_size)
    clf.fit(train_texts, train_labels, epochs=epochs)

    print("Generating explanations...")
    start = time.time()
    attributions = [clf.explain(text, top_k=max(k_values)) for text in test_texts]
    explanation_time = time.time() - start

    print("Computing faithfulness metrics...")
    result = BenchmarkResult(name="SPLADE")
    result.comprehensiveness = comprehensiveness(clf, test_texts, attributions, k_values)
    result.sufficiency = sufficiency(clf, test_texts, attributions, k_values)
    result.monotonicity = monotonicity(clf, test_texts, attributions)
    result.aopc = aopc(clf, test_texts, attributions, max(k_values))
    result.explanation_time = explanation_time

    if human_rationales:
        agreement = rationale_agreement(attributions, human_rationales, k=max(k_values))
        result.rationale_f1 = agreement["f1"]

    return result, clf


def benchmark_baseline(
    explainer_class: type,
    explainer_name: str,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    num_labels: int,
    epochs: int,
    batch_size: int,
    k_values: list[int],
    human_rationales: list[list[str]] | None = None,
) -> BenchmarkResult:
    """Benchmark a baseline explainer."""
    print(f"\n" + "="*60)
    print(f"{explainer_name}")
    print("="*60)

    explainer = explainer_class(num_labels=num_labels)
    explainer.fit(train_texts, train_labels, epochs=epochs, batch_size=batch_size)

    print("Generating explanations...")
    start = time.time()
    attributions = [explainer.explain(text, top_k=max(k_values)) for text in test_texts]
    explanation_time = time.time() - start

    print("Computing faithfulness metrics...")
    result = BenchmarkResult(name=explainer_name)
    result.comprehensiveness = comprehensiveness(explainer, test_texts, attributions, k_values)
    result.sufficiency = sufficiency(explainer, test_texts, attributions, k_values)
    result.monotonicity = monotonicity(explainer, test_texts, attributions)
    result.aopc = aopc(explainer, test_texts, attributions, max(k_values))
    result.explanation_time = explanation_time

    if human_rationales:
        agreement = rationale_agreement(attributions, human_rationales, k=max(k_values))
        result.rationale_f1 = agreement["f1"]

    return result


def print_results(results: list[BenchmarkResult], k_values: list[int]):
    """Print benchmark results as comparison table."""
    print("\n" + "="*80)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("="*80)

    k_display = k_values[len(k_values)//2] if k_values else 10
    print(f"\n{'Method':<30} {'Comp@'+str(k_display):>10} {'Suff@'+str(k_display):>10} {'Mono':>10} {'AOPC':>10} {'Time':>10}")
    print("-" * 82)

    for result in results:
        comp = result.comprehensiveness.get(k_display, 0)
        suff = result.sufficiency.get(k_display, 0)
        mono = result.monotonicity
        aopc_val = result.aopc
        time_str = f"{result.explanation_time:.1f}s"
        print(f"{result.name:<30} {comp:>10.4f} {suff:>10.4f} {mono:>10.4f} {aopc_val:>10.4f} {time_str:>10}")

    print("\n" + "-"*82)
    print("Interpretation:")
    print("  Comprehensiveness: Higher = better (removing important tokens hurts prediction)")
    print("  Sufficiency: Lower = better (top tokens alone predict well)")
    print("  Monotonicity: Higher = better (consistent importance ordering)")
    print("  AOPC: Higher = better (aggregate faithfulness)")

    has_rationale = any(r.rationale_f1 is not None for r in results)
    if has_rationale:
        print(f"\n{'Method':<30} {'Rationale F1':>15}")
        print("-" * 47)
        for result in results:
            if result.rationale_f1 is not None:
                print(f"{result.name:<30} {result.rationale_f1:>15.4f}")

    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)

    valid_results = [r for r in results if r.aopc > 0]
    if valid_results:
        best_comp = max(valid_results, key=lambda r: r.comprehensiveness.get(k_display, 0))
        best_suff = min(valid_results, key=lambda r: r.sufficiency.get(k_display, float('inf')))
        best_mono = max(valid_results, key=lambda r: r.monotonicity)
        best_aopc = max(valid_results, key=lambda r: r.aopc)

        print(f"  Best Comprehensiveness@{k_display}: {best_comp.name}")
        print(f"  Best Sufficiency@{k_display}: {best_suff.name}")
        print(f"  Best Monotonicity: {best_mono.name}")
        print(f"  Best AOPC: {best_aopc.name}")


def run_benchmark(
    dataset: str,
    train_samples: int,
    test_samples: int,
    epochs: int,
    batch_size: int,
) -> list[BenchmarkResult]:
    """Run full interpretability benchmark."""
    k_values = [1, 5, 10, 20]
    results = []
    human_rationales = None

    print(f"\nLoading dataset: {dataset}")
    if dataset == "hatexplain":
        train_texts, train_labels, _, num_labels = load_hatexplain("train", train_samples)
        test_texts, test_labels, human_rationales, _ = load_hatexplain("test", test_samples)
    else:
        train_texts, train_labels, num_labels = load_classification_data(
            dataset if dataset != "sst2" else "glue",
            split="train",
            max_samples=train_samples,
            subset="sst2" if dataset == "sst2" else None,
            text_column="sentence" if dataset == "sst2" else "text",
        )
        test_split = "validation" if dataset == "sst2" else "test"
        test_texts, test_labels, _ = load_classification_data(
            dataset if dataset != "sst2" else "glue",
            split=test_split,
            max_samples=test_samples,
            subset="sst2" if dataset == "sst2" else None,
            text_column="sentence" if dataset == "sst2" else "text",
        )

    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: {num_labels}")

    splade_result, _ = benchmark_splade(
        train_texts, train_labels, test_texts, num_labels,
        epochs, batch_size, k_values, human_rationales
    )
    results.append(splade_result)

    results.append(benchmark_baseline(
        AttentionExplainer, "Attention",
        train_texts, train_labels, test_texts, num_labels,
        epochs, batch_size, k_values, human_rationales
    ))

    results.append(benchmark_baseline(
        LIMEExplainer, "LIME",
        train_texts, train_labels, test_texts, num_labels,
        epochs, batch_size, k_values, human_rationales
    ))

    results.append(benchmark_baseline(
        SHAPExplainer, "SHAP",
        train_texts, train_labels, test_texts, num_labels,
        epochs, batch_size, k_values, human_rationales
    ))

    results.append(benchmark_baseline(
        IntegratedGradientsExplainer, "Integrated Gradients",
        train_texts, train_labels, test_texts, num_labels,
        epochs, batch_size, k_values, human_rationales
    ))

    print_results(results, k_values)
    return results


def main():
    parser = argparse.ArgumentParser(description="Interpretability benchmark: SPLADE vs post-hoc methods")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "ag_news", "imdb", "hatexplain"])
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
