"""Interpretability benchmark: SPLADE vs post-hoc explanation methods."""

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.adversarial import compute_adversarial_sensitivity
from src.baselines import (
    AttentionExplainer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
    train_shared_model,
)
from src.cuda import set_seed
from src.data import compute_rationale_agreement, load_benchmark_data, load_hatexplain
from src.faithfulness import (
    compute_comprehensiveness,
    compute_filler_comprehensiveness,
    compute_monotonicity,
    compute_normalized_aopc,
    compute_sufficiency,
)
from src.models import SPLADEClassifier


@dataclass
class InterpretabilityResult:
    """Benchmark metrics for an explainer."""
    name: str
    comprehensiveness: dict[int, float] = field(default_factory=dict)
    sufficiency: dict[int, float] = field(default_factory=dict)
    monotonicity: float = 0.0
    aopc: float = 0.0
    naopc: float = 0.0
    naopc_lower: float = 0.0
    naopc_upper: float = 0.0
    f_comprehensiveness: dict[int, float] = field(default_factory=dict)
    f_sufficiency: dict[int, float] = field(default_factory=dict)
    filler_comprehensiveness: dict[int, float] = field(default_factory=dict)
    adversarial_sensitivity: float = 0.0
    adversarial_mean_tau: float = 0.0
    rationale_f1: float | None = None
    explanation_time: float = 0.0
    accuracy: float = 0.0


def print_interpretability_results(results: list[InterpretabilityResult], k_values: list[int]) -> None:
    """Print benchmark results comparison table."""
    print("\n" + "=" * 80)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("=" * 80)

    k_display = k_values[len(k_values) // 2]

    # Primary metrics first (NAOPC + Filler Tokens -- robust to OOD artifacts)
    print(f"\n--- PRIMARY METRICS (robust to OOD artifacts) ---")
    print(f"\n{'Method':<30} {'NAOPC':>10} {'Filler@' + str(k_display):>12} {'Mono':>10} {'Accuracy':>10}")
    print("-" * 74)
    for result in results:
        filler = result.filler_comprehensiveness.get(k_display, 0.0)
        print(
            f"{result.name:<30} {result.naopc:>10.4f} {filler:>12.4f} "
            f"{result.monotonicity:>10.4f} {result.accuracy:>10.4f}"
        )

    print(f"\n{'Method':<30} {'AOPC_lo':>10} {'AOPC_hi':>10} {'Adv Sens':>10} {'Mean Tau':>10}")
    print("-" * 72)
    for result in results:
        print(
            f"{result.name:<30} {result.naopc_lower:>10.4f} "
            f"{result.naopc_upper:>10.4f} {result.adversarial_sensitivity:>10.4f} "
            f"{result.adversarial_mean_tau:>10.4f}"
        )

    # Supplementary ERASER metrics (with caveat)
    print(f"\n--- SUPPLEMENTARY ERASER METRICS ---")
    print("  Note: Comp/Suff can be inflated by OOD artifacts (arXiv:2308.14272).")
    print(f"\n{'Method':<30} {'Comp@' + str(k_display):>10} {'Suff@' + str(k_display):>10} "
          f"{'F-Comp@' + str(k_display):>12} {'F-Suff@' + str(k_display):>12} {'Time':>10}")
    print("-" * 86)
    for result in results:
        comp = result.comprehensiveness[k_display]
        suff = result.sufficiency[k_display]
        f_comp = result.f_comprehensiveness[k_display]
        f_suff = result.f_sufficiency[k_display]
        time_str = f"{result.explanation_time:.1f}s"
        print(
            f"{result.name:<30} {comp:>10.4f} {suff:>10.4f} "
            f"{f_comp:>12.4f} {f_suff:>12.4f} {time_str:>10}"
        )

    print("\n" + "-" * 82)
    print("Interpretation:")
    print("  NAOPC: Normalized AOPC (0-1, higher = better, cross-model comparable)")
    print("  Filler: Filler-token comprehensiveness (higher = better, OOD-robust)")
    print("  Monotonicity: Higher = better (consistent importance ordering)")
    print("  Comp/Suff: ERASER metrics (higher comp / lower suff = better)")
    print("  F-Comp/F-Suff: F-Fidelity bounded variants (reduced OOD effects)")
    print("  Adv Sens: Adversarial sensitivity (higher = more faithful)")

    if any(r.rationale_f1 is not None for r in results):
        print(f"\n{'Method':<30} {'Rationale F1':>15}")
        print("-" * 47)
        for result in results:
            if result.rationale_f1 is not None:
                print(f"{result.name:<30} {result.rationale_f1:>15.4f}")

    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    best_naopc = max(results, key=lambda r: r.naopc)
    best_filler = max(results, key=lambda r: r.filler_comprehensiveness.get(k_display, 0.0))
    best_mono = max(results, key=lambda r: r.monotonicity)
    best_comp = max(results, key=lambda r: r.comprehensiveness[k_display])

    print(f"  Best NAOPC: {best_naopc.name}")
    print(f"  Best Filler Comp@{k_display}: {best_filler.name}")
    print(f"  Best Monotonicity: {best_mono.name}")
    print(f"  Best Comprehensiveness@{k_display}: {best_comp.name}")


def benchmark_explainer(
    clf, name: str, explain_fn,
    test_texts: list[str], k_values: list[int],
    human_rationales: list[list[str]] | None = None,
) -> InterpretabilityResult:
    """Evaluate a single explainer on faithfulness metrics."""
    print(f"\nGenerating explanations for {name}...")
    start = time.time()
    attributions = [explain_fn(text, max(k_values)) for text in tqdm(test_texts, desc="Explaining")]
    explanation_time = time.time() - start

    print("Computing metrics...")
    result = InterpretabilityResult(name=name, explanation_time=explanation_time)
    result.comprehensiveness = compute_comprehensiveness(clf, test_texts, attributions, k_values)
    result.sufficiency = compute_sufficiency(clf, test_texts, attributions, k_values)
    result.monotonicity = compute_monotonicity(clf, test_texts, attributions)
    aopc_scores = compute_comprehensiveness(clf, test_texts, attributions, list(range(1, max(k_values) + 1)))
    result.aopc = float(np.mean(list(aopc_scores.values())))

    naopc_result = compute_normalized_aopc(clf, test_texts, attributions, k_max=max(k_values))
    result.naopc, result.naopc_lower, result.naopc_upper = naopc_result["naopc"], naopc_result["aopc_lower"], naopc_result["aopc_upper"]

    result.f_comprehensiveness = compute_comprehensiveness(clf, test_texts, attributions, k_values, beta=0.5)
    result.f_sufficiency = compute_sufficiency(clf, test_texts, attributions, k_values, beta=0.5)
    result.filler_comprehensiveness = compute_filler_comprehensiveness(clf, test_texts, attributions, k_values)

    adv_result = compute_adversarial_sensitivity(clf, explain_fn, test_texts[:50], top_k=max(k_values))
    result.adversarial_sensitivity, result.adversarial_mean_tau = adv_result["adversarial_sensitivity"], adv_result["mean_tau"]

    if human_rationales:
        result.rationale_f1 = compute_rationale_agreement(attributions, human_rationales, k=max(k_values))

    return result


def print_cvb_analysis(
    clf: SPLADEClassifier,
    train_texts: list[str], train_labels: list[int],
    test_texts: list[str], test_labels: list[int],
) -> None:
    """Print Concept Vocabulary Bottleneck analysis statistics."""
    print("\n" + "=" * 80)
    print("CONCEPT VOCABULARY BOTTLENECK ANALYSIS")
    print("=" * 80)

    # Sparsity
    test_sparse = clf.transform(test_texts)
    nonzero_counts = np.count_nonzero(test_sparse, axis=1)
    vocab_size = test_sparse.shape[1]
    print(f"\nSparsity: {1 - np.mean(nonzero_counts) / vocab_size:.4f} ({np.mean(nonzero_counts):.1f} terms)")

    # Global concepts
    print("\nTop 20 Global Concepts:")
    train_sparse = clf.transform(train_texts)
    importance = np.abs(train_sparse).mean(axis=0)
    top_indices = np.argsort(importance)[-50:][::-1]
    for i, idx in enumerate(top_indices[:20]):
        print(f"  {i + 1:2}. {clf.tokenizer.decode([idx]):<15} {importance[idx]:.4f}")

    # Concept completeness
    print("\nConcept Completeness:")
    test_importance = np.abs(test_sparse).mean(axis=0)
    n_folds = min(5, len(test_labels))
    for k in [10, 50, 100, 500]:
        mask = np.zeros(test_sparse.shape[1])
        mask[np.argsort(test_importance)[-k:]] = 1
        lr = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(lr, test_sparse * mask, test_labels, cv=n_folds, scoring="accuracy")
        print(f"  Top-{k:4}: {scores.mean():.4f}")

    # Per-class concepts
    print("\nTop Concepts per Class:")
    labels_arr = np.array(train_labels)
    for class_idx in np.unique(labels_arr):
        class_importance = np.abs(train_sparse[labels_arr == class_idx]).mean(axis=0)
        top_idx = np.argsort(class_importance)[-20:][::-1]
        concepts = [clf.tokenizer.decode([i]) for i in top_idx[:5]]
        print(f"  Class {int(class_idx)}: {', '.join(concepts)}")

    # Classifier weights
    print("\nClassifier Weights:")
    with torch.no_grad():
        weights = clf.model.classifier.weight.cpu().numpy()
    for class_idx in range(weights.shape[0]):
        class_weights = weights[class_idx]
        top_idx = np.argsort(np.abs(class_weights))[-10:][::-1]
        terms = [(clf.tokenizer.decode([i]), float(class_weights[i])) for i in top_idx]
        pos = ", ".join([f"{t}(+{w:.2f})" for t, w in terms if w > 0][:3])
        neg = ", ".join([f"{t}({w:.2f})" for t, w in terms if w < 0][:3])
        print(f"  Class {class_idx}: {pos} | {neg}")


def run_single_seed_benchmark(
    dataset: str, train_samples: int, test_samples: int,
    epochs: int, batch_size: int,
    run_cvb_analysis: bool = False, seed: int = 42,
) -> list[InterpretabilityResult]:
    set_seed(seed)
    k_values = [1, 5, 10, 20]
    human_rationales = None

    if dataset == "hatexplain":
        train_texts, train_labels, _, num_labels = load_hatexplain("train", train_samples)
        test_texts, test_labels, human_rationales, _ = load_hatexplain("test", test_samples)
    else:
        train_texts, train_labels, test_texts, test_labels, num_labels = load_benchmark_data(dataset, train_samples, test_samples)

    clf = SPLADEClassifier(num_labels=num_labels, batch_size=batch_size)
    clf.fit(train_texts, train_labels, epochs=epochs)
    accuracy = clf.score(test_texts, test_labels)

    results = [benchmark_explainer(
        clf, "SPLADE", lambda text, top_k: clf.explain(text, top_k=top_k),
        test_texts, k_values, human_rationales,
    )]
    results[0].accuracy = accuracy

    print("\nTraining shared baseline model...")
    shared_model, shared_tokenizer = train_shared_model(
        "distilbert-base-uncased", num_labels, train_texts, train_labels,
        epochs=epochs, batch_size=batch_size,
    )
    baseline_accuracy = sum(
        int(np.argmax(p)) == t
        for p, t in zip(
            AttentionExplainer(shared_model, shared_tokenizer, num_labels).predict_proba(test_texts),
            test_labels,
        )
    ) / len(test_labels)

    baselines = [
        (AttentionExplainer, "Attention"),
        (LIMEExplainer, "LIME"),
        (SHAPExplainer, "SHAP"),
        (IntegratedGradientsExplainer, "Integrated Gradients"),
    ]
    for explainer_class, name in baselines:
        explainer = explainer_class(shared_model, shared_tokenizer, num_labels)
        result = benchmark_explainer(explainer, name, lambda text, top_k, e=explainer: e.explain(text, top_k=top_k), test_texts, k_values, human_rationales)
        result.accuracy = baseline_accuracy
        results.append(result)

    print_interpretability_results(results, k_values)
    if run_cvb_analysis:
        print_cvb_analysis(clf, train_texts, train_labels, test_texts, test_labels)
    return results


def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot_means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_bootstrap)]
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def run_multi_seed_benchmark(
    dataset: str, train_samples: int, test_samples: int,
    epochs: int, batch_size: int,
    seeds: list[int] = [42, 123, 456],
    run_cvb_analysis: bool = False,
) -> dict[int, list[InterpretabilityResult]]:
    """Run benchmark across multiple seeds and report statistics."""
    all_results: dict[int, list[InterpretabilityResult]] = {}
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        all_results[seed] = run_single_seed_benchmark(
            dataset, train_samples, test_samples,
            epochs, batch_size, run_cvb_analysis, seed,
        )

    # Aggregate statistics
    method_names = [r.name for r in all_results[seeds[0]]]
    k_display = 10

    print(f"\n{'='*80}")
    print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*80}")

    print(f"\n{'Method':<25} {'NAOPC':>18} {'Filler@10':>18} {'Mono':>18}")
    print("-" * 81)
    splade_naopc_vals = []
    for method_idx, method_name in enumerate(method_names):
        naopc_vals = [all_results[s][method_idx].naopc for s in seeds]
        filler_vals = [all_results[s][method_idx].filler_comprehensiveness.get(k_display, 0.0) for s in seeds]
        mono_vals = [all_results[s][method_idx].monotonicity for s in seeds]

        if method_name == "SPLADE":
            splade_naopc_vals = naopc_vals

        naopc_ci = bootstrap_ci(naopc_vals)
        filler_ci = bootstrap_ci(filler_vals)
        mono_ci = bootstrap_ci(mono_vals)

        print(
            f"{method_name:<25} "
            f"{np.mean(naopc_vals):.4f}+/-{np.std(naopc_vals):.4f} "
            f"{np.mean(filler_vals):.4f}+/-{np.std(filler_vals):.4f} "
            f"{np.mean(mono_vals):.4f}+/-{np.std(mono_vals):.4f}"
        )

    # Paired t-tests vs SPLADE
    if len(seeds) >= 3:
        print(f"\nPaired t-test vs SPLADE (NAOPC):")
        for method_idx, method_name in enumerate(method_names):
            if method_name == "SPLADE":
                continue
            baseline_vals = [all_results[s][method_idx].naopc for s in seeds]
            if len(set(splade_naopc_vals)) > 1 and len(set(baseline_vals)) > 1:
                t_stat, p_value = scipy_stats.ttest_rel(splade_naopc_vals, baseline_vals)
                sig = "*" if p_value < 0.05 else ""
                print(f"  SPLADE vs {method_name}: t={t_stat:.3f}, p={p_value:.4f} {sig}")
            else:
                print(f"  SPLADE vs {method_name}: insufficient variance for t-test")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Interpretability benchmark: SPLADE vs post-hoc methods")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "ag_news", "imdb", "hatexplain"])
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cvb", action="store_true", help="Run Concept Vocabulary Bottleneck analysis")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds (e.g., '42,123,456')")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if len(seeds) == 1:
        run_single_seed_benchmark(
            dataset=args.dataset, train_samples=args.train_samples, test_samples=args.test_samples,
            epochs=args.epochs, batch_size=args.batch_size, run_cvb_analysis=args.cvb, seed=seeds[0],
        )
    else:
        run_multi_seed_benchmark(
            dataset=args.dataset, train_samples=args.train_samples, test_samples=args.test_samples,
            epochs=args.epochs, batch_size=args.batch_size, seeds=seeds, run_cvb_analysis=args.cvb,
        )


if __name__ == "__main__":
    main()
