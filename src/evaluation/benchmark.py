"""Interpretability benchmark: evaluation helpers and result reporting.

All explanation methods explain the same SPLADE model via adapter classes,
ensuring fair faithfulness comparisons.

All evaluation parameters are sourced from the central EvalConfig dataclass.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from src.config.eval_config import EvalConfig
from src.evaluation.adversarial import compute_adversarial_sensitivity
from src.evaluation.concept_analysis import concept_intervention, concept_necessity, concept_sufficiency
from src.evaluation.faithfulness import (
    UnigramSampler,
    compute_comprehensiveness,
    compute_filler_comprehensiveness,
    compute_monotonicity,
    compute_normalized_aopc,
    compute_soft_comprehensiveness,
    compute_soft_sufficiency,
    compute_sufficiency,
)
from src.data.loader import compute_rationale_agreement


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
    ffidelity_comp: dict[int, float] = field(default_factory=dict)
    ffidelity_suff: dict[int, float] = field(default_factory=dict)
    filler_comprehensiveness: dict[int, float] = field(default_factory=dict)
    soft_comprehensiveness: float = 0.0
    soft_sufficiency: float = 0.0
    adversarial_sensitivity: float = 0.0
    adversarial_mean_tau: float = 0.0
    rationale_f1: float | None = None
    explanation_time: float = 0.0
    accuracy: float = 0.0


def print_interpretability_results(results: list[InterpretabilityResult], config: EvalConfig) -> None:
    """Print benchmark results comparison table."""
    print("\n" + "=" * 80)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("=" * 80)

    k_display = config.k_display

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
        comp = result.comprehensiveness.get(k_display, 0.0)
        suff = result.sufficiency.get(k_display, 0.0)
        f_comp = result.f_comprehensiveness.get(k_display, 0.0)
        f_suff = result.f_sufficiency.get(k_display, 0.0)
        time_str = f"{result.explanation_time:.1f}s"
        print(
            f"{result.name:<30} {comp:>10.4f} {suff:>10.4f} "
            f"{f_comp:>12.4f} {f_suff:>12.4f} {time_str:>10}"
        )

    # Soft perturbation metrics
    print(f"\n--- SOFT PERTURBATION METRICS (arXiv:2305.10496) ---")
    print(f"  Note: Probabilistic masking avoids OOD artifacts of hard masking.")
    print(f"\n{'Method':<30} {'Soft Comp':>12} {'Soft Suff':>12}")
    print("-" * 56)
    for result in results:
        print(
            f"{result.name:<30} {result.soft_comprehensiveness:>12.4f} "
            f"{result.soft_sufficiency:>12.4f}"
        )

    # F-Fidelity (fine-tuned model) metrics
    if any(result.ffidelity_comp for result in results):
        print(f"\n--- F-FIDELITY METRICS (fine-tuned model, arXiv:2410.02970) ---")
        print(f"\n{'Method':<30} {'FF-Comp@' + str(k_display):>12} {'FF-Suff@' + str(k_display):>12}")
        print("-" * 56)
        for result in results:
            ff_comp = result.ffidelity_comp.get(k_display, 0.0)
            ff_suff = result.ffidelity_suff.get(k_display, 0.0)
            print(f"{result.name:<30} {ff_comp:>12.4f} {ff_suff:>12.4f}")

    print("\n" + "-" * 82)
    print("Interpretation:")
    print("  NAOPC: Normalized AOPC (0-1, higher = better, per-example normalized)")
    print("  Filler: Filler-token comprehensiveness (higher = better, OOD-robust)")
    print("  Monotonicity: Higher = better (consistent importance ordering)")
    print("  Soft Comp: Soft comprehensiveness (higher = better, probabilistic masking)")
    print("  Soft Suff: Soft sufficiency (lower = better, probabilistic retention)")
    print("  Comp/Suff: ERASER metrics (higher comp / lower suff = better)")
    print("  F-Comp/F-Suff: Beta-bounded variants (reduced OOD effects)")
    print("  FF-Comp/FF-Suff: F-Fidelity with fine-tuned model (proper OOD handling)")
    print("  Adv Sens: Adversarial sensitivity with multi-attack + tau-hat")

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
    best_comp = max(results, key=lambda r: r.comprehensiveness.get(k_display, 0.0))
    best_soft_comp = max(results, key=lambda r: r.soft_comprehensiveness)

    print(f"  Best NAOPC: {best_naopc.name}")
    print(f"  Best Filler Comp@{k_display}: {best_filler.name}")
    print(f"  Best Soft Comp: {best_soft_comp.name}")
    print(f"  Best Monotonicity: {best_mono.name}")
    print(f"  Best Comprehensiveness@{k_display}: {best_comp.name}")


def benchmark_explainer(
    clf, name: str, explain_fn,
    test_texts: list[str],
    config: EvalConfig,
    mask_token: str,
    human_rationales: list[list[str]] | None = None,
    attacks: list | None = None,
    sampler: UnigramSampler | None = None,
    ftuned_clf=None,
) -> InterpretabilityResult:
    """Evaluate a single explainer on faithfulness metrics."""
    k_values = list(config.k_values)

    print(f"\nGenerating explanations for {name}...")
    start = time.time()
    attributions = [explain_fn(text, config.k_max) for text in tqdm(test_texts, desc="Explaining")]
    explanation_time = time.time() - start

    print("Computing metrics...")
    result = InterpretabilityResult(name=name, explanation_time=explanation_time)
    result.comprehensiveness = compute_comprehensiveness(
        clf, test_texts, attributions, k_values, mask_token,
    )
    result.sufficiency = compute_sufficiency(
        clf, test_texts, attributions, k_values, mask_token,
    )
    result.monotonicity = compute_monotonicity(
        clf, test_texts, attributions, config.monotonicity_steps, mask_token,
    )
    aopc_scores = compute_comprehensiveness(
        clf, test_texts, attributions, list(range(1, config.k_max + 1)), mask_token,
    )
    result.aopc = float(np.mean(list(aopc_scores.values())))

    naopc_result = compute_normalized_aopc(
        clf, test_texts, attributions,
        k_max=config.k_max, beam_size=config.naopc_beam_size, mask_token=mask_token,
    )
    result.naopc = naopc_result["naopc"]
    result.naopc_lower = naopc_result["aopc_lower"]
    result.naopc_upper = naopc_result["aopc_upper"]

    # Beta-bounded comprehensiveness/sufficiency
    result.f_comprehensiveness = compute_comprehensiveness(
        clf, test_texts, attributions, k_values, mask_token, beta=config.ffidelity_beta,
    )
    result.f_sufficiency = compute_sufficiency(
        clf, test_texts, attributions, k_values, mask_token, beta=config.ffidelity_beta,
    )

    # F-Fidelity with fine-tuned model (arXiv:2410.02970)
    if ftuned_clf is not None:
        result.ffidelity_comp = compute_comprehensiveness(
            ftuned_clf, test_texts, attributions, k_values, mask_token,
            beta=config.ffidelity_beta,
        )
        result.ffidelity_suff = compute_sufficiency(
            ftuned_clf, test_texts, attributions, k_values, mask_token,
            beta=config.ffidelity_beta,
        )

    # Distribution-preserving filler comprehensiveness
    if sampler is not None:
        result.filler_comprehensiveness = compute_filler_comprehensiveness(
            clf, test_texts, attributions, k_values, sampler,
        )

    # Soft perturbation metrics (arXiv:2305.10496)
    result.soft_comprehensiveness = compute_soft_comprehensiveness(
        clf, test_texts, attributions, mask_token,
        n_samples=config.soft_metric_n_samples, seed=config.seed,
    )
    result.soft_sufficiency = compute_soft_sufficiency(
        clf, test_texts, attributions, mask_token,
        n_samples=config.soft_metric_n_samples, seed=config.seed,
    )

    # Multi-attack adversarial sensitivity with tau-hat
    adv_result = compute_adversarial_sensitivity(
        clf, explain_fn, test_texts[:config.adversarial_test_samples],
        attacks=attacks,
        max_changes=config.adversarial_max_changes,
        mcp_threshold=config.adversarial_mcp_threshold,
        top_k=config.k_max,
        seed=config.seed,
    )
    result.adversarial_sensitivity = adv_result["adversarial_sensitivity"]
    result.adversarial_mean_tau = adv_result["mean_tau"]

    if human_rationales:
        result.rationale_f1 = compute_rationale_agreement(
            attributions, human_rationales, k=config.k_max,
        )

    return result


def print_cvb_analysis(
    clf,
    train_texts: list[str], train_labels: list[int],
    test_texts: list[str], test_labels: list[int],
    config: EvalConfig,
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
    print(f"\nTop {config.display_top_n_concepts} Global Concepts:")
    train_sparse = clf.transform(train_texts)
    importance = np.abs(train_sparse).mean(axis=0)
    top_indices = np.argsort(importance)[-config.concept_top_n:][::-1]
    for i, idx in enumerate(top_indices[:config.display_top_n_concepts]):
        print(f"  {i + 1:2}. {clf.tokenizer.decode([idx]):<15} {importance[idx]:.4f}")

    # Concept completeness (linear probe)
    print("\nConcept Completeness (Linear Probe):")
    test_importance = np.abs(test_sparse).mean(axis=0)
    n_folds = min(5, len(test_labels))
    for k in config.concept_top_k_values:
        mask = np.zeros(test_sparse.shape[1])
        mask[np.argsort(test_importance)[-k:]] = 1
        lr = LogisticRegression(max_iter=config.linear_probe_max_iter, random_state=config.seed)
        scores = cross_val_score(lr, test_sparse * mask, test_labels, cv=n_folds, scoring="accuracy")
        print(f"  Top-{k:4}: {scores.mean():.4f}")

    # CB-LLM Concept Analysis (arXiv:2412.07992)
    print("\nConcept Sufficiency (arXiv:2412.07992):")
    suff_results = concept_sufficiency(
        clf, test_texts, test_labels, top_k_values=list(config.concept_top_k_values),
    )
    for k, acc in suff_results.items():
        print(f"  Top-{k:4} concepts: {acc:.4f} accuracy")

    print("\nConcept Necessity (accuracy drop when top-k removed):")
    nec_results = concept_necessity(
        clf, test_texts, test_labels, top_k_values=list(config.concept_top_k_values),
    )
    for k, drop in nec_results.items():
        print(f"  Top-{k:4} concepts: {drop:.4f} accuracy drop")

    top_concept_indices = np.argsort(importance)[-config.concept_top_n:].tolist()
    print(f"\nConcept Intervention (zeroing top-{config.concept_top_n} concepts):")
    interv = concept_intervention(
        clf, test_texts, test_labels, top_concept_indices,
        num_trials=config.concept_intervention_trials, seed=config.seed,
    )
    print(f"  Original accuracy: {interv['accuracy_original']:.4f}")
    print(f"  After intervention: {interv['accuracy_intervened']:.4f}")
    print(f"  Accuracy drop: {interv['accuracy_drop']:.4f}")

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
    with torch.inference_mode():
        weights = clf.model.classifier.weight.cpu().numpy()
    for class_idx in range(weights.shape[0]):
        class_weights = weights[class_idx]
        top_idx = np.argsort(np.abs(class_weights))[-10:][::-1]
        terms = [(clf.tokenizer.decode([i]), float(class_weights[i])) for i in top_idx]
        pos = ", ".join([f"{t}(+{w:.2f})" for t, w in terms if w > 0][:3])
        neg = ", ".join([f"{t}({w:.2f})" for t, w in terms if w < 0][:3])
        print(f"  Class {class_idx}: {pos} | {neg}")
