import math
import time
from dataclasses import dataclass, field

import numpy
from tqdm import tqdm

from splade.evaluation.adversarial import compute_adversarial_sensitivity
from splade.evaluation.causal import (MLMCounterfactualGenerator,
                                      compute_causal_faithfulness)
from splade.evaluation.constants import (ADVERSARIAL_MAX_CHANGES,
                                         ADVERSARIAL_MCP_THRESHOLD,
                                         ADVERSARIAL_TEST_SAMPLES,
                                         FFIDELITY_BETA, FFIDELITY_N_SAMPLES,
                                         K_MAX, K_VALUES, MONOTONICITY_STEPS,
                                         NAOPC_BEAM_SIZE,
                                         SOFT_METRIC_N_SAMPLES)
from splade.evaluation.faithfulness import (
    UnigramSampler, compute_comprehensiveness,
    compute_ffidelity_comprehensiveness, compute_ffidelity_sufficiency,
    compute_filler_comprehensiveness, compute_monotonicity,
    compute_normalized_aopc, compute_soft_metrics, compute_sufficiency)
from splade.evaluation.token_alignment import normalize_attributions_to_words


@dataclass
class InterpretabilityResult:
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
    explanation_time: float = 0.0
    accuracy: float = 0.0
    inference_latency: float = 0.0
    causal_faithfulness: float = 0.0


def aggregate_results(results_list: list[list[InterpretabilityResult]]) -> list[dict]:
    if not results_list:
        return []

    methods = [r.name for r in results_list[0]]
    aggregated = []

    for i, name in enumerate(methods):
        method_seeds = [seeds[i] for seeds in results_list]
        stats = {"name": name}

        metrics = [
            "monotonicity", "aopc", "naopc", "naopc_lower", "naopc_upper",
            "soft_comprehensiveness", "soft_sufficiency", "adversarial_sensitivity",
            "adversarial_mean_tau", "accuracy", "inference_latency", "causal_faithfulness"
        ]

        for metric in metrics:
            vals = [getattr(s, metric) for s in method_seeds]
            stats[f"{metric}_mean"] = float(numpy.nanmean(vals))
            stats[f"{metric}_std"] = float(numpy.nanstd(vals))

        k_val = K_VALUES[len(K_VALUES) // 2]
        stats["k"] = k_val
        for metric in ["comprehensiveness", "sufficiency", "filler_comprehensiveness"]:
            vals = [getattr(s, metric).get(k_val, 0.0) for s in method_seeds]
            stats[f"{metric}_mean"] = float(numpy.mean(vals))
            stats[f"{metric}_std"] = float(numpy.std(vals))

        aggregated.append(stats)
    return aggregated


def print_aggregated_results(aggregated: list[dict]):
    print("\n" + "=" * 140)
    print(f"{'Method':<20} {'Accuracy':>12} {'NAOPC':>12} {'Filler':>12} {'Latency (ms)':>15} {'Causal':>12}")
    print("-" * 140)
    for res in aggregated:
        print(
            f"{res['name']:<20} "
            f"{res['accuracy_mean']:>6.4f}+/-{res['accuracy_std']:>4.4f} "
            f"{res['naopc_mean']:>6.4f}+/-{res['naopc_std']:>4.4f} "
            f"{res['filler_comprehensiveness_mean']:>6.4f}+/-{res['filler_comprehensiveness_std']:>4.4f} "
            f"{res['inference_latency_mean']*1000:>8.2f}+/-{res['inference_latency_std']*1000:>4.2f} "
            f"{res['causal_faithfulness_mean']:>6.4f}+/-{res['causal_faithfulness_std']:>4.4f}"
        )
    print("=" * 140)


def print_interpretability_results(results: list[InterpretabilityResult]) -> None:
    k_display = K_VALUES[len(K_VALUES) // 2]

    print("\n" + "=" * 80)
    print("INTERPRETABILITY BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\n--- PRIMARY METRICS (robust to OOD artifacts) ---")
    print(f"\n{'Method':<30} {'NAOPC':>10} {'Filler@' + str(k_display):>12} {'Latency':>12} {'Causal':>10} {'Accuracy':>10}")
    print("-" * 96)
    for result in results:
        filler = result.filler_comprehensiveness.get(k_display, 0.0)
        print(
            f"{result.name:<30} {result.naopc:>10.4f} {filler:>12.4f} "
            f"{result.inference_latency*1000:>11.2f}ms {result.causal_faithfulness:>10.4f} {result.accuracy:>10.4f}"
        )

    print(f"\n{'Method':<30} {'AOPC_lo':>10} {'AOPC_hi':>10} {'Adv Sens':>10} {'Mean Tau':>10}")
    print("-" * 72)
    for result in results:
        print(
            f"{result.name:<30} {result.naopc_lower:>10.4f} "
            f"{result.naopc_upper:>10.4f} {result.adversarial_sensitivity:>10.4f} "
            f"{result.adversarial_mean_tau:>10.4f}"
        )

    print(f"\n--- SUPPLEMENTARY ERASER METRICS ---")
    print("  Note: Comp/Suff can be inflated by OOD artifacts (arXiv:2308.14272).")
    print(
        f"\n{'Method':<30} {'Comp@' + str(k_display):>10} {'Suff@' + str(k_display):>10} "
        f"{'F-Comp@' + str(k_display):>12} {'F-Suff@' + str(k_display):>12} {'Time':>10}"
    )
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

    print(f"\n--- SOFT PERTURBATION METRICS (arXiv:2305.10496) ---")
    print(f"  Note: Probabilistic masking avoids OOD artifacts of hard masking.")
    print(f"\n{'Method':<30} {'Soft Comp':>12} {'Soft Suff':>12}")
    print("-" * 56)
    for result in results:
        print(
            f"{result.name:<30} {result.soft_comprehensiveness:>12.4f} "
            f"{result.soft_sufficiency:>12.4f}"
        )

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
    print("  Causal: Counterfactual Consistency (higher = better, semantic alignment)")
    print("  Latency: Inference time per sample in ms (lower = better)")
    print("  Soft Comp: Soft comprehensiveness (higher = better, probabilistic masking)")
    print("  F-Comp/F-Suff: Beta-bounded variants (reduced OOD effects)")
    print("  FF-Comp/FF-Suff: F-Fidelity with fine-tuned model (proper OOD handling)")
    print("  Adv Sens: Adversarial sensitivity with multi-attack + tau-hat")

    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    def _safe_key(val):
        return val if not math.isnan(val) else float('-inf')

    best_naopc = max(results, key=lambda result: _safe_key(result.naopc))
    best_filler = max(results, key=lambda result: _safe_key(result.filler_comprehensiveness.get(k_display, 0.0)))
    best_mono = max(results, key=lambda result: _safe_key(result.monotonicity))
    best_causal = max(results, key=lambda result: _safe_key(result.causal_faithfulness))
    best_comp = max(results, key=lambda result: _safe_key(result.comprehensiveness.get(k_display, 0.0)))
    best_soft_comp = max(results, key=lambda result: _safe_key(result.soft_comprehensiveness))

    print(f"  Best NAOPC: {best_naopc.name}")
    print(f"  Best Filler Comp@{k_display}: {best_filler.name}")
    print(f"  Best Causal Faithfulness: {best_causal.name}")
    print(f"  Best Soft Comp: {best_soft_comp.name}")
    print(f"  Best Monotonicity: {best_mono.name}")
    print(f"  Best Comprehensiveness@{k_display}: {best_comp.name}")


def benchmark_explainer(
    clf,
    name: str,
    explain_fn,
    batch_explain_fn,
    test_texts: list[str],
    mask_token: str,
    seed: int,
    attacks: list,
    sampler: UnigramSampler,
    ftuned_clf,
    tokenizer,
    max_length: int = 128,
) -> InterpretabilityResult:
    k_values = list(K_VALUES)

    print(f"\nGenerating explanations for {name}...")
    start = time.time()
    raw_attributions = batch_explain_fn(test_texts, K_MAX)
    explanation_time = time.time() - start

    inference_latency = explanation_time / len(test_texts) if test_texts else 0.0

    attributions = [
        normalize_attributions_to_words(text, attrib, tokenizer)
        for text, attrib in zip(test_texts, raw_attributions)
    ]

    print("Computing metrics...")
    result = InterpretabilityResult(
        name=name,
        explanation_time=explanation_time,
        inference_latency=inference_latency
    )

    original_probs = clf.predict_proba(test_texts)

    aopc_k_values = list(range(1, K_MAX + 1))
    all_comp_k = sorted(set(k_values) | set(aopc_k_values))
    all_comp_scores = compute_comprehensiveness(clf, test_texts, attributions, all_comp_k, mask_token, original_probs=original_probs)
    result.comprehensiveness = {k: all_comp_scores[k] for k in k_values}
    result.aopc = float(numpy.mean([all_comp_scores[k] for k in aopc_k_values]))

    result.sufficiency = compute_sufficiency(clf, test_texts, attributions, k_values, mask_token, original_probs=original_probs)
    result.monotonicity = compute_monotonicity(clf, test_texts, attributions, MONOTONICITY_STEPS, mask_token, original_probs=original_probs)

    naopc_result = compute_normalized_aopc(
        clf, test_texts, attributions,
        k_max=K_MAX, beam_size=NAOPC_BEAM_SIZE, mask_token=mask_token,
        original_probs=original_probs,
    )
    result.naopc = naopc_result["naopc"]
    result.naopc_lower = naopc_result["aopc_lower"]
    result.naopc_upper = naopc_result["aopc_upper"]

    result.f_comprehensiveness = compute_comprehensiveness(
        clf, test_texts, attributions, k_values, mask_token,
        beta=FFIDELITY_BETA, original_probs=original_probs,
    )
    result.f_sufficiency = compute_sufficiency(
        clf, test_texts, attributions, k_values, mask_token,
        beta=FFIDELITY_BETA, original_probs=original_probs,
    )

    ft_original_probs = ftuned_clf.predict_proba(test_texts)
    result.ffidelity_comp = compute_ffidelity_comprehensiveness(
        ftuned_clf, test_texts, attributions, k_values, mask_token,
        beta=FFIDELITY_BETA, n_samples=FFIDELITY_N_SAMPLES, seed=seed,
        original_probs=ft_original_probs,
    )
    result.ffidelity_suff = compute_ffidelity_sufficiency(
        ftuned_clf, test_texts, attributions, k_values, mask_token,
        beta=FFIDELITY_BETA, n_samples=FFIDELITY_N_SAMPLES, seed=seed,
        original_probs=ft_original_probs,
    )

    result.filler_comprehensiveness = compute_filler_comprehensiveness(
        clf, test_texts, attributions, k_values, sampler,
        original_probs=original_probs,
    )

    result.soft_comprehensiveness, result.soft_sufficiency = compute_soft_metrics(
        clf, test_texts, attributions, mask_token,
        n_samples=SOFT_METRIC_N_SAMPLES, seed=seed,
        tokenizer=tokenizer, max_length=max_length,
        original_probs=original_probs,
    )

    def normalized_explain_fn(text, top_k):
        raw = explain_fn(text, top_k)
        return normalize_attributions_to_words(text, raw, tokenizer)
    adv_explain_fn = normalized_explain_fn

    adv_subset_probs = original_probs[:ADVERSARIAL_TEST_SAMPLES]
    adv_result = compute_adversarial_sensitivity(
        clf, adv_explain_fn,
        test_texts[:ADVERSARIAL_TEST_SAMPLES],
        attacks=attacks,
        max_changes=ADVERSARIAL_MAX_CHANGES,
        mcp_threshold=ADVERSARIAL_MCP_THRESHOLD,
        top_k=K_MAX,
        seed=seed,
        original_probs=adv_subset_probs,
    )
    result.adversarial_sensitivity = adv_result["adversarial_sensitivity"]
    result.adversarial_mean_tau = adv_result["mean_tau"]

    result.causal_faithfulness = compute_causal_faithfulness(
        clf.model,
        clf.tokenizer,
        test_texts,
        attributions,
        clf.max_length,
        generator=MLMCounterfactualGenerator(clf.tokenizer.name_or_path),
    )

    return result
