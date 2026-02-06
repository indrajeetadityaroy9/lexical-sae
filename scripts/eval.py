"""Run interpretability benchmark: SPLADE vs post-hoc explanation methods.

Thin entry point — all reusable logic lives in src/.
"""

import argparse

import numpy as np
from scipy import stats as scipy_stats

from src.baselines import (
    SPLADEAttentionExplainer,
    SPLADEIntegratedGradientsExplainer,
    SPLADELIMEExplainer,
)
from src.config import EvalConfig
from src.data import load_benchmark_data, load_hatexplain
from src.evaluation.adversarial import (
    CharacterAttack,
    TextFoolerAttack,
    WordNetAttack,
)
from src.evaluation.benchmark import (
    benchmark_explainer,
    print_cvb_analysis,
    print_interpretability_results,
)
from src.evaluation.faithfulness import UnigramSampler
from src.models import SPLADEClassifier
from src.training.finetune import finetune_splade_for_ffidelity
from src.utils import set_seed


def run_single_seed_benchmark(
    dataset: str, train_samples: int, test_samples: int,
    epochs: int, batch_size: int,
    run_cvb_analysis: bool = False, seed: int = 42,
    config: EvalConfig = EvalConfig(),
) -> list:
    set_seed(seed)
    human_rationales = None

    if dataset == "hatexplain":
        train_texts, train_labels, _, num_labels = load_hatexplain("train", train_samples)
        test_texts, test_labels, human_rationales, _ = load_hatexplain("test", test_samples)
    else:
        train_texts, train_labels, test_texts, test_labels, num_labels = load_benchmark_data(dataset, train_samples, test_samples)

    # Train single SPLADE model -- all methods explain THIS model (C4 fix)
    clf = SPLADEClassifier(num_labels=num_labels, batch_size=batch_size)
    clf.fit(train_texts, train_labels, epochs=epochs)
    accuracy = clf.score(test_texts, test_labels)
    mask_token = clf.tokenizer.mask_token

    # Build distribution-preserving filler sampler (L5)
    sampler = UnigramSampler(test_texts, seed=config.seed)

    # Fine-tune model copy for F-Fidelity (C3)
    print("\nFine-tuning model copy for F-Fidelity...")
    ftuned_clf = finetune_splade_for_ffidelity(
        clf, train_texts, train_labels,
        beta=config.ffidelity_beta,
        ft_epochs=config.ffidelity_ft_epochs,
        ft_lr=config.ffidelity_ft_lr,
        batch_size=config.ffidelity_ft_batch_size,
        mask_token=mask_token,
        seed=config.seed,
    )

    # Build multi-attack suite (H1)
    attacks = [
        WordNetAttack(max_changes=config.adversarial_max_changes),
        TextFoolerAttack(clf, max_changes=config.adversarial_max_changes),
        CharacterAttack(max_changes=config.adversarial_max_changes),
    ]

    # SPLADE native explanations (C1: now classifier-weighted)
    results = [benchmark_explainer(
        clf, "SPLADE (CVB)", lambda text, top_k: clf.explain(text, top_k=top_k),
        test_texts, config, mask_token, human_rationales,
        attacks=attacks, sampler=sampler, ftuned_clf=ftuned_clf,
    )]
    results[0].accuracy = accuracy

    # Adapter-based baselines -- all explain the same SPLADE model
    adapter_baselines = [
        (SPLADEAttentionExplainer(clf), "Attention (SPLADE)"),
        (SPLADELIMEExplainer(clf, num_samples=config.lime_num_samples), "LIME (SPLADE)"),
        (SPLADEIntegratedGradientsExplainer(clf, n_steps=config.ig_n_steps), "IntGrad (SPLADE)"),
    ]
    for explainer, name in adapter_baselines:
        result = benchmark_explainer(
            clf, name,
            lambda text, top_k, e=explainer: e.explain(text, top_k=top_k),
            test_texts, config, mask_token, human_rationales,
            attacks=attacks, sampler=sampler, ftuned_clf=ftuned_clf,
        )
        result.accuracy = accuracy
        results.append(result)

    print_interpretability_results(results, config)
    if run_cvb_analysis:
        print_cvb_analysis(clf, train_texts, train_labels, test_texts, test_labels, config)
    return results


def run_multi_seed_benchmark(
    dataset: str, train_samples: int, test_samples: int,
    epochs: int, batch_size: int,
    run_cvb_analysis: bool = False,
    config: EvalConfig = EvalConfig(),
) -> dict:
    """Run benchmark across multiple seeds and report statistics."""
    seeds = list(config.multi_seed_seeds)
    all_results: dict = {}
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}")
        all_results[seed] = run_single_seed_benchmark(
            dataset, train_samples, test_samples,
            epochs, batch_size, run_cvb_analysis, seed, config,
        )

    # Aggregate statistics
    method_names = [r.name for r in all_results[seeds[0]]]
    k_display = config.k_display

    print(f"\n{'='*80}")
    print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*80}")

    print(f"\n{'Method':<25} {'NAOPC':>18} {'Filler@' + str(k_display):>18} {'Mono':>18}")
    print("-" * 81)
    splade_naopc_vals = []
    for method_idx, method_name in enumerate(method_names):
        naopc_vals = [all_results[s][method_idx].naopc for s in seeds]
        filler_vals = [all_results[s][method_idx].filler_comprehensiveness.get(k_display, 0.0) for s in seeds]
        mono_vals = [all_results[s][method_idx].monotonicity for s in seeds]

        if method_name == "SPLADE (CVB)":
            splade_naopc_vals = naopc_vals

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
            if method_name == "SPLADE (CVB)":
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
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    if args.config is not None:
        from src.config import load_experiment_config
        raw = load_experiment_config(args.config)
        eval_fields = raw.get("evaluation", {})
        config = EvalConfig(**{k: v for k, v in eval_fields.items() if hasattr(EvalConfig, k)})
    else:
        config = EvalConfig()

    if args.multi_seed:
        run_multi_seed_benchmark(
            dataset=args.dataset, train_samples=args.train_samples, test_samples=args.test_samples,
            epochs=args.epochs, batch_size=args.batch_size, run_cvb_analysis=args.cvb, config=config,
        )
    else:
        run_single_seed_benchmark(
            dataset=args.dataset, train_samples=args.train_samples, test_samples=args.test_samples,
            epochs=args.epochs, batch_size=args.batch_size, run_cvb_analysis=args.cvb,
            seed=config.seed, config=config,
        )


if __name__ == "__main__":
    main()
