"""Experiment B: Surgical Concept Removal (Bias Lobotomy).

Trains Lexical-SAE on CivilComments or BeaverTails, identifies correlated tokens
in the target class attribution, surgically suppresses them, and measures
bias reduction (FPR gap) with accuracy preservation.

Supports two modes:
  - CivilComments: suppress identity-correlated tokens in Toxic class
  - BeaverTails: suppress harm-category-correlated tokens for disentangled surgery
"""

import argparse
import json
import os
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.data.loader import (
    load_beavertails_with_categories,
    load_civilcomments_with_identity,
)
from splade.evaluation.leace import LEACEWrappedModel, fit_leace_eraser
from splade.inference import score_model
from splade.intervene import (
    SuppressedModel,
    analyze_weff_sign_flips,
    evaluate_bias,
    get_top_tokens,
)
from splade.pipelines import setup_and_train


# Identity-related tokens commonly found in toxicity models
IDENTITY_TOKEN_CANDIDATES = [
    "muslim", "islam", "islamic", "mosque", "quran",
    "jewish", "jew", "synagogue", "torah",
    "christian", "church", "bible",
    "gay", "lesbian", "homosexual", "transgender", "bisexual",
    "black", "african", "white", "asian", "latino", "hispanic",
    "female", "woman", "women", "male", "man", "men",
    "disability", "disabled", "blind", "deaf",
]

# Harm-category-related tokens for BeaverTails disentangled surgery
HARM_TOKEN_CANDIDATES = [
    "kill", "murder", "attack", "weapon", "gun", "bomb", "shoot", "stab",
    "hate", "slur", "racist", "sexist", "bigot",
    "suicide", "harm", "hurt", "cut",
    "drug", "cocaine", "heroin", "meth",
    "steal", "fraud", "theft", "rob",
    "terrorist", "terror", "extremist",
    "sex", "porn", "nude", "explicit",
    "abuse", "torture", "violent", "violence",
]


def _load_surgery_data(config, dataset_name: str):
    """Load dataset with metadata for surgery experiment.

    Returns: (test_texts, test_labels, test_metadata, token_candidates)
    """
    seed = config.seed
    if dataset_name == "civilcomments":
        _, _, test_texts, test_labels, _, _, test_meta = load_civilcomments_with_identity(
            train_samples=0,
            test_samples=config.data.test_samples,
            seed=seed,
        )
        return test_texts, test_labels, test_meta, IDENTITY_TOKEN_CANDIDATES
    elif dataset_name == "beavertails":
        _, _, test_texts, test_labels, _, _, test_meta = load_beavertails_with_categories(
            train_samples=0,
            test_samples=config.data.test_samples,
            seed=seed,
        )
        return test_texts, test_labels, test_meta, HARM_TOKEN_CANDIDATES
    else:
        raise ValueError(f"Surgery not supported for dataset: {dataset_name}")


def _load_train_data_for_leace(config, dataset_name: str):
    """Load training data for LEACE baseline fitting."""
    seed = config.seed
    if dataset_name == "civilcomments":
        train_texts, train_labels, _, _, _, _, _ = load_civilcomments_with_identity(
            train_samples=config.data.train_samples,
            test_samples=0,
            seed=seed,
        )
    elif dataset_name == "beavertails":
        train_texts, train_labels, _, _, _, _, _ = load_beavertails_with_categories(
            train_samples=config.data.train_samples,
            test_samples=0,
            seed=seed,
        )
    else:
        raise ValueError(f"LEACE not supported for dataset: {dataset_name}")
    return train_texts, train_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B: Surgical Bias Removal")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset for surgery (civilcomments or beavertails). "
             "Defaults to config's dataset_name.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = args.dataset or config.data.dataset_name
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    print(f"\n{'=' * 60}")
    print(f"SURGICAL BIAS REMOVAL EXPERIMENT ({dataset_name})")
    print(f"{'=' * 60}")

    # Train on target dataset
    print(f"\n--- Training Lexical-SAE on {dataset_name} ---")
    exp = setup_and_train(config, seed=config.seed)
    print(f"Test accuracy: {exp.accuracy:.4f}")

    # Load metadata for bias evaluation
    print("\n--- Loading metadata annotations ---")
    test_texts, test_labels, test_meta, token_candidates = _load_surgery_data(config, dataset_name)

    # Get top tokens for target class (class_idx=1: toxic/unsafe)
    print("\n--- Top attributed tokens for target class ---")
    top_tokens = get_top_tokens(
        exp.model, exp.tokenizer, class_idx=1,
        centroid_tracker=exp.centroid_tracker, top_k=50,
    )
    print(f"{'Token':<20} {'Score':>10}")
    print("-" * 30)
    for token, score in top_tokens[:20]:
        clean_token = token.lower().lstrip("\u0120").strip("##")
        marker = " *" if clean_token in token_candidates else ""
        print(f"{token:<20} {score:>10.4f}{marker}")
    print("(* = candidate for suppression)")

    # Identify which candidate tokens appear in top attributions
    top_token_names = {t for t, _ in top_tokens}
    tokens_to_suppress = [t for t in token_candidates if t in top_token_names]
    print(f"\nCandidate tokens found in top-50 attributions: {tokens_to_suppress}")

    # Evaluate BEFORE suppression
    print("\n--- Bias evaluation BEFORE suppression ---")
    bias_before = evaluate_bias(
        exp.model, exp.tokenizer, test_texts, test_labels,
        test_meta, exp.max_length, batch_size=exp.batch_size,
    )
    _print_bias_results(bias_before)

    # Apply surgical suppression (reversible wrapper)
    print(f"\n--- Suppressing {len(tokens_to_suppress)} tokens ---")
    token_ids = exp.tokenizer.convert_tokens_to_ids(tokens_to_suppress)
    token_ids = [tid for tid in token_ids if tid != exp.tokenizer.unk_token_id]
    suppressed_model = SuppressedModel(exp.model, token_ids)

    # Evaluate AFTER suppression
    print("\n--- Bias evaluation AFTER suppression ---")
    bias_after = evaluate_bias(
        suppressed_model, exp.tokenizer, test_texts, test_labels,
        test_meta, exp.max_length, batch_size=exp.batch_size,
    )
    _print_bias_results(bias_after)

    # LEACE baseline: concept erasure via covariance-based projection
    print("\n--- LEACE concept erasure baseline ---")
    train_texts_leace, train_labels_leace = _load_train_data_for_leace(config, dataset_name)
    eraser = fit_leace_eraser(
        exp.model, exp.tokenizer, train_texts_leace, train_labels_leace,
        exp.max_length, batch_size=exp.batch_size,
    )
    leace_model = LEACEWrappedModel(exp.model, eraser)

    print("\n--- Bias evaluation with LEACE ---")
    bias_leace = evaluate_bias(
        leace_model, exp.tokenizer, test_texts, test_labels,
        test_meta, exp.max_length, batch_size=exp.batch_size,
    )
    _print_bias_results(bias_leace)

    # Summary
    acc_drop = bias_before["overall_accuracy"] - bias_after["overall_accuracy"]
    max_gap_before = max(abs(m["fpr_gap"]) for m in bias_before["per_identity"].values()) if bias_before["per_identity"] else 0
    max_gap_after = max(abs(m["fpr_gap"]) for m in bias_after["per_identity"].values()) if bias_after["per_identity"] else 0
    max_gap_leace = max(abs(m["fpr_gap"]) for m in bias_leace["per_identity"].values()) if bias_leace["per_identity"] else 0
    acc_drop_leace = bias_before["overall_accuracy"] - bias_leace["overall_accuracy"]

    print(f"\n--- Summary ---")
    print(f"{'Method':<25} {'Acc Drop':>10} {'Max FPR Gap':>12}")
    print("-" * 47)
    print(f"{'Before':<25} {'—':>10} {max_gap_before:>12.4f}")
    print(f"{'Surgical suppression':<25} {acc_drop:>+10.4f} {max_gap_after:>12.4f}")
    print(f"{'LEACE erasure':<25} {acc_drop_leace:>+10.4f} {max_gap_leace:>12.4f}")
    print(f"Tokens suppressed: {len(token_ids)}")

    # Collateral damage analysis
    print(f"\n--- Collateral Damage ---")
    print(f"{'Method':<25} {'NonTox+Identity':>16} {'NonTox-NoIdent':>16} {'Gap':>8}")
    print("-" * 65)
    for label, bias in [("Before", bias_before), ("Surgical", bias_after), ("LEACE", bias_leace)]:
        ni_acc = bias.get("nontoxic_identity_accuracy", 0.0)
        nni_acc = bias.get("nontoxic_noidentity_accuracy", 0.0)
        gap = bias.get("collateral_gap", 0.0)
        print(f"{label:<25} {ni_acc:>16.4f} {nni_acc:>16.4f} {gap:>+8.4f}")

    # W_eff sign analysis on original model
    print(f"\n--- W_eff Sign Analysis (target tokens) ---")
    sign_tokens = token_candidates[:10]
    sign_results = analyze_weff_sign_flips(
        exp.model, exp.tokenizer, test_texts, test_labels,
        sign_tokens, exp.max_length, batch_size=exp.batch_size,
    )
    if sign_results:
        print(f"{'Token':<20} {'Pos%':>8} {'Neg%':>8} {'Active':>8} {'|W_eff|':>10}")
        print("-" * 54)
        for token, info in sorted(sign_results.items(), key=lambda x: -x[1]["n_active"]):
            print(
                f"{token:<20} {info['positive_frac']:>7.1%} {info['negative_frac']:>7.1%} "
                f"{info['n_active']:>8d} {info['mean_magnitude']:>10.4f}"
            )
    else:
        print("  No target tokens active in test data.")

    results = {
        "seed": config.seed,
        "dataset": dataset_name,
        "accuracy_before": bias_before["overall_accuracy"],
        "accuracy_after": bias_after["overall_accuracy"],
        "tokens_suppressed": tokens_to_suppress,
        "bias_before": bias_before,
        "bias_after": bias_after,
        "bias_leace": bias_leace,
        "weff_sign_analysis": sign_results,
    }

    output_path = os.path.join(config.output_dir, "surgery_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def _print_bias_results(bias: dict) -> None:
    print(f"Overall accuracy: {bias['overall_accuracy']:.4f}")
    print(f"Overall FPR: {bias['overall_fpr']:.4f}")
    if "collateral_gap" in bias:
        print(f"Collateral gap: {bias['collateral_gap']:+.4f} "
              f"(identity={bias['nontoxic_identity_accuracy']:.4f}, "
              f"no-identity={bias['nontoxic_noidentity_accuracy']:.4f})")
    for name, metrics in sorted(bias["per_identity"].items(), key=lambda x: -abs(x[1]["fpr_gap"])):
        print(f"  {name:<35} FPR={metrics['fpr']:.4f}  gap={metrics['fpr_gap']:+.4f}  n={metrics['count']}")


if __name__ == "__main__":
    main()
