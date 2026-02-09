"""Experiment B: Surgical Concept Removal (Bias Lobotomy).

Trains Lexical-SAE on CivilComments, identifies identity-correlated tokens
in the "Toxic" class attribution, surgically suppresses them, and measures
bias reduction (FPR gap) with accuracy preservation.
"""

import argparse
import json
import os
from dataclasses import asdict

import yaml

from splade.config.load import load_config
from splade.data.loader import load_civilcomments_with_identity
from splade.inference import score_model
from splade.intervene import (
    SuppressedModel,
    evaluate_bias,
    get_top_tokens,
    suppress_tokens_by_name,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B: Surgical Bias Removal")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "resolved_config.yaml"), "w") as f:
        yaml.dump(asdict(config), f)

    print(f"\n{'=' * 60}")
    print("SURGICAL BIAS REMOVAL EXPERIMENT")
    print(f"{'=' * 60}")

    # Train on CivilComments
    print("\n--- Training Lexical-SAE on CivilComments ---")
    exp = setup_and_train(config, seed=config.evaluation.seeds[0])
    print(f"Test accuracy: {exp.accuracy:.4f}")

    # Load identity annotations for bias evaluation
    print("\n--- Loading identity annotations ---")
    _, _, test_texts, test_labels, _, _, test_identities = load_civilcomments_with_identity(
        train_samples=0,
        test_samples=config.data.test_samples,
        seed=config.evaluation.seeds[0],
    )

    # Get top tokens for Toxic class (class_idx=1)
    print("\n--- Top attributed tokens for Toxic class ---")
    if exp.centroid_tracker is not None:
        top_tokens = get_top_tokens(
            exp.model, exp.tokenizer, class_idx=1,
            centroid_tracker=exp.centroid_tracker, top_k=50,
        )
        print(f"{'Token':<20} {'Score':>10}")
        print("-" * 30)
        for token, score in top_tokens[:20]:
            clean_token = token.lower().lstrip("\u0120").strip("##")
            marker = " *" if clean_token in IDENTITY_TOKEN_CANDIDATES else ""
            print(f"{token:<20} {score:>10.4f}{marker}")
        print("(* = identity-correlated)")

    # Identify which identity tokens appear in top attributions
    top_token_names = {t for t, _ in top_tokens} if exp.centroid_tracker else set()
    tokens_to_suppress = [t for t in IDENTITY_TOKEN_CANDIDATES if t in top_token_names]
    print(f"\nIdentity tokens found in top-50 Toxic attributions: {tokens_to_suppress}")

    # Evaluate BEFORE suppression
    print("\n--- Bias evaluation BEFORE suppression ---")
    bias_before = evaluate_bias(
        exp.model, exp.tokenizer, test_texts, test_labels,
        test_identities, exp.max_length, batch_size=exp.batch_size,
    )
    print(f"Overall accuracy: {bias_before['overall_accuracy']:.4f}")
    print(f"Overall FPR: {bias_before['overall_fpr']:.4f}")
    for name, metrics in sorted(bias_before["per_identity"].items(), key=lambda x: -abs(x[1]["fpr_gap"])):
        print(f"  {name:<35} FPR={metrics['fpr']:.4f}  gap={metrics['fpr_gap']:+.4f}  n={metrics['count']}")

    # Apply surgical suppression (reversible wrapper)
    print(f"\n--- Suppressing {len(tokens_to_suppress)} identity tokens ---")
    token_ids = exp.tokenizer.convert_tokens_to_ids(tokens_to_suppress)
    token_ids = [tid for tid in token_ids if tid != exp.tokenizer.unk_token_id]
    suppressed_model = SuppressedModel(exp.model, token_ids)

    # Evaluate AFTER suppression
    print("\n--- Bias evaluation AFTER suppression ---")
    bias_after = evaluate_bias(
        suppressed_model, exp.tokenizer, test_texts, test_labels,
        test_identities, exp.max_length, batch_size=exp.batch_size,
    )
    print(f"Overall accuracy: {bias_after['overall_accuracy']:.4f}")
    print(f"Overall FPR: {bias_after['overall_fpr']:.4f}")
    for name, metrics in sorted(bias_after["per_identity"].items(), key=lambda x: -abs(x[1]["fpr_gap"])):
        print(f"  {name:<35} FPR={metrics['fpr']:.4f}  gap={metrics['fpr_gap']:+.4f}  n={metrics['count']}")

    # Summary
    acc_drop = bias_before["overall_accuracy"] - bias_after["overall_accuracy"]
    max_gap_before = max(abs(m["fpr_gap"]) for m in bias_before["per_identity"].values()) if bias_before["per_identity"] else 0
    max_gap_after = max(abs(m["fpr_gap"]) for m in bias_after["per_identity"].values()) if bias_after["per_identity"] else 0

    print(f"\n--- Summary ---")
    print(f"Accuracy drop: {acc_drop:+.4f}")
    print(f"Max FPR gap: {max_gap_before:.4f} -> {max_gap_after:.4f}")
    print(f"Tokens suppressed: {len(token_ids)}")

    results = {
        "seed": config.evaluation.seeds[0],
        "accuracy_before": bias_before["overall_accuracy"],
        "accuracy_after": bias_after["overall_accuracy"],
        "tokens_suppressed": tokens_to_suppress,
        "bias_before": bias_before,
        "bias_after": bias_after,
    }

    output_path = os.path.join(config.output_dir, "surgery_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
