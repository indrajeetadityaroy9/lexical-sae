"""Experiment C: Long-Context Needle in Haystack.

Tests whether Lexical-SAE's sparse bottleneck naturally localizes signal
in long documents. Trains on short texts, then evaluates on progressively
longer inputs (padding with irrelevant text). ModernBERT supports up to 8192 tokens.
"""

import random

from cajt.config import Config
from cajt.evaluation.collect import score_model
from cajt.training.pipeline import setup_and_train


def _pad_texts_to_length(
    texts: list[str],
    target_word_count: int,
    filler_texts: list[str],
    seed: int = 42,
) -> list[str]:
    """Pad each text to approximately target_word_count by appending filler."""
    rng = random.Random(seed)
    padded = []
    for text in texts:
        words = text.split()
        while len(words) < target_word_count and filler_texts:
            filler = rng.choice(filler_texts)
            words.extend(filler.split())
        padded.append(" ".join(words[:target_word_count]))
    return padded


def run(config: Config) -> dict:
    """Run long-context needle-in-haystack experiment."""
    print(f"\n{'=' * 60}")
    print("LONG-CONTEXT NEEDLE IN HAYSTACK EXPERIMENT")
    print(f"{'=' * 60}")

    seed = config.seed

    # Train on standard-length texts
    print("\n--- Training Lexical-SAE ---")
    exp = setup_and_train(config, seed)
    print(f"Baseline accuracy (original length): {exp.accuracy:.4f}")

    # Load filler texts for padding (use train texts as filler)
    filler_texts = exp.train_texts[:500]

    # Test at increasing lengths from config
    target_lengths = config.long_context.target_word_counts
    max_token_lengths = config.long_context.max_token_lengths

    results = {
        "seed": seed,
        "baseline_accuracy": exp.accuracy,
        "length_results": [],
    }

    print(f"\n{'Word Count':<15} {'Max Tokens':<15} {'Accuracy':>10}")
    print("-" * 40)

    for word_count, max_tokens in zip(target_lengths, max_token_lengths):
        padded_texts = _pad_texts_to_length(
            exp.test_texts, word_count, filler_texts, seed=seed,
        )
        acc = score_model(
            exp.model, exp.tokenizer, padded_texts, exp.test_labels,
            max_length=max_tokens, batch_size=exp.batch_size,
            num_labels=exp.num_labels,
        )
        print(f"{word_count:<15} {max_tokens:<15} {acc:>10.4f}")
        results["length_results"].append({
            "word_count": word_count,
            "max_tokens": max_tokens,
            "accuracy": acc,
        })

    return results
