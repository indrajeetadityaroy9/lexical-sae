"""Faithfulness metrics for interpretability evaluation.

Reference: DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP Models" (2020)
"""

from typing import Protocol

import numpy as np


class Predictor(Protocol):
    """Protocol for models that can predict probabilities."""
    def predict_proba(self, texts: list[str]) -> list[list[float]]: ...


def _mask_tokens(text: str, tokens_to_mask: set[str], mask_token: str) -> str:
    """Replace specified tokens in text with mask token."""
    words = text.split()
    masked_words = []
    for word in words:
        word_lower = word.lower().strip('.,!?;:"\'-')
        if word_lower in tokens_to_mask or any(t in word_lower for t in tokens_to_mask):
            masked_words.append(mask_token)
        else:
            masked_words.append(word)
    return ' '.join(masked_words)


def _keep_only_tokens(text: str, tokens_to_keep: set[str], mask_token: str) -> str:
    """Keep only specified tokens, mask everything else."""
    words = text.split()
    masked_words = []
    for word in words:
        word_lower = word.lower().strip('.,!?;:"\'-')
        if word_lower in tokens_to_keep or any(t in word_lower for t in tokens_to_keep):
            masked_words.append(word)
        else:
            masked_words.append(mask_token)
    return ' '.join(masked_words)


def _get_predicted_class_prob(proba: list[float]) -> tuple[int, float]:
    """Get predicted class and its probability."""
    pred_class = int(np.argmax(proba))
    return pred_class, proba[pred_class]


def comprehensiveness(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: list[int] = [1, 5, 10, 20],
    mask_token: str = "[MASK]",
) -> dict[int, float]:
    """Compute comprehensiveness: prediction drop when top-k tokens removed.

    Higher comprehensiveness = more faithful (removing important tokens hurts more)
    """
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    for text, attrib, orig_proba in zip(texts, attributions, original_probas):
        pred_class, orig_conf = _get_predicted_class_prob(orig_proba)
        tokens = [t for t, w in attrib if w > 0]

        for k in k_values:
            if k > len(tokens):
                top_k_tokens = set(t.lower() for t in tokens)
            else:
                top_k_tokens = set(t.lower() for t in tokens[:k])

            if not top_k_tokens:
                results[k].append(0.0)
                continue

            masked_text = _mask_tokens(text, top_k_tokens, mask_token)
            masked_proba = model.predict_proba([masked_text])[0]
            masked_conf = masked_proba[pred_class]
            results[k].append(orig_conf - masked_conf)

    return {k: float(np.mean(scores)) if scores else 0.0 for k, scores in results.items()}


def sufficiency(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: list[int] = [1, 5, 10, 20],
    mask_token: str = "[MASK]",
) -> dict[int, float]:
    """Compute sufficiency: how well top-k tokens alone predict output.

    Lower sufficiency = more faithful (top tokens alone predict well)
    """
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    for text, attrib, orig_proba in zip(texts, attributions, original_probas):
        pred_class, orig_conf = _get_predicted_class_prob(orig_proba)
        tokens = [t for t, w in attrib if w > 0]

        for k in k_values:
            if k > len(tokens):
                top_k_tokens = set(t.lower() for t in tokens)
            else:
                top_k_tokens = set(t.lower() for t in tokens[:k])

            if not top_k_tokens:
                results[k].append(orig_conf)
                continue

            reduced_text = _keep_only_tokens(text, top_k_tokens, mask_token)
            reduced_proba = model.predict_proba([reduced_text])[0]
            reduced_conf = reduced_proba[pred_class]
            results[k].append(orig_conf - reduced_conf)

    return {k: float(np.mean(scores)) if scores else 0.0 for k, scores in results.items()}


def monotonicity(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    steps: int = 10,
    mask_token: str = "[MASK]",
) -> float:
    """Compute monotonicity: fraction of steps where confidence decreases.

    Higher monotonicity = more faithful (consistent importance ordering)
    """
    total_monotonic = 0
    total_steps = 0

    for text, attrib in zip(texts, attributions):
        tokens = [t for t, w in attrib if w > 0]
        if len(tokens) < 2:
            continue

        orig_proba = model.predict_proba([text])[0]
        pred_class, prev_conf = _get_predicted_class_prob(orig_proba)

        actual_steps = min(steps, len(tokens))
        step_size = max(1, len(tokens) // actual_steps)

        removed_tokens = set()
        for i in range(0, len(tokens), step_size):
            for t in tokens[i:i+step_size]:
                removed_tokens.add(t.lower())

            masked_text = _mask_tokens(text, removed_tokens, mask_token)
            masked_proba = model.predict_proba([masked_text])[0]
            curr_conf = masked_proba[pred_class]

            if curr_conf <= prev_conf:
                total_monotonic += 1
            total_steps += 1
            prev_conf = curr_conf

    return total_monotonic / total_steps if total_steps > 0 else 0.0


def aopc(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_max: int = 20,
    mask_token: str = "[MASK]",
) -> float:
    """Compute Area Over Perturbation Curve (aggregate comprehensiveness).

    Higher AOPC = more faithful explanations overall.
    """
    k_values = list(range(1, k_max + 1))
    comp_scores = comprehensiveness(model, texts, attributions, k_values, mask_token)
    return float(np.mean(list(comp_scores.values())))
