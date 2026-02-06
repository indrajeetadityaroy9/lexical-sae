"""Faithfulness metrics for interpretability evaluation."""

import random
from typing import Protocol

import numpy as np

_FILLER_TOKENS = ["the", "a", "an", "is", "was", "and", "or", "but", "in", "on"]


class Predictor(Protocol):
    def predict_proba(self, texts: list[str]) -> list[list[float]]: ...


def _top_k_tokens(attrib: list[tuple[str, float]], k: int) -> set[str]:
    """Select top-k tokens by weight, preserving importance ordering."""
    seen: set[str] = set()
    result: set[str] = set()
    for t, w in attrib:
        if w <= 0:
            continue
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.add(tl)
            if len(result) >= k:
                break
    return result


def _mask_by_token_set(
    text: str,
    token_set: set[str],
    mask_token: str,
    mode: str = "remove",
    max_fraction: float = 1.0,
) -> str:
    """Mask tokens in text based on a token set."""
    normalized = {t.lstrip("#").lower() for t in token_set}
    words = text.split()
    max_masks = int(len(words) * max_fraction) if max_fraction < 1.0 else len(words)

    if mode == "remove":
        mask_positions = [
            i for i, w in enumerate(words)
            if w.lower().strip('.,!?;:"\'-') in normalized
        ][:max_masks]
    else:  # mode == "keep"
        mask_positions = [
            i for i, w in enumerate(words)
            if w.lower().strip('.,!?;:"\'-') not in normalized
        ][:max_masks]

    masked_words = list(words)
    for pos in mask_positions:
        masked_words[pos] = mask_token
    return ' '.join(masked_words)


def compute_comprehensiveness(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: list[int] = [1, 5, 10, 20], mask_token: str = "[MASK]",
    beta: float = 1.0,
) -> dict[int, float]:
    """Prediction drop when top-k tokens removed (ERASER/F-Fidelity)."""
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    for text, attrib, orig_proba in zip(texts, attributions, original_probas):
        pred_class = int(np.argmax(orig_proba))
        orig_conf = orig_proba[pred_class]
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            masked_text = _mask_by_token_set(
                text, top_tokens, mask_token,
                mode="remove", max_fraction=beta,
            )
            masked_conf = model.predict_proba([masked_text])[0][pred_class]
            results[k].append(orig_conf - masked_conf)

    return {k: float(np.mean(scores)) for k, scores in results.items()}


def compute_sufficiency(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: list[int] = [1, 5, 10, 20], mask_token: str = "[MASK]",
    beta: float = 1.0,
) -> dict[int, float]:
    """How well top-k tokens alone predict output (ERASER/F-Fidelity)."""
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    for text, attrib, orig_proba in zip(texts, attributions, original_probas):
        pred_class = int(np.argmax(orig_proba))
        orig_conf = orig_proba[pred_class]
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            reduced_text = _mask_by_token_set(
                text, top_tokens, mask_token,
                mode="keep", max_fraction=beta,
            )
            reduced_conf = model.predict_proba([reduced_text])[0][pred_class]
            results[k].append(orig_conf - reduced_conf)

    return {k: float(np.mean(scores)) for k, scores in results.items()}


def compute_monotonicity(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    steps: int = 10, mask_token: str = "[MASK]",
) -> float:
    """Fraction of removal steps where confidence decreases."""
    total_monotonic, total_steps = 0, 0

    for text, attrib in zip(texts, attributions):
        tokens = [t for t, w in attrib if w > 0]
        orig_proba = model.predict_proba([text])[0]
        pred_class = int(np.argmax(orig_proba))
        prev_conf = orig_proba[pred_class]

        actual_steps = min(steps, len(tokens))
        step_size = max(1, len(tokens) // actual_steps)
        removed_tokens = set()

        for i in range(0, len(tokens), step_size):
            for t in tokens[i:i+step_size]:
                removed_tokens.add(t.lower())
            masked_text = _mask_by_token_set(text, removed_tokens, mask_token, mode="remove")
            curr_conf = model.predict_proba([masked_text])[0][pred_class]
            if curr_conf <= prev_conf:
                total_monotonic += 1
            total_steps += 1
            prev_conf = curr_conf

    return total_monotonic / total_steps


def _compute_aopc_for_ordering(
    model: Predictor, text: str, ordering: list[str], mask_token: str = "[MASK]",
) -> float:
    orig_proba = model.predict_proba([text])[0]
    pred_class = int(np.argmax(orig_proba))
    orig_conf = orig_proba[pred_class]
    total_drop = 0.0
    removed_tokens = set()
    for token in ordering:
        removed_tokens.add(token.lower())
        masked_conf = model.predict_proba(
            [_mask_by_token_set(text, removed_tokens, mask_token, mode="remove")]
        )[0][pred_class]
        total_drop += orig_conf - masked_conf
    return total_drop / len(ordering)


def _beam_search_ordering(
    model: Predictor, text: str, tokens: list[str],
    beam_size: int = 5, mask_token: str = "[MASK]",
) -> float:
    orig_proba = model.predict_proba([text])[0]
    pred_class = int(np.argmax(orig_proba))
    orig_conf = orig_proba[pred_class]
    beams = [(set(), [], 0.0)]

    for _ in range(len(tokens)):
        candidates = []
        for removed_set, ordering, cum_drop in beams:
            for token in [t for t in tokens if t.lower() not in removed_set]:
                new_removed = removed_set | {token.lower()}
                masked_conf = model.predict_proba(
                    [_mask_by_token_set(text, new_removed, mask_token, mode="remove")]
                )[0][pred_class]
                candidates.append((new_removed, ordering + [token], cum_drop + orig_conf - masked_conf))
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_size]

    return beams[0][2] / len(tokens)


def compute_normalized_aopc(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_max: int = 20, beam_size: int = 15, n_random_samples: int = 10,
    mask_token: str = "[MASK]",
) -> dict[str, float]:
    """Normalized AOPC (arXiv:2408.08137)."""
    aopc_scores, lower_scores, upper_scores = [], [], []
    rng = np.random.default_rng(42)

    for text, attrib in zip(texts, attributions):
        tokens = [t for t, w in attrib if w > 0][:k_max]
        attr_ordering = [t for t, _ in attrib if t in tokens]
        aopc_scores.append(_compute_aopc_for_ordering(model, text, attr_ordering, mask_token))

        random_total = 0.0
        for _ in range(n_random_samples):
            ordering = list(tokens)
            rng.shuffle(ordering)
            random_total += _compute_aopc_for_ordering(model, text, ordering, mask_token)
        lower_scores.append(random_total / n_random_samples)

        upper_scores.append(_beam_search_ordering(model, text, tokens, beam_size, mask_token))

    aopc_mean, lower_mean, upper_mean = np.mean(aopc_scores), np.mean(lower_scores), np.mean(upper_scores)
    naopc = (aopc_mean - lower_mean) / (upper_mean - lower_mean)

    return {
        "naopc": float(np.clip(naopc, 0.0, 1.0)),
        "aopc_lower": float(lower_mean),
        "aopc_upper": float(upper_mean),
    }


def compute_filler_comprehensiveness(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: list[int] = [1, 5, 10, 20],
    seed: int = 42,
) -> dict[int, float]:
    """Comprehensiveness using filler tokens instead of [MASK] (arXiv:2502.18848).

    Replaces top-k tokens with neutral filler words to avoid OOD artifacts
    that inflate mask-based metrics (Goodhart's Law, arXiv:2308.14272).
    """
    rng = random.Random(seed)
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    for text, attrib, orig_proba in zip(texts, attributions, original_probas):
        pred_class = int(np.argmax(orig_proba))
        orig_conf = orig_proba[pred_class]
        top_tokens_all = _top_k_tokens(attrib, max(k_values))

        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            normalized = {t.lstrip("#").lower() for t in top_tokens}
            words = text.split()
            filled_words = [
                rng.choice(_FILLER_TOKENS) if w.lower().strip('.,!?;:"\'-') in normalized else w
                for w in words
            ]
            filled_text = ' '.join(filled_words)
            filled_conf = model.predict_proba([filled_text])[0][pred_class]
            results[k].append(orig_conf - filled_conf)

    return {k: float(np.mean(scores)) for k, scores in results.items()}
