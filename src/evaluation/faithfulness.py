"""Faithfulness metrics for interpretability evaluation."""

from collections import Counter
from typing import Protocol

import numpy as np


class UnigramSampler:
    """Sample replacement tokens from unigram distribution (arXiv:2308.14272).

    Preserving the token distribution avoids OOD artifacts when evaluating
    comprehensiveness with filler tokens.
    """

    def __init__(self, texts: list[str], seed: int):
        counts: Counter[str] = Counter()
        for text in texts:
            for word in text.lower().split():
                w = word.strip('.,!?;:"\'-')
                if w:
                    counts[w] += 1
        total = sum(counts.values())
        self.words = list(counts.keys())
        self.probs = np.array([counts[w] / total for w in self.words])
        self.rng = np.random.default_rng(seed)

    def sample(self) -> str:
        return str(self.rng.choice(self.words, p=self.probs))


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


def _compute_masking_metric(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    mode: str,
    beta: float = 1.0,
) -> dict[int, float]:
    """Shared implementation for comprehensiveness and sufficiency (batched)."""
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    # Phase 1: collect all masked texts
    masked_texts = []
    index_map = []  # (text_idx, k)
    for text_idx, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            masked_text = _mask_by_token_set(
                text, top_tokens, mask_token, mode=mode, max_fraction=beta,
            )
            masked_texts.append(masked_text)
            index_map.append((text_idx, k))

    # Phase 2: single batch call
    if masked_texts:
        all_probas = model.predict_proba(masked_texts)
    else:
        all_probas = []

    # Phase 3: distribute results
    for i, (text_idx, k) in enumerate(index_map):
        orig_proba = original_probas[text_idx]
        pred_class = int(np.argmax(orig_proba))
        orig_conf = orig_proba[pred_class]
        masked_conf = all_probas[i][pred_class]
        results[k].append(orig_conf - masked_conf)

    return {k: float(np.mean(scores)) for k, scores in results.items()}


def compute_comprehensiveness(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 1.0,
) -> dict[int, float]:
    """Prediction drop when top-k tokens removed (ERASER/F-Fidelity)."""
    return _compute_masking_metric(model, texts, attributions, k_values, mask_token, "remove", beta)


def compute_sufficiency(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 1.0,
) -> dict[int, float]:
    """How well top-k tokens alone predict output (ERASER/F-Fidelity)."""
    return _compute_masking_metric(model, texts, attributions, k_values, mask_token, "keep", beta)


def compute_monotonicity(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    steps: int,
    mask_token: str,
) -> float:
    """Fraction of removal steps where confidence decreases (batched)."""
    # Phase 1: build all masked texts
    all_masked_texts = []
    text_meta = []  # (text_idx, step_count)

    for text_idx, (text, attrib) in enumerate(zip(texts, attributions)):
        tokens = [t for t, w in attrib if w > 0]
        if not tokens:
            continue

        actual_steps = min(steps, len(tokens))
        step_size = max(1, len(tokens) // actual_steps)
        removed_tokens: set[str] = set()
        step_count = 0

        for i in range(0, len(tokens), step_size):
            for t in tokens[i:i + step_size]:
                removed_tokens.add(t.lower())
            masked_text = _mask_by_token_set(text, removed_tokens, mask_token, mode="remove")
            all_masked_texts.append(masked_text)
            step_count += 1

        text_meta.append((text_idx, step_count))

    if not all_masked_texts:
        return 0.0

    # Phase 2: batch calls for originals and masked texts
    orig_texts = [texts[idx] for idx, _ in text_meta]
    original_probas = model.predict_proba(orig_texts)
    all_probas = model.predict_proba(all_masked_texts)

    # Phase 3: walk sequentially to check monotonicity
    total_monotonic, total_steps = 0, 0
    proba_idx = 0

    for meta_idx, (_, step_count) in enumerate(text_meta):
        orig_proba = original_probas[meta_idx]
        pred_class = int(np.argmax(orig_proba))
        prev_conf = orig_proba[pred_class]
        for _ in range(step_count):
            curr_conf = all_probas[proba_idx][pred_class]
            if curr_conf <= prev_conf:
                total_monotonic += 1
            total_steps += 1
            prev_conf = curr_conf
            proba_idx += 1

    return total_monotonic / total_steps if total_steps > 0 else 0.0


def _compute_aopc_for_ordering(
    model: Predictor, text: str, ordering: list[str], mask_token: str,
) -> float:
    """AOPC for a given token removal ordering (batched)."""
    if not ordering:
        return 0.0

    # Build all incremental-removal texts + original
    masked_texts = []
    removed_tokens: set[str] = set()
    for token in ordering:
        removed_tokens.add(token.lower())
        masked_texts.append(_mask_by_token_set(text, removed_tokens, mask_token, mode="remove"))

    all_texts = [text] + masked_texts
    all_probas = model.predict_proba(all_texts)

    pred_class = int(np.argmax(all_probas[0]))
    orig_conf = all_probas[0][pred_class]
    total_drop = sum(orig_conf - all_probas[i + 1][pred_class] for i in range(len(ordering)))
    return total_drop / len(ordering)


def _beam_search_ordering(
    model: Predictor, text: str, tokens: list[str],
    beam_size: int, mask_token: str,
    maximize: bool = True,
) -> float:
    """Beam search for token removal ordering (batched per step)."""
    orig_proba = model.predict_proba([text])[0]
    pred_class = int(np.argmax(orig_proba))
    orig_conf = orig_proba[pred_class]
    beams: list[tuple[set[str], list[str], float]] = [(set(), [], 0.0)]

    for _ in range(len(tokens)):
        candidate_texts = []
        candidate_meta = []

        for removed_set, ordering, cum_drop in beams:
            for token in tokens:
                if token.lower() in removed_set:
                    continue
                new_removed = removed_set | {token.lower()}
                masked_text = _mask_by_token_set(text, new_removed, mask_token, mode="remove")
                candidate_texts.append(masked_text)
                candidate_meta.append((new_removed, ordering + [token], cum_drop))

        if not candidate_texts:
            break

        # Single batch call per step
        all_probas = model.predict_proba(candidate_texts)

        candidates = []
        for i, (new_removed, new_ordering, cum_drop) in enumerate(candidate_meta):
            masked_conf = all_probas[i][pred_class]
            candidates.append((new_removed, new_ordering, cum_drop + orig_conf - masked_conf))

        candidates.sort(key=lambda x: x[2], reverse=maximize)
        beams = candidates[:beam_size]
        if not beams:
            break

    return beams[0][2] / len(tokens) if beams else 0.0


def compute_normalized_aopc(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_max: int,
    beam_size: int,
    mask_token: str,
) -> dict[str, float]:
    """Normalized AOPC per Eq 8 of arXiv:2408.08137 (ACL 2025).

    Per-example normalization: NAOPC = mean_x[(AOPC_x - lower_x) / (upper_x - lower_x)]
    Both bounds use beam search (maximize=True for upper, maximize=False for lower).
    """
    naopc_scores = []
    aopc_scores, lower_scores, upper_scores = [], [], []

    for text, attrib in zip(texts, attributions):
        # Deduplicate by lowercase (masking operates on lowercased token sets)
        seen: set[str] = set()
        tokens = []
        for t, w in attrib:
            if w > 0 and t.lower() not in seen:
                seen.add(t.lower())
                tokens.append(t)
                if len(tokens) >= k_max:
                    break
        if len(tokens) < 2:
            continue
        token_set = set(tokens)
        attr_ordering = [t for t, _ in attrib if t in token_set]
        aopc_x = _compute_aopc_for_ordering(model, text, attr_ordering, mask_token)

        # Lower bound: beam search for MINIMUM drop ordering
        lower_x = _beam_search_ordering(
            model, text, tokens, beam_size, mask_token, maximize=False,
        )
        # Upper bound: beam search for MAXIMUM drop ordering
        upper_x = _beam_search_ordering(
            model, text, tokens, beam_size, mask_token, maximize=True,
        )

        aopc_scores.append(aopc_x)
        lower_scores.append(lower_x)
        upper_scores.append(upper_x)

        # Per-example normalization (Eq 8)
        denom = upper_x - lower_x
        if denom > 1e-8:
            naopc_scores.append((aopc_x - lower_x) / denom)

    naopc = float(np.mean(naopc_scores)) if naopc_scores else 0.0
    return {
        "naopc": float(np.clip(naopc, 0.0, 1.0)),
        "aopc_lower": float(np.mean(lower_scores)) if lower_scores else 0.0,
        "aopc_upper": float(np.mean(upper_scores)) if upper_scores else 0.0,
    }


def compute_filler_comprehensiveness(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    sampler: UnigramSampler,
) -> dict[int, float]:
    """Comprehensiveness using filler tokens instead of [MASK] (batched).

    Replaces top-k tokens with corpus-sampled filler words to avoid OOD artifacts
    that inflate mask-based metrics (Goodhart's Law, arXiv:2308.14272).
    """
    results = {k: [] for k in k_values}
    original_probas = model.predict_proba(texts)

    # Phase 1: collect all filled texts
    filled_texts = []
    index_map = []  # (text_idx, k)
    for text_idx, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            normalized = {t.lstrip("#").lower() for t in top_tokens}
            words = text.split()
            filled_words = [
                sampler.sample()
                if w.lower().strip('.,!?;:"\'-') in normalized else w
                for w in words
            ]
            filled_texts.append(' '.join(filled_words))
            index_map.append((text_idx, k))

    # Phase 2: single batch call
    if filled_texts:
        all_probas = model.predict_proba(filled_texts)
    else:
        all_probas = []

    # Phase 3: distribute results
    for i, (text_idx, k) in enumerate(index_map):
        orig_proba = original_probas[text_idx]
        pred_class = int(np.argmax(orig_proba))
        orig_conf = orig_proba[pred_class]
        filled_conf = all_probas[i][pred_class]
        results[k].append(orig_conf - filled_conf)

    return {k: float(np.mean(scores)) for k, scores in results.items()}


def _build_word_importance_map(text: str, attrib: list[tuple[str, float]]) -> dict[int, float]:
    """Map word positions in text to attribution importance scores."""
    attrib_dict: dict[str, float] = {}
    for token, weight in attrib:
        key = token.lstrip("#").lower()
        if key not in attrib_dict:
            attrib_dict[key] = abs(weight)

    word_importance: dict[int, float] = {}
    for i, word in enumerate(text.split()):
        clean = word.lower().strip('.,!?;:"\'-')
        if clean in attrib_dict:
            word_importance[i] = attrib_dict[clean]
    return word_importance


def compute_soft_comprehensiveness(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
) -> float:
    """Soft Comprehensiveness via probabilistic token masking (batched, arXiv:2305.10496).

    Each token is masked with probability proportional to its normalized importance.
    Reports average confidence drop across Monte Carlo samples.
    Higher = better explanations.
    """
    rng = np.random.default_rng(seed)
    original_probas = model.predict_proba(texts)

    # Phase 1: generate all masked texts
    all_masked_texts = []
    meta = []  # (text_idx, pred_class)

    for text_idx, (text, attrib, orig_proba) in enumerate(zip(texts, attributions, original_probas)):
        pred_class = int(np.argmax(orig_proba))
        word_importance = _build_word_importance_map(text, attrib)
        words = text.split()
        if not words:
            continue

        importances = np.array([word_importance.get(i, 0.0) for i in range(len(words))])
        max_imp = importances.max()
        if max_imp > 0:
            importances = importances / max_imp

        for _ in range(n_samples):
            mask_flags = rng.random(len(words)) < importances
            masked_words = [
                mask_token if mask_flags[i] else words[i]
                for i in range(len(words))
            ]
            all_masked_texts.append(' '.join(masked_words))
            meta.append((text_idx, pred_class))

    if not all_masked_texts:
        return 0.0

    # Phase 2: single batch call
    all_probas = model.predict_proba(all_masked_texts)

    # Phase 3: aggregate per-text
    scores_by_text: dict[int, list[float]] = {}
    for i, (text_idx, pred_class) in enumerate(meta):
        orig_conf = original_probas[text_idx][pred_class]
        masked_conf = all_probas[i][pred_class]
        if text_idx not in scores_by_text:
            scores_by_text[text_idx] = []
        scores_by_text[text_idx].append(orig_conf - masked_conf)

    scores = [float(np.mean(drops)) for drops in scores_by_text.values()]
    return float(np.mean(scores)) if scores else 0.0


def compute_soft_sufficiency(
    model: Predictor, texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
) -> float:
    """Soft Sufficiency via probabilistic token retention (batched, arXiv:2305.10496).

    Each token is retained with probability proportional to its normalized importance.
    Reports average confidence drop across Monte Carlo samples.
    Lower = better (retaining important tokens should preserve confidence).
    """
    rng = np.random.default_rng(seed)
    original_probas = model.predict_proba(texts)

    # Phase 1: generate all masked texts
    all_masked_texts = []
    meta = []  # (text_idx, pred_class)

    for text_idx, (text, attrib, orig_proba) in enumerate(zip(texts, attributions, original_probas)):
        pred_class = int(np.argmax(orig_proba))
        word_importance = _build_word_importance_map(text, attrib)
        words = text.split()
        if not words:
            continue

        importances = np.array([word_importance.get(i, 0.0) for i in range(len(words))])
        max_imp = importances.max()
        if max_imp > 0:
            importances = importances / max_imp

        for _ in range(n_samples):
            retain_flags = rng.random(len(words)) < importances
            masked_words = [
                words[i] if retain_flags[i] else mask_token
                for i in range(len(words))
            ]
            all_masked_texts.append(' '.join(masked_words))
            meta.append((text_idx, pred_class))

    if not all_masked_texts:
        return 0.0

    # Phase 2: single batch call
    all_probas = model.predict_proba(all_masked_texts)

    # Phase 3: aggregate per-text
    scores_by_text: dict[int, list[float]] = {}
    for i, (text_idx, pred_class) in enumerate(meta):
        orig_conf = original_probas[text_idx][pred_class]
        masked_conf = all_probas[i][pred_class]
        if text_idx not in scores_by_text:
            scores_by_text[text_idx] = []
        scores_by_text[text_idx].append(orig_conf - masked_conf)

    scores = [float(np.mean(drops)) for drops in scores_by_text.values()]
    return float(np.mean(scores)) if scores else 0.0
