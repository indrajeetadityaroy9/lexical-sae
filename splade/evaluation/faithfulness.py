"""Faithfulness metrics for token attribution methods."""

from collections import Counter
from typing import Protocol

import numpy
import torch


class UnigramSampler:
    def __init__(self, texts: list[str], seed: int):
        counts: Counter[str] = Counter()
        for text in texts:
            for word in text.lower().split():
                normalized_word = word.strip('.,!?;:"\'-')
                if normalized_word:
                    counts[normalized_word] += 1
        total = sum(counts.values())
        self.words = list(counts.keys())
        self.probs = numpy.array([counts[word] / total for word in self.words])
        self.rng = numpy.random.default_rng(seed)

    def sample(self) -> str:
        return str(self.rng.choice(self.words, p=self.probs))


class Predictor(Protocol):
    def predict_proba(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingPredictor(Protocol):
    """Extended predictor with embedding-level access for soft metrics."""
    def predict_proba(self, texts: list[str]) -> list[list[float]]: ...
    def get_embeddings(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]: ...
    def predict_proba_from_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[list[float]]: ...


def _top_k_tokens(attrib: list[tuple[str, float]], k: int) -> set[str]:
    seen: set[str] = set()
    result: set[str] = set()
    for token, weight in attrib:
        if weight <= 0:
            continue
        lowered = token.lower().strip('.,!?;:"\'-')
        if lowered not in seen:
            seen.add(lowered)
            result.add(lowered)
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
    normalized_set = {token.strip('.,!?;:"\'-').lower() for token in token_set}
    words = text.split()
    max_masks = int(len(words) * max_fraction) if max_fraction < 1.0 else len(words)

    if mode == "remove":
        mask_positions = [
            index
            for index, word in enumerate(words)
            if word.lower().strip('.,!?;:"\'-') in normalized_set
        ][:max_masks]
    else:
        mask_positions = [
            index
            for index, word in enumerate(words)
            if word.lower().strip('.,!?;:"\'-') not in normalized_set
        ][:max_masks]

    masked_words = list(words)
    for position in mask_positions:
        masked_words[position] = mask_token
    return " ".join(masked_words)


def _mask_by_attribution_budget(
    text: str,
    attrib: list[tuple[str, float]],
    mask_token: str,
    mode: str,
    beta: float,
) -> str:
    positive_attribs = [(token, weight) for token, weight in attrib if weight > 0]
    total_mass = sum(weight for _, weight in positive_attribs)
    if total_mass <= 0:
        return text

    budget = beta * total_mass
    cumulative = 0.0
    selected_tokens: set[str] = set()
    for token, weight in positive_attribs:
        if cumulative >= budget:
            break
        selected_tokens.add(token.strip('.,!?;:"\'-').lower())
        cumulative += weight

    return _mask_by_token_set(text, selected_tokens, mask_token, mode=mode)


def _compute_masking_metric(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    mode: str,
    beta: float = 1.0,
    beta_mode: str = "token_fraction",
) -> dict[int, float]:
    results = {k: [] for k in k_values}
    original_probabilities = model.predict_proba(texts)

    masked_texts = []
    index_map = []
    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            if beta_mode == "attribution_mass":
                masked_text = _mask_by_attribution_budget(text, attrib, mask_token, mode, beta)
            else:
                top_tokens = _top_k_tokens(attrib, k)
                masked_text = _mask_by_token_set(
                    text,
                    top_tokens,
                    mask_token,
                    mode=mode,
                    max_fraction=beta,
                )
            masked_texts.append(masked_text)
            index_map.append((text_index, k))

    all_probabilities = model.predict_proba(masked_texts) if masked_texts else []

    for index, (text_index, k) in enumerate(index_map):
        original_probability = original_probabilities[text_index]
        predicted_class = int(numpy.argmax(original_probability))
        original_confidence = original_probability[predicted_class]
        masked_confidence = all_probabilities[index][predicted_class]
        results[k].append(original_confidence - masked_confidence)

    return {k: float(numpy.mean(scores)) for k, scores in results.items()}


def compute_comprehensiveness(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 1.0,
    beta_mode: str = "token_fraction",
) -> dict[int, float]:
    return _compute_masking_metric(model, texts, attributions, k_values, mask_token, "remove", beta, beta_mode)


def compute_sufficiency(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 1.0,
    beta_mode: str = "token_fraction",
) -> dict[int, float]:
    return _compute_masking_metric(model, texts, attributions, k_values, mask_token, "keep", beta, beta_mode)


def compute_monotonicity(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    steps: int,
    mask_token: str,
) -> float:
    all_masked_texts = []
    text_meta = []

    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        tokens = [token for token, weight in attrib if weight > 0]
        if not tokens:
            continue

        actual_steps = min(steps, len(tokens))
        step_size = max(1, len(tokens) // actual_steps)
        removed_tokens: set[str] = set()
        step_count = 0

        for index in range(0, len(tokens), step_size):
            for token in tokens[index : index + step_size]:
                removed_tokens.add(token.lower())
            masked_text = _mask_by_token_set(text, removed_tokens, mask_token, mode="remove")
            all_masked_texts.append(masked_text)
            step_count += 1

        text_meta.append((text_index, step_count))

    if not all_masked_texts:
        return 0.0

    original_texts = [texts[index] for index, _ in text_meta]
    original_probabilities = model.predict_proba(original_texts)
    all_probabilities = model.predict_proba(all_masked_texts)

    total_monotonic = 0
    total_steps = 0
    probability_index = 0

    for meta_index, (_, step_count) in enumerate(text_meta):
        original_probability = original_probabilities[meta_index]
        predicted_class = int(numpy.argmax(original_probability))
        previous_confidence = original_probability[predicted_class]
        for _ in range(step_count):
            current_confidence = all_probabilities[probability_index][predicted_class]
            if current_confidence <= previous_confidence:
                total_monotonic += 1
            total_steps += 1
            previous_confidence = current_confidence
            probability_index += 1

    return total_monotonic / total_steps if total_steps > 0 else 0.0


def _compute_aopc_for_ordering(
    model: Predictor,
    text: str,
    ordering: list[str],
    mask_token: str,
) -> float:
    if not ordering:
        return 0.0

    masked_texts = []
    removed_tokens: set[str] = set()
    for token in ordering:
        removed_tokens.add(token.lower())
        masked_texts.append(_mask_by_token_set(text, removed_tokens, mask_token, mode="remove"))

    all_texts = [text] + masked_texts
    all_probabilities = model.predict_proba(all_texts)

    predicted_class = int(numpy.argmax(all_probabilities[0]))
    original_confidence = all_probabilities[0][predicted_class]
    total_drop = sum(
        original_confidence - all_probabilities[index + 1][predicted_class]
        for index in range(len(ordering))
    )
    return total_drop / (len(ordering) + 1)


def _beam_search_ordering(
    model: Predictor,
    text: str,
    tokens: list[str],
    beam_size: int,
    mask_token: str,
    maximize: bool = True,
) -> float:
    original_probability = model.predict_proba([text])[0]
    predicted_class = int(numpy.argmax(original_probability))
    original_confidence = original_probability[predicted_class]
    beams: list[tuple[set[str], list[str], float]] = [(set(), [], 0.0)]

    for _ in range(len(tokens)):
        candidate_texts = []
        candidate_meta = []

        for removed_set, ordering, cumulative_drop in beams:
            for token in tokens:
                if token.lower() in removed_set:
                    continue
                new_removed = removed_set | {token.lower()}
                masked_text = _mask_by_token_set(text, new_removed, mask_token, mode="remove")
                candidate_texts.append(masked_text)
                candidate_meta.append((new_removed, ordering + [token], cumulative_drop))

        if not candidate_texts:
            break

        all_probabilities = model.predict_proba(candidate_texts)
        candidates = []
        for index, (new_removed, new_ordering, cumulative_drop) in enumerate(candidate_meta):
            masked_confidence = all_probabilities[index][predicted_class]
            candidates.append((new_removed, new_ordering, cumulative_drop + original_confidence - masked_confidence))

        candidates.sort(key=lambda candidate: candidate[2], reverse=maximize)
        beams = candidates[:beam_size]
        if not beams:
            break

    return beams[0][2] / (len(tokens) + 1) if beams else 0.0


def compute_normalized_aopc(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_max: int,
    beam_size: int,
    mask_token: str,
) -> dict[str, float]:
    """Compute normalized AOPC from per-example beam-search bounds."""
    naopc_scores = []
    aopc_scores = []
    lower_scores = []
    upper_scores = []

    for text, attrib in zip(texts, attributions):
        seen: set[str] = set()
        tokens = []
        for token, weight in attrib:
            if weight > 0 and token.lower() not in seen:
                seen.add(token.lower())
                tokens.append(token)
                if len(tokens) >= k_max:
                    break
        if len(tokens) < 2:
            continue

        token_set = set(tokens)
        attr_ordering = [token for token, _ in attrib if token in token_set]
        aopc_x = _compute_aopc_for_ordering(model, text, attr_ordering, mask_token)
        lower_x = _beam_search_ordering(model, text, tokens, beam_size, mask_token, maximize=False)
        upper_x = _beam_search_ordering(model, text, tokens, beam_size, mask_token, maximize=True)

        aopc_scores.append(aopc_x)
        lower_scores.append(lower_x)
        upper_scores.append(upper_x)

        denom = upper_x - lower_x
        if denom > 1e-8:
            naopc_scores.append((aopc_x - lower_x) / denom)

    naopc = float(numpy.mean(naopc_scores)) if naopc_scores else 0.0
    return {
        "naopc": float(numpy.clip(naopc, 0.0, 1.0)),
        "aopc_lower": float(numpy.mean(lower_scores)) if lower_scores else 0.0,
        "aopc_upper": float(numpy.mean(upper_scores)) if upper_scores else 0.0,
    }


def compute_filler_comprehensiveness(
    model: Predictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    sampler: UnigramSampler,
) -> dict[int, float]:
    """Comprehensiveness with corpus-sampled filler replacements."""
    results = {k: [] for k in k_values}
    original_probabilities = model.predict_proba(texts)

    filled_texts = []
    index_map = []
    for text_index, (text, attrib) in enumerate(zip(texts, attributions)):
        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            normalized = {token.strip('.,!?;:"\'-').lower() for token in top_tokens}
            words = text.split()
            filled_words = [
                sampler.sample() if word.lower().strip('.,!?;:"\'-') in normalized else word
                for word in words
            ]
            filled_texts.append(" ".join(filled_words))
            index_map.append((text_index, k))

    all_probabilities = model.predict_proba(filled_texts) if filled_texts else []

    for index, (text_index, k) in enumerate(index_map):
        original_probability = original_probabilities[text_index]
        predicted_class = int(numpy.argmax(original_probability))
        original_confidence = original_probability[predicted_class]
        filled_confidence = all_probabilities[index][predicted_class]
        results[k].append(original_confidence - filled_confidence)

    return {k: float(numpy.mean(scores)) for k, scores in results.items()}


def _clean_subword(token: str) -> str:
    """Strip subword markers from any tokenizer family."""
    if token.startswith("##"):
        return token[2:]
    if token.startswith("\u2581"):
        return token[1:]
    if token.startswith("\u0120"):
        return token[1:]
    return token


def _build_word_importance_map(text: str, attrib: list[tuple[str, float]]) -> dict[int, float]:
    attrib_dict: dict[str, float] = {}
    for token, weight in attrib:
        key = _clean_subword(token).lower()
        if key not in attrib_dict:
            attrib_dict[key] = abs(weight)

    word_importance: dict[int, float] = {}
    for index, word in enumerate(text.split()):
        clean = word.lower().strip('.,!?;:"\'-')
        if clean in attrib_dict:
            word_importance[index] = attrib_dict[clean]
    return word_importance


def _build_token_importances(
    text: str,
    attrib: list[tuple[str, float]],
    tokenizer,
    max_length: int,
) -> numpy.ndarray:
    """Map word-level attributions to subword-token-level importances.

    Returns an array of shape (max_length,) with normalized importances
    aligned to the tokenizer's subword positions. Special tokens get 0.
    """
    word_importance = _build_word_importance_map(text, attrib)
    words = text.split()

    # Tokenize with offset mapping to find subword → word alignment
    encoding = tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    # Build character-position → word-index map
    char_to_word: dict[int, int] = {}
    pos = 0
    for word_idx, word in enumerate(words):
        start = text.find(word, pos)
        if start == -1:
            continue
        for c in range(start, start + len(word)):
            char_to_word[c] = word_idx
        pos = start + len(word)

    importances = numpy.zeros(len(input_ids), dtype=numpy.float64)
    for tok_idx, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue  # special token
        mid = (start + end) // 2
        word_idx = char_to_word.get(mid)
        if word_idx is not None and word_idx in word_importance:
            importances[tok_idx] = word_importance[word_idx]

    max_imp = importances.max()
    if max_imp > 0:
        importances = importances / max_imp
    return importances


def compute_soft_comprehensiveness(
    model: EmbeddingPredictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 128,
) -> float:
    """Embedding-level Bernoulli dropout comprehensiveness (arXiv:2305.10496, Eq. 4).

    For each token position i, each embedding dimension is independently zeroed
    with probability a_i (the normalized attribution), then re-scored. The baseline
    is a zero-tensor forward pass (empty rationale).
    """
    rng = numpy.random.default_rng(seed)
    original_probs = model.predict_proba(texts)

    # Get embeddings for all texts at once
    embeddings, attention_masks = model.get_embeddings(texts)
    embed_dim = embeddings.shape[-1]

    # Zero-tensor baseline (arXiv:2305.10496, §3.2)
    zero_emb = torch.zeros_like(embeddings)
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks)

    scores = []
    for text_idx in range(len(texts)):
        if tokenizer is not None:
            token_importances = _build_token_importances(
                texts[text_idx], attributions[text_idx], tokenizer, max_length
            )
        else:
            # Fallback: uniform zero (no attribution info at token level)
            seq_len = embeddings.shape[1]
            token_importances = numpy.zeros(seq_len)

        predicted_class = int(numpy.argmax(original_probs[text_idx]))
        orig_conf = original_probs[text_idx][predicted_class]
        base_conf = baseline_probs[text_idx][predicted_class]

        drops = []
        emb_i = embeddings[text_idx]  # [L, H]
        mask_i = attention_masks[text_idx].unsqueeze(0)  # [1, L]

        for _ in range(n_samples):
            # Per-dimension Bernoulli: zero out dimension j of token i with P=a_i
            keep_mask = torch.ones_like(emb_i)
            for tok_idx in range(min(len(token_importances), emb_i.shape[0])):
                imp = token_importances[tok_idx]
                if imp > 0:
                    dim_keep = torch.from_numpy(
                        (rng.random(embed_dim) >= imp).astype(numpy.float32)
                    ).to(emb_i.device)
                    keep_mask[tok_idx] = dim_keep

            perturbed = emb_i * keep_mask
            pert_probs = model.predict_proba_from_embeddings(
                perturbed.unsqueeze(0), mask_i
            )
            drops.append(orig_conf - pert_probs[0][predicted_class])

        # Normalize by baseline (Eq. 4)
        raw_comp = float(numpy.mean(drops))
        baseline_suff = 1.0 - max(0.0, orig_conf - base_conf)
        denom = 1.0 - baseline_suff
        scores.append(raw_comp / denom if denom > 1e-8 else 0.0)

    return float(numpy.mean(scores)) if scores else 0.0


def compute_soft_sufficiency(
    model: EmbeddingPredictor,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 128,
) -> float:
    """Embedding-level Bernoulli retention sufficiency (arXiv:2305.10496, Eq. 5).

    For each token position i, each embedding dimension is retained with
    probability a_i and zeroed with probability (1 - a_i).
    """
    rng = numpy.random.default_rng(seed)
    original_probs = model.predict_proba(texts)

    embeddings, attention_masks = model.get_embeddings(texts)
    embed_dim = embeddings.shape[-1]

    # Zero-tensor baseline
    zero_emb = torch.zeros_like(embeddings)
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks)

    scores = []
    for text_idx in range(len(texts)):
        if tokenizer is not None:
            token_importances = _build_token_importances(
                texts[text_idx], attributions[text_idx], tokenizer, max_length
            )
        else:
            seq_len = embeddings.shape[1]
            token_importances = numpy.zeros(seq_len)

        predicted_class = int(numpy.argmax(original_probs[text_idx]))
        orig_conf = original_probs[text_idx][predicted_class]
        base_conf = baseline_probs[text_idx][predicted_class]

        drops = []
        emb_i = embeddings[text_idx]
        mask_i = attention_masks[text_idx].unsqueeze(0)

        for _ in range(n_samples):
            # Retain dimension j of token i with P=a_i, zero with P=(1-a_i)
            retain_mask = torch.zeros_like(emb_i)
            for tok_idx in range(min(len(token_importances), emb_i.shape[0])):
                imp = token_importances[tok_idx]
                if imp > 0:
                    dim_retain = torch.from_numpy(
                        (rng.random(embed_dim) < imp).astype(numpy.float32)
                    ).to(emb_i.device)
                    retain_mask[tok_idx] = dim_retain

            perturbed = emb_i * retain_mask
            pert_probs = model.predict_proba_from_embeddings(
                perturbed.unsqueeze(0), mask_i
            )
            drops.append(orig_conf - pert_probs[0][predicted_class])

        # Normalize by baseline (Eq. 5)
        raw_suff = 1.0 - max(0.0, float(numpy.mean(drops)))
        baseline_suff = 1.0 - max(0.0, orig_conf - base_conf)
        denom = 1.0 - baseline_suff
        scores.append((raw_suff - baseline_suff) / denom if denom > 1e-8 else 0.0)

    return float(numpy.mean(scores)) if scores else 0.0
