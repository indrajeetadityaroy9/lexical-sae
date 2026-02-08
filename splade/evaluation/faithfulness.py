from collections import Counter

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
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    mode: str,
    beta: float = 1.0,
    beta_mode: str = "token_fraction",
    original_probs: list[list[float]],
) -> dict[int, float]:
    results = {k: [] for k in k_values}
    original_probabilities = original_probs

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
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 1.0,
    beta_mode: str = "token_fraction",
    original_probs: list[list[float]],
) -> dict[int, float]:
    return _compute_masking_metric(model, texts, attributions, k_values, mask_token, "remove", beta, beta_mode, original_probs=original_probs)


def compute_sufficiency(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 1.0,
    beta_mode: str = "token_fraction",
    original_probs: list[list[float]],
) -> dict[int, float]:
    return _compute_masking_metric(model, texts, attributions, k_values, mask_token, "keep", beta, beta_mode, original_probs=original_probs)


def compute_monotonicity(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    steps: int,
    mask_token: str,
    original_probs: list[list[float]],
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

    original_probabilities = [original_probs[index] for index, _ in text_meta]
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
    model,
    text: str,
    ordering: list[str],
    mask_token: str,
    original_prob: list[float],
) -> float:
    if not ordering:
        return 0.0

    masked_texts = []
    removed_tokens: set[str] = set()
    for token in ordering:
        removed_tokens.add(token.lower())
        masked_texts.append(_mask_by_token_set(text, removed_tokens, mask_token, mode="remove"))

    all_probabilities = [original_prob] + model.predict_proba(masked_texts)

    predicted_class = int(numpy.argmax(all_probabilities[0]))
    original_confidence = all_probabilities[0][predicted_class]
    total_drop = sum(
        original_confidence - all_probabilities[index + 1][predicted_class]
        for index in range(len(ordering))
    )
    return total_drop / len(ordering)


def _beam_search_ordering(
    model,
    text: str,
    tokens: list[str],
    beam_size: int,
    mask_token: str,
    maximize: bool = True,
    original_prob: list[float],
) -> float:
    original_probability = original_prob
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

    return beams[0][2] / len(tokens) if beams else 0.0


def compute_normalized_aopc(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_max: int,
    beam_size: int,
    mask_token: str,
    original_probs: list[list[float]],
) -> dict[str, float]:
    naopc_scores = []
    aopc_scores = []
    lower_scores = []
    upper_scores = []

    for text_idx, (text, attrib) in enumerate(zip(texts, attributions)):
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

        orig_prob_i = original_probs[text_idx]
        token_set = set(tokens)
        attr_ordering = [token for token, _ in attrib if token in token_set]
        aopc_x = _compute_aopc_for_ordering(model, text, attr_ordering, mask_token, original_prob=orig_prob_i)
        lower_x = _beam_search_ordering(model, text, tokens, beam_size, mask_token, maximize=False, original_prob=orig_prob_i)
        upper_x = _beam_search_ordering(model, text, tokens, beam_size, mask_token, maximize=True, original_prob=orig_prob_i)

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
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    sampler: UnigramSampler,
    original_probs: list[list[float]],
) -> dict[int, float]:
    results = {k: [] for k in k_values}
    original_probabilities = original_probs

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


def _stochastic_mask_text(
    text: str,
    attrib_token_set: set[str],
    mask_token: str,
    mode: str,
    beta: float,
    rng: numpy.random.Generator,
) -> str:
    normalized_set = {t.strip('.,!?;:"\'-').lower() for t in attrib_token_set}
    words = text.split()
    result_words = []
    for word in words:
        clean = word.lower().strip('.,!?;:"\'-')
        is_attributed = clean in normalized_set
        if mode == "remove":
            if is_attributed:
                result_words.append(mask_token)
            elif rng.random() < beta:
                result_words.append(mask_token)
            else:
                result_words.append(word)
        else:
            if is_attributed:
                result_words.append(word)
            elif rng.random() < beta:
                result_words.append(mask_token)
            else:
                result_words.append(word)
    return " ".join(result_words)


def compute_ffidelity_metric(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    mode: str,
    beta: float = 0.1,
    n_samples: int = 50,
    seed: int = 42,
    original_probs: list[list[float]],
) -> dict[int, float]:
    rng = numpy.random.default_rng(seed)
    results = {k: [] for k in k_values}
    original_probabilities = original_probs

    all_masked_texts = []
    index_map = []
    for text_idx, (text, attrib) in enumerate(zip(texts, attributions)):
        original_probability = original_probabilities[text_idx]
        predicted_class = int(numpy.argmax(original_probability))
        original_confidence = original_probability[predicted_class]

        for k in k_values:
            top_tokens = _top_k_tokens(attrib, k)
            for _ in range(n_samples):
                all_masked_texts.append(
                    _stochastic_mask_text(text, top_tokens, mask_token, mode, beta, rng)
                )
            index_map.append((text_idx, k, predicted_class, original_confidence))

    if all_masked_texts:
        all_probs = model.predict_proba(all_masked_texts)
    else:
        all_probs = []

    prob_idx = 0
    for text_idx, k, predicted_class, original_confidence in index_map:
        sample_probs = all_probs[prob_idx : prob_idx + n_samples]
        prob_idx += n_samples
        avg_confidence = float(numpy.mean([p[predicted_class] for p in sample_probs]))
        results[k].append(original_confidence - avg_confidence)

    return {k: float(numpy.mean(scores)) for k, scores in results.items()}


def compute_ffidelity_comprehensiveness(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 0.1,
    n_samples: int = 50,
    seed: int = 42,
    original_probs: list[list[float]],
) -> dict[int, float]:
    return compute_ffidelity_metric(
        model, texts, attributions, k_values, mask_token,
        mode="remove", beta=beta, n_samples=n_samples, seed=seed,
        original_probs=original_probs,
    )


def compute_ffidelity_sufficiency(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    k_values: tuple[int, ...] | list[int],
    mask_token: str,
    beta: float = 0.1,
    n_samples: int = 50,
    seed: int = 42,
    original_probs: list[list[float]],
) -> dict[int, float]:
    return compute_ffidelity_metric(
        model, texts, attributions, k_values, mask_token,
        mode="keep", beta=beta, n_samples=n_samples, seed=seed,
        original_probs=original_probs,
    )


def _clean_subword(token: str) -> str:
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
    word_importance = _build_word_importance_map(text, attrib)
    words = text.split()

    encoding = tokenizer(
        text, max_length=max_length, padding="max_length",
        truncation=True, return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

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
            continue
        mid = (start + end) // 2
        word_idx = char_to_word.get(mid)
        if word_idx is not None and word_idx in word_importance:
            importances[tok_idx] = word_importance[word_idx]

    max_imp = importances.max()
    if max_imp > 1e-12:
        importances = importances / max_imp
    return importances


def compute_soft_comprehensiveness(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 128,
    original_probs: list[list[float]],
) -> float:
    orig_probs = original_probs

    embeddings, attention_masks = model.get_embeddings(texts)
    embed_dim = embeddings.shape[-1]

    zero_emb = torch.zeros_like(embeddings)
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks)

    scores = []
    for text_idx in range(len(texts)):
        token_importances = _build_token_importances(
            texts[text_idx], attributions[text_idx], tokenizer, max_length
        )

        predicted_class = int(numpy.argmax(orig_probs[text_idx]))
        orig_conf = orig_probs[text_idx][predicted_class]
        base_conf = baseline_probs[text_idx][predicted_class]

        emb_i = embeddings[text_idx]
        mask_i = attention_masks[text_idx]
        full_L = emb_i.shape[0]

        seq_len = min(len(token_importances), full_L)
        imp_tensor = torch.tensor(
            token_importances[:seq_len], dtype=torch.float32, device=emb_i.device
        ).unsqueeze(1)

        rand_all = torch.rand(n_samples, seq_len, embed_dim, device=emb_i.device)
        keep_masks = (rand_all >= imp_tensor.unsqueeze(0)).float()

        if seq_len < full_L:
            pad = torch.ones(n_samples, full_L - seq_len, embed_dim, device=emb_i.device)
            keep_masks = torch.cat([keep_masks, pad], dim=1)

        perturbed_batch = emb_i.unsqueeze(0) * keep_masks
        mask_batch = mask_i.unsqueeze(0).expand(n_samples, -1)

        pert_probs_t = model.predict_proba_from_embeddings_tensor(perturbed_batch, mask_batch)
        mean_drop = float((orig_conf - pert_probs_t[:, predicted_class]).mean().item())

        raw_comp = max(0.0, mean_drop)
        baseline_suff = 1.0 - max(0.0, orig_conf - base_conf)
        denom = 1.0 - baseline_suff
        scores.append(raw_comp / denom if denom > 1e-8 else 0.0)

    return float(numpy.mean(scores)) if scores else 0.0


def compute_soft_sufficiency(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 128,
    original_probs: list[list[float]],
) -> float:
    orig_probs = original_probs

    embeddings, attention_masks = model.get_embeddings(texts)
    embed_dim = embeddings.shape[-1]

    zero_emb = torch.zeros_like(embeddings)
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks)

    scores = []
    for text_idx in range(len(texts)):
        token_importances = _build_token_importances(
            texts[text_idx], attributions[text_idx], tokenizer, max_length
        )

        predicted_class = int(numpy.argmax(orig_probs[text_idx]))
        orig_conf = orig_probs[text_idx][predicted_class]
        base_conf = baseline_probs[text_idx][predicted_class]

        emb_i = embeddings[text_idx]
        mask_i = attention_masks[text_idx]
        full_L = emb_i.shape[0]

        seq_len = min(len(token_importances), full_L)
        imp_tensor = torch.tensor(
            token_importances[:seq_len], dtype=torch.float32, device=emb_i.device
        ).unsqueeze(1)

        rand_all = torch.rand(n_samples, seq_len, embed_dim, device=emb_i.device)
        retain_masks = (rand_all < imp_tensor.unsqueeze(0)).float()

        if seq_len < full_L:
            pad = torch.zeros(n_samples, full_L - seq_len, embed_dim, device=emb_i.device)
            retain_masks = torch.cat([retain_masks, pad], dim=1)

        perturbed_batch = emb_i.unsqueeze(0) * retain_masks
        mask_batch = mask_i.unsqueeze(0).expand(n_samples, -1)

        pert_probs_t = model.predict_proba_from_embeddings_tensor(perturbed_batch, mask_batch)
        mean_drop = float((orig_conf - pert_probs_t[:, predicted_class]).mean().item())

        raw_suff = 1.0 - max(0.0, mean_drop)
        baseline_suff = 1.0 - max(0.0, orig_conf - base_conf)
        denom = 1.0 - baseline_suff
        scores.append(max(0.0, raw_suff - baseline_suff) / denom if denom > 1e-8 else 0.0)

    return float(numpy.mean(scores)) if scores else 0.0


def compute_soft_metrics(
    model,
    texts: list[str],
    attributions: list[list[tuple[str, float]]],
    mask_token: str,
    n_samples: int = 20,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 128,
    original_probs: list[list[float]],
) -> tuple[float, float]:
    orig_probs = original_probs

    embeddings, attention_masks = model.get_embeddings(texts)
    embed_dim = embeddings.shape[-1]
    zero_emb = torch.zeros_like(embeddings)
    baseline_probs = model.predict_proba_from_embeddings(zero_emb, attention_masks)

    comp_scores = []
    suff_scores = []
    for text_idx in range(len(texts)):
        token_importances = _build_token_importances(
            texts[text_idx], attributions[text_idx], tokenizer, max_length
        )

        predicted_class = int(numpy.argmax(orig_probs[text_idx]))
        orig_conf = orig_probs[text_idx][predicted_class]
        base_conf = baseline_probs[text_idx][predicted_class]

        emb_i = embeddings[text_idx]
        mask_i = attention_masks[text_idx]
        full_L = emb_i.shape[0]

        seq_len = min(len(token_importances), full_L)
        imp_tensor = torch.tensor(
            token_importances[:seq_len], dtype=torch.float32, device=emb_i.device
        ).unsqueeze(1)

        rand_comp = torch.rand(n_samples, seq_len, embed_dim, device=emb_i.device)
        keep_masks = (rand_comp >= imp_tensor.unsqueeze(0)).float()
        if seq_len < full_L:
            keep_masks = torch.cat([keep_masks, torch.ones(n_samples, full_L - seq_len, embed_dim, device=emb_i.device)], dim=1)
        pert_comp = emb_i.unsqueeze(0) * keep_masks
        mask_batch = mask_i.unsqueeze(0).expand(n_samples, -1)

        rand_suff = torch.rand(n_samples, seq_len, embed_dim, device=emb_i.device)
        retain_masks = (rand_suff < imp_tensor.unsqueeze(0)).float()
        if seq_len < full_L:
            retain_masks = torch.cat([retain_masks, torch.zeros(n_samples, full_L - seq_len, embed_dim, device=emb_i.device)], dim=1)
        pert_suff = emb_i.unsqueeze(0) * retain_masks

        comp_probs_t = model.predict_proba_from_embeddings_tensor(pert_comp, mask_batch)
        comp_drop = float((orig_conf - comp_probs_t[:, predicted_class]).mean().item())
        suff_probs_t = model.predict_proba_from_embeddings_tensor(pert_suff, mask_batch)
        suff_drop = float((orig_conf - suff_probs_t[:, predicted_class]).mean().item())

        raw_comp = max(0.0, comp_drop)
        baseline_suff = 1.0 - max(0.0, orig_conf - base_conf)
        denom = 1.0 - baseline_suff
        comp_scores.append(raw_comp / denom if denom > 1e-8 else 0.0)

        raw_suff = 1.0 - max(0.0, suff_drop)
        suff_scores.append(max(0.0, raw_suff - baseline_suff) / denom if denom > 1e-8 else 0.0)

    soft_comp = float(numpy.mean(comp_scores)) if comp_scores else 0.0
    soft_suff = float(numpy.mean(suff_scores)) if suff_scores else 0.0
    return soft_comp, soft_suff
