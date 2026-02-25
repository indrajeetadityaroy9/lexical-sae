"""Contextual Consistency Score: polysemy defense metric.

Measures whether the model differentiates word senses via context by
computing Jaccard similarity of active sparse sets for the same word
across different label contexts.

Low mean Jaccard = context-sensitive (good): same word activates
different sparse dimensions depending on surrounding context.
High mean Jaccard = bag-of-words (bad): same word always activates
the same dimensions regardless of context.
"""

import re

import torch
from torch.utils.data import DataLoader, TensorDataset

from cajt.runtime import autocast, DEVICE


POLYSEMOUS_WORDS = [
    "right", "light", "well", "bank", "bat",
    "match", "spring", "fair", "crane", "rock",
]


def find_word_occurrences(
    texts: list[str],
    target_words: list[str],
    min_occurrences: int = 5,
) -> dict[str, list[int]]:
    """Find text indices containing each target word.

    Uses regex word boundaries to handle punctuation and capitalization.

    Args:
        texts: List of input texts.
        target_words: Words to search for.
        min_occurrences: Minimum occurrences to include a word.

    Returns:
        {word: [text_indices]} for words with >= min_occurrences.
    """
    result = {}
    for word in target_words:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        indices = [i for i, text in enumerate(texts) if pattern.search(text)]
        if len(indices) >= min_occurrences:
            result[word] = indices
    return result


def compute_active_sets(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    text_indices: list[int],
    max_length: int,
    batch_size: int = 64,
) -> list[set[int]]:
    """Compute active sparse dimension sets for selected texts.

    Args:
        model: LexicalSAE model.
        tokenizer: HuggingFace tokenizer.
        texts: Full text list.
        text_indices: Indices of texts to process.
        max_length: Tokenizer max length.
        batch_size: Inference batch size.

    Returns:
        List of sets of active (non-zero) sparse dimension indices.
    """

    selected_texts = [texts[i] for i in text_indices]

    encoding = tokenizer(
        selected_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    all_active_sets = []
    with torch.inference_mode(), autocast():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            sparse_seq, *_ = model(batch_ids, batch_mask)
            sparse_vector = model.to_pooled(sparse_seq, batch_mask)
            for i in range(sparse_vector.shape[0]):
                active = (sparse_vector[i] > 0).nonzero(as_tuple=True)[0]
                all_active_sets.append(set(active.cpu().tolist()))

    return all_active_sets


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_contextual_consistency_score(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    max_length: int,
    target_words: list[str] | None = None,
    min_occurrences: int = 5,
    batch_size: int = 64,
) -> dict:
    """Compute Contextual Consistency Score across label contexts.

    Splits word occurrences by label (proxy for different semantic contexts),
    computes cross-context pairwise Jaccard similarity of active sparse sets.

    Args:
        model: LexicalSAE model.
        tokenizer: HuggingFace tokenizer.
        texts: Input texts.
        labels: Corresponding labels.
        max_length: Tokenizer max length.
        target_words: Words to evaluate (defaults to POLYSEMOUS_WORDS).
        min_occurrences: Minimum occurrences per word per label group.
        batch_size: Inference batch size.

    Returns:
        {mean_jaccard, per_word: {word: {jaccard, n_pairs, n_occurrences}},
         num_words_evaluated}
    """
    if target_words is None:
        target_words = POLYSEMOUS_WORDS

    occurrences = find_word_occurrences(texts, target_words, min_occurrences=1)

    per_word = {}
    all_jaccards = []

    for word, indices in occurrences.items():
        # Split indices by label
        label_groups = {}
        for idx in indices:
            lbl = labels[idx]
            label_groups.setdefault(lbl, []).append(idx)

        # Need at least 2 label groups with enough occurrences
        valid_groups = {
            lbl: idxs for lbl, idxs in label_groups.items()
            if len(idxs) >= min_occurrences
        }
        if len(valid_groups) < 2:
            continue

        # Compute active sets for all occurrences of this word
        all_indices = []
        index_to_label = {}
        for lbl, idxs in valid_groups.items():
            for idx in idxs:
                index_to_label[len(all_indices)] = lbl
                all_indices.append(idx)

        active_sets = compute_active_sets(
            model, tokenizer, texts, all_indices, max_length, batch_size,
        )

        # Compute cross-context Jaccard (between different label groups)
        cross_jaccards = []
        for i in range(len(active_sets)):
            for j in range(i + 1, len(active_sets)):
                if index_to_label[i] != index_to_label[j]:
                    j_sim = jaccard_similarity(active_sets[i], active_sets[j])
                    cross_jaccards.append(j_sim)

        if cross_jaccards:
            mean_j = sum(cross_jaccards) / len(cross_jaccards)
            per_word[word] = {
                "jaccard": mean_j,
                "n_pairs": len(cross_jaccards),
                "n_occurrences": len(all_indices),
            }
            all_jaccards.extend(cross_jaccards)

    mean_jaccard = sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0.0

    return {
        "mean_jaccard": mean_jaccard,
        "per_word": per_word,
        "num_words_evaluated": len(per_word),
    }
