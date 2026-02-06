"""Adversarial sensitivity metrics (arXiv:2409.17774)."""

import random as _random_module
from functools import lru_cache
from typing import Callable

import numpy as np
from nltk.corpus import wordnet as wn
from scipy import stats

from src.faithfulness import Predictor


@lru_cache(maxsize=4096)
def _get_wordnet_synonyms(word: str) -> list[str]:
    """Fetch single-word synonyms from WordNet."""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name()
            if "_" not in name and name.lower() != word.lower():
                synonyms.add(name.lower())
    return list(synonyms)[:20]


def _generate_adversarial_text(text: str, max_changes: int = 3, rng: _random_module.Random | None = None) -> str:
    """Generate adversarial text via synonym substitution."""
    if rng is None:
        rng = _random_module.Random(42)
    words = text.split()
    modified_words, change_count = list(words), 0

    indices = list(range(len(words)))
    rng.shuffle(indices)

    for idx in indices:
        if change_count >= max_changes:
            break
        word = words[idx].lower().strip('.,!?;:"\'-')
        syns = _get_wordnet_synonyms(word)
        if syns:
            replacement = rng.choice(syns)
            if words[idx][0].isupper():
                replacement = replacement.capitalize()
            modified_words[idx] = replacement
            change_count += 1

    return ' '.join(w for w in modified_words if w)


def compute_adversarial_sensitivity(
    model: Predictor,
    explainer_fn: Callable[[str, int], list[tuple[str, float]]],
    texts: list[str],
    max_changes: int = 3,
    mcp_threshold: float = 0.7,
    top_k: int = 20,
    seed: int = 42,
) -> dict:
    """Measure how explanation rankings change under adversarial attack (arXiv:2409.17774)."""
    rng = _random_module.Random(seed)
    original_probas = model.predict_proba(texts)

    filtered = []
    for text, orig_proba in zip(texts, original_probas):
        if max(orig_proba) < mcp_threshold:
            continue
        adv_text = _generate_adversarial_text(text, max_changes, rng)
        filtered.append({"original_text": text, "adversarial_text": adv_text})

    tau_scores = []
    for ex in filtered:
        exp_orig = explainer_fn(ex["original_text"], top_k)
        exp_adv = explainer_fn(ex["adversarial_text"], top_k)

        orig_tokens = {t.lower(): i for i, (t, _) in enumerate(exp_orig)}
        adv_tokens = {t.lower(): i for i, (t, _) in enumerate(exp_adv)}
        common = set(orig_tokens) & set(adv_tokens)

        if len(common) < 2:
            tau_scores.append(0.0)
            continue

        tau, _ = stats.kendalltau([orig_tokens[t] for t in common], [adv_tokens[t] for t in common])
        tau_scores.append(0.0 if np.isnan(tau) else float(tau))

    if not tau_scores:
        return {"adversarial_sensitivity": 0.0, "mean_tau": 0.0}

    sensitivity_scores = [1 - (tau + 1) / 2 for tau in tau_scores]
    return {
        "adversarial_sensitivity": float(np.mean(sensitivity_scores)),
        "mean_tau": float(np.mean(tau_scores)),
    }
