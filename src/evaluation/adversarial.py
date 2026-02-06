"""Adversarial sensitivity metrics (arXiv:2409.17774).

Supports multiple attack strategies: WordNet synonyms, TextFooler-style
importance-based replacement, and character-level perturbations.
Uses generalized Kendall tau-hat for incomplete rankings.
"""

import random as _random_module
import string
from functools import lru_cache
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from nltk.corpus import wordnet as wn

from src.evaluation.faithfulness import Predictor


# ---------------------------------------------------------------------------
# Attack strategies
# ---------------------------------------------------------------------------

@runtime_checkable
class AttackStrategy(Protocol):
    def attack(self, text: str, rng: _random_module.Random) -> str: ...


class WordNetAttack:
    """Synonym substitution via WordNet."""

    def __init__(self, max_changes: int = 3):
        self.max_changes = max_changes

    def attack(self, text: str, rng: _random_module.Random) -> str:
        words = text.split()
        modified_words, change_count = list(words), 0
        indices = list(range(len(words)))
        rng.shuffle(indices)
        for idx in indices:
            if change_count >= self.max_changes:
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


class TextFoolerAttack:
    """TextFooler-style importance-based word replacement (Jin et al. 2020).

    Ranks words by leave-one-out confidence drop, then replaces the most
    important words with WordNet synonyms (approximating counter-fitted embeddings).
    """

    def __init__(self, model: Predictor, max_changes: int = 3):
        self.model = model
        self.max_changes = max_changes

    def attack(self, text: str, rng: _random_module.Random) -> str:
        words = text.split()
        if not words:
            return text

        orig_proba = self.model.predict_proba([text])[0]
        pred_class = int(np.argmax(orig_proba))
        orig_conf = orig_proba[pred_class]

        # Rank words by importance (leave-one-out confidence drop)
        importance = []
        for i in range(len(words)):
            reduced = ' '.join(words[:i] + words[i+1:])
            if not reduced.strip():
                importance.append((i, 0.0))
                continue
            reduced_conf = self.model.predict_proba([reduced])[0][pred_class]
            importance.append((i, orig_conf - reduced_conf))
        importance.sort(key=lambda x: x[1], reverse=True)

        modified_words = list(words)
        change_count = 0
        for idx, _ in importance:
            if change_count >= self.max_changes:
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


class CharacterAttack:
    """Character-level perturbations: swap, insert, delete."""

    def __init__(self, max_changes: int = 3):
        self.max_changes = max_changes

    def attack(self, text: str, rng: _random_module.Random) -> str:
        words = text.split()
        if not words:
            return text

        # Pick random words to perturb
        eligible = [i for i, w in enumerate(words) if len(w) > 2]
        if not eligible:
            return text

        modified_words = list(words)
        indices = rng.sample(eligible, min(self.max_changes, len(eligible)))

        for idx in indices:
            word = modified_words[idx]
            op = rng.choice(['swap', 'insert', 'delete'])
            chars = list(word)
            if op == 'swap' and len(chars) > 1:
                i = rng.randint(0, len(chars) - 2)
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif op == 'insert':
                i = rng.randint(0, len(chars))
                chars.insert(i, rng.choice(string.ascii_lowercase))
            elif op == 'delete' and len(chars) > 1:
                i = rng.randint(0, len(chars) - 1)
                chars.pop(i)
            modified_words[idx] = ''.join(chars)

        return ' '.join(modified_words)


# ---------------------------------------------------------------------------
# Kendall tau-hat for incomplete rankings
# ---------------------------------------------------------------------------

def _kendall_tau_hat(
    ranking_a: list[tuple[str, float]],
    ranking_b: list[tuple[str, float]],
) -> float:
    """Generalized Kendall tau for incomplete rankings (tau-hat_x).

    Handles disjoint elements by treating missing items as tied at the end
    of the ranking they're absent from.
    """
    all_items = list(set(t.lower() for t, _ in ranking_a) | set(t.lower() for t, _ in ranking_b))
    n = len(all_items)
    if n < 2:
        return 0.0

    rank_a = {t.lower(): i for i, (t, _) in enumerate(ranking_a)}
    rank_b = {t.lower(): i for i, (t, _) in enumerate(ranking_b)}
    default_a = len(ranking_a)
    default_b = len(ranking_b)

    concordant, discordant = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            ra_i = rank_a.get(all_items[i], default_a)
            ra_j = rank_a.get(all_items[j], default_a)
            rb_i = rank_b.get(all_items[i], default_b)
            rb_j = rank_b.get(all_items[j], default_b)
            diff_a = ra_i - ra_j
            diff_b = rb_i - rb_j
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main metric
# ---------------------------------------------------------------------------

def compute_adversarial_sensitivity(
    model: Predictor,
    explainer_fn: Callable[[str, int], list[tuple[str, float]]],
    texts: list[str],
    attacks: list[AttackStrategy] | None,
    max_changes: int,
    mcp_threshold: float,
    top_k: int,
    seed: int,
) -> dict:
    """Measure how explanation rankings change under adversarial attack (arXiv:2409.17774).

    Supports multiple attack strategies and uses generalized Kendall tau-hat
    for incomplete rankings.
    """
    if attacks is None:
        attacks = [WordNetAttack(max_changes)]

    rng = _random_module.Random(seed)
    original_probas = model.predict_proba(texts)

    # Filter to high-confidence examples
    filtered = []
    for text, orig_proba in zip(texts, original_probas):
        if max(orig_proba) >= mcp_threshold:
            filtered.append(text)

    all_tau_scores = []
    per_attack_results = {}

    for attack in attacks:
        attack_name = type(attack).__name__
        tau_scores = []

        for text in filtered:
            adv_text = attack.attack(text, rng)
            exp_orig = explainer_fn(text, top_k)
            exp_adv = explainer_fn(adv_text, top_k)

            tau = _kendall_tau_hat(exp_orig, exp_adv)
            tau_scores.append(tau)

        all_tau_scores.extend(tau_scores)
        if tau_scores:
            per_attack_results[attack_name] = {
                "mean_tau": float(np.mean(tau_scores)),
                "sensitivity": float(np.mean([1 - (t + 1) / 2 for t in tau_scores])),
            }

    if not all_tau_scores:
        return {"adversarial_sensitivity": 0.0, "mean_tau": 0.0, "per_attack": {}}

    sensitivity_scores = [1 - (tau + 1) / 2 for tau in all_tau_scores]
    return {
        "adversarial_sensitivity": float(np.mean(sensitivity_scores)),
        "mean_tau": float(np.mean(all_tau_scores)),
        "per_attack": per_attack_results,
    }
