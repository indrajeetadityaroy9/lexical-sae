"""Adversarial sensitivity metrics for explanation rankings."""

import random
import string
from functools import lru_cache
from typing import Callable, Protocol, runtime_checkable

import numpy
from nltk.corpus import wordnet

from splade.evaluation.faithfulness import Predictor


@runtime_checkable
class AttackStrategy(Protocol):
    def attack(self, text: str, rng: random.Random) -> str: ...


class WordNetAttack:
    """Synonym substitution via WordNet."""

    def __init__(self, max_changes: int = 3):
        self.max_changes = max_changes

    def attack(self, text: str, rng: random.Random) -> str:
        words = text.split()
        modified_words = list(words)
        change_count = 0
        indices = list(range(len(words)))
        rng.shuffle(indices)
        for index in indices:
            if change_count >= self.max_changes:
                break
            source = words[index].lower().strip('.,!?;:"\'-')
            synonyms = _get_wordnet_synonyms(source)
            if synonyms:
                replacement = rng.choice(synonyms)
                if words[index][0].isupper():
                    replacement = replacement.capitalize()
                modified_words[index] = replacement
                change_count += 1
        return " ".join(word for word in modified_words if word)


class TextFoolerAttack:
    """Leave-one-out ranking followed by synonym replacement."""

    def __init__(self, model: Predictor, max_changes: int = 3):
        self.model = model
        self.max_changes = max_changes

    def attack(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if not words:
            return text

        original_probabilities = self.model.predict_proba([text])[0]
        predicted_class = int(numpy.argmax(original_probabilities))
        original_confidence = original_probabilities[predicted_class]

        # Batch all leave-one-out texts into a single predict_proba call
        reduced_texts = []
        reduced_indices = []
        for index in range(len(words)):
            reduced_text = " ".join(words[:index] + words[index + 1:])
            if not reduced_text.strip():
                continue
            reduced_texts.append(reduced_text)
            reduced_indices.append(index)

        importance = []
        if reduced_texts:
            all_reduced_probs = self.model.predict_proba(reduced_texts)
            ri = 0
            for index in range(len(words)):
                if ri < len(reduced_indices) and reduced_indices[ri] == index:
                    reduced_confidence = all_reduced_probs[ri][predicted_class]
                    importance.append((index, original_confidence - reduced_confidence))
                    ri += 1
                else:
                    importance.append((index, 0.0))
        else:
            importance = [(index, 0.0) for index in range(len(words))]
        importance.sort(key=lambda pair: pair[1], reverse=True)

        modified_words = list(words)
        change_count = 0
        for index, _ in importance:
            if change_count >= self.max_changes:
                break
            source = words[index].lower().strip('.,!?;:"\'-')
            synonyms = _get_wordnet_synonyms(source)
            if synonyms:
                replacement = rng.choice(synonyms)
                if words[index][0].isupper():
                    replacement = replacement.capitalize()
                modified_words[index] = replacement
                change_count += 1

        return " ".join(word for word in modified_words if word)


class CharacterAttack:
    """Character-level perturbations: swap, insert, delete."""

    def __init__(self, max_changes: int = 3):
        self.max_changes = max_changes

    def attack(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if not words:
            return text

        eligible = [index for index, word in enumerate(words) if len(word) > 2]
        if not eligible:
            return text

        modified_words = list(words)
        indices = rng.sample(eligible, min(self.max_changes, len(eligible)))

        for index in indices:
            word = modified_words[index]
            operation = rng.choice(["swap", "insert", "delete"])
            characters = list(word)
            if operation == "swap" and len(characters) > 1:
                swap_index = rng.randint(0, len(characters) - 2)
                characters[swap_index], characters[swap_index + 1] = (
                    characters[swap_index + 1],
                    characters[swap_index],
                )
            elif operation == "insert":
                insert_index = rng.randint(0, len(characters))
                characters.insert(insert_index, rng.choice(string.ascii_lowercase))
            elif operation == "delete" and len(characters) > 1:
                delete_index = rng.randint(0, len(characters) - 1)
                characters.pop(delete_index)
            modified_words[index] = "".join(characters)

        return " ".join(modified_words)


def _kendall_tau_hat(
    ranking_a: list[tuple[str, float]],
    ranking_b: list[tuple[str, float]],
) -> float:
    """Generalized Kendall tau for incomplete rankings."""
    all_items = list({token.lower() for token, _ in ranking_a} | {token.lower() for token, _ in ranking_b})
    item_count = len(all_items)
    if item_count < 2:
        return 0.0

    rank_a = {token.lower(): index for index, (token, _) in enumerate(ranking_a)}
    rank_b = {token.lower(): index for index, (token, _) in enumerate(ranking_b)}
    default_a = len(ranking_a)
    default_b = len(ranking_b)

    concordant = 0
    discordant = 0
    for left in range(item_count):
        for right in range(left + 1, item_count):
            rank_a_left = rank_a.get(all_items[left], default_a)
            rank_a_right = rank_a.get(all_items[right], default_a)
            rank_b_left = rank_b.get(all_items[left], default_b)
            rank_b_right = rank_b.get(all_items[right], default_b)
            diff_a = rank_a_left - rank_a_right
            diff_b = rank_b_left - rank_b_right
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1

    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


@lru_cache(maxsize=4096)
def _get_wordnet_synonyms(word: str) -> list[str]:
    """Fetch single-word synonyms from WordNet."""
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name()
            if "_" not in name and name.lower() != word.lower():
                synonyms.add(name.lower())
    return list(synonyms)[:20]


def compute_adversarial_sensitivity(
    model: Predictor,
    explainer_fn: Callable[[str, int], list[tuple[str, float]]],
    texts: list[str],
    attacks: list[AttackStrategy] | None,
    max_changes: int,
    mcp_threshold: float,
    top_k: int,
    seed: int,
    original_probs: list[list[float]] | None = None,
) -> dict:
    """Measure ranking instability under one or more attacks."""
    if attacks is None:
        attacks = [WordNetAttack(max_changes)]

    rng = random.Random(seed)
    original_probabilities = original_probs if original_probs is not None else model.predict_proba(texts)
    filtered_texts = [
        text
        for text, probabilities in zip(texts, original_probabilities)
        if max(probabilities) >= mcp_threshold
    ]

    all_tau_scores = []
    per_attack_results = {}

    for attack in attacks:
        attack_name = type(attack).__name__
        tau_scores = []
        for text in filtered_texts:
            adversarial_text = attack.attack(text, rng)
            original_explanation = explainer_fn(text, top_k)
            adversarial_explanation = explainer_fn(adversarial_text, top_k)
            tau = _kendall_tau_hat(original_explanation, adversarial_explanation)
            tau_scores.append(tau)

        all_tau_scores.extend(tau_scores)
        if tau_scores:
            per_attack_results[attack_name] = {
                "mean_tau": float(numpy.mean(tau_scores)),
                "sensitivity": float(numpy.mean([1 - (tau + 1) / 2 for tau in tau_scores])),
            }

    if not all_tau_scores:
        return {"adversarial_sensitivity": 0.0, "mean_tau": 0.0, "per_attack": {}}

    sensitivity_scores = [1 - (tau + 1) / 2 for tau in all_tau_scores]
    return {
        "adversarial_sensitivity": float(numpy.mean(sensitivity_scores)),
        "mean_tau": float(numpy.mean(all_tau_scores)),
        "per_attack": per_attack_results,
    }
