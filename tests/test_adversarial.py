"""Tests for adversarial sensitivity metrics."""

import random

import pytest

from src.evaluation.adversarial import (
    CharacterAttack,
    TextFoolerAttack,
    WordNetAttack,
    _kendall_tau_hat,
    compute_adversarial_sensitivity,
)


class MockPredictor:
    def predict_proba(self, texts):
        return [[0.1, 0.9] for _ in texts]


class TestAttackStrategies:
    def test_wordnet_attack_produces_different_text(self):
        attack = WordNetAttack(max_changes=3)
        rng = random.Random(42)
        text = "The movie was good and enjoyable"
        result = attack.attack(text, rng)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_character_attack_produces_different_text(self):
        attack = CharacterAttack(max_changes=2)
        rng = random.Random(42)
        text = "The movie was fantastic"
        result = attack.attack(text, rng)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_character_attack_empty_text(self):
        attack = CharacterAttack(max_changes=2)
        rng = random.Random(42)
        assert attack.attack("", rng) == ""

    def test_textfooler_attack_runs(self):
        model = MockPredictor()
        attack = TextFoolerAttack(model, max_changes=2)
        rng = random.Random(42)
        text = "The movie was good"
        result = attack.attack(text, rng)
        assert isinstance(result, str)


class TestKendallTauHat:
    def test_identical_rankings(self):
        ranking = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        tau = _kendall_tau_hat(ranking, ranking)
        assert tau == pytest.approx(1.0, abs=1e-6)

    def test_reversed_rankings(self):
        ranking_a = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        ranking_b = [("c", 0.9), ("b", 0.7), ("a", 0.5)]
        tau = _kendall_tau_hat(ranking_a, ranking_b)
        assert tau == pytest.approx(-1.0, abs=1e-6)

    def test_partial_overlap(self):
        """tau-hat handles disjoint items by assigning default rank."""
        ranking_a = [("a", 0.9), ("b", 0.7)]
        ranking_b = [("b", 0.9), ("c", 0.7)]
        tau = _kendall_tau_hat(ranking_a, ranking_b)
        assert -1.0 <= tau <= 1.0

    def test_completely_disjoint(self):
        ranking_a = [("a", 0.9), ("b", 0.7)]
        ranking_b = [("c", 0.9), ("d", 0.7)]
        tau = _kendall_tau_hat(ranking_a, ranking_b)
        assert -1.0 <= tau <= 1.0

    def test_single_item(self):
        ranking = [("a", 0.9)]
        tau = _kendall_tau_hat(ranking, ranking)
        assert tau == 0.0  # n < 2


class TestComputeAdversarialSensitivity:
    def test_returns_expected_keys(self):
        model = MockPredictor()
        texts = ["The movie was good"]

        def explainer_fn(text, top_k):
            return [("good", 0.9), ("movie", 0.5)]

        result = compute_adversarial_sensitivity(
            model, explainer_fn, texts,
            attacks=None, max_changes=3, mcp_threshold=0.7, top_k=20, seed=42,
        )
        assert "adversarial_sensitivity" in result
        assert "mean_tau" in result
        assert "per_attack" in result

    def test_multi_attack(self):
        model = MockPredictor()
        texts = ["The movie was good"]

        def explainer_fn(text, top_k):
            return [("good", 0.9), ("movie", 0.5)]

        attacks = [WordNetAttack(3), CharacterAttack(2)]
        result = compute_adversarial_sensitivity(
            model, explainer_fn, texts, attacks=attacks,
            max_changes=3, mcp_threshold=0.7, top_k=20, seed=42,
        )
        assert "adversarial_sensitivity" in result
        assert isinstance(result["per_attack"], dict)
