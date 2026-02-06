"""Tests for faithfulness metrics."""

import pytest

from src.faithfulness import (
    _mask_by_token_set,
    _top_k_tokens,
    compute_comprehensiveness,
    compute_filler_comprehensiveness,
    compute_sufficiency,
)


class MockPredictor:
    """Deterministic predictor: high confidence if 'good' present, low otherwise."""
    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            if "good" in text.lower():
                results.append([0.1, 0.9])
            else:
                results.append([0.7, 0.3])
        return results


class TestTopKTokens:
    def test_preserves_importance_order(self):
        """Top-k must select the k most important tokens."""
        attrib = [("important", 0.9), ("medium", 0.5), ("low", 0.1)]
        result = _top_k_tokens(attrib, k=2)
        assert result == {"important", "medium"}

    def test_skips_zero_weight(self):
        attrib = [("a", 0.9), ("b", 0.0), ("c", 0.5)]
        result = _top_k_tokens(attrib, k=3)
        assert result == {"a", "c"}

    def test_deduplicates_case_insensitive(self):
        attrib = [("Word", 0.9), ("word", 0.8), ("other", 0.5)]
        result = _top_k_tokens(attrib, k=2)
        assert result == {"word", "other"}

    def test_k_larger_than_available(self):
        attrib = [("a", 0.9), ("b", 0.5)]
        result = _top_k_tokens(attrib, k=10)
        assert result == {"a", "b"}

    def test_empty_attrib(self):
        result = _top_k_tokens([], k=5)
        assert result == set()


class TestMaskByTokenSet:
    def test_remove_mode(self):
        text = "This is a good movie"
        masked = _mask_by_token_set(text, {"good"}, "[MASK]", mode="remove")
        assert masked == "This is a [MASK] movie"

    def test_keep_mode(self):
        text = "This is a good movie"
        masked = _mask_by_token_set(text, {"good"}, "[MASK]", mode="keep")
        assert masked == "[MASK] [MASK] [MASK] good [MASK]"

    def test_max_fraction(self):
        text = "a b c d e f g h i j"  # 10 words
        # Remove all but max_fraction=0.3 -> at most 3 tokens masked
        masked = _mask_by_token_set(
            text, {"a", "b", "c", "d", "e"}, "[MASK]",
            mode="remove", max_fraction=0.3,
        )
        mask_count = masked.count("[MASK]")
        assert mask_count == 3


class TestComprehensiveness:
    def test_removing_key_token_drops_confidence(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        result = compute_comprehensiveness(model, texts, attributions, k_values=[1])
        # Removing "good" -> confidence drops from 0.9 to 0.3 = 0.6 drop
        assert result[1] == pytest.approx(0.6, abs=0.01)

    def test_removing_irrelevant_token_no_drop(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("this", 0.9), ("good", 0.1)]]
        result = compute_comprehensiveness(model, texts, attributions, k_values=[1])
        # Removing "this" (not the keyword) -> "is good" still has "good" -> conf stays at 0.9
        assert result[1] == pytest.approx(0.0, abs=0.01)


class TestSufficiency:
    def test_keeping_key_token_preserves_confidence(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        result = compute_sufficiency(model, texts, attributions, k_values=[1])
        # Keeping only "good" -> "good" still present -> conf = 0.9 -> drop = 0.0
        assert result[1] == pytest.approx(0.0, abs=0.01)


class TestFillerComprehensiveness:
    def test_returns_dict_with_k_values(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        result = compute_filler_comprehensiveness(model, texts, attributions, k_values=[1, 5])
        assert 1 in result
        assert 5 in result
