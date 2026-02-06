"""Tests for faithfulness metrics."""

import pytest

from src.evaluation.faithfulness import (
    UnigramSampler,
    _mask_by_token_set,
    _top_k_tokens,
    compute_comprehensiveness,
    compute_filler_comprehensiveness,
    compute_soft_comprehensiveness,
    compute_soft_sufficiency,
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
        result = compute_comprehensiveness(model, texts, attributions, k_values=[1], mask_token="[MASK]")
        # Removing "good" -> confidence drops from 0.9 to 0.3 = 0.6 drop
        assert result[1] == pytest.approx(0.6, abs=0.01)

    def test_removing_irrelevant_token_no_drop(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("this", 0.9), ("good", 0.1)]]
        result = compute_comprehensiveness(model, texts, attributions, k_values=[1], mask_token="[MASK]")
        # Removing "this" (not the keyword) -> "is good" still has "good" -> conf stays at 0.9
        assert result[1] == pytest.approx(0.0, abs=0.01)


class TestSufficiency:
    def test_keeping_key_token_preserves_confidence(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        result = compute_sufficiency(model, texts, attributions, k_values=[1], mask_token="[MASK]")
        # Keeping only "good" -> "good" still present -> conf = 0.9 -> drop = 0.0
        assert result[1] == pytest.approx(0.0, abs=0.01)


class TestFillerComprehensiveness:
    def test_returns_dict_with_k_values(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        sampler = UnigramSampler(["the cat sat on the mat", "a dog ran in the park"], seed=42)
        result = compute_filler_comprehensiveness(model, texts, attributions, k_values=[1, 5], sampler=sampler)
        assert 1 in result
        assert 5 in result

    def test_with_unigram_sampler(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        sampler = UnigramSampler(["the cat sat on the mat", "a dog ran in the park"], seed=42)
        result = compute_filler_comprehensiveness(
            model, texts, attributions, k_values=[1], sampler=sampler,
        )
        assert 1 in result
        assert isinstance(result[1], float)


class TestUnigramSampler:
    def test_samples_from_corpus(self):
        sampler = UnigramSampler(["the cat sat", "a dog ran"], seed=42)
        for _ in range(10):
            word = sampler.sample()
            assert isinstance(word, str)
            assert len(word) > 0

    def test_deterministic_with_seed(self):
        sampler1 = UnigramSampler(["the cat sat"], seed=42)
        sampler2 = UnigramSampler(["the cat sat"], seed=42)
        assert sampler1.sample() == sampler2.sample()


class TestSoftComprehensiveness:
    def test_returns_float(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        result = compute_soft_comprehensiveness(
            model, texts, attributions, "[MASK]", n_samples=5, seed=42,
        )
        assert isinstance(result, float)

    def test_removing_key_token_drops_confidence(self):
        model = MockPredictor()
        texts = ["This is good"]
        # importance=1.0 → always masked
        attributions = [[("good", 1.0), ("this", 0.0)]]
        result = compute_soft_comprehensiveness(
            model, texts, attributions, "[MASK]", n_samples=50, seed=42,
        )
        # "good" always masked → conf drops from 0.9 to 0.3 = 0.6
        assert result == pytest.approx(0.6, abs=0.05)

    def test_deterministic_with_seed(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.3)]]
        r1 = compute_soft_comprehensiveness(model, texts, attributions, "[MASK]", n_samples=10, seed=42)
        r2 = compute_soft_comprehensiveness(model, texts, attributions, "[MASK]", n_samples=10, seed=42)
        assert r1 == r2

    def test_empty_texts(self):
        model = MockPredictor()
        result = compute_soft_comprehensiveness(model, [], [], "[MASK]", n_samples=5, seed=42)
        assert result == 0.0


class TestBatchedCallCount:
    """Verify batched implementations reduce predict_proba call count."""

    def test_masking_metric_batched(self):
        """_compute_masking_metric should make exactly 2 predict_proba calls."""
        call_count = [0]

        class CountingPredictor:
            def predict_proba(self, texts):
                call_count[0] += 1
                return [[0.1, 0.9] if "good" in t.lower() else [0.7, 0.3] for t in texts]

        model = CountingPredictor()
        texts = ["This is good", "That was bad", "Very good indeed"]
        attributions = [
            [("good", 0.9), ("this", 0.1)],
            [("bad", 0.9), ("that", 0.1)],
            [("good", 0.9), ("very", 0.1)],
        ]
        compute_comprehensiveness(model, texts, attributions, k_values=[1, 3, 5], mask_token="[MASK]")
        # 1 call for originals + 1 call for all masked texts = 2
        assert call_count[0] == 2

    def test_soft_comp_batched(self):
        """compute_soft_comprehensiveness should make exactly 2 predict_proba calls."""
        call_count = [0]

        class CountingPredictor:
            def predict_proba(self, texts):
                call_count[0] += 1
                return [[0.1, 0.9] if "good" in t.lower() else [0.7, 0.3] for t in texts]

        model = CountingPredictor()
        texts = ["This is good", "That was bad"]
        attributions = [
            [("good", 0.9), ("this", 0.1)],
            [("bad", 0.9), ("that", 0.1)],
        ]
        compute_soft_comprehensiveness(model, texts, attributions, "[MASK]", n_samples=10, seed=42)
        # 1 call for originals + 1 call for all masked texts = 2
        assert call_count[0] == 2

    def test_beam_search_batched(self):
        """_beam_search_ordering should make 1 + len(tokens) predict_proba calls."""
        from src.evaluation.faithfulness import _beam_search_ordering
        call_count = [0]

        class CountingPredictor:
            def predict_proba(self, texts):
                call_count[0] += 1
                return [[0.1, 0.9] if "good" in t.lower() else [0.7, 0.3] for t in texts]

        model = CountingPredictor()
        tokens = ["good", "this", "is", "very"]
        _beam_search_ordering(model, "This is very good", tokens, beam_size=2, mask_token="[MASK]")
        # 1 call for original + 4 steps (1 batch call each) = 5
        assert call_count[0] == 5


class TestSoftSufficiency:
    def test_returns_float(self):
        model = MockPredictor()
        texts = ["This is good"]
        attributions = [[("good", 0.9), ("this", 0.1)]]
        result = compute_soft_sufficiency(
            model, texts, attributions, "[MASK]", n_samples=5, seed=42,
        )
        assert isinstance(result, float)

    def test_keeping_key_token_preserves_confidence(self):
        model = MockPredictor()
        texts = ["This is good"]
        # importance=1.0 → always retained; 0.0 → always masked
        attributions = [[("good", 1.0), ("this", 0.0)]]
        result = compute_soft_sufficiency(
            model, texts, attributions, "[MASK]", n_samples=50, seed=42,
        )
        # "good" always retained → conf stays 0.9 → drop ≈ 0.0
        assert result == pytest.approx(0.0, abs=0.05)

    def test_empty_texts(self):
        model = MockPredictor()
        result = compute_soft_sufficiency(model, [], [], "[MASK]", n_samples=5, seed=42)
        assert result == 0.0
