"""Tests for data loading and rationale agreement metrics."""

import pytest

from src.data import compute_rationale_agreement


class TestRationaleAgreement:
    def test_perfect_overlap(self):
        """Model and human pick exactly the same tokens → F1 = 1.0."""
        attribs = [[("good", 0.9), ("movie", 0.5), ("great", 0.3)]]
        human = [["good", "movie", "great"]]
        result = compute_rationale_agreement(attribs, human, k=3)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        """Zero common tokens → F1 = 0.0."""
        attribs = [[("good", 0.9), ("movie", 0.5)]]
        human = [["bad", "terrible"]]
        result = compute_rationale_agreement(attribs, human, k=2)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        """Partial overlap gives intermediate F1."""
        attribs = [[("good", 0.9), ("movie", 0.5)]]
        human = [["good", "terrible"]]
        # model_tokens = {"good", "movie"}, human_set = {"good", "terrible"}
        # intersection = {"good"}, precision = 1/2, recall = 1/2 → F1 = 0.5
        result = compute_rationale_agreement(attribs, human, k=2)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_empty_model_tokens(self):
        """No positive-weight attributions → 0.0, no crash."""
        attribs = [[("bad", -0.1), ("worse", 0.0)]]
        human = [["good"]]
        result = compute_rationale_agreement(attribs, human, k=2)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_empty_human_rationale(self):
        """Empty human rationale → 0.0, no crash."""
        attribs = [[("good", 0.9)]]
        human = [[]]
        result = compute_rationale_agreement(attribs, human, k=1)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_both_empty(self):
        """Both model and human empty → 0.0, no crash."""
        attribs = [[("x", 0.0)]]
        human = [[]]
        result = compute_rationale_agreement(attribs, human, k=1)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_case_insensitive(self):
        """Token matching is case-insensitive."""
        attribs = [[("Good", 0.9), ("Movie", 0.5)]]
        human = [["good", "movie"]]
        result = compute_rationale_agreement(attribs, human, k=2)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_k_limits_model_tokens(self):
        """k controls how many attributions are considered."""
        attribs = [[("good", 0.9), ("movie", 0.5), ("great", 0.3)]]
        human = [["great"]]
        # k=1 → only "good" considered → no overlap → F1=0
        result_k1 = compute_rationale_agreement(attribs, human, k=1)
        assert result_k1 == pytest.approx(0.0, abs=0.01)
        # k=3 → "good", "movie", "great" → overlap on "great"
        result_k3 = compute_rationale_agreement(attribs, human, k=3)
        assert result_k3 > 0.0

    def test_multiple_examples(self):
        """Averages correctly over multiple examples."""
        attribs = [
            [("good", 0.9)],
            [("bad", 0.9)],
        ]
        human = [
            ["good"],  # perfect match
            ["terrible"],  # no match
        ]
        # Ex 1: P=1, R=1; Ex 2: P=0, R=0
        # avg P=0.5, avg R=0.5, F1 = 0.5
        result = compute_rationale_agreement(attribs, human, k=1)
        assert result == pytest.approx(0.5, abs=0.01)
