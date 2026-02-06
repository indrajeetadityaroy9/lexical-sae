"""Tests for SPLADEClassifier model API."""

import pytest
import numpy as np
import torch

from src.models import SPLADEClassifier


@pytest.fixture(scope="module")
def clf():
    """Create a classifier (untrained) for API tests."""
    return SPLADEClassifier(num_labels=2, model_name="distilbert-base-uncased")


class TestInputValidation:
    def test_fit_empty_X(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.fit([], [])

    def test_fit_length_mismatch(self, clf):
        with pytest.raises(ValueError, match="same length"):
            clf.fit(["a"], [0, 1])

    def test_fit_non_string(self, clf):
        with pytest.raises(TypeError, match="strings"):
            clf.fit([123], [0])

    def test_predict_empty(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.predict([])

    def test_predict_proba_empty(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.predict_proba([])

    def test_predict_non_string(self, clf):
        with pytest.raises(TypeError, match="strings"):
            clf.predict([42])

    def test_score_empty_y(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.score(["text"], [])

    def test_transform_empty(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.transform([])

    def test_explain_empty_string(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.explain("")

    def test_explain_whitespace_only(self, clf):
        with pytest.raises(ValueError, match="non-empty"):
            clf.explain("   ")

    def test_explain_top_k_zero(self, clf):
        with pytest.raises(ValueError, match="top_k"):
            clf.explain("hello", top_k=0)

    def test_explain_top_k_negative(self, clf):
        with pytest.raises(ValueError, match="top_k"):
            clf.explain("hello", top_k=-1)


class TestPredictShapes:
    """Test output shapes and types (untrained model is fine for shape checks)."""

    def test_predict_returns_list_of_ints(self, clf):
        preds = clf.predict(["Hello world", "Test sentence"])
        assert isinstance(preds, list)
        assert len(preds) == 2
        assert all(isinstance(p, int) for p in preds)

    def test_predict_valid_class_indices(self, clf):
        preds = clf.predict(["Hello world"])
        assert all(0 <= p < clf.num_labels for p in preds)

    def test_predict_proba_shape(self, clf):
        probs = clf.predict_proba(["Hello world", "Test sentence"])
        assert len(probs) == 2
        assert len(probs[0]) == clf.num_labels

    def test_predict_proba_sums_to_one(self, clf):
        probs = clf.predict_proba(["Hello world"])
        assert sum(probs[0]) == pytest.approx(1.0, abs=1e-4)

    def test_predict_proba_non_negative(self, clf):
        probs = clf.predict_proba(["Hello world"])
        assert all(p >= 0 for p in probs[0])

    def test_transform_shape(self, clf):
        sparse = clf.transform(["Hello world", "Test"])
        assert isinstance(sparse, np.ndarray)
        assert sparse.shape[0] == 2
        assert sparse.shape[1] == clf.model.vocab_size


class TestExplain:
    def test_no_special_tokens(self, clf):
        """explain() must not return [CLS], [SEP], [UNK], [MASK], [PAD]."""
        results = clf.explain("This is a great movie and I loved it", top_k=10)
        special = {"[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"}
        for token, weight in results:
            assert token not in special, f"Special token {token} in explain output"

    def test_returns_list_of_tuples(self, clf):
        results = clf.explain("Hello world", top_k=5)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_weights_non_negative(self, clf):
        results = clf.explain("This is a test sentence", top_k=5)
        for _, weight in results:
            assert weight > 0

    def test_respects_top_k(self, clf):
        results = clf.explain("This is a test sentence for explanation", top_k=3)
        assert len(results) <= 3

    def test_weights_descending(self, clf):
        results = clf.explain("The movie was absolutely fantastic and wonderful", top_k=5)
        if len(results) > 1:
            weights = [w for _, w in results]
            for i in range(len(weights) - 1):
                assert weights[i] >= weights[i + 1], "Weights should be in descending order"
