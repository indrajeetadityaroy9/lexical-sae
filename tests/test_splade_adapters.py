"""Tests for SPLADE adapter explainers (fair baseline comparison)."""

import pytest

from src.models import SPLADEClassifier
from src.baselines.splade_adapters import (
    SPLADEAttentionExplainer,
    SPLADEIntegratedGradientsExplainer,
    SPLADELIMEExplainer,
)


@pytest.fixture(scope="module")
def clf():
    """Untrained SPLADE classifier for API tests."""
    return SPLADEClassifier(num_labels=2, model_name="distilbert-base-uncased")


class TestSPLADEAttentionExplainer:
    def test_explain_returns_list_of_tuples(self, clf):
        adapter = SPLADEAttentionExplainer(clf)
        result = adapter.explain("This is a test sentence", top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_num_labels_matches_clf(self, clf):
        adapter = SPLADEAttentionExplainer(clf)
        assert adapter.num_labels == clf.num_labels


class TestSPLADEIntegratedGradientsExplainer:
    def test_num_labels_matches_clf(self, clf):
        adapter = SPLADEIntegratedGradientsExplainer(clf, n_steps=5)
        assert adapter.num_labels == clf.num_labels


class TestSPLADELIMEExplainer:
    def test_explain_returns_list_of_tuples(self, clf):
        adapter = SPLADELIMEExplainer(clf, num_samples=50)
        result = adapter.explain("This is a test sentence", top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_explain_sorted_by_abs_weight(self, clf):
        adapter = SPLADELIMEExplainer(clf, num_samples=50)
        result = adapter.explain("The movie was absolutely wonderful", top_k=5)
        if len(result) > 1:
            abs_weights = [abs(w) for _, w in result]
            for i in range(len(abs_weights) - 1):
                assert abs_weights[i] >= abs_weights[i + 1]


class TestAllAdaptersShareModel:
    """Verify all adapters reference the same underlying SPLADE model."""

    def test_adapters_share_same_clf_instance(self, clf):
        attention = SPLADEAttentionExplainer(clf)
        lime_exp = SPLADELIMEExplainer(clf, num_samples=50)
        ig_exp = SPLADEIntegratedGradientsExplainer(clf, n_steps=5)

        assert attention.clf is clf
        assert lime_exp.clf is clf
        assert ig_exp.clf is clf
