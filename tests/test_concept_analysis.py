"""Tests for concept analysis metrics."""

import numpy as np
import pytest
import torch

from src.evaluation.concept_analysis import concept_intervention, concept_necessity, concept_sufficiency
from src.models import SPLADEClassifier


@pytest.fixture(scope="module")
def clf():
    """Untrained SPLADE classifier for shape/API tests."""
    return SPLADEClassifier(num_labels=2, model_name="distilbert-base-uncased")


class TestConceptSufficiency:
    def test_returns_dict_with_k_values(self, clf):
        texts = ["This is good", "This is bad"]
        labels = [1, 0]
        result = concept_sufficiency(clf, texts, labels, top_k_values=[10, 50])
        assert 10 in result
        assert 50 in result
        for k, acc in result.items():
            assert 0.0 <= acc <= 1.0


class TestConceptNecessity:
    def test_returns_dict_with_k_values(self, clf):
        texts = ["This is good", "This is bad"]
        labels = [1, 0]
        result = concept_necessity(clf, texts, labels, top_k_values=[10, 50])
        assert 10 in result
        assert 50 in result
        for k, drop in result.items():
            assert isinstance(drop, float)


class TestConceptIntervention:
    def test_returns_expected_keys(self, clf):
        texts = ["This is good", "This is bad"]
        labels = [1, 0]
        result = concept_intervention(
            clf, texts, labels,
            concept_indices=[100, 200, 300],
            num_trials=5, seed=42,
        )
        assert "accuracy_original" in result
        assert "accuracy_intervened" in result
        assert "accuracy_drop" in result
