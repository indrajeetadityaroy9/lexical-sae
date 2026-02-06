"""Integration tests for the full benchmark pipeline.

These tests require a GPU and train real (tiny) models. They are slower
than unit tests but verify end-to-end correctness.
"""

import pytest
import numpy as np

from src.adversarial import compute_adversarial_sensitivity
from src.baselines import (
    AttentionExplainer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
    train_shared_model,
)
from src.faithfulness import compute_comprehensiveness, compute_sufficiency


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRAIN_TEXTS = [
    "I love this movie it is great",
    "Terrible film awful acting",
    "Amazing performance wonderful story",
    "Horrible waste of time",
    "Best movie I have ever seen",
    "Worst film in years",
    "Brilliant direction superb screenplay",
    "Dreadful plot boring dialogue",
] * 5  # 40 samples

TRAIN_LABELS = [1, 0, 1, 0, 1, 0, 1, 0] * 5

TEST_TEXTS = [
    "This is a good movie",
    "This is a bad movie",
    "Fantastic acting and story",
    "Terrible and boring film",
]
TEST_LABELS = [1, 0, 1, 0]


@pytest.fixture(scope="module")
def shared_model():
    """Train a shared DistilBERT model (small, 1 epoch)."""
    model, tokenizer = train_shared_model(
        "distilbert-base-uncased", num_labels=2,
        texts=TRAIN_TEXTS, labels=TRAIN_LABELS,
        epochs=1, batch_size=8,
    )
    return model, tokenizer


@pytest.fixture(scope="module")
def all_explainers(shared_model):
    """Create all four explainer instances from the shared model."""
    model, tokenizer = shared_model
    return {
        "attention": AttentionExplainer(model, tokenizer, num_labels=2),
        "lime": LIMEExplainer(model, tokenizer, num_labels=2),
        "shap": SHAPExplainer(model, tokenizer, num_labels=2),
        "ig": IntegratedGradientsExplainer(model, tokenizer, num_labels=2),
    }


# ---------------------------------------------------------------------------
# Shared model: all baselines use the same model
# ---------------------------------------------------------------------------

class TestSharedModel:
    def test_all_baselines_same_predictions(self, all_explainers):
        """All baselines must produce identical predict_proba (same model)."""
        explainers = list(all_explainers.values())
        reference = explainers[0].predict_proba(TEST_TEXTS)

        for explainer in explainers[1:]:
            probs = explainer.predict_proba(TEST_TEXTS)
            for ref_row, test_row in zip(reference, probs):
                np.testing.assert_allclose(
                    ref_row, test_row, atol=1e-5,
                    err_msg=f"Predictions differ between baselines",
                )

    def test_all_baselines_same_accuracy(self, all_explainers):
        """All baselines must report the same accuracy (shared model)."""
        accuracies = []
        for name, explainer in all_explainers.items():
            probs = explainer.predict_proba(TEST_TEXTS)
            preds = [int(np.argmax(p)) for p in probs]
            acc = sum(p == t for p, t in zip(preds, TEST_LABELS)) / len(TEST_LABELS)
            accuracies.append(acc)
        assert all(a == accuracies[0] for a in accuracies), \
            f"Accuracies differ: {accuracies}"


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

class TestExplainerOutput:
    def test_attention_explains(self, all_explainers):
        result = all_explainers["attention"].explain("This is a good movie", top_k=5)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(t, str) and isinstance(w, float) for t, w in result)

    def test_lime_explains(self, all_explainers):
        result = all_explainers["lime"].explain("This is a good movie", top_k=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_shap_explains(self, all_explainers):
        result = all_explainers["shap"].explain("This is a good movie", top_k=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_ig_explains(self, all_explainers):
        result = all_explainers["ig"].explain("This is a good movie", top_k=5)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Faithfulness metrics with real explainers
# ---------------------------------------------------------------------------

class TestFaithfulnessWithRealModels:
    def test_comprehensiveness_returns_valid(self, all_explainers):
        explainer = all_explainers["attention"]
        texts = TEST_TEXTS[:2]
        attribs = [explainer.explain(t, top_k=10) for t in texts]
        result = compute_comprehensiveness(explainer, texts, attribs, k_values=[1, 5])
        assert 1 in result
        assert 5 in result
        for k, v in result.items():
            assert isinstance(v, float)
            assert not np.isnan(v)

    def test_sufficiency_returns_valid(self, all_explainers):
        explainer = all_explainers["attention"]
        texts = TEST_TEXTS[:2]
        attribs = [explainer.explain(t, top_k=10) for t in texts]
        result = compute_sufficiency(explainer, texts, attribs, k_values=[1, 5])
        assert 1 in result
        assert 5 in result
        for k, v in result.items():
            assert isinstance(v, float)
            assert not np.isnan(v)


# ---------------------------------------------------------------------------
# Adversarial sensitivity determinism
# ---------------------------------------------------------------------------

class TestAdversarialDeterminism:
    def test_same_seed_same_result(self, all_explainers):
        """Two calls with the same seed must produce identical results."""
        explainer = all_explainers["attention"]
        texts = TEST_TEXTS[:2]

        result1 = compute_adversarial_sensitivity(
            explainer,
            lambda text, top_k: explainer.explain(text, top_k=top_k),
            texts, seed=42,
        )
        result2 = compute_adversarial_sensitivity(
            explainer,
            lambda text, top_k: explainer.explain(text, top_k=top_k),
            texts, seed=42,
        )
        assert result1["adversarial_sensitivity"] == pytest.approx(
            result2["adversarial_sensitivity"], abs=1e-6,
        )
        assert result1["mean_tau"] == pytest.approx(
            result2["mean_tau"], abs=1e-6,
        )

    def test_different_seed_may_differ(self, all_explainers):
        """Different seeds should (generally) produce different results."""
        explainer = all_explainers["attention"]
        texts = TEST_TEXTS[:2]

        result1 = compute_adversarial_sensitivity(
            explainer,
            lambda text, top_k: explainer.explain(text, top_k=top_k),
            texts, seed=42,
        )
        result2 = compute_adversarial_sensitivity(
            explainer,
            lambda text, top_k: explainer.explain(text, top_k=top_k),
            texts, seed=999,
        )
        # Not asserting inequality (seeds could produce same result),
        # just verifying both run without error
        assert isinstance(result1["adversarial_sensitivity"], float)
        assert isinstance(result2["adversarial_sensitivity"], float)
