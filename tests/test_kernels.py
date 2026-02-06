"""Tests for GPU kernels and regularization."""

import torch
import pytest

from src.kernels import (
    DocumentFrequencyTracker,
    DFFlopsRegFunction,
    SpladeAggregateFunction,
    splade_aggregate,
)


class TestSpladeAggregate:
    def test_output_matches_custom_function(self):
        """Forward pass of splade_aggregate matches SpladeAggregateFunction."""
        torch.manual_seed(42)
        logits = torch.randn(2, 5, 100, device="cuda", requires_grad=False)
        mask = torch.ones(2, 5, device="cuda")
        mask[0, 3:] = 0  # mask padding

        result_pytorch = splade_aggregate(logits, mask)
        result_custom = SpladeAggregateFunction.apply(logits, mask)

        torch.testing.assert_close(result_pytorch, result_custom, atol=1e-5, rtol=1e-5)

    def test_masking_zeroes_padded_positions(self):
        """Padded positions should not contribute to sparse vector."""
        logits = torch.ones(1, 4, 10, device="cuda") * 5.0
        mask = torch.tensor([[1, 1, 0, 0]], device="cuda", dtype=torch.float32)

        result = splade_aggregate(logits, mask)
        # All values should be log1p(relu(5.0)) = log(6) ≈ 1.79
        expected_val = torch.log1p(torch.tensor(5.0, device="cuda"))
        assert result.shape == (1, 10)
        torch.testing.assert_close(result[0, 0], expected_val, atol=1e-4, rtol=1e-4)

    def test_output_shape(self):
        logits = torch.randn(3, 7, 50, device="cuda")
        mask = torch.ones(3, 7, device="cuda")
        result = splade_aggregate(logits, mask)
        assert result.shape == (3, 50)


class TestDFFlopsReg:
    def test_gradient_finite(self):
        """DFFlopsRegFunction gradient should be finite."""
        activations = torch.randn(4, 100, device="cuda", requires_grad=True)
        df_weights = torch.rand(100, device="cuda")

        loss = DFFlopsRegFunction.apply(activations, df_weights)
        loss.backward()

        assert activations.grad is not None
        assert torch.isfinite(activations.grad).all()

    def test_zero_activations_zero_loss(self):
        """Zero activations should produce zero loss."""
        activations = torch.zeros(4, 100, device="cuda", requires_grad=True)
        df_weights = torch.rand(100, device="cuda")

        loss = DFFlopsRegFunction.apply(activations, df_weights)
        assert loss.item() == 0.0


class TestDocumentFrequencyTracker:
    def test_update_and_counts(self):
        tracker = DocumentFrequencyTracker(vocab_size=5, device="cuda")
        vectors = torch.tensor([
            [1.0, 0.0, 2.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ], device="cuda")

        tracker.update(vectors)

        assert tracker.doc_count == 2
        expected_df = torch.tensor([1, 0, 2, 0, 1], device="cuda", dtype=torch.float32)
        torch.testing.assert_close(tracker.df_counts, expected_df)

    def test_get_weights_no_nan(self):
        tracker = DocumentFrequencyTracker(vocab_size=10, device="cuda")
        vectors = torch.rand(20, 10, device="cuda")
        tracker.update(vectors)

        for alpha in [0.01, 0.1, 0.5, 0.99]:
            weights = tracker.get_weights(alpha=alpha)
            assert torch.isfinite(weights).all(), f"NaN/Inf with alpha={alpha}"
            assert (weights >= 0).all()

    def test_get_weights_alpha_one(self):
        """alpha=1.0 should not raise (epsilon guards log(1)=0)."""
        tracker = DocumentFrequencyTracker(vocab_size=5, device="cuda")
        vectors = torch.ones(3, 5, device="cuda")
        tracker.update(vectors)

        weights = tracker.get_weights(alpha=1.0)
        assert torch.isfinite(weights).all()

    def test_get_stats(self):
        tracker = DocumentFrequencyTracker(vocab_size=5, device="cuda")
        vectors = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        ], device="cuda")
        tracker.update(vectors)

        stats = tracker.get_stats()
        assert stats["doc_count"] == 2
        assert stats["top1_df_pct"] == 100.0  # term 0 and 2 appear in both docs
