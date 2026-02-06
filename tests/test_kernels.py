"""Tests for GPU kernels and regularization."""

import torch
import pytest

from src.models.components import (
    SpladeAggregateFunction,
    splade_aggregate,
    splade_aggregate_gated,
)
from src.training.losses import (
    DFFlopsRegFunction,
    DocumentFrequencyTracker,
    sparse_contrastive_loss,
)


class TestForwardKernel:
    def test_matches_pytorch_reference(self):
        """Triton forward kernel matches PyTorch log1p(relu) + mask + max."""
        torch.manual_seed(42)
        logits = torch.randn(2, 5, 100, device="cuda")
        mask = torch.ones(2, 5, device="cuda")
        mask[0, 3:] = 0

        result_triton = splade_aggregate(logits, mask)

        # PyTorch reference
        activated = torch.log1p(torch.relu(logits.float()))
        activated = activated.masked_fill(~mask.unsqueeze(-1).bool(), 0.0)
        result_pytorch = activated.max(dim=1).values

        torch.testing.assert_close(result_triton, result_pytorch, atol=1e-5, rtol=1e-5)

    def test_all_masked_returns_zeros(self):
        """All-masked input should produce zero output."""
        logits = torch.randn(1, 4, 50, device="cuda")
        mask = torch.zeros(1, 4, device="cuda")
        result = splade_aggregate(logits, mask)
        assert (result == 0.0).all()

    def test_output_shape(self):
        logits = torch.randn(3, 7, 50, device="cuda")
        mask = torch.ones(3, 7, device="cuda")
        result = splade_aggregate(logits, mask)
        assert result.shape == (3, 50)

    def test_non_negative_output(self):
        """log1p(relu(x)) >= 0, so output should be non-negative."""
        torch.manual_seed(123)
        logits = torch.randn(4, 10, 200, device="cuda")
        mask = torch.ones(4, 10, device="cuda")
        result = splade_aggregate(logits, mask)
        assert (result >= 0).all()


class TestGatedForwardKernel:
    def test_matches_separate_aggregate_and_gate(self):
        """Gated kernel should match: aggregate then apply (gate > 0)."""
        torch.manual_seed(42)
        logits = torch.randn(2, 5, 100, device="cuda")
        mask = torch.ones(2, 5, device="cuda")
        mask[0, 3:] = 0
        gate_logits = torch.randn(100, device="cuda")

        result_fused = splade_aggregate_gated(logits, mask, gate_logits)

        # Reference: separate aggregate + manual gating
        result_separate = splade_aggregate(logits, mask)
        gate_mask = (gate_logits > 0).float()
        result_separate = result_separate * gate_mask

        torch.testing.assert_close(result_fused, result_separate, atol=1e-5, rtol=1e-5)

    def test_all_gates_open(self):
        """All-positive gate_logits should match plain aggregate."""
        torch.manual_seed(42)
        logits = torch.randn(2, 5, 50, device="cuda")
        mask = torch.ones(2, 5, device="cuda")
        gate_logits = torch.ones(50, device="cuda")

        result_gated = splade_aggregate_gated(logits, mask, gate_logits)
        result_plain = splade_aggregate(logits, mask)

        torch.testing.assert_close(result_gated, result_plain, atol=1e-5, rtol=1e-5)

    def test_all_gates_closed(self):
        """All-negative gate_logits should produce zeros."""
        logits = torch.randn(2, 5, 50, device="cuda")
        mask = torch.ones(2, 5, device="cuda")
        gate_logits = -torch.ones(50, device="cuda")

        result = splade_aggregate_gated(logits, mask, gate_logits)
        assert (result == 0.0).all()


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

    def test_reset(self):
        """reset() should zero out DF counts and doc_count."""
        tracker = DocumentFrequencyTracker(vocab_size=5, device="cuda")
        vectors = torch.ones(3, 5, device="cuda")
        tracker.update(vectors)
        assert tracker.doc_count == 3

        tracker.reset()
        assert tracker.doc_count == 0
        assert tracker.df_counts.sum().item() == 0.0


class TestSparseContrastiveLoss:
    def test_single_sample_returns_zero(self):
        vecs = torch.randn(1, 100, device="cuda")
        labels = torch.tensor([0], device="cuda")
        loss = sparse_contrastive_loss(vecs, labels)
        assert loss.item() == 0.0

    def test_all_same_class_finite(self):
        vecs = torch.randn(4, 100, device="cuda")
        labels = torch.zeros(4, device="cuda", dtype=torch.long)
        loss = sparse_contrastive_loss(vecs, labels)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_all_different_classes_returns_zero(self):
        vecs = torch.randn(4, 100, device="cuda")
        labels = torch.arange(4, device="cuda")
        loss = sparse_contrastive_loss(vecs, labels)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        vecs = torch.randn(4, 100, device="cuda", requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1], device="cuda")
        loss = sparse_contrastive_loss(vecs, labels)
        loss.backward()
        assert vecs.grad is not None
        assert torch.isfinite(vecs.grad).all()

    def test_identical_vectors_finite(self):
        vecs = torch.ones(4, 100, device="cuda")
        labels = torch.tensor([0, 0, 1, 1], device="cuda")
        loss = sparse_contrastive_loss(vecs, labels)
        assert torch.isfinite(loss)

    def test_binary_classification(self):
        vecs = torch.randn(8, 50, device="cuda", requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device="cuda")
        loss = sparse_contrastive_loss(vecs, labels)
        assert loss.item() > 0
        loss.backward()
        assert torch.isfinite(vecs.grad).all()
