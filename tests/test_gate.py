"""Tests for Gumbel-Softmax sparse gate."""

import pytest
import torch

from src.models.gate import GumbelSparseGate, _gumbel_sigmoid


class TestGumbelSigmoid:
    def test_output_shape(self):
        logits = torch.randn(100)
        result = _gumbel_sigmoid(logits, temperature=1.0)
        assert result.shape == (100,)

    def test_binary_output(self):
        """Straight-through should produce 0/1 in forward."""
        logits = torch.randn(1000)
        result = _gumbel_sigmoid(logits, temperature=1.0)
        unique = torch.unique(result)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_zero_temperature_hard(self):
        logits = torch.tensor([1.0, -1.0, 0.5])
        result = _gumbel_sigmoid(logits, temperature=0.0)
        assert result.tolist() == [1.0, 0.0, 1.0]

    def test_gradient_flows(self):
        logits = torch.randn(50, requires_grad=True)
        result = _gumbel_sigmoid(logits, temperature=1.0)
        result.sum().backward()
        assert logits.grad is not None


class TestGumbelSparseGate:
    def test_output_shape(self):
        gate = GumbelSparseGate(100)
        x = torch.randn(4, 100)
        out = gate(x, temperature=1.0, training=True)
        assert out.shape == (4, 100)

    def test_inference_binary(self):
        """At inference, gate produces hard 0/1 mask."""
        gate = GumbelSparseGate(100)
        x = torch.ones(2, 100)
        out = gate(x, temperature=1.0, training=False)
        # ones init → all gate_logits = 1.0 > 0 → all open
        assert (out == x).all()

    def test_ones_init_all_open(self):
        """Ones initialization means all gates open at inference."""
        gate = GumbelSparseGate(50)
        x = torch.ones(1, 50)
        out = gate(x, training=False)
        assert (out == 1.0).all()

    def test_gradient_flows_through_gate(self):
        gate = GumbelSparseGate(50)
        x = torch.randn(2, 50, requires_grad=True)
        out = gate(x, temperature=1.0, training=True)
        out.sum().backward()
        assert x.grad is not None
        assert gate.gate_logits.grad is not None

    def test_negative_logits_gate_closed(self):
        """Negative logits should close gates at inference."""
        gate = GumbelSparseGate(10)
        gate.gate_logits.data = -torch.ones(10)
        x = torch.ones(1, 10)
        out = gate(x, training=False)
        assert (out == 0.0).all()

    def test_sparse_input_preserved(self):
        """Gated output should be at most as active as input."""
        gate = GumbelSparseGate(100)
        x = torch.zeros(1, 100)
        x[0, :10] = 1.0  # only 10 active
        out = gate(x, training=False)
        assert out.count_nonzero() <= 10
