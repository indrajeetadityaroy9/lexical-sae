"""Tests for baseline explainer methods."""

import torch
import pytest

from splade.evaluation.baselines import (
    attention_attribution,
    dla_attribution,
    gradient_attribution,
    integrated_gradients_attribution,
)
from splade.evaluation.compare_explainers import _EXPLAINERS


class FakeLexicalSAE(torch.nn.Module):
    """Minimal model mimicking LexicalSAE's interface for testing."""

    def __init__(self, vocab_size=100, num_labels=2, hidden=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.classifier_fc1 = torch.nn.Linear(vocab_size, hidden)
        self.classifier_fc2 = torch.nn.Linear(hidden, num_labels)
        # Fixed embedding for deterministic forward pass
        self._embedding = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        B = input_ids.shape[0]
        # Deterministic sparse_vector from input_ids (simulates SPLADE head)
        emb = self._embedding(input_ids.clamp(max=self.vocab_size - 1))
        sparse_vector = emb.mean(dim=1)  # [B, V]

        # ReLU MLP classifier with W_eff
        pre_relu = self.classifier_fc1(sparse_vector)
        activation_mask = (pre_relu > 0).float()
        hidden = pre_relu * activation_mask
        logits = self.classifier_fc2(hidden)

        W1 = self.classifier_fc1.weight
        W2 = self.classifier_fc2.weight
        b1 = self.classifier_fc1.bias
        b2 = self.classifier_fc2.bias
        masked_W1 = activation_mask.unsqueeze(-1) * W1.unsqueeze(0)
        W_eff = torch.matmul(W2.unsqueeze(0), masked_W1)
        b_eff = torch.matmul(activation_mask * b1, W2.T) + b2

        return logits, sparse_vector, W_eff, b_eff

    def classifier_logits_only(self, sparse_vector):
        return self.classifier_fc2(torch.relu(self.classifier_fc1(sparse_vector)))


@pytest.fixture
def fake_model():
    torch.manual_seed(42)
    return FakeLexicalSAE()


@pytest.fixture
def fake_inputs():
    B = 4
    return (
        torch.randint(0, 100, (B, 16)),      # input_ids
        torch.ones(B, 16, dtype=torch.long),  # attention_mask
        torch.zeros(B, dtype=torch.long),     # target_classes
    )


class TestGradientAttribution:
    def test_output_shape(self, fake_model, fake_inputs):
        input_ids, attention_mask, targets = fake_inputs
        attr = gradient_attribution(fake_model, input_ids, attention_mask, targets)
        assert attr.shape == (4, 100)

    def test_nonzero(self, fake_model, fake_inputs):
        input_ids, attention_mask, targets = fake_inputs
        attr = gradient_attribution(fake_model, input_ids, attention_mask, targets)
        assert attr.abs().sum() > 0


class TestIntegratedGradients:
    def test_output_shape(self, fake_model, fake_inputs):
        input_ids, attention_mask, targets = fake_inputs
        attr = integrated_gradients_attribution(
            fake_model, input_ids, attention_mask, targets, steps=5,
        )
        assert attr.shape == (4, 100)

    def test_one_step_approximates_gradient(self, fake_model, fake_inputs):
        input_ids, attention_mask, targets = fake_inputs
        ig_attr = integrated_gradients_attribution(
            fake_model, input_ids, attention_mask, targets, steps=1,
        )
        grad_attr = gradient_attribution(fake_model, input_ids, attention_mask, targets)
        # With 1 step at alpha=1.0, IG should exactly equal gradient x input
        assert torch.allclose(ig_attr, grad_attr, atol=1e-5)


class TestDLAAttribution:
    def test_matches_canonical(self, fake_model, fake_inputs):
        """DLA wrapper should match compute_attribution_tensor directly."""
        from splade.mechanistic.attribution import compute_attribution_tensor

        input_ids, attention_mask, targets = fake_inputs
        wrapper_attr = dla_attribution(fake_model, input_ids, attention_mask, targets)

        with torch.inference_mode():
            _, sparse, W_eff, _ = fake_model(input_ids, attention_mask)
        canonical_attr = compute_attribution_tensor(sparse, W_eff, targets).float()

        assert torch.allclose(wrapper_attr, canonical_attr, atol=1e-5)


class _FakeAttention(torch.nn.Module):
    """Fake attention module with Q/K projections (DistilBERT-style)."""

    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.q_lin = torch.nn.Linear(hidden_size, hidden_size)
        self.k_lin = torch.nn.Linear(hidden_size, hidden_size)
        self.n_heads = num_heads


class _FakeTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.attention = _FakeAttention(hidden_size, num_heads)

    def forward(self, x, *args, **kwargs):
        return x  # identity — just need the layer for hooks


class _FakeTransformerBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.layer = torch.nn.ModuleList([_FakeTransformerLayer(hidden_size, num_heads)])


class _FakeBERTWithAttention(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads=2):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = _FakeTransformerBlock(hidden_size, num_heads)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.embeddings(input_ids)
        for layer in self.transformer.layer:
            hidden = layer(hidden)

        class _Out:
            last_hidden_state = hidden

        return _Out()


class FakeLexicalSAEWithAttention(torch.nn.Module):
    """FakeLexicalSAE with BERT-like structure for testing attention_attribution."""

    def __init__(self, vocab_size=100, num_labels=2, hidden=32):
        super().__init__()
        self.vocab_size = vocab_size
        self._encoder = _FakeBERTWithAttention(hidden, vocab_size)
        self.vocab_transform = torch.nn.Linear(hidden, hidden)
        self.vocab_projector = torch.nn.Linear(hidden, vocab_size)
        self.vocab_layer_norm = torch.nn.LayerNorm(hidden)

        class _Act(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        self.activation = _Act()
        self.classifier_fc1 = torch.nn.Linear(vocab_size, hidden)
        self.classifier_fc2 = torch.nn.Linear(hidden, num_labels)

    @property
    def encoder(self):
        return self._encoder

    def _compute_sparse_sequence(self, attention_mask, *, input_ids=None, **kwargs):
        hidden = self._encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        transformed = self.vocab_transform(hidden)
        transformed = torch.nn.functional.gelu(transformed)
        transformed = self.vocab_layer_norm(transformed)
        mlm_logits = self.vocab_projector(transformed)
        activated = self.activation(mlm_logits)
        log_act = torch.log1p(activated)
        return log_act.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)


class TestAttentionAttribution:
    def test_output_shape(self):
        torch.manual_seed(42)
        model = FakeLexicalSAEWithAttention()
        input_ids = torch.randint(0, 100, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        targets = torch.zeros(4, dtype=torch.long)
        attr = attention_attribution(model, input_ids, attention_mask, targets)
        assert attr.shape == (4, 100)

    def test_nonzero(self):
        torch.manual_seed(42)
        model = FakeLexicalSAEWithAttention()
        input_ids = torch.randint(0, 100, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        targets = torch.zeros(4, dtype=torch.long)
        attr = attention_attribution(model, input_ids, attention_mask, targets)
        assert attr.abs().sum() > 0


class TestExplainerRegistry:
    def test_all_registered(self):
        expected = {"dla", "gradient", "integrated_gradients", "attention"}
        assert set(_EXPLAINERS.keys()) == expected

    def test_all_callable(self):
        for name, fn in _EXPLAINERS.items():
            assert callable(fn), f"{name} is not callable"
