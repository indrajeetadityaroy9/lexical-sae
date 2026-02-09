"""Tests for layerwise attribution decomposition."""

import torch
import pytest

from splade.mechanistic.layerwise import (
    _get_transformer_layers,
    decompose_sparse_vector_by_layer,
    compute_layerwise_attribution,
)


class FakeTransformerLayer(torch.nn.Module):
    """Minimal transformer layer with residual connection."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, *args, **kwargs):
        # Residual connection: output = input + transform(input)
        return x + torch.relu(self.linear(x))


class FakeTransformer(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layer = torch.nn.ModuleList(
            [FakeTransformerLayer(hidden_size) for _ in range(num_layers)]
        )


class FakeBERT(torch.nn.Module):
    def __init__(self, hidden_size=64, vocab_size=100, num_layers=3):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = FakeTransformer(hidden_size, num_layers)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.embeddings(input_ids)
        for layer in self.transformer.layer:
            hidden = layer(hidden)

        class Output:
            last_hidden_state = hidden

        return Output()


class FakeLexicalSAEWithBERT(torch.nn.Module):
    """Minimal LexicalSAE with fake BERT for testing layerwise decomposition."""

    def __init__(self, hidden_size=64, vocab_size=100, num_layers=3, num_labels=2):
        super().__init__()
        self._encoder = FakeBERT(hidden_size, vocab_size, num_layers)
        self.vocab_size = vocab_size
        self.vocab_transform = torch.nn.Linear(hidden_size, hidden_size)
        self.vocab_projector = torch.nn.Linear(hidden_size, vocab_size)
        self.vocab_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.classifier_fc1 = torch.nn.Linear(vocab_size, 32)
        self.classifier_fc2 = torch.nn.Linear(32, num_labels)

        class _DReLU(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        self.activation = _DReLU()

    @property
    def encoder(self):
        return self._encoder

    def forward(self, input_ids, attention_mask):
        hidden = self._encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        transformed = self.vocab_transform(hidden)
        transformed = torch.nn.functional.gelu(transformed)
        transformed = self.vocab_layer_norm(transformed)
        mlm_logits = self.vocab_projector(transformed)
        activated = self.activation(mlm_logits)
        log_act = torch.log1p(activated)
        log_act = log_act.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0.0)
        sparse_vector = log_act.max(dim=1).values

        pre_relu = self.classifier_fc1(sparse_vector)
        activation_mask = (pre_relu > 0).float()
        h = pre_relu * activation_mask
        logits = self.classifier_fc2(h)

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
def model():
    torch.manual_seed(42)
    return FakeLexicalSAEWithBERT(num_layers=3)


@pytest.fixture
def inputs():
    B = 2
    return (
        torch.randint(0, 100, (B, 8)),
        torch.ones(B, 8, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
    )


class TestGetTransformerLayers:
    def test_finds_layers(self, model):
        layers = _get_transformer_layers(model)
        assert len(layers) == 3


class TestDecomposeByLayer:
    def test_output_shape(self, model, inputs):
        input_ids, attention_mask, _ = inputs
        contributions = decompose_sparse_vector_by_layer(model, input_ids, attention_mask)
        assert contributions.shape == (2, 3, 100)  # [B, num_layers, V]

    def test_contributions_nonzero(self, model, inputs):
        input_ids, attention_mask, _ = inputs
        contributions = decompose_sparse_vector_by_layer(model, input_ids, attention_mask)
        assert contributions.abs().sum() > 0


class TestLayerwiseAttribution:
    def test_output_shape(self, model, inputs):
        input_ids, attention_mask, targets = inputs
        importance = compute_layerwise_attribution(
            model, input_ids, attention_mask, targets,
        )
        assert importance.shape == (2, 3)  # [B, num_layers]

    def test_importance_nonnegative(self, model, inputs):
        input_ids, attention_mask, targets = inputs
        importance = compute_layerwise_attribution(
            model, input_ids, attention_mask, targets,
        )
        assert (importance >= 0).all()
