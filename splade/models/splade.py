"""SPLADE model definition with shifted-ReLU activation on H100."""

import warnings

import torch
import torch.nn
import torch.nn.functional
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from splade.models.layers.activation import DReLU


def _get_nested_attr(obj, path: str):
    """Resolve a dot-separated attribute path."""
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


class SpladeModel(torch.nn.Module):
    """Encoder and classifier head for SPLADE features using H100 optimizations."""

    # Known MLM head attribute paths per model family
    _MLM_HEAD_PATHS = [
        # DistilBERT
        ("vocab_transform", "vocab_layer_norm", "vocab_projector"),
        # BERT / ALBERT
        ("cls.predictions.transform.dense", "cls.predictions.transform.LayerNorm", "cls.predictions.decoder"),
    ]

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size
        # Pad to multiple of 8 for Tensor Core alignment (30522 → 30528)
        self.padded_vocab_size = (self.vocab_size + 7) // 8 * 8

        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")

        # MLM head layers (padded for Tensor Core alignment)
        self.vocab_transform = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_projector = torch.nn.Linear(config.hidden_size, self.padded_vocab_size)
        self.vocab_layer_norm = torch.nn.LayerNorm(config.hidden_size)

        # Shifted-ReLU activation with learnable threshold (padded)
        self.activation = DReLU(self.padded_vocab_size)

        # Zero padded regions so they produce zero sparse activations
        if self.padded_vocab_size > self.vocab_size:
            with torch.no_grad():
                self.vocab_projector.weight[self.vocab_size:].zero_()
                self.vocab_projector.bias[self.vocab_size:].zero_()

        # Initialize weights from pre-trained MLM head into padded layers
        masked_lm = AutoModelForMaskedLM.from_pretrained(model_name)
        loaded = False
        for transform_path, norm_path, proj_path in self._MLM_HEAD_PATHS:
            try:
                self.vocab_transform.load_state_dict(
                    _get_nested_attr(masked_lm, transform_path).state_dict()
                )
                self.vocab_layer_norm.load_state_dict(
                    _get_nested_attr(masked_lm, norm_path).state_dict()
                )
                # Copy pretrained [V, H] weights into padded [V_pad, H] layer
                pretrained_proj = _get_nested_attr(masked_lm, proj_path)
                with torch.no_grad():
                    self.vocab_projector.weight[:self.vocab_size].copy_(pretrained_proj.weight)
                    self.vocab_projector.bias[:self.vocab_size].copy_(pretrained_proj.bias)
                loaded = True
                break
            except (AttributeError, RuntimeError):
                continue
        if not loaded:
            warnings.warn(
                f"Could not load MLM head weights from {model_name}; using random initialization"
            )
        del masked_lm

        self.classifier = torch.nn.Linear(self.padded_vocab_size, num_labels)

    def _splade_head(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MLM head → shifted-ReLU → log1p → max-pool → classify."""
        transformed = self.vocab_transform(hidden)
        transformed = torch.nn.functional.gelu(transformed)
        transformed = self.vocab_layer_norm(transformed)
        mlm_logits = self.vocab_projector(transformed)

        activated = self.activation(mlm_logits)

        log_activations = torch.log1p(activated)
        masked_activations = log_activations.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        sparse_vector = masked_activations.max(dim=1).values

        return self.classifier(sparse_vector), sparse_vector

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self._splade_head(hidden, attention_mask)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings before the transformer layers."""
        return self.bert.embeddings(input_ids)

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass starting from pre-computed embeddings."""
        hidden = self.bert(
            inputs_embeds=embeddings, attention_mask=attention_mask
        ).last_hidden_state
        return self._splade_head(hidden, attention_mask)