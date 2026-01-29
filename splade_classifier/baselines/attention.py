"""Attention-based explainer using DistilBERT attention weights."""

import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer

from splade_classifier.baselines.base import BaseExplainer


class AttentionExplainer(BaseExplainer):
    """Generate explanations from DistilBERT attention weights.

    Extracts attention from the final layer and aggregates across heads
    to produce token-level importance scores.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        device: str = "auto",
        aggregation: str = "mean",
        layer: int = -1,
    ):
        """Initialize attention explainer.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            max_length: Maximum sequence length
            device: Device specification
            aggregation: How to aggregate attention heads ("mean", "max", "cls")
            layer: Which layer to extract attention from (-1 = last)
        """
        # Initialize base (creates model without output_attentions)
        super().__init__(model_name, num_labels, max_length, device)

        self.aggregation = aggregation
        self.layer = layer

        # Recreate model with attention output enabled
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True,
        ).to(self.device)

    def _get_attention_weights(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Extract attention weights for a single text.

        Returns:
            Tuple of (attention_weights, tokens) where attention_weights
            is shape [seq_len] and tokens is the list of token strings.
        """
        # Tokenize
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Get attention from specified layer
        # Shape: [batch, heads, seq_len, seq_len]
        attentions = outputs.attentions[self.layer][0]  # Remove batch dim

        # Get actual sequence length (non-padding)
        seq_len = attention_mask.sum().item()

        # Aggregate across heads
        if self.aggregation == "mean":
            # Mean across all heads
            attn = attentions.mean(dim=0)  # [seq_len, seq_len]
        elif self.aggregation == "max":
            # Max across heads
            attn = attentions.max(dim=0).values
        elif self.aggregation == "cls":
            # Attention from [CLS] token only
            attn = attentions.mean(dim=0)  # Average heads first
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # For classification, use attention TO [CLS] or FROM [CLS]
        # We use attention FROM [CLS] to each token as importance
        if self.aggregation == "cls":
            token_importance = attn[0, :seq_len]  # [CLS] attending to others
        else:
            # Use column sum (how much each token is attended to)
            token_importance = attn[:seq_len, :seq_len].sum(dim=0)

        # Convert to numpy and get tokens
        importance = token_importance.cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())

        return importance, tokens

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate attention-based explanation.

        Args:
            text: Input text
            top_k: Number of top tokens to return

        Returns:
            List of (token, weight) tuples sorted by attention weight
        """
        importance, tokens = self._get_attention_weights(text)

        # Create (token, weight) pairs, excluding special tokens
        explanations = []
        for token, weight in zip(tokens, importance):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                # Clean up subword tokens
                clean_token = token.replace("##", "")
                explanations.append((clean_token, float(weight)))

        # Sort by weight descending and take top-k
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]
