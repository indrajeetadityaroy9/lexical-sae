"""Attention-based explainer using DistilBERT attention weights."""

import numpy as np
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from src.baselines.base import BaseExplainer
from src.cuda import DEVICE


class AttentionExplainer(BaseExplainer):
    """Explanations from DistilBERT last-layer attention weights."""

    def __init__(
        self, model: DistilBertForSequenceClassification,
        tokenizer: AutoTokenizer, num_labels: int, max_length: int = 128,
    ):
        super().__init__(model, tokenizer, num_labels, max_length)
        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Extract and aggregate attention weights for token importance."""
        enc = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

        attentions = outputs.attentions[-1][0]
        seq_len = int(attention_mask.sum().item())
        attn = attentions.mean(dim=0)
        token_importance = attn[:seq_len, :seq_len].sum(dim=0)

        importance = token_importance.cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())

        explanations = [
            (token.replace("##", ""), float(weight))
            for token, weight in zip(tokens, importance)
            if token not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]
