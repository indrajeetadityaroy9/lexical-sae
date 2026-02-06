"""Integrated Gradients explainer using Captum."""

import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from src.baselines.base import BaseExplainer
from src.cuda import DEVICE


class IntegratedGradientsExplainer(BaseExplainer):
    """Generate explanations using Integrated Gradients."""

    def __init__(
        self, model: DistilBertForSequenceClassification,
        tokenizer: AutoTokenizer, num_labels: int, max_length: int = 128,
        n_steps: int = 50,
    ):
        super().__init__(model, tokenizer, num_labels, max_length)
        self.n_steps = n_steps

        def forward_func(input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        self._lig = LayerIntegratedGradients(forward_func, self.model.distilbert.embeddings.word_embeddings)

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Compute integrated gradients relative to a pad-token baseline."""
        enc = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)

        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        with torch.no_grad():
            pred_class = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1).item()

        attributions = self._lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred_class,
            n_steps=self.n_steps,
        )

        token_attributions = attributions.sum(dim=-1).squeeze(0)
        seq_len = int(attention_mask.sum().item())

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())
        attrs = token_attributions[:seq_len].cpu().detach().numpy()

        explanations = [
            (token.replace("##", ""), float(abs(attr)))
            for token, attr in zip(tokens, attrs)
            if token not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]
