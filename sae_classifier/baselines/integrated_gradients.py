"""Integrated Gradients explainer using Captum."""

import torch
from captum.attr import LayerIntegratedGradients

from sae_classifier.baselines.base import BaseExplainer


class IntegratedGradientsExplainer(BaseExplainer):
    """Generate explanations using Integrated Gradients."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        device: str = "auto",
        n_steps: int = 50,
    ):
        super().__init__(model_name, num_labels, max_length, device)
        self.n_steps = n_steps
        self._lig = None

    def _init_lig(self):
        """Initialize LayerIntegratedGradients."""
        def forward_func(inputs_embeds, attention_mask):
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return outputs.logits

        embedding_layer = self.model.distilbert.embeddings
        self._lig = LayerIntegratedGradients(forward_func, embedding_layer)

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input IDs."""
        return self.model.distilbert.embeddings(input_ids)

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate Integrated Gradients explanation."""
        if self._lig is None:
            self._init_lig()

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        input_embeds = self._get_embeddings(input_ids)

        pad_id = self.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = self._get_embeddings(baseline_ids)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred_class = outputs.logits.argmax(dim=-1).item()

        attributions = self._lig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(attention_mask,),
            target=pred_class,
            n_steps=self.n_steps,
        )

        token_attributions = attributions.sum(dim=-1).squeeze(0)
        seq_len = attention_mask.sum().item()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())
        attrs = token_attributions[:seq_len].cpu().numpy()

        explanations = []
        for token, attr in zip(tokens, attrs):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                clean_token = token.replace("##", "")
                explanations.append((clean_token, float(abs(attr))))

        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]
