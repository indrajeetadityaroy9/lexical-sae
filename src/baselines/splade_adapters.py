"""SPLADE adapter explainers for fair baseline comparison.

All adapters explain the SPLADE model's predictions (using its internal
DistilBERT), so faithfulness metrics are measured against the same model.
This eliminates the confound of comparing explanations across different models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from src.baselines import SPECIAL_TOKENS
from src.utils.cuda import DEVICE

if TYPE_CHECKING:
    from src.models import SPLADEClassifier


class SPLADEAttentionExplainer:
    """Attention-based explanations using SPLADE's internal DistilBERT."""

    def __init__(self, clf: SPLADEClassifier):
        self.clf = clf
        self.tokenizer = clf.tokenizer
        self.num_labels = clf.num_labels

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        enc = self.tokenizer(
            text, max_length=self.clf.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)

        with torch.inference_mode():
            outputs = self.clf.model.bert(
                input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=True,
            )

        attentions = outputs.attentions[-1][0]  # [heads, seq, seq]
        seq_len = int(attention_mask.sum().item())
        attn = attentions.mean(dim=0)  # [seq, seq]
        token_importance = attn[:seq_len, :seq_len].sum(dim=0)

        importance = token_importance.cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())

        explanations = [
            (token.replace("##", ""), float(weight))
            for token, weight in zip(tokens, importance)
            if token not in SPECIAL_TOKENS
        ]
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]


class SPLADEIntegratedGradientsExplainer:
    """Integrated Gradients explanations using SPLADE's DistilBERT embeddings."""

    def __init__(self, clf: SPLADEClassifier, n_steps: int = 50):
        self.clf = clf
        self.tokenizer = clf.tokenizer
        self.num_labels = clf.num_labels
        self.n_steps = n_steps

        from captum.attr import LayerIntegratedGradients

        def forward_func(input_ids, attention_mask):
            logits, _ = self.clf.model(input_ids, attention_mask)
            return logits

        # torch.compile wraps bert in OptimizedModule; Captum hooks need the
        # original submodule to fire during forward passes.
        bert = getattr(clf.model.bert, '_orig_mod', clf.model.bert)
        self._lig = LayerIntegratedGradients(
            forward_func, bert.embeddings.word_embeddings,
        )

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        enc = self.tokenizer(
            text, max_length=self.clf.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        with torch.no_grad():
            logits, _ = self.clf.model(input_ids, attention_mask)
            pred_class = logits.argmax(dim=-1).item()

        # Temporarily unwrap compiled BERT so Captum hooks fire correctly
        compiled_bert = self.clf.model.bert
        orig_bert = getattr(compiled_bert, '_orig_mod', compiled_bert)
        self.clf.model.bert = orig_bert
        try:
            attributions = self._lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=pred_class,
                n_steps=self.n_steps,
            )
        finally:
            self.clf.model.bert = compiled_bert

        token_attributions = attributions.sum(dim=-1).squeeze(0)
        seq_len = int(attention_mask.sum().item())
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())
        attrs = token_attributions[:seq_len].cpu().detach().numpy()

        explanations = [
            (token.replace("##", ""), float(abs(attr)))
            for token, attr in zip(tokens, attrs)
            if token not in SPECIAL_TOKENS
        ]
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]


class SPLADELIMEExplainer:
    """LIME explanations using SPLADE as the black-box model."""

    def __init__(self, clf: SPLADEClassifier, num_samples: int = 500):
        self.clf = clf
        self.tokenizer = clf.tokenizer
        self.num_labels = clf.num_labels
        self.num_samples = num_samples
        self.lime_explainer = LimeTextExplainer(
            class_names=[f"class_{i}" for i in range(clf.num_labels)],
        )

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        def predict_fn(texts):
            text_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
            return np.array(self.clf.predict_proba(text_list))

        probs = self.clf.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))

        exp = self.lime_explainer.explain_instance(
            text, predict_fn, num_features=top_k,
            num_samples=self.num_samples, labels=[pred_class],
        )
        explanations = exp.as_list(label=pred_class)
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return [(word, float(weight)) for word, weight in explanations[:top_k]]
