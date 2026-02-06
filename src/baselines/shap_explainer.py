"""SHAP-based explainer for text classification."""

import numpy as np
import shap
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from src.baselines.base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """Generate explanations using SHAP."""

    def __init__(
        self, model: DistilBertForSequenceClassification,
        tokenizer: AutoTokenizer, num_labels: int, max_length: int = 128,
        max_evals: int = 500,
    ):
        super().__init__(model, tokenizer, num_labels, max_length)
        self.max_evals = max_evals
        self._shap_explainer = shap.Explainer(
            self._predict_fn, self.tokenizer,
            output_names=[f"class_{i}" for i in range(self.num_labels)],
        )

    def _predict_fn(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.array(self.predict_proba(list(texts)))

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Compute SHAP values for the predicted class."""
        probs = self.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))

        shap_values = self._shap_explainer([text], max_evals=self.max_evals)

        values = shap_values.values[0][:, pred_class]

        tokens = shap_values.data[0]

        explanations = [
            (str(token), float(value))
            for token, value in zip(tokens, values)
            if token and token.strip()
        ]
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanations[:top_k]
