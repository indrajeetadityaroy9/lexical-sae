"""SHAP-based explainer for text classification."""

import numpy as np
import shap

from sae_classifier.baselines.base import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """Generate explanations using SHAP."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        device: str = "auto",
        max_evals: int = 500,
    ):
        super().__init__(model_name, num_labels, max_length, device)
        self.max_evals = max_evals
        self._shap_explainer = None

    def _init_shap_explainer(self):
        """Initialize SHAP explainer."""
        self._shap_explainer = shap.Explainer(
            self._predict_fn,
            self.tokenizer,
            output_names=[f"class_{i}" for i in range(self.num_labels)],
        )

    def _predict_fn(self, texts):
        """Prediction function for SHAP."""
        if isinstance(texts, str):
            texts = [texts]
        elif hasattr(texts, 'tolist'):
            texts = texts.tolist()
        probs = self.predict_proba(list(texts))
        return np.array(probs)

    def fit(self, texts: list[str], labels: list[int], epochs: int = 3,
            batch_size: int = 32, learning_rate: float = 2e-5) -> "SHAPExplainer":
        """Fine-tune model and initialize SHAP explainer."""
        super().fit(texts, labels, epochs, batch_size, learning_rate)
        self._init_shap_explainer()
        return self

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate SHAP explanation."""
        if self._shap_explainer is None:
            self._init_shap_explainer()

        probs = self.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))

        shap_values = self._shap_explainer([text], max_evals=self.max_evals)

        if hasattr(shap_values, 'values'):
            values = shap_values.values[0]
            if len(values.shape) > 1:
                values = values[:, pred_class]
        else:
            values = shap_values[0]

        if hasattr(shap_values, 'data'):
            tokens = shap_values.data[0]
        else:
            tokens = text.split()

        explanations = []
        for token, value in zip(tokens, values):
            if token and token.strip():
                explanations.append((str(token), float(value)))

        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanations[:top_k]
