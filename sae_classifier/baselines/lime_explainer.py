"""LIME-based explainer for text classification."""

import numpy as np
from lime.lime_text import LimeTextExplainer

from sae_classifier.baselines.base import BaseExplainer


class LIMEExplainer(BaseExplainer):
    """Generate explanations using LIME."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        device: str = "auto",
        num_samples: int = 500,
        class_names: list[str] | None = None,
    ):
        super().__init__(model_name, num_labels, max_length, device)

        self.num_samples = num_samples
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_labels)]
        self.lime_explainer = LimeTextExplainer(class_names=class_names)

    def _predict_fn(self, texts: list[str]) -> np.ndarray:
        """Prediction function for LIME."""
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        probs = self.predict_proba(texts)
        return np.array(probs)

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate LIME explanation."""
        probs = self.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))

        exp = self.lime_explainer.explain_instance(
            text,
            self._predict_fn,
            num_features=top_k,
            num_samples=self.num_samples,
            labels=[pred_class],
        )

        explanations = exp.as_list(label=pred_class)
        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return [(word, float(weight)) for word, weight in explanations[:top_k]]
