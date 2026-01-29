"""SPLADE-based explainer as baseline for comparison.

Wraps the original SPLADE classifier to provide explanations based on
sparse vocabulary projections. This serves as a baseline to compare
against SAE-based explanations.
"""

import sys
import torch
import numpy as np
from pathlib import Path

from sae_classifier.baselines.base import BaseExplainer


class SPLADEExplainer(BaseExplainer):
    """Generate explanations using SPLADE sparse vocabulary representations.

    SPLADE projects text to sparse vocabulary-sized vectors where each dimension
    corresponds to a vocabulary term. Explanations are the top-k terms by weight.

    This is included as a baseline to compare against SAE-based explanations.
    Unlike SAE features which are learned, SPLADE dimensions are predetermined
    by the tokenizer vocabulary.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        device: str = "auto",
        target_sparsity: float = 0.95,
    ):
        """Initialize SPLADE explainer.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            max_length: Maximum sequence length
            device: Device specification
            target_sparsity: Target sparsity for SPLADE regularization
        """
        # Don't call super().__init__ as we use a different model architecture
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.target_sparsity = target_sparsity

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._splade_clf = None
        self._is_fitted = False

    def _ensure_splade_imported(self):
        """Lazily import SPLADE classifier."""
        if self._splade_clf is not None:
            return

        # Add splade_classifier to path if needed
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from splade_classifier import SPLADEClassifier
            self._splade_clf = SPLADEClassifier(
                num_labels=self.num_labels,
                model_name=self.model_name,
                max_length=self.max_length,
                target_sparsity=self.target_sparsity,
                device=str(self.device),
            )
        except ImportError as e:
            raise ImportError(
                "SPLADE classifier not available. "
                "Ensure splade_classifier package is in the path."
            ) from e

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float | None = None,
    ) -> "SPLADEExplainer":
        """Fine-tune SPLADE model on classification data."""
        self._ensure_splade_imported()

        self._splade_clf.batch_size = batch_size
        self._splade_clf.epochs = epochs
        if learning_rate is not None:
            self._splade_clf.learning_rate = learning_rate

        self._splade_clf.fit(texts, labels)
        self._is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        """Get class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._splade_clf.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[int]:
        """Get class predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._splade_clf.predict(texts)

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate SPLADE explanation (top vocabulary terms by weight).

        Args:
            text: Input text
            top_k: Number of top terms to return

        Returns:
            List of (token, weight) tuples sorted by weight descending
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._splade_clf.explain(text, top_k=top_k)

    def get_expansion_terms(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Get SPLADE semantic expansion terms (terms not in input)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._splade_clf.get_expansion_terms(text, top_k=top_k)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Get sparse SPLADE vectors [n_samples, vocab_size]."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._splade_clf.transform(texts)
