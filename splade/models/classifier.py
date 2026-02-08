"""Sklearn-style wrapper for SPLADE classifier."""

import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoTokenizer

from splade.models.splade import SpladeModel
from splade.inference import (
    predict_model,
    predict_proba_model,
    transform_model,
    explain_model,
    score_model,
)
from splade.data.loader import infer_max_length
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import DEVICE, set_seed


class SPLADEClassifier(BaseEstimator, ClassifierMixin):
    """SPLADE classifier with sklearn-compatible API.

    All training hyperparameters are auto-inferred from data and model
    architecture. No manual tuning required.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.max_length = None
        self.batch_size = None

    def _ensure_initialized(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = SpladeModel(self.model_name, self.num_labels).to(DEVICE)
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def fit(self, X: list[str], y: list[int], validation_data=None):
        """Train the SPLADE model. All hyperparameters are auto-inferred."""
        self._ensure_initialized()

        from splade.training.loop import train_model

        # Auto-infer max_length and batch_size from data
        self.max_length = infer_max_length(X, self.tokenizer)
        self.batch_size = _infer_batch_size(self.model_name, self.max_length)

        val_texts, val_labels = None, None
        if validation_data:
            val_texts, val_labels = validation_data

        train_model(
            self.model, self.tokenizer, X, y,
            model_name=self.model_name, num_labels=self.num_labels,
            val_texts=val_texts, val_labels=val_labels,
        )
        return self

    def predict(self, X: list[str]) -> list[int]:
        self._ensure_initialized()
        return predict_model(
            self.model, self.tokenizer, X,
            self.max_length or 128, self.batch_size or 32, self.num_labels,
        )

    def predict_proba(self, X: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        return predict_proba_model(
            self.model, self.tokenizer, X,
            self.max_length or 128, self.batch_size or 32, self.num_labels,
        )

    def transform(self, X: list[str]) -> numpy.ndarray:
        """Return the sparse SPLADE embeddings."""
        self._ensure_initialized()
        return transform_model(
            self.model, self.tokenizer, X,
            self.max_length or 128, self.batch_size or 32,
        )

    def score(self, X: list[str], y: list[int], sample_weight=None) -> float:
        self._ensure_initialized()
        return score_model(
            self.model, self.tokenizer, X, y,
            self.max_length or 128, self.batch_size or 32, self.num_labels,
        )

    def explain(self, text: str, top_k: int = 10, target_class: int | None = None) -> list[tuple[str, float]]:
        """Return lexical explanations for a single instance."""
        self._ensure_initialized()
        return explain_model(
            self.model, self.tokenizer, text,
            self.max_length or 128, top_k, target_class, input_only=True,
        )

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self._ensure_initialized()
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
