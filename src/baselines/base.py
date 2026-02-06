"""Base class for post-hoc explainers."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DistilBertForSequenceClassification

from src.cuda import DEVICE


def train_shared_model(
    model_name: str, num_labels: int, texts: list[str], labels: list[int],
    epochs: int = 3, batch_size: int = 32, learning_rate: float = 2e-5,
    max_length: int = 128,
) -> tuple[DistilBertForSequenceClassification, AutoTokenizer]:
    """Train a single DistilBERT model shared by all baseline explainers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(DEVICE)

    enc = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(enc["input_ids"], enc["attention_mask"], labels_tensor),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, batch_labels in tqdm(loader, desc=f"Shared model epoch {epoch+1}"):
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        print(f"Shared model epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    model.eval()
    return model, tokenizer


class BaseExplainer(ABC):
    """Base class for all explainers using a shared pre-trained model."""

    def __init__(
        self, model: DistilBertForSequenceClassification,
        tokenizer: AutoTokenizer, num_labels: int, max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length
        self.model.eval()

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        """Predict class probabilities."""
        enc = self._tokenize(texts)
        with torch.no_grad():
            input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = enc["attention_mask"].to(DEVICE, non_blocking=True)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs.cpu().tolist()

    @abstractmethod
    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate token-level explanations."""
        pass
