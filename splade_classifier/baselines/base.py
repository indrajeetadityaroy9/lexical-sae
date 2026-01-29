"""Base class for explainers."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm


class BaseExplainer(ABC):
    """Base class for all explainers."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 128,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize texts."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
    ) -> "BaseExplainer":
        """Fine-tune the model on classification data."""
        enc = self._tokenize(texts)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], labels_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for input_ids, attention_mask, batch_labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

        self.model.eval()
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        """Get class probabilities."""
        enc = self._tokenize(texts)

        with torch.no_grad():
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().tolist()

    def predict(self, texts: list[str]) -> list[int]:
        """Get class predictions."""
        probs = self.predict_proba(texts)
        return [max(range(len(p)), key=lambda i: p[i]) for p in probs]

    @abstractmethod
    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Generate explanation for a single text."""
        pass
