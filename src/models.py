"""SPLADE classifier for sklearn-compatible text classification."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, DistilBertForMaskedLM

from src.cuda import COMPUTE_DTYPE, DEVICE
from src.kernels import (
    _EPS,
    DFFlopsRegFunction,
    DocumentFrequencyTracker,
    SpladeAggregateFunction,
    splade_aggregate,
)


class _LRScheduler:
    def __init__(self, base_lr: float, num_samples: int, batch_size: int, epochs: int):
        steps_per_epoch = -(-num_samples // batch_size)
        self.base_lr = base_lr
        self.warmup_steps = int(steps_per_epoch * 0.1)
        self.total_steps = steps_per_epoch * epochs
        self._step = 0

    def step(self) -> float:
        s = self._step
        self._step += 1
        if s < self.warmup_steps:
            return self.base_lr * (s / self.warmup_steps)
        progress = (s - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class _LambdaSchedule:
    """Quadratic lambda warmup for FLOPS regularization (arXiv:2505.15070)."""

    def __init__(self, warmup_steps: int, lambda_init: float = 0.1, lambda_peak: float = 10.0):
        self.lambda_init = lambda_init
        self.lambda_peak = lambda_peak
        self.warmup_steps = warmup_steps
        self._step = 0
        self._current_sparsity = 0.0

    def compute_lambda(self, activations: torch.Tensor) -> float:
        with torch.no_grad():
            self._current_sparsity = (activations.abs() < _EPS[activations.dtype]['div']).float().mean().item()

        if self._step < self.warmup_steps:
            progress = self._step / self.warmup_steps
            lam = self.lambda_init + (self.lambda_peak - self.lambda_init) * (progress ** 2)
        else:
            lam = self.lambda_peak

        self._step += 1
        return lam


class _SpladeEncoder(nn.Module):
    def __init__(self, model_name: str, num_labels: int, classifier_dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = config.vocab_size
        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size)
        self.vocab_projector = nn.Linear(config.hidden_size, self.vocab_size)

        mlm = DistilBertForMaskedLM.from_pretrained(model_name)
        self.vocab_transform.load_state_dict(mlm.vocab_transform.state_dict())
        self.vocab_layer_norm.load_state_dict(mlm.vocab_layer_norm.state_dict())
        self.vocab_projector.load_state_dict(mlm.vocab_projector.state_dict())
        del mlm

        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.vocab_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        transformed = self.vocab_layer_norm(F.gelu(self.vocab_transform(hidden)))
        mlm_logits = self.vocab_projector(transformed)

        sparse_vec = SpladeAggregateFunction.apply(mlm_logits, attention_mask) if self.training else splade_aggregate(mlm_logits, attention_mask)
        return self.classifier(self.classifier_dropout(sparse_vec)), sparse_vec


class SPLADEClassifier:
    """sklearn-compatible SPLADE classifier with DF-FLOPS regularization (arXiv:2505.15070)."""

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        df_alpha: float = 0.1,
        df_beta: float = 5.0,
        classifier_dropout: float = 0.1,
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.df_alpha = df_alpha
        self.df_beta = df_beta

        self.model = _SpladeEncoder(model_name, num_labels, classifier_dropout).to(DEVICE)
        self.model.bert = torch.compile(self.model.bert, mode="reduce-overhead", fullgraph=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )

    def _run_inference_loop(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        extract_sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        loader = DataLoader(
            TensorDataset(input_ids, attention_mask),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
        all_logits = []
        all_sparse = [] if extract_sparse else None

        with torch.inference_mode():
            for batch_ids, batch_mask in loader:
                batch_ids = batch_ids.to(DEVICE, non_blocking=True)
                batch_mask = batch_mask.to(DEVICE, non_blocking=True)
                logits, sparse = self.model(batch_ids, batch_mask)
                all_logits.append(logits.cpu())
                if extract_sparse:
                    all_sparse.append(sparse.cpu())

        logits = torch.cat(all_logits, dim=0)
        sparse = torch.cat(all_sparse, dim=0) if extract_sparse else None
        return logits, sparse

    @staticmethod
    def _validate_texts(X: list[str], name: str = "X") -> None:
        if not X:
            raise ValueError(f"{name} must be a non-empty list of strings")
        if not all(isinstance(x, str) for x in X):
            raise TypeError(f"All elements of {name} must be strings")

    def fit(self, X: list[str], y, epochs: int | None = None) -> "SPLADEClassifier":
        self._validate_texts(X)
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
        if epochs is None:
            epochs = self.epochs

        steps_per_epoch = -(-len(X) // self.batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(int(total_steps * 0.3), 1)
        self._lambda_schedule = _LambdaSchedule(warmup_steps=warmup_steps)
        self._df_tracker = DocumentFrequencyTracker(
            vocab_size=self.model.vocab_size, device=DEVICE,
        )

        y_tensor = torch.tensor(y, dtype=torch.float32 if self.num_labels == 1 else torch.long)
        enc = self._tokenize(X)
        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], y_tensor),
            batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True,
        )

        no_decay = {"bias", "LayerNorm.weight", "vocab_layer_norm.weight"}
        param_groups = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)
        self._lr_scheduler = _LRScheduler(
            base_lr=self.learning_rate, num_samples=len(X),
            batch_size=self.batch_size, epochs=epochs,
        )
        criterion = nn.CrossEntropyLoss() if self.num_labels > 1 else nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss, num_batches = 0.0, 0
            for input_ids, mask, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = input_ids.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                lr = self._lr_scheduler.step()
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', dtype=COMPUTE_DTYPE):
                    logits, sparse = self.model(input_ids, mask)
                    lam = self._lambda_schedule.compute_lambda(sparse)
                    self._df_tracker.update(sparse)
                    df_weights = self._df_tracker.get_weights(alpha=self.df_alpha, beta=self.df_beta)
                    reg_loss = lam * DFFlopsRegFunction.apply(sparse, df_weights)
                    cls_loss = criterion(logits.squeeze(-1), labels) if self.num_labels == 1 else criterion(logits, labels.view(-1))
                    loss = cls_loss + reg_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            sparsity = self._lambda_schedule._current_sparsity
            stats = self._df_tracker.get_stats()
            print(f"Epoch {epoch+1}: Loss = {total_loss/num_batches:.4f}, Sparsity: {sparsity:.2%}, Top-1 DF: {stats['top1_df_pct']:.1f}%")

        self.model.eval()
        return self

    def predict_proba(self, X: list[str]) -> list[list[float]]:
        self._validate_texts(X)
        enc = self._tokenize(X)
        logits, _ = self._run_inference_loop(enc["input_ids"], enc["attention_mask"])
        if self.num_labels == 1:
            p = torch.sigmoid(logits).squeeze(-1)
            probs = torch.stack([1 - p, p], dim=1)
        else:
            probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    def predict(self, X: list[str]) -> list[int]:
        self._validate_texts(X)
        enc = self._tokenize(X)
        logits, _ = self._run_inference_loop(enc["input_ids"], enc["attention_mask"])
        if self.num_labels == 1:
            return (logits.squeeze(-1) > 0).int().tolist()
        return logits.argmax(dim=1).tolist()

    def score(self, X: list[str], y) -> float:
        self._validate_texts(X)
        if not y:
            raise ValueError("y must be non-empty")
        return sum(p == t for p, t in zip(self.predict(X), y)) / len(y)

    def transform(self, X: list[str]) -> np.ndarray:
        """Returns sparse SPLADE vectors [n_samples, vocab_size]."""
        self._validate_texts(X)
        enc = self._tokenize(X)
        _, sparse = self._run_inference_loop(enc["input_ids"], enc["attention_mask"], extract_sparse=True)
        return sparse.numpy()

    _SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"}

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns top-k (token, weight) pairs explaining the prediction."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        sparse_vec = self.transform([text])[0]
        top_indices = np.argsort(sparse_vec)[-(top_k + len(self._SPECIAL_TOKENS)):][::-1]
        results = []
        for idx in top_indices:
            if sparse_vec[idx] <= 0:
                break
            token = self.tokenizer.convert_ids_to_tokens(int(idx))
            if token not in self._SPECIAL_TOKENS:
                results.append((token, float(sparse_vec[idx])))
            if len(results) >= top_k:
                break
        return results

