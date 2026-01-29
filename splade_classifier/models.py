"""SPLADE classifier with sklearn-compatible API."""

import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, DistilBertForMaskedLM
from tqdm import tqdm

from splade_classifier.kernels import splade_aggregate, SpladeAggregateFunction, FlopsRegFunction, SelfNormalizingFlopsRegFunction
from splade_classifier.adaptive import StableOps, compute_base_lr, AdaptiveLRScheduler, AdaptiveFlopsReg


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_device(device: str) -> torch.device:
    """Get torch device from string specification.

    Args:
        device: "auto", "cuda", "cuda:N", or "cpu"

    Returns:
        torch.device object
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _SpladeEncoder(nn.Module):
    """DistilBERT + MLM head -> log1p(relu) -> max-pool -> classifier."""

    def __init__(self, model_name: str, num_labels: int):
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

        self.classifier = nn.Linear(self.vocab_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        transformed = self.vocab_layer_norm(F.gelu(self.vocab_transform(hidden)))
        mlm_logits = self.vocab_projector(transformed)

        sparse_vec = SpladeAggregateFunction.apply(mlm_logits, attention_mask) if self.training else splade_aggregate(mlm_logits, attention_mask)
        return self.classifier(sparse_vec), sparse_vec


class SPLADEClassifier:
    """sklearn-compatible SPLADE classifier.

    A lean, parameter-free text classifier using sparse vocabulary representations.
    Learning rate and FLOPS regularization are automatically tuned from data.

    Args:
        num_labels: Number of output classes (default: 2)
        model_name: HuggingFace model name (default: "distilbert-base-uncased")
        batch_size: Training and inference batch size (default: 32)
        epochs: Number of training epochs (default: 3)
        learning_rate: Learning rate. None = auto-compute from model statistics (default: None)
        target_sparsity: Target sparsity for adaptive FLOPS regularization (default: 0.95)
        max_length: Maximum sequence length for tokenization (default: 128)
        device: Device specification: "auto", "cuda", "cuda:N", or "cpu" (default: "auto")

    Example:
        >>> clf = SPLADEClassifier(num_labels=4)
        >>> clf.fit(texts, labels)
        >>> predictions = clf.predict(test_texts)
        >>> explanation = clf.explain("sample text")
    """

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float | None = None,
        target_sparsity: float = 0.95,
        max_length: int = 128,
        device: str = "auto",
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.target_sparsity = target_sparsity
        self.max_length = max_length

        # Device setup
        self.device = _get_device(device)
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        # Model setup
        self.model = _SpladeEncoder(model_name, num_labels)
        self.model = self.model.to(self.device)

        # Apply torch.compile on CUDA
        if self.device.type == "cuda":
            self.model.bert = torch.compile(self.model.bert, mode="reduce-overhead", fullgraph=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Adaptive components (initialized in fit())
        self._lr_scheduler = None
        self._adaptive_flops = None
        self._computed_lr = None

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize texts."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _run_inference_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        extract_sparse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Unified inference loop."""
        loader = DataLoader(
            TensorDataset(input_ids, attention_mask),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        all_logits = []
        all_sparse = [] if extract_sparse else None

        with torch.inference_mode():
            for batch_ids, batch_mask in loader:
                batch_ids = batch_ids.to(self.device, non_blocking=True)
                batch_mask = batch_mask.to(self.device, non_blocking=True)
                logits, sparse = self.model(batch_ids, batch_mask)
                all_logits.append(logits.cpu())
                if extract_sparse:
                    all_sparse.append(sparse.cpu())

        logits = torch.cat(all_logits, dim=0)
        sparse = torch.cat(all_sparse, dim=0) if extract_sparse else None

        return logits, sparse

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities."""
        if self.num_labels == 1:
            p = torch.sigmoid(logits).squeeze(-1)
            return torch.stack([1 - p, p], dim=1)
        return F.softmax(logits, dim=-1)

    def fit(self, X: list[str], y, epochs: int | None = None) -> "SPLADEClassifier":
        """Train the classifier.

        Args:
            X: List of input texts
            y: Labels (list or array)
            epochs: Override instance epochs (optional)

        Returns:
            self
        """
        if epochs is None:
            epochs = self.epochs

        # Auto-compute learning rate if not specified
        if self.learning_rate is None:
            self._computed_lr = compute_base_lr(self.model, self.batch_size)
        else:
            self._computed_lr = self.learning_rate

        # Initialize adaptive FLOPS regularization
        self._adaptive_flops = AdaptiveFlopsReg(target_sparsity=self.target_sparsity)

        # Data preparation
        y_tensor = torch.tensor(y, dtype=torch.float32 if self.num_labels == 1 else torch.long)
        enc = self._tokenize(X)
        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self._computed_lr)
        self._lr_scheduler = AdaptiveLRScheduler(
            base_lr=self._computed_lr,
            num_samples=len(X),
            batch_size=self.batch_size,
            epochs=epochs,
        )

        # Loss function
        criterion = nn.CrossEntropyLoss() if self.num_labels > 1 else nn.BCEWithLogitsLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss, n = 0.0, 0
            for input_ids, mask, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = input_ids.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Update learning rate
                lr = self._lr_scheduler.step()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                    logits, sparse = self.model(input_ids, mask)

                    # Self-normalizing FLOPS with adaptive lambda
                    reg_loss = SelfNormalizingFlopsRegFunction.apply(sparse)
                    adaptive_lambda = self._adaptive_flops.compute_lambda(sparse)
                    reg_loss = adaptive_lambda * reg_loss

                    # Classification loss
                    if self.num_labels == 1:
                        cls_loss = criterion(logits, labels)
                    else:
                        cls_loss = criterion(logits, labels.view(-1))

                    loss = cls_loss + reg_loss

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n += 1

            sparsity = self._adaptive_flops.get_current_sparsity()
            print(f"Epoch {epoch+1}: Loss = {total_loss/n:.4f}, Sparsity: {sparsity:.2%}")

        self.model.eval()
        return self

    def _infer(self, X: list[str]) -> torch.Tensor:
        """Run inference and return probabilities."""
        enc = self._tokenize(X)
        logits, _ = self._run_inference_loop(enc["input_ids"], enc["attention_mask"], extract_sparse=False)
        return self._logits_to_probs(logits)

    def predict_proba(self, X: list[str]) -> list[list[float]]:
        """Predict class probabilities."""
        return self._infer(X).tolist()

    def predict(self, X: list[str]) -> list[int]:
        """Predict class labels."""
        return self._infer(X).argmax(dim=1).tolist()

    def score(self, X: list[str], y) -> float:
        """Compute accuracy score."""
        return sum(p == t for p, t in zip(self.predict(X), y)) / len(y)

    def transform(self, X: list[str]) -> np.ndarray:
        """Return sparse SPLADE vectors [n_samples, vocab_size]."""
        enc = self._tokenize(X)
        _, sparse = self._run_inference_loop(enc["input_ids"], enc["attention_mask"], extract_sparse=True)
        return sparse.numpy()

    def explain(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (token, weight) pairs explaining the prediction.

        Args:
            text: Single text to explain
            top_k: Number of top terms to return (default: 10)

        Returns:
            List of (token_string, weight) tuples sorted by weight descending
        """
        sparse_vec = self.transform([text])[0]
        top_indices = np.argsort(sparse_vec)[-top_k:][::-1]

        explanations = []
        for idx in top_indices:
            if sparse_vec[idx] > 0:
                token = self.tokenizer.convert_ids_to_tokens(int(idx))
                explanations.append((token, float(sparse_vec[idx])))

        return explanations

    def print_explanation(self, text: str, top_k: int = 10) -> None:
        """Pretty-print explanation for a prediction."""
        proba = self.predict_proba([text])[0]
        pred_class = int(np.argmax(proba))
        confidence = proba[pred_class]

        explanations = self.explain(text, top_k)

        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Prediction: Class {pred_class} (confidence: {confidence:.2%})")
        print(f"\nTop {len(explanations)} contributing terms:")
        print("-" * 40)
        for token, weight in explanations:
            bar = "#" * int(min(weight * 10, 30))
            print(f"  {token:20s} {weight:6.3f} {bar}")

    def explain_tokens(self, text: str) -> list[tuple[str, float, int]]:
        """Return token-level attributions aligned to input sequence.

        Maps input token IDs directly to their weights in the sparse vector.
        This shows which input tokens contribute to the prediction.

        Args:
            text: Input text

        Returns:
            List of (token_string, weight, position) tuples sorted by weight.
            Position is the token index in the input sequence.
        """
        # Tokenize to get input IDs
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0].tolist()

        # Get sparse vector
        sparse_vec = self.transform([text])[0]

        # Map input tokens to their weights
        attributions = []
        for pos, token_id in enumerate(input_ids):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            # Skip special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            weight = sparse_vec[token_id]
            if weight > 0:
                # Clean up subword tokens
                clean_token = token.replace("##", "")
                attributions.append((clean_token, float(weight), pos))

        # Sort by weight descending
        attributions.sort(key=lambda x: -x[1])
        return attributions

    def get_expansion_terms(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return vocabulary terms NOT in input that have non-zero weight.

        These are SPLADE's semantic expansions - related terms that the model
        activates even though they don't appear in the input text.

        Args:
            text: Input text
            top_k: Number of top expansion terms to return

        Returns:
            List of (token_string, weight) tuples for expansion terms
        """
        # Get input token IDs
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids_set = set(enc["input_ids"][0].tolist())

        # Get sparse vector
        sparse_vec = self.transform([text])[0]

        # Find non-zero dimensions that are NOT in the input
        expansions = []
        for idx, weight in enumerate(sparse_vec):
            if weight > 0 and idx not in input_ids_set:
                token = self.tokenizer.convert_ids_to_tokens(idx)
                # Skip special tokens and subwords
                if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"] and not token.startswith("##"):
                    expansions.append((token, float(weight)))

        # Sort by weight and return top-k
        expansions.sort(key=lambda x: -x[1])
        return expansions[:top_k]

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn compatibility)."""
        return {
            'num_labels': self.num_labels,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'target_sparsity': self.target_sparsity,
            'max_length': self.max_length,
            'device': str(self.device),
        }

    def set_params(self, **params) -> "SPLADEClassifier":
        """Set parameters for this estimator (sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def save(self, path: str) -> None:
        """Save model state and configuration to disk."""
        if not path.endswith('.pt'):
            path = path + '.pt'

        state_dict = self.model.state_dict()

        checkpoint = {
            'state_dict': state_dict,
            'config': {
                'num_labels': self.num_labels,
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'target_sparsity': self.target_sparsity,
                'max_length': self.max_length,
            },
            'version': '1.0.0',
        }

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "SPLADEClassifier":
        """Load a saved model from disk."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        clf = cls(**checkpoint['config'], device=device)

        # Handle torch.compile prefix
        state_dict = checkpoint['state_dict']
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('bert._orig_mod.'):
                cleaned_state_dict['bert.' + k[15:]] = v
            elif '._orig_mod.' in k:
                cleaned_state_dict[k.replace('._orig_mod.', '.')] = v
            else:
                cleaned_state_dict[k] = v

        clf.model.load_state_dict(cleaned_state_dict, strict=False)
        clf.model.eval()

        return clf
