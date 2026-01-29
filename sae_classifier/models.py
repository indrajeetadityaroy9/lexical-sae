"""SAE-based classifier with sklearn-compatible API."""

import random
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from sae_classifier.sae import TopKSAE, BatchTopKSAE, create_sae, load_sae
from sae_classifier.adaptive import StableOps, compute_base_lr, AdaptiveLRScheduler


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_device(device: str) -> torch.device:
    """Get torch device from string specification."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _ActivationExtractor:
    """Extract activations from a specific transformer layer."""

    def __init__(self, model: nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        self._hook = None
        self._register_hook(model)

    def _register_hook(self, model: nn.Module):
        """Register forward hook on target layer."""
        if hasattr(model, "transformer"):
            layer = model.transformer.layer[self.layer_idx]
        elif hasattr(model, "encoder"):
            layer = model.encoder.layer[self.layer_idx]
        elif hasattr(model, "distilbert"):
            layer = model.distilbert.transformer.layer[self.layer_idx]
        else:
            raise ValueError("Unsupported model architecture")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations = hidden

        self._hook = layer.register_forward_hook(hook_fn)

    def get_pooled_activations(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get mean-pooled activations over non-padding tokens."""
        if self.activations is None:
            raise RuntimeError("No activations captured. Run forward pass first.")

        # Mean pool over sequence length (excluding padding)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (self.activations * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def remove_hook(self):
        """Remove the forward hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class SAEClassifier:
    """SAE-based interpretable text classifier with sklearn API.

    Uses Sparse Autoencoder features from transformer activations as the
    representation for classification. Explanations are faithful by construction
    since SAE features ARE the input to the classifier.

    Args:
        num_labels: Number of output classes (default: 2)
        model_name: HuggingFace model name (default: "distilbert-base-uncased")
        sae_layer: Which transformer layer to extract activations from (default: 5)
        sae_expansion: SAE expansion factor (d_sae = d_model * expansion) (default: 8)
        sae_k: Number of active SAE features per sample (default: 32)
        sae_variant: SAE variant ("topk" or "batchtopk") (default: "topk")
        batch_size: Training and inference batch size (default: 32)
        epochs: Number of training epochs (default: 3)
        sae_epochs: Number of SAE pretraining epochs (default: 5)
        learning_rate: Classifier learning rate (None = auto) (default: None)
        max_length: Maximum sequence length (default: 128)
        device: Device specification (default: "auto")
        sae_pretrained_path: Path to pretrained SAE checkpoint (default: None)

    Example:
        >>> clf = SAEClassifier(num_labels=2)
        >>> clf.fit(texts, labels)
        >>> predictions = clf.predict(test_texts)
        >>> explanation = clf.explain("sample text")
    """

    def __init__(
        self,
        num_labels: int = 2,
        model_name: str = "distilbert-base-uncased",
        sae_layer: int = 5,
        sae_expansion: int = 8,
        sae_k: int = 32,
        sae_variant: str = "topk",
        batch_size: int = 32,
        epochs: int = 3,
        sae_epochs: int = 5,
        learning_rate: float | None = None,
        max_length: int = 128,
        device: str = "auto",
        sae_pretrained_path: str | Path | None = None,
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.sae_layer = sae_layer
        self.sae_expansion = sae_expansion
        self.sae_k = sae_k
        self.sae_variant = sae_variant
        self.batch_size = batch_size
        self.epochs = epochs
        self.sae_epochs = sae_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = _get_device(device)
        self.sae_pretrained_path = sae_pretrained_path

        # Initialize transformer backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.d_model = self.config.hidden_size
        self.backbone = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        self.backbone = self.backbone.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize or load SAE
        if sae_pretrained_path is not None:
            self.sae = load_sae(sae_pretrained_path, device=str(self.device))
        else:
            self.sae = create_sae(
                variant=sae_variant,
                d_model=self.d_model,
                expansion_factor=sae_expansion,
                k=sae_k,
            )
            self.sae = self.sae.to(self.device)

        # Initialize classifier head (SAE latents -> classes)
        self.classifier = nn.Linear(self.sae.d_sae, num_labels).to(self.device)

        # Activation extractor
        self._extractor = _ActivationExtractor(self.backbone, sae_layer)

        # Training state
        self._is_fitted = False

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize texts."""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _get_sae_features(self, texts: list[str]) -> torch.Tensor:
        """Extract SAE features for texts."""
        self.backbone.eval()
        self.sae.eval()

        enc = self._tokenize(texts)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            # Forward through backbone (hook captures activations)
            self.backbone(input_ids=input_ids, attention_mask=attention_mask)

            # Get pooled activations
            activations = self._extractor.get_pooled_activations(attention_mask)

            # Encode through SAE
            latents = self.sae.encode(activations)

        return latents

    def _train_sae(self, texts: list[str]) -> None:
        """Pretrain SAE on transformer activations."""
        print(f"Pretraining SAE on {len(texts)} texts...")
        self.backbone.eval()

        # Collect activations
        all_activations = []
        loader = DataLoader(range(len(texts)), batch_size=self.batch_size, shuffle=False)

        for batch_indices in tqdm(loader, desc="Collecting activations"):
            batch_texts = [texts[i] for i in batch_indices]
            enc = self._tokenize(batch_texts)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                activations = self._extractor.get_pooled_activations(attention_mask)
                all_activations.append(activations.cpu())

        all_activations = torch.cat(all_activations, dim=0)
        print(f"Collected {all_activations.shape[0]} activation vectors")

        # Train SAE
        self.sae.train()
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=1e-4)

        dataset = TensorDataset(all_activations)
        train_loader = DataLoader(dataset, batch_size=min(4096, len(all_activations)), shuffle=True)

        for epoch in range(self.sae_epochs):
            total_loss = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                output = self.sae(batch)
                loss = output.loss + output.aux_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            dead_frac = self.sae.get_dead_latent_fraction()
            print(f"SAE Epoch {epoch+1}/{self.sae_epochs}: Loss={avg_loss:.4f}, Dead={dead_frac:.1%}")
            self.sae.reset_stats()

        self.sae.eval()

    def fit(self, X: list[str], y: list[int], epochs: int | None = None) -> "SAEClassifier":
        """Train the SAE classifier.

        Two-stage training:
        1. Pretrain SAE on transformer activations (if not using pretrained)
        2. Train linear classifier on SAE features

        Args:
            X: List of input texts
            y: List of labels
            epochs: Override default epochs

        Returns:
            self
        """
        if epochs is None:
            epochs = self.epochs

        # Stage 1: Pretrain SAE (if needed)
        if self.sae_pretrained_path is None:
            self._train_sae(X)

        # Stage 2: Train classifier
        print(f"Training classifier on {len(X)} samples...")

        # Extract SAE features for all training data
        all_features = []
        loader = DataLoader(range(len(X)), batch_size=self.batch_size, shuffle=False)

        self.backbone.eval()
        self.sae.eval()
        for batch_indices in tqdm(loader, desc="Extracting features"):
            batch_texts = [X[i] for i in batch_indices]
            features = self._get_sae_features(batch_texts)
            all_features.append(features.cpu())

        all_features = torch.cat(all_features, dim=0)
        labels_tensor = torch.tensor(y, dtype=torch.long)

        # Train classifier
        self.classifier.train()
        train_dataset = TensorDataset(all_features, labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        lr = self.learning_rate if self.learning_rate else 1e-3
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for features, batch_labels in train_loader:
                features = features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.classifier(features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

            acc = correct / total
            print(f"Classifier Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}")

        self.classifier.eval()
        self._is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        """Get class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        self.classifier.eval()
        all_probs = []

        loader = DataLoader(range(len(texts)), batch_size=self.batch_size, shuffle=False)
        for batch_indices in loader:
            batch_texts = [texts[i] for i in batch_indices]
            features = self._get_sae_features(batch_texts)

            with torch.no_grad():
                logits = self.classifier(features)
                probs = F.softmax(logits, dim=-1)
                all_probs.extend(probs.cpu().tolist())

        return all_probs

    def predict(self, texts: list[str]) -> list[int]:
        """Get class predictions."""
        probs = self.predict_proba(texts)
        return [max(range(len(p)), key=lambda i: p[i]) for p in probs]

    def score(self, X: list[str], y: list[int]) -> float:
        """Compute accuracy on test data."""
        preds = self.predict(X)
        return sum(p == t for p, t in zip(preds, y)) / len(y)

    def explain(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Get top-k SAE feature explanations.

        Returns feature indices and their activation values. Unlike SPLADE
        which returns vocabulary terms, SAE returns learned feature indices.

        Args:
            text: Input text
            top_k: Number of top features to return

        Returns:
            List of (feature_index, activation) tuples sorted by activation
        """
        features = self._get_sae_features([text])[0]  # [d_sae]

        # Get top-k active features
        values, indices = torch.topk(features, min(top_k, (features > 0).sum().item()))

        return [(int(idx), float(val)) for idx, val in zip(indices.tolist(), values.tolist())]

    def explain_tokens(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Get token-level explanations via attention attribution.

        Maps SAE feature importance back to input tokens using attention weights.
        This provides a token-level explanation compatible with faithfulness metrics.

        Args:
            text: Input text
            top_k: Number of top tokens to return

        Returns:
            List of (token, weight) tuples sorted by importance
        """
        enc = self._tokenize([text])
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Get SAE features and their weights in classifier
        with torch.no_grad():
            self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            activations = self._extractor.get_pooled_activations(attention_mask)
            latents = self.sae.encode(activations)[0]  # [d_sae]

            # Get classifier weights for predicted class
            logits = self.classifier(latents.unsqueeze(0))
            pred_class = logits.argmax(dim=-1).item()
            class_weights = self.classifier.weight[pred_class]  # [d_sae]

            # Feature importance = latent activation * class weight
            feature_importance = latents * class_weights  # [d_sae]

            # Project back to activation space via SAE decoder
            activation_importance = feature_importance @ self.sae.W_dec_normalized  # [d_model]

            # Get per-token activations (before pooling)
            token_acts = self._extractor.activations[0]  # [seq_len, d_model]
            seq_len = attention_mask.sum().item()
            token_acts = token_acts[:seq_len]

            # Token importance = dot product with activation importance direction
            token_importance = (token_acts * activation_importance.unsqueeze(0)).sum(dim=-1)
            token_importance = token_importance.abs()

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())

        # Create explanations
        explanations = []
        for token, weight in zip(tokens, token_importance.cpu().tolist()):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                clean_token = token.replace("##", "")
                explanations.append((clean_token, weight))

        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:top_k]

    def get_feature_activations(self, texts: list[str]) -> np.ndarray:
        """Get SAE feature activations for texts.

        Args:
            texts: List of input texts

        Returns:
            Array of shape [num_texts, d_sae] with SAE latent activations
        """
        features = self._get_sae_features(texts)
        return features.cpu().numpy()

    def save(self, path: str | Path) -> None:
        """Save classifier checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save SAE
        torch.save({
            "sae_state_dict": self.sae.state_dict(),
            "sae_config": {
                "d_model": self.sae.d_model,
                "d_sae": self.sae.d_sae,
                "k": self.sae.k,
                "normalize_decoder": self.sae.normalize_decoder,
                "aux_loss_coef": self.sae.aux_loss_coef,
                "variant": "batchtopk" if isinstance(self.sae, BatchTopKSAE) else "topk",
            },
        }, path / "sae.pt")

        # Save classifier
        torch.save({
            "classifier_state_dict": self.classifier.state_dict(),
            "config": {
                "num_labels": self.num_labels,
                "model_name": self.model_name,
                "sae_layer": self.sae_layer,
                "sae_expansion": self.sae_expansion,
                "sae_k": self.sae_k,
                "sae_variant": self.sae_variant,
                "max_length": self.max_length,
            },
        }, path / "classifier.pt")

        print(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "SAEClassifier":
        """Load classifier from checkpoint."""
        path = Path(path)

        # Load classifier config
        clf_checkpoint = torch.load(path / "classifier.pt", map_location="cpu")
        config = clf_checkpoint["config"]

        # Create instance
        instance = cls(
            num_labels=config["num_labels"],
            model_name=config["model_name"],
            sae_layer=config["sae_layer"],
            sae_expansion=config["sae_expansion"],
            sae_k=config["sae_k"],
            sae_variant=config["sae_variant"],
            max_length=config["max_length"],
            device=device,
            sae_pretrained_path=path / "sae.pt",
        )

        # Load classifier weights
        instance.classifier.load_state_dict(clf_checkpoint["classifier_state_dict"])
        instance._is_fitted = True

        print(f"Loaded checkpoint from {path}")
        return instance

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn compatibility."""
        return {
            "num_labels": self.num_labels,
            "model_name": self.model_name,
            "sae_layer": self.sae_layer,
            "sae_expansion": self.sae_expansion,
            "sae_k": self.sae_k,
            "sae_variant": self.sae_variant,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "sae_epochs": self.sae_epochs,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
        }

    def set_params(self, **params) -> "SAEClassifier":
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
