"""SAE training loop for transformer activations.

Handles:
- Activation extraction from transformer layers
- SAE training with reconstruction + auxiliary loss
- Dead latent monitoring and resampling
- Checkpointing and evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Iterator
import json

from sae_classifier.sae.architecture import TopKSAE, BatchTopKSAE, create_sae


class ActivationBuffer:
    """Buffer for collecting activations from a transformer layer.

    Hooks into a specific layer and accumulates activations during forward passes.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        max_samples: int = 100_000,
        d_model: int = 768,
    ):
        self.layer_idx = layer_idx
        self.max_samples = max_samples
        self.d_model = d_model

        # Storage for activations
        self.activations: list[torch.Tensor] = []
        self.total_tokens = 0

        # Register hook on the target layer
        self.hook = None
        self._register_hook(model)

    def _register_hook(self, model: nn.Module):
        """Register forward hook on target layer."""
        # For DistilBERT: model.transformer.layer[layer_idx]
        if hasattr(model, "transformer"):
            layer = model.transformer.layer[self.layer_idx]
        elif hasattr(model, "encoder"):
            layer = model.encoder.layer[self.layer_idx]
        else:
            raise ValueError("Unsupported model architecture")

        def hook_fn(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Flatten batch and sequence dimensions, keep d_model
            # hidden: [batch, seq, d_model] -> [batch * seq, d_model]
            flat = hidden.detach().view(-1, self.d_model)

            # Only keep up to max_samples
            remaining = self.max_samples - self.total_tokens
            if remaining > 0:
                to_keep = min(flat.shape[0], remaining)
                self.activations.append(flat[:to_keep].cpu())
                self.total_tokens += to_keep

        self.hook = layer.register_forward_hook(hook_fn)

    def get_activations(self) -> torch.Tensor:
        """Return all collected activations as single tensor."""
        if not self.activations:
            raise RuntimeError("No activations collected")
        return torch.cat(self.activations, dim=0)

    def clear(self):
        """Clear stored activations."""
        self.activations = []
        self.total_tokens = 0

    def remove_hook(self):
        """Remove the forward hook."""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def is_full(self) -> bool:
        """Check if buffer has reached max_samples."""
        return self.total_tokens >= self.max_samples


def collect_activations(
    model_name: str = "distilbert-base-uncased",
    layer_idx: int = 5,
    texts: list[str] | None = None,
    dataset_name: str = "wikitext",
    dataset_split: str = "train",
    max_samples: int = 100_000,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Collect activations from a transformer layer.

    Args:
        model_name: HuggingFace model name
        layer_idx: Which layer to extract activations from (0-indexed)
        texts: Optional list of texts (if None, loads from dataset_name)
        dataset_name: HuggingFace dataset to use if texts not provided
        dataset_split: Dataset split
        max_samples: Maximum number of token activations to collect
        batch_size: Batch size for forward passes
        max_length: Maximum sequence length
        device: Device to run model on

    Returns:
        Tensor of activations [num_samples, d_model]
    """
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    d_model = model.config.hidden_size

    # Create activation buffer
    buffer = ActivationBuffer(model, layer_idx, max_samples, d_model)

    # Load texts if not provided
    if texts is None:
        from datasets import load_dataset
        if dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-v1", split=dataset_split)
            texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples // 10]
        else:
            ds = load_dataset(dataset_name, split=dataset_split)
            text_col = "text" if "text" in ds.column_names else ds.column_names[0]
            texts = ds[text_col][:max_samples // 10]

    # Process texts in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
            if buffer.is_full():
                break

            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            model(**inputs)

    buffer.remove_hook()
    activations = buffer.get_activations()

    print(f"Collected {activations.shape[0]} activations from layer {layer_idx}")
    return activations


class SAETrainer:
    """Trainer for Sparse Autoencoders on transformer activations.

    Handles training loop, evaluation, and checkpointing.
    """

    def __init__(
        self,
        sae: TopKSAE | BatchTopKSAE,
        lr: float = 1e-4,
        lr_warmup_steps: int = 1000,
        weight_decay: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.sae = sae.to(device)
        self.device = device
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps

        # Optimizer
        self.optimizer = torch.optim.Adam(
            sae.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # LR scheduler with warmup
        def lr_lambda(step):
            if step < lr_warmup_steps:
                return step / lr_warmup_steps
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

    def train_epoch(
        self,
        activations: torch.Tensor,
        batch_size: int = 4096,
        shuffle: bool = True,
    ) -> dict[str, float]:
        """Train for one epoch on activation data.

        Args:
            activations: Tensor of activations [num_samples, d_model]
            batch_size: Training batch size
            shuffle: Whether to shuffle data

        Returns:
            Dictionary of metrics (loss, aux_loss, dead_fraction)
        """
        self.sae.train()

        dataset = TensorDataset(activations)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc="Training")
        for (batch,) in pbar:
            batch = batch.to(self.device)

            # Forward pass
            output = self.sae(batch)

            # Combined loss
            loss = output.loss + output.aux_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += output.loss.item()
            total_aux_loss += output.aux_loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                "loss": f"{output.loss.item():.4f}",
                "aux": f"{output.aux_loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        metrics = {
            "loss": total_loss / num_batches,
            "aux_loss": total_aux_loss / num_batches,
            "dead_fraction": self.sae.get_dead_latent_fraction(),
        }

        return metrics

    @torch.no_grad()
    def evaluate(self, activations: torch.Tensor, batch_size: int = 4096) -> dict[str, float]:
        """Evaluate SAE on held-out activations.

        Args:
            activations: Tensor of activations [num_samples, d_model]
            batch_size: Evaluation batch size

        Returns:
            Dictionary of metrics
        """
        self.sae.eval()

        dataset = TensorDataset(activations)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        total_l0 = 0.0  # Average number of active features
        num_batches = 0

        for (batch,) in loader:
            batch = batch.to(self.device)
            output = self.sae(batch)

            total_loss += output.loss.item()
            total_l0 += (output.latents > 0).float().sum(dim=-1).mean().item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "avg_l0": total_l0 / num_batches,
            "dead_fraction": self.sae.get_dead_latent_fraction(),
        }

    def train(
        self,
        activations: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 4096,
        val_split: float = 0.1,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            activations: Tensor of activations [num_samples, d_model]
            epochs: Number of training epochs
            batch_size: Training batch size
            val_split: Fraction of data for validation
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary of metric histories
        """
        # Split into train/val
        n_val = int(len(activations) * val_split)
        perm = torch.randperm(len(activations))
        val_activations = activations[perm[:n_val]]
        train_activations = activations[perm[n_val:]]

        print(f"Training on {len(train_activations)} samples, validating on {len(val_activations)}")

        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history = {"loss": [], "aux_loss": [], "val_loss": [], "dead_fraction": []}

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Reset activation stats each epoch
            self.sae.reset_stats()

            # Train
            train_metrics = self.train_epoch(train_activations, batch_size)

            # Evaluate
            val_metrics = self.evaluate(val_activations, batch_size)

            # Log
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  Avg L0:     {val_metrics['avg_l0']:.1f}")
            print(f"  Dead:       {val_metrics['dead_fraction']:.1%}")

            # Track history
            history["loss"].append(train_metrics["loss"])
            history["aux_loss"].append(train_metrics["aux_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["dead_fraction"].append(val_metrics["dead_fraction"])

            # Checkpoint if best
            if checkpoint_dir and val_metrics["val_loss"] < self.best_loss:
                self.best_loss = val_metrics["val_loss"]
                self.save(checkpoint_dir / "best.pt")
                print(f"  Saved best checkpoint (val_loss={self.best_loss:.4f})")

        # Save final
        if checkpoint_dir:
            self.save(checkpoint_dir / "final.pt")

        return history

    def save(self, path: str | Path):
        """Save SAE checkpoint."""
        path = Path(path)
        torch.save({
            "sae_state_dict": self.sae.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "config": {
                "d_model": self.sae.d_model,
                "d_sae": self.sae.d_sae,
                "k": self.sae.k,
                "normalize_decoder": self.sae.normalize_decoder,
                "aux_loss_coef": self.sae.aux_loss_coef,
                "variant": "batchtopk" if isinstance(self.sae, BatchTopKSAE) else "topk",
            },
        }, path)

    def load(self, path: str | Path):
        """Load SAE checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.sae.load_state_dict(checkpoint["sae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]


def load_sae(path: str | Path, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> TopKSAE | BatchTopKSAE:
    """Load a trained SAE from checkpoint.

    Args:
        path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded SAE model
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    sae = create_sae(
        variant=config["variant"],
        d_model=config["d_model"],
        expansion_factor=config["d_sae"] // config["d_model"],
        k=config["k"],
        normalize_decoder=config["normalize_decoder"],
        aux_loss_coef=config["aux_loss_coef"],
    )

    sae.load_state_dict(checkpoint["sae_state_dict"])
    sae = sae.to(device)
    sae.eval()

    return sae


def main():
    """CLI for SAE training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train SAE on transformer activations")
    parser.add_argument("--model", default="distilbert-base-uncased", help="HuggingFace model")
    parser.add_argument("--layer", type=int, default=5, help="Layer to extract activations from")
    parser.add_argument("--expansion", type=int, default=8, help="SAE expansion factor")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument("--variant", choices=["topk", "batchtopk"], default="topk")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=100_000, help="Max activation samples")
    parser.add_argument("--checkpoint-dir", default="checkpoints/sae", help="Checkpoint directory")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    # Collect activations
    print(f"Collecting activations from {args.model} layer {args.layer}...")
    activations = collect_activations(
        model_name=args.model,
        layer_idx=args.layer,
        max_samples=args.max_samples,
        device=device,
    )

    # Create SAE
    d_model = activations.shape[1]
    sae = create_sae(
        variant=args.variant,
        d_model=d_model,
        expansion_factor=args.expansion,
        k=args.k,
    )
    print(f"Created {args.variant} SAE: {d_model} -> {sae.d_sae} (k={args.k})")

    # Train
    trainer = SAETrainer(sae, lr=args.lr, device=device)
    history = trainer.train(
        activations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\nTraining complete!")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Dead latents: {history['dead_fraction'][-1]:.1%}")


if __name__ == "__main__":
    main()
