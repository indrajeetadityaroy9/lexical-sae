"""
Train Sparse Autoencoder (SAE) on extracted SPLADE vectors.

The SAE learns to reconstruct SPLADE vectors using a sparse hidden layer,
enabling interpretability through monosemantic feature discovery.

Usage:
    python -m src.train_sae --vectors_path outputs/vectors.pt --output_dir outputs/sae
"""

import argparse
import os

import torch
import torch.optim as optim

from src.interpretability.sparse_autoencoder import SparseAutoencoder
from src.utils import get_device


def train_sae(
    vectors_path: str,
    output_dir: str,
    hidden_dim: int = 4096,
    k: int = 32,
    sparsity_coefficient: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """
    Train SAE on extracted SPLADE vectors.

    Args:
        vectors_path: Path to extracted vectors file
        output_dir: Directory to save trained SAE
        hidden_dim: Dimension of SAE hidden layer
        k: Number of active features per sample (TopK activation)
        sparsity_coefficient: Weight for L1 sparsity loss
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load vectors
    print(f"Loading vectors from {vectors_path}...")
    data = torch.load(vectors_path)
    train_vectors = data['train_vectors']
    test_vectors = data['test_vectors']
    vocab_size = data['vocab_size']

    print(f"Train vectors: {train_vectors.shape}")
    print(f"Test vectors: {test_vectors.shape}")
    print(f"Vocab size: {vocab_size}")

    # Create SAE
    sae = SparseAutoencoder(
        input_dim=vocab_size,
        hidden_dim=hidden_dim,
        k=k,
        sparsity_coefficient=sparsity_coefficient,
        tied_weights=True,
        normalize_decoder=True
    )
    sae.to(device)

    # Print model info
    num_params = sum(p.numel() for p in sae.parameters())
    print(f"SAE parameters: {num_params:,}")
    print(f"Hidden dim: {hidden_dim}, K: {k}")

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_vectors)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(test_vectors)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Training loop
    print(f"\nStarting SAE training for {epochs} epochs...")
    best_test_loss = float('inf')

    for epoch in range(epochs):
        sae.train()
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass - returns SAEOutput dataclass
            output = sae(batch, return_loss=True)

            output.loss.backward()
            optimizer.step()

            total_loss += output.loss.item()
            total_recon_loss += output.reconstruction_loss.item()
            total_sparsity_loss += output.sparsity_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_sparsity = total_sparsity_loss / len(train_loader)

        # Evaluate on test set
        sae.eval()
        test_loss = 0
        test_hidden_sparsity = 0

        with torch.no_grad():
            for (batch,) in test_loader:
                batch = batch.to(device)
                output = sae(batch, return_loss=True)
                test_loss += output.reconstruction_loss.item()

                # Calculate hidden layer sparsity
                sparsity = (output.hidden == 0).float().mean().item()
                test_hidden_sparsity += sparsity

        test_loss /= len(test_loader)
        test_hidden_sparsity /= len(test_loader)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, Sparse: {avg_sparsity:.4f}) | "
              f"Test Loss: {test_loss:.6f} | Hidden Sparsity: {test_hidden_sparsity*100:.1f}%")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': sae.state_dict(),
                'config': {
                    'input_dim': vocab_size,
                    'hidden_dim': hidden_dim,
                    'k': k,
                    'sparsity_coefficient': sparsity_coefficient
                },
                'best_test_loss': best_test_loss
            }, os.path.join(output_dir, 'sae_best.pt'))

    # Save final model
    torch.save({
        'model_state_dict': sae.state_dict(),
        'config': {
            'input_dim': vocab_size,
            'hidden_dim': hidden_dim,
            'k': k,
            'sparsity_coefficient': sparsity_coefficient
        },
        'final_test_loss': test_loss
    }, os.path.join(output_dir, 'sae_final.pt'))

    print(f"\n=== Training Complete ===")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Models saved to {output_dir}")

    return sae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_path', type=str, default='outputs/vectors.pt')
    parser.add_argument('--output_dir', type=str, default='outputs/sae')
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--sparsity_coefficient', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    train_sae(
        vectors_path=args.vectors_path,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        k=args.k,
        sparsity_coefficient=args.sparsity_coefficient,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
