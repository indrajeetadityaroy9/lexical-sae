import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random
import numpy as np
from src.models import DistilBERTSparseClassifier
from src.data import get_data_loaders
from src.regularizers import flops_regularization
from src.evaluation import evaluate
import time


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(args):
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    train_loader, test_loader = get_data_loaders(
        os.path.join(args.data_dir, 'movie_reviews_train.txt'),
        os.path.join(args.data_dir, 'movie_reviews_test.txt'),
        batch_size=args.batch_size
    )

    # 2. Init Model
    model = DistilBERTSparseClassifier()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Optimizer (Transformer needs smaller LR, usually)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting training for {args.epochs} epochs with FLOPS lambda={args.flops_lambda}...")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_flops_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['label'].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits, sparse_vec = model(input_ids, attention_mask)
            
            # Loss Calculation
            # Task Loss (Classification)
            bce_loss = criterion(logits, y)
            
            # Sparsity Loss (FLOPS Regularization)
            flops_loss = args.flops_lambda * flops_regularization(sparse_vec)
            
            loss = bce_loss + flops_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_flops_loss += flops_loss.item()
            
        # Evaluation
        acc, f1, sparsity = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss:.4f} (FLOPS: {total_flops_loss:.4f}) | Test Acc: {acc:.4f} | F1: {f1:.4f} | Sparsity: {sparsity:.2f}%")

    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save Model
    os.makedirs(args.output_dir, exist_ok=True)
    # Save the whole model or state dict (state dict preferred)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data', help='Directory containing train/test files')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate (default: 2e-5 for Transformers)")
    parser.add_argument('--flops_lambda', type=float, default=1e-4, help='Regularization strength for FLOPS sparsity')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    train(args)
