"""
Analyze SAE features for interpretability.

This script loads a trained SAE and analyzes the learned features,
identifying which vocabulary terms are associated with each feature
and generating an interpretability report.

Usage:
    python -m src.analyze_sae --sae_path outputs/sae/sae_best.pt --vectors_path outputs/vectors.pt
"""

import torch
import argparse
import os
from transformers import AutoTokenizer
from collections import defaultdict

from src.interpretability.sparse_autoencoder import SparseAutoencoder


def load_sae(sae_path: str, device: torch.device) -> SparseAutoencoder:
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(sae_path, map_location=device)
    config = checkpoint['config']

    sae = SparseAutoencoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        k=config['k'],
        sparsity_coefficient=config.get('sparsity_coefficient', 1e-3),
        tied_weights=True,
        normalize_decoder=True
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()

    return sae


def analyze_feature_tokens(
    sae: SparseAutoencoder,
    tokenizer,
    top_k: int = 15
) -> dict:
    """
    Analyze which vocabulary tokens are associated with each SAE feature.

    For each feature, we look at the decoder weights to find which
    vocabulary terms have the strongest association.

    Returns:
        dict mapping feature_idx to list of (token, weight) tuples
    """
    feature_tokens = {}

    # Get decoder weights (tied weights = encoder.weight)
    # encoder.weight is [hidden_dim, input_dim]
    weights = sae.encoder.weight.data  # [hidden_dim, input_dim]

    hidden_dim = weights.shape[0]

    print(f"Analyzing {hidden_dim} SAE features...")

    for feat_idx in range(hidden_dim):
        # Get weights for this feature
        feat_weights = weights[feat_idx]  # [input_dim]

        # Get top-k tokens by absolute weight
        values, indices = torch.topk(feat_weights.abs(), top_k)

        tokens = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            token = tokenizer.decode([idx]).strip()
            actual_weight = feat_weights[idx].item()
            tokens.append((token, actual_weight))

        feature_tokens[feat_idx] = tokens

    return feature_tokens


def analyze_feature_activations(
    sae: SparseAutoencoder,
    vectors: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Analyze which features activate most often and their correlation with labels.

    Returns:
        dict with activation statistics per feature
    """
    sae.eval()
    vectors = vectors.to(device)

    with torch.no_grad():
        output = sae(vectors, return_loss=False)
        hidden = output.hidden  # [num_samples, hidden_dim]

    # Move to CPU for analysis
    hidden = hidden.cpu()
    labels = labels.cpu()

    hidden_dim = hidden.shape[1]
    stats = {}

    for feat_idx in range(hidden_dim):
        feat_activations = hidden[:, feat_idx]

        # Activation frequency (how often is this feature non-zero)
        freq = (feat_activations.abs() > 1e-6).float().mean().item()

        # Mean activation when active
        active_mask = feat_activations.abs() > 1e-6
        if active_mask.any():
            mean_when_active = feat_activations[active_mask].mean().item()
        else:
            mean_when_active = 0.0

        # Correlation with positive sentiment (label=1)
        if freq > 0.01:  # Only compute for features that fire enough
            pos_mask = labels == 1
            neg_mask = labels == 0

            pos_activation = feat_activations[pos_mask].mean().item()
            neg_activation = feat_activations[neg_mask].mean().item()

            # Simple sentiment correlation score
            sentiment_diff = pos_activation - neg_activation
        else:
            pos_activation = 0.0
            neg_activation = 0.0
            sentiment_diff = 0.0

        stats[feat_idx] = {
            'activation_freq': freq,
            'mean_activation': mean_when_active,
            'pos_sentiment_activation': pos_activation,
            'neg_sentiment_activation': neg_activation,
            'sentiment_correlation': sentiment_diff
        }

    return stats


def generate_report(
    feature_tokens: dict,
    activation_stats: dict,
    tokenizer,
    output_path: str,
    top_features: int = 50
):
    """
    Generate interpretability report with top features.
    """
    # Sort features by activation frequency
    sorted_by_freq = sorted(
        activation_stats.items(),
        key=lambda x: x[1]['activation_freq'],
        reverse=True
    )

    # Sort features by sentiment correlation (absolute value)
    sorted_by_sentiment = sorted(
        activation_stats.items(),
        key=lambda x: abs(x[1]['sentiment_correlation']),
        reverse=True
    )

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SAE INTERPRETABILITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    total_features = len(activation_stats)
    active_features = sum(1 for s in activation_stats.values() if s['activation_freq'] > 0.01)

    report_lines.append(f"Total SAE features: {total_features}")
    report_lines.append(f"Active features (>1% activation): {active_features}")
    report_lines.append("")

    # Top features by frequency
    report_lines.append("-" * 80)
    report_lines.append("TOP FEATURES BY ACTIVATION FREQUENCY")
    report_lines.append("-" * 80)
    report_lines.append("")

    for feat_idx, stats in sorted_by_freq[:top_features]:
        if stats['activation_freq'] < 0.001:
            break

        tokens = feature_tokens.get(feat_idx, [])
        token_str = ", ".join([f"{t[0]}({t[1]:.3f})" for t in tokens[:8]])

        report_lines.append(f"Feature {feat_idx:4d} | Freq: {stats['activation_freq']*100:5.2f}% | "
                          f"Sentiment: {stats['sentiment_correlation']:+.4f}")
        report_lines.append(f"  Top tokens: {token_str}")
        report_lines.append("")

    # Top features by sentiment correlation
    report_lines.append("-" * 80)
    report_lines.append("TOP FEATURES BY SENTIMENT CORRELATION")
    report_lines.append("-" * 80)
    report_lines.append("")

    report_lines.append("Positive sentiment features (activate more on positive reviews):")
    pos_features = [(i, s) for i, s in sorted_by_sentiment if s['sentiment_correlation'] > 0.001][:20]
    for feat_idx, stats in pos_features:
        tokens = feature_tokens.get(feat_idx, [])
        token_str = ", ".join([f"{t[0]}" for t in tokens[:6]])
        report_lines.append(f"  Feature {feat_idx:4d} | Corr: {stats['sentiment_correlation']:+.4f} | Tokens: {token_str}")

    report_lines.append("")
    report_lines.append("Negative sentiment features (activate more on negative reviews):")
    neg_features = [(i, s) for i, s in sorted_by_sentiment if s['sentiment_correlation'] < -0.001][:20]
    for feat_idx, stats in neg_features:
        tokens = feature_tokens.get(feat_idx, [])
        token_str = ", ".join([f"{t[0]}" for t in tokens[:6]])
        report_lines.append(f"  Feature {feat_idx:4d} | Corr: {stats['sentiment_correlation']:+.4f} | Tokens: {token_str}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to {output_path}")
    print("\n" + report_text)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load SAE
    print(f"Loading SAE from {args.sae_path}...")
    sae = load_sae(args.sae_path, device)
    print(f"SAE hidden dim: {sae.hidden_dim}, K: {sae.k}")

    # Load vectors
    print(f"Loading vectors from {args.vectors_path}...")
    data = torch.load(args.vectors_path)
    test_vectors = data['test_vectors']
    test_labels = data['test_labels']
    print(f"Test vectors: {test_vectors.shape}")

    # Analyze feature tokens
    print("\nAnalyzing feature-token associations...")
    feature_tokens = analyze_feature_tokens(sae, tokenizer, top_k=15)

    # Analyze feature activations
    print("\nAnalyzing feature activations on test set...")
    activation_stats = analyze_feature_activations(sae, test_vectors, test_labels, device)

    # Generate report
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    generate_report(feature_tokens, activation_stats, tokenizer, args.output_path, top_features=args.top_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_path', type=str, default='outputs/sae/sae_best.pt')
    parser.add_argument('--vectors_path', type=str, default='outputs/vectors.pt')
    parser.add_argument('--output_path', type=str, default='outputs/sae/interpretability_report.txt')
    parser.add_argument('--top_features', type=int, default=30)

    args = parser.parse_args()
    main(args)
