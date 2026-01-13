"""
Visualization module for SPLADE and SAE interpretability.

Provides functions for creating visual explanations of:
1. SPLADE term weights for documents
2. SAE feature activations
3. Feature clustering and relationships
4. Training metrics
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn

# Visualization imports (optional - gracefully handle missing)
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


class InterpretabilityVisualizer:
    """
    Visualization tools for SPLADE and SAE interpretability.

    Provides various plotting functions for understanding
    what the models have learned.
    """

    def __init__(
        self,
        tokenizer,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "whitegrid"
    ):
        """
        Initialize visualizer.

        Args:
            tokenizer: HuggingFace tokenizer for decoding
            figsize: Default figure size
            style: Seaborn style (if available)
        """
        check_matplotlib()
        self.tokenizer = tokenizer
        self.figsize = figsize

        if HAS_SEABORN:
            sns.set_style(style)

    def plot_term_weights(
        self,
        sparse_vec: torch.Tensor,
        title: str = "SPLADE Term Weights",
        top_k: int = 30,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot top-k term weights from a SPLADE vector.

        Args:
            sparse_vec: SPLADE vector [vocab_size] or [1, vocab_size]
            title: Plot title
            top_k: Number of top terms to show
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure
        """
        if sparse_vec.dim() == 2:
            sparse_vec = sparse_vec.squeeze(0)

        sparse_vec = sparse_vec.cpu()

        # Get top-k terms
        values, indices = torch.topk(sparse_vec, top_k)

        # Decode tokens
        tokens = [self.tokenizer.decode([idx]).strip() for idx in indices.tolist()]
        weights = values.tolist()

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=self.figsize)

        y_pos = np.arange(len(tokens))
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(tokens)))[::-1]

        ax.barh(y_pos, weights, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()
        ax.set_xlabel("Weight")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_feature_activations(
        self,
        text: str,
        feature_activations: List[Tuple[int, float, List[str]]],
        title: str = "SAE Feature Activations",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot which SAE features activate for a given text.

        Args:
            text: Input text (for display)
            feature_activations: List of (feature_idx, activation, top_tokens)
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Extract data
        feature_ids = [f"F{f[0]}" for f in feature_activations]
        activations = [f[1] for f in feature_activations]
        token_labels = [", ".join(f[2][:3]) for f in feature_activations]

        y_pos = np.arange(len(feature_ids))
        colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(feature_ids)))[::-1]

        bars = ax.barh(y_pos, activations, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{fid}: {lbl}" for fid, lbl in zip(feature_ids, token_labels)])
        ax.invert_yaxis()
        ax.set_xlabel("Activation")
        ax.set_title(f"{title}\nText: \"{text[:50]}...\"" if len(text) > 50 else f"{title}\nText: \"{text}\"")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_sparsity_distribution(
        self,
        vectors: torch.Tensor,
        title: str = "Sparsity Distribution",
        threshold: float = 1e-2,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of sparsity across documents.

        Args:
            vectors: SPLADE vectors [num_docs, vocab_size]
            title: Plot title
            threshold: Threshold for considering value as zero
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        # Calculate sparsity per document
        active_counts = (vectors.abs() > threshold).sum(dim=1).cpu().numpy()
        total_dim = vectors.shape[1]
        sparsities = 1.0 - (active_counts / total_dim)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of sparsity
        axes[0].hist(sparsities * 100, bins=50, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("Sparsity (%)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Document Sparsity")
        axes[0].axvline(np.mean(sparsities) * 100, color="red", linestyle="--",
                        label=f"Mean: {np.mean(sparsities)*100:.1f}%")
        axes[0].legend()

        # Histogram of active dimensions
        axes[1].hist(active_counts, bins=50, color="coral", edgecolor="white")
        axes[1].set_xlabel("Number of Active Dimensions")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Active Dimensions")
        axes[1].axvline(np.mean(active_counts), color="red", linestyle="--",
                        label=f"Mean: {np.mean(active_counts):.0f}")
        axes[1].legend()

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_feature_frequency(
        self,
        feature_stats: Dict[str, torch.Tensor],
        top_k: int = 50,
        title: str = "Feature Activation Frequency",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot activation frequency of top SAE features.

        Args:
            feature_stats: Statistics from FeatureAnalyzer
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        frequencies = feature_stats["frequency"].cpu().numpy()

        # Get top-k features by frequency
        top_indices = np.argsort(frequencies)[::-1][:top_k]
        top_freqs = frequencies[top_indices]

        fig, ax = plt.subplots(figsize=self.figsize)

        x_pos = np.arange(len(top_indices))
        ax.bar(x_pos, top_freqs * 100, color="teal", edgecolor="white")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Activation Frequency (%)")
        ax.set_title(title)
        ax.set_xticks(x_pos[::5])
        ax.set_xticklabels([str(i) for i in top_indices[::5]], rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_feature_clustering(
        self,
        activations: torch.Tensor,
        labels: Optional[np.ndarray] = None,
        title: str = "SAE Feature Space (UMAP)",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot UMAP clustering of documents in SAE feature space.

        Args:
            activations: SAE hidden activations [num_docs, hidden_dim]
            labels: Document labels for coloring (optional)
            title: Plot title
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        if not HAS_UMAP:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")

        # Reduce dimensionality with UMAP
        activations_np = activations.cpu().numpy()

        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
        embedding = reducer.fit_transform(activations_np)

        fig, ax = plt.subplots(figsize=self.figsize)

        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.6,
                s=10
            )
            plt.colorbar(scatter, ax=ax, label="Label")
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                alpha=0.6,
                s=10,
                color="steelblue"
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_training_history(
        self,
        history: List[Dict[str, float]],
        metrics: List[str] = ["loss", "reconstruction_loss", "active_features"],
        title: str = "SAE Training History",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot SAE training history.

        Args:
            history: List of epoch metrics dictionaries
            metrics: Which metrics to plot
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))

        if num_metrics == 1:
            axes = [axes]

        epochs = range(1, len(history) + 1)

        for ax, metric in zip(axes, metrics):
            values = [h.get(metric, 0) for h in history]
            ax.plot(epochs, values, marker="o", markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_decoder_weights_heatmap(
        self,
        sae: nn.Module,
        feature_indices: List[int],
        top_k_tokens: int = 20,
        title: str = "SAE Feature-Token Associations",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of decoder weights for selected features.

        Args:
            sae: Trained SparseAutoencoder
            feature_indices: Which features to visualize
            top_k_tokens: Number of top tokens per feature
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        if not HAS_SEABORN:
            raise ImportError("seaborn is required. Install with: pip install seaborn")

        # Collect top tokens across all features
        all_tokens = set()
        feature_tokens = {}

        for feat_idx in feature_indices:
            top_tokens = sae.get_top_tokens_for_feature(feat_idx, self.tokenizer, top_k_tokens)
            feature_tokens[feat_idx] = {t[0]: t[1] for t in top_tokens}
            all_tokens.update([t[0] for t in top_tokens])

        # Create weight matrix
        tokens_list = sorted(all_tokens)
        weight_matrix = np.zeros((len(feature_indices), len(tokens_list)))

        for i, feat_idx in enumerate(feature_indices):
            for j, token in enumerate(tokens_list):
                weight_matrix[i, j] = feature_tokens[feat_idx].get(token, 0)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(tokens_list) * 0.4), len(feature_indices) * 0.5 + 2))

        sns.heatmap(
            weight_matrix,
            xticklabels=tokens_list,
            yticklabels=[f"F{idx}" for idx in feature_indices],
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Weight"}
        )

        ax.set_xlabel("Token")
        ax.set_ylabel("Feature")
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
