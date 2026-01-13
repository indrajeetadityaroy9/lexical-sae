"""
Feature Analysis for Sparse Autoencoder interpretability.

Analyzes SAE features to identify monosemantic concepts and
generate human-readable interpretations of what each feature represents.

Key analyses:
1. Activation frequency: How often does each feature fire?
2. Token association: Which vocabulary terms activate each feature?
3. Document clustering: Do features correspond to semantic clusters?
4. Monosemanticity scoring: How "clean" is each feature?

Reference:
    Sparse Autoencoders Find Highly Interpretable Features in Language Models
    https://arxiv.org/abs/2309.08600
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


@dataclass
class FeatureInfo:
    """Information about a single SAE feature."""
    feature_idx: int
    activation_frequency: float  # Fraction of docs where feature fires
    mean_activation: float  # Average activation when active
    top_tokens: List[Tuple[str, float]]  # Top associated vocab tokens
    top_documents: List[Tuple[int, float]]  # Top activating doc indices
    monosemanticity_score: float  # How clean/interpretable the feature is
    interpretation: Optional[str] = None  # Human-readable interpretation


class FeatureAnalyzer:
    """
    Analyzes SAE features to identify monosemantic concepts.

    Given a trained SAE and a corpus of documents encoded as SPLADE vectors,
    this analyzer computes various statistics and interpretability metrics
    for each learned feature.
    """

    def __init__(
        self,
        sae: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the feature analyzer.

        Args:
            sae: Trained SparseAutoencoder
            tokenizer: HuggingFace tokenizer for decoding tokens
            device: Device for computation
        """
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sae.to(self.device)
        self.sae.eval()

        # Cache for feature activations
        self._activation_cache: Optional[torch.Tensor] = None
        self._feature_stats: Dict[int, FeatureInfo] = {}

    def compute_activations(
        self,
        vectors: torch.Tensor,
        batch_size: int = 256
    ) -> torch.Tensor:
        """
        Compute SAE hidden activations for all vectors.

        Args:
            vectors: SPLADE vectors [num_docs, vocab_size]
            batch_size: Batch size for processing

        Returns:
            activations: SAE hidden activations [num_docs, hidden_dim]
        """
        self.sae.eval()
        all_activations = []

        with torch.no_grad():
            for i in tqdm(range(0, len(vectors), batch_size), desc="Computing activations"):
                batch = vectors[i:i + batch_size].to(self.device)
                output = self.sae(batch, return_loss=False)
                all_activations.append(output.hidden.cpu())

        self._activation_cache = torch.cat(all_activations, dim=0)
        return self._activation_cache

    def get_activation_statistics(
        self,
        activations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute global statistics about feature activations.

        Returns:
            Dictionary containing:
            - frequency: Fraction of docs where each feature is active
            - mean_when_active: Mean activation value when active
            - max_activation: Maximum activation value
            - std_when_active: Std of activation when active
        """
        if activations is None:
            activations = self._activation_cache

        if activations is None:
            raise ValueError("No activations available. Call compute_activations first.")

        # Binary mask of active features
        active_mask = activations > 1e-6

        # Frequency: fraction of documents where feature fires
        frequency = active_mask.float().mean(dim=0)

        # Mean activation when active
        masked_activations = activations * active_mask.float()
        sum_activations = masked_activations.sum(dim=0)
        count_active = active_mask.sum(dim=0).clamp(min=1)
        mean_when_active = sum_activations / count_active

        # Max activation
        max_activation = activations.max(dim=0).values

        return {
            "frequency": frequency,
            "mean_when_active": mean_when_active,
            "max_activation": max_activation,
            "active_count": active_mask.sum(dim=0)
        }

    def analyze_feature(
        self,
        feature_idx: int,
        activations: Optional[torch.Tensor] = None,
        top_k_tokens: int = 20,
        top_k_docs: int = 10
    ) -> FeatureInfo:
        """
        Analyze a single SAE feature in detail.

        Args:
            feature_idx: Index of the feature to analyze
            activations: Pre-computed activations (or use cached)
            top_k_tokens: Number of top tokens to retrieve
            top_k_docs: Number of top-activating documents

        Returns:
            FeatureInfo with detailed analysis
        """
        if activations is None:
            activations = self._activation_cache

        if activations is None:
            raise ValueError("No activations available. Call compute_activations first.")

        # Feature activations across all documents
        feature_acts = activations[:, feature_idx]

        # Activation frequency
        active_mask = feature_acts > 1e-6
        activation_frequency = active_mask.float().mean().item()

        # Mean activation when active
        if active_mask.sum() > 0:
            mean_activation = feature_acts[active_mask].mean().item()
        else:
            mean_activation = 0.0

        # Top associated vocabulary tokens (from decoder weights)
        top_tokens = self.sae.get_top_tokens_for_feature(
            feature_idx,
            self.tokenizer,
            k=top_k_tokens
        )

        # Top activating documents
        top_values, top_indices = torch.topk(feature_acts, min(top_k_docs, len(feature_acts)))
        top_documents = [
            (idx.item(), val.item())
            for idx, val in zip(top_indices, top_values)
        ]

        # Monosemanticity score
        monosemanticity = self._compute_monosemanticity(feature_idx, activations)

        info = FeatureInfo(
            feature_idx=feature_idx,
            activation_frequency=activation_frequency,
            mean_activation=mean_activation,
            top_tokens=top_tokens,
            top_documents=top_documents,
            monosemanticity_score=monosemanticity
        )

        self._feature_stats[feature_idx] = info
        return info

    def _compute_monosemanticity(
        self,
        feature_idx: int,
        activations: torch.Tensor
    ) -> float:
        """
        Compute monosemanticity score for a feature.

        A monosemantic feature has low entropy in its token associations,
        meaning it corresponds to a single, clear concept.

        Score is based on:
        1. Entropy of decoder weights (lower = more focused)
        2. Consistency of activating documents (higher = more consistent)

        Returns:
            Score between 0 (polysemantic) and 1 (monosemantic)
        """
        # Get decoder weights for this feature
        weights = self.sae.get_feature_weights(feature_idx)
        abs_weights = weights.abs()

        # Normalize to probability distribution
        prob = abs_weights / (abs_weights.sum() + 1e-10)

        # Compute entropy (lower = more monosemantic)
        log_prob = torch.log(prob + 1e-10)
        entropy = -(prob * log_prob).sum().item()

        # Normalize entropy to [0, 1] (max entropy for uniform distribution)
        max_entropy = np.log(len(weights))
        normalized_entropy = entropy / max_entropy

        # Invert so higher = more monosemantic
        monosemanticity = 1.0 - normalized_entropy

        # Scale to reasonable range (pure entropy score tends to be very high)
        monosemanticity = min(1.0, monosemanticity * 2)

        return monosemanticity

    def analyze_all_features(
        self,
        activations: Optional[torch.Tensor] = None,
        top_k: int = 100,
        min_frequency: float = 0.001
    ) -> List[FeatureInfo]:
        """
        Analyze all SAE features and return top-k by activation frequency.

        Args:
            activations: Pre-computed activations
            top_k: Number of top features to analyze in detail
            min_frequency: Minimum activation frequency to consider

        Returns:
            List of FeatureInfo for top features
        """
        if activations is None:
            activations = self._activation_cache

        if activations is None:
            raise ValueError("No activations available. Call compute_activations first.")

        # Get global statistics
        stats = self.get_activation_statistics(activations)

        # Find features that fire often enough
        frequencies = stats["frequency"]
        active_features = (frequencies > min_frequency).nonzero().squeeze(-1)

        # Sort by frequency
        feature_freqs = [(idx.item(), frequencies[idx].item()) for idx in active_features]
        feature_freqs.sort(key=lambda x: x[1], reverse=True)

        # Analyze top-k features
        results = []
        for feature_idx, _ in tqdm(feature_freqs[:top_k], desc="Analyzing features"):
            info = self.analyze_feature(feature_idx, activations)
            results.append(info)

        return results

    def generate_feature_report(
        self,
        features: Optional[List[FeatureInfo]] = None,
        top_k: int = 50
    ) -> str:
        """
        Generate human-readable report of analyzed features.

        Args:
            features: List of FeatureInfo (or use cached)
            top_k: Number of features to include

        Returns:
            Formatted string report
        """
        if features is None:
            features = list(self._feature_stats.values())

        if not features:
            return "No features analyzed yet. Call analyze_all_features first."

        # Sort by monosemanticity score
        features = sorted(features, key=lambda x: x.monosemanticity_score, reverse=True)[:top_k]

        lines = [
            "=" * 80,
            "SPARSE AUTOENCODER FEATURE ANALYSIS REPORT",
            "=" * 80,
            f"Total features analyzed: {len(self._feature_stats)}",
            f"Top {top_k} most monosemantic features:",
            "-" * 80,
        ]

        for rank, info in enumerate(features, 1):
            lines.append(f"\n[Feature {info.feature_idx}] (Rank #{rank})")
            lines.append(f"  Monosemanticity Score: {info.monosemanticity_score:.4f}")
            lines.append(f"  Activation Frequency: {info.activation_frequency:.4%}")
            lines.append(f"  Mean Activation: {info.mean_activation:.4f}")
            lines.append(f"  Top Tokens: {', '.join([f'{t[0]}({t[1]:.2f})' for t in info.top_tokens[:10]])}")

            if info.interpretation:
                lines.append(f"  Interpretation: {info.interpretation}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def find_similar_features(
        self,
        feature_idx: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find features with similar decoder weight patterns.

        Args:
            feature_idx: Reference feature index
            top_k: Number of similar features to return

        Returns:
            List of (feature_idx, similarity) tuples
        """
        ref_weights = self.sae.get_feature_weights(feature_idx)
        ref_weights = ref_weights / (ref_weights.norm() + 1e-10)

        similarities = []
        for i in range(self.sae.hidden_dim):
            if i == feature_idx:
                continue
            other_weights = self.sae.get_feature_weights(i)
            other_weights = other_weights / (other_weights.norm() + 1e-10)
            sim = (ref_weights * other_weights).sum().item()
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_feature_activation_for_text(
        self,
        text: str,
        vectorizer: nn.Module,
        top_k: int = 10
    ) -> List[Tuple[int, float, List[str]]]:
        """
        Get which SAE features activate for a given text.

        Args:
            text: Input text
            vectorizer: SPLADE vectorizer model
            top_k: Number of top features to return

        Returns:
            List of (feature_idx, activation, top_tokens) tuples
        """
        vectorizer.eval()
        self.sae.eval()

        # Get SPLADE vector for text
        with torch.no_grad():
            splade_vec = vectorizer.encode_text(text, device=self.device)
            output = self.sae(splade_vec, return_loss=False)
            hidden = output.hidden.squeeze(0)

        # Get top-k active features
        values, indices = torch.topk(hidden, top_k)

        results = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            if val > 1e-6:
                top_tokens = self.sae.get_top_tokens_for_feature(idx, self.tokenizer, k=5)
                token_strs = [t[0] for t in top_tokens]
                results.append((idx, val, token_strs))

        return results
