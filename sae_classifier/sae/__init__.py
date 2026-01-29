"""Sparse Autoencoder module for interpretable feature extraction."""

from sae_classifier.sae.architecture import BatchTopKSAE, TopKSAE, create_sae, SAEOutput
from sae_classifier.sae.trainer import SAETrainer, collect_activations, load_sae
from sae_classifier.sae.autointerpret import (
    FeatureInterpretation,
    collect_feature_activations,
    generate_feature_description,
    score_feature_description,
    interpret_feature,
    compute_interpretability_scores,
    create_openai_llm_fn,
    create_anthropic_llm_fn,
)

__all__ = [
    # Architecture
    "BatchTopKSAE",
    "TopKSAE",
    "create_sae",
    "SAEOutput",
    # Training
    "SAETrainer",
    "collect_activations",
    "load_sae",
    # Autointerpretability
    "FeatureInterpretation",
    "collect_feature_activations",
    "generate_feature_description",
    "score_feature_description",
    "interpret_feature",
    "compute_interpretability_scores",
    "create_openai_llm_fn",
    "create_anthropic_llm_fn",
]
