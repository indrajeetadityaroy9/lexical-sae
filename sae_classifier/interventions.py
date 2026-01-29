"""Intervention-based evaluation metrics for SAE interpretability.

These metrics go beyond perturbation-based faithfulness by directly
intervening on SAE feature activations and measuring causal effects.

References:
    - Heimersheim & Nanda, "How to use and interpret activation patching" (2024)
    - Turner et al., "Steering Language Models With Activation Engineering" (2023)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Protocol
from tqdm import tqdm


class SAEPredictor(Protocol):
    """Protocol for SAE-based classifiers."""
    def predict_proba(self, texts: list[str]) -> list[list[float]]: ...
    def explain(self, text: str, top_k: int) -> list[tuple[int, float]]: ...
    def _get_sae_features(self, texts: list[str]) -> torch.Tensor: ...
    sae: torch.nn.Module
    classifier: torch.nn.Module
    device: torch.device


def activation_patching(
    model: SAEPredictor,
    text: str,
    feature_idx: int,
    patch_value: float = 0.0,
) -> dict[str, float]:
    """Measure causal effect of a single SAE feature by patching its activation.

    Replaces the activation of feature_idx with patch_value and measures
    the change in prediction.

    Args:
        model: SAE-based classifier
        text: Input text
        feature_idx: Index of SAE feature to patch
        patch_value: Value to set the feature to (default: 0 = ablation)

    Returns:
        Dictionary with:
            - original_prob: Confidence before patching
            - patched_prob: Confidence after patching
            - causal_effect: |original - patched|
            - pred_class: Predicted class
    """
    # Get original prediction
    original_probs = model.predict_proba([text])[0]
    pred_class = int(np.argmax(original_probs))
    original_conf = original_probs[pred_class]

    # Get SAE features
    features = model._get_sae_features([text])[0].clone()  # [d_sae]

    # Patch the feature
    original_value = features[feature_idx].item()
    features[feature_idx] = patch_value

    # Get patched prediction
    with torch.no_grad():
        logits = model.classifier(features.unsqueeze(0).to(model.device))
        patched_probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

    patched_conf = patched_probs[pred_class]

    return {
        "original_prob": original_conf,
        "patched_prob": patched_conf,
        "causal_effect": abs(original_conf - patched_conf),
        "original_feature_value": original_value,
        "pred_class": pred_class,
    }


def feature_ablation_curve(
    model: SAEPredictor,
    text: str,
    max_features: int = 20,
) -> dict[str, list[float]]:
    """Measure prediction change as top features are progressively ablated.

    Similar to comprehensiveness but operates on SAE features directly
    rather than input tokens.

    Args:
        model: SAE-based classifier
        text: Input text
        max_features: Maximum number of features to ablate

    Returns:
        Dictionary with:
            - k_values: Number of features ablated
            - confidences: Confidence at each k
            - effects: Cumulative causal effect at each k
    """
    # Get original prediction
    original_probs = model.predict_proba([text])[0]
    pred_class = int(np.argmax(original_probs))
    original_conf = original_probs[pred_class]

    # Get top features
    explanations = model.explain(text, top_k=max_features)

    k_values = []
    confidences = [original_conf]
    effects = [0.0]

    # Get SAE features
    features = model._get_sae_features([text])[0].clone()

    for k, (feat_idx, _) in enumerate(explanations, 1):
        # Ablate this feature
        features[feat_idx] = 0.0

        # Get new prediction
        with torch.no_grad():
            logits = model.classifier(features.unsqueeze(0).to(model.device))
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

        conf = probs[pred_class]
        k_values.append(k)
        confidences.append(conf)
        effects.append(original_conf - conf)

    return {
        "k_values": k_values,
        "confidences": confidences[1:],  # Exclude original
        "effects": effects[1:],
        "original_conf": original_conf,
    }


def feature_necessity(
    model: SAEPredictor,
    texts: list[str],
    k_values: list[int] = [1, 5, 10, 20],
) -> dict[int, float]:
    """Compute feature necessity: prediction drop when top-k SAE features ablated.

    This is the SAE-feature analog of comprehensiveness. Higher values indicate
    that the identified features are necessary for the prediction.

    Args:
        model: SAE-based classifier
        texts: List of input texts
        k_values: Number of top features to ablate

    Returns:
        Dictionary mapping k -> average confidence drop
    """
    results = {k: [] for k in k_values}

    for text in tqdm(texts, desc="Computing feature necessity"):
        curve = feature_ablation_curve(model, text, max_features=max(k_values))

        for k in k_values:
            if k <= len(curve["effects"]):
                results[k].append(curve["effects"][k - 1])
            else:
                results[k].append(curve["effects"][-1] if curve["effects"] else 0.0)

    return {k: float(np.mean(scores)) if scores else 0.0 for k, scores in results.items()}


def feature_sufficiency(
    model: SAEPredictor,
    texts: list[str],
    k_values: list[int] = [1, 5, 10, 20],
) -> dict[int, float]:
    """Compute feature sufficiency: prediction using ONLY top-k SAE features.

    Zeroes all features except the top-k and measures how well the model
    can still predict. Lower values indicate the top features are sufficient.

    Args:
        model: SAE-based classifier
        texts: List of input texts
        k_values: Number of top features to keep

    Returns:
        Dictionary mapping k -> average confidence drop (original - with_only_top_k)
    """
    results = {k: [] for k in k_values}

    for text in tqdm(texts, desc="Computing feature sufficiency"):
        # Get original prediction
        original_probs = model.predict_proba([text])[0]
        pred_class = int(np.argmax(original_probs))
        original_conf = original_probs[pred_class]

        # Get SAE features and top-k indices
        features = model._get_sae_features([text])[0]
        explanations = model.explain(text, top_k=max(k_values))
        top_indices = [idx for idx, _ in explanations]

        for k in k_values:
            # Keep only top-k features
            masked_features = torch.zeros_like(features)
            for idx in top_indices[:k]:
                masked_features[idx] = features[idx]

            # Get prediction with only top-k
            with torch.no_grad():
                logits = model.classifier(masked_features.unsqueeze(0).to(model.device))
                probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

            conf = probs[pred_class]
            results[k].append(original_conf - conf)

    return {k: float(np.mean(scores)) if scores else 0.0 for k, scores in results.items()}


def steering_effectiveness(
    model: SAEPredictor,
    texts: list[str],
    feature_idx: int,
    steering_strengths: list[float] = [-2.0, -1.0, 1.0, 2.0],
) -> dict[str, float]:
    """Measure how adding/subtracting a feature direction affects predictions.

    For each text, adds steering_strength * decoder_direction to the
    activation and measures prediction change.

    Args:
        model: SAE-based classifier
        texts: List of input texts
        feature_idx: Index of feature to steer with
        steering_strengths: Multipliers for steering direction

    Returns:
        Dictionary with average effects at each steering strength
    """
    results = {f"strength_{s}": [] for s in steering_strengths}

    for text in texts:
        original_probs = model.predict_proba([text])[0]
        pred_class = int(np.argmax(original_probs))
        original_conf = original_probs[pred_class]

        features = model._get_sae_features([text])[0].clone()

        for strength in steering_strengths:
            # Add steering to the feature
            steered_features = features.clone()
            steered_features[feature_idx] = steered_features[feature_idx] + strength

            with torch.no_grad():
                logits = model.classifier(steered_features.unsqueeze(0).to(model.device))
                probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

            steered_conf = probs[pred_class]
            results[f"strength_{strength}"].append(original_conf - steered_conf)

    return {k: float(np.mean(v)) for k, v in results.items()}


def compute_intervention_metrics(
    model: SAEPredictor,
    texts: list[str],
    k_values: list[int] = [1, 5, 10],
) -> dict[str, float | dict]:
    """Compute all intervention-based metrics.

    Args:
        model: SAE-based classifier
        texts: List of test texts
        k_values: k values for necessity/sufficiency

    Returns:
        Dictionary with all metrics
    """
    necessity = feature_necessity(model, texts, k_values)
    sufficiency = feature_sufficiency(model, texts, k_values)

    # Compute average ablation effect for top-5 features across all texts
    avg_ablation_effects = []
    for text in tqdm(texts[:50], desc="Computing ablation effects"):  # Limit for speed
        explanations = model.explain(text, top_k=5)
        for feat_idx, _ in explanations:
            result = activation_patching(model, text, feat_idx)
            avg_ablation_effects.append(result["causal_effect"])

    return {
        "feature_necessity": necessity,
        "feature_sufficiency": sufficiency,
        "avg_ablation_effect": float(np.mean(avg_ablation_effects)) if avg_ablation_effects else 0.0,
    }
