"""Automated interpretability scoring for SAE features.

Uses LLMs to generate human-readable descriptions of SAE features
and scores how well those descriptions predict feature activations.

References:
    - Bills et al., "Language models can explain neurons in language models" (2023)
    - EleutherAI automated interpretability: https://blog.eleuther.ai/autointerp/

Note: Requires optional dependencies. Install with:
    pip install openai anthropic
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable
from tqdm import tqdm


@dataclass
class FeatureInterpretation:
    """Interpretation result for a single SAE feature."""
    feature_idx: int
    description: str
    detection_score: float  # F1 of LLM predictions vs actual activations
    top_activating_texts: list[tuple[str, float]]
    activation_threshold: float


def collect_feature_activations(
    sae: torch.nn.Module,
    get_activations_fn: Callable[[list[str]], torch.Tensor],
    texts: list[str],
    feature_idx: int,
    top_k: int = 20,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Collect texts that maximally activate a specific SAE feature.

    Args:
        sae: Sparse autoencoder model
        get_activations_fn: Function that takes texts and returns activations [n, d_model]
        texts: Corpus of texts to search
        feature_idx: Index of feature to find activations for
        top_k: Number of top-activating texts to return
        batch_size: Batch size for processing

    Returns:
        List of (text, activation_value) tuples sorted by activation descending
    """
    all_activations = []

    sae.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            activations = get_activations_fn(batch_texts)  # [batch, d_model]

            # Encode through SAE
            latents = sae.encode(activations)  # [batch, d_sae]
            feature_acts = latents[:, feature_idx].cpu().numpy()

            for text, act in zip(batch_texts, feature_acts):
                all_activations.append((text, float(act)))

    # Sort by activation and return top-k
    all_activations.sort(key=lambda x: x[1], reverse=True)
    return all_activations[:top_k]


def generate_feature_description(
    activating_texts: list[tuple[str, float]],
    llm_fn: Callable[[str], str],
    num_examples: int = 10,
) -> str:
    """Use LLM to generate a description of what a feature represents.

    Args:
        activating_texts: List of (text, activation) tuples
        llm_fn: Function that takes a prompt and returns LLM response
        num_examples: Number of examples to show the LLM

    Returns:
        Human-readable description of the feature
    """
    # Format examples for the prompt
    examples = "\n".join([
        f"- \"{text[:200]}...\" (activation: {act:.2f})"
        if len(text) > 200 else f"- \"{text}\" (activation: {act:.2f})"
        for text, act in activating_texts[:num_examples]
    ])

    prompt = f"""Below are text examples that highly activate a specific feature in a neural network.
Your task is to identify the common pattern or concept that these texts share.

Examples that activate this feature:
{examples}

Based on these examples, provide a concise description (1-2 sentences) of what concept or pattern this feature detects.
Focus on the semantic meaning, not surface-level patterns like punctuation or formatting.

Feature description:"""

    return llm_fn(prompt).strip()


def score_feature_description(
    description: str,
    test_texts: list[str],
    actual_activations: list[float],
    llm_fn: Callable[[str], str],
    activation_threshold: float | None = None,
) -> float:
    """Score how well a description predicts feature activations.

    Uses the LLM to predict which texts should activate the feature based
    on the description, then computes F1 against actual activations.

    Args:
        description: Natural language description of the feature
        test_texts: Texts to evaluate
        actual_activations: Actual activation values for each text
        llm_fn: Function that takes a prompt and returns LLM response
        activation_threshold: Threshold for "active" (default: median of non-zero)

    Returns:
        F1 score between LLM predictions and actual activations
    """
    # Determine activation threshold
    if activation_threshold is None:
        non_zero = [a for a in actual_activations if a > 0]
        activation_threshold = np.median(non_zero) if non_zero else 0.1

    # Get actual binary labels
    actual_labels = [1 if a > activation_threshold else 0 for a in actual_activations]

    # Ask LLM to predict for each text
    predicted_labels = []
    for text in test_texts:
        prompt = f"""A neural network feature is described as: "{description}"

Does the following text likely activate this feature? Answer only "yes" or "no".

Text: "{text[:500]}"

Answer:"""

        response = llm_fn(prompt).strip().lower()
        predicted_labels.append(1 if "yes" in response else 0)

    # Compute F1 score
    tp = sum(1 for p, a in zip(predicted_labels, actual_labels) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(predicted_labels, actual_labels) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(predicted_labels, actual_labels) if p == 0 and a == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def interpret_feature(
    sae: torch.nn.Module,
    get_activations_fn: Callable[[list[str]], torch.Tensor],
    texts: list[str],
    feature_idx: int,
    llm_fn: Callable[[str], str],
    top_k_examples: int = 20,
    n_test_samples: int = 50,
) -> FeatureInterpretation:
    """Generate and score interpretation for a single SAE feature.

    Args:
        sae: Sparse autoencoder model
        get_activations_fn: Function to get model activations
        texts: Corpus of texts
        feature_idx: Feature to interpret
        llm_fn: LLM function for generation/scoring
        top_k_examples: Number of top-activating examples
        n_test_samples: Number of samples for scoring

    Returns:
        FeatureInterpretation with description and scores
    """
    # Collect activating texts
    activating_texts = collect_feature_activations(
        sae, get_activations_fn, texts, feature_idx, top_k=top_k_examples
    )

    if not activating_texts or activating_texts[0][1] == 0:
        return FeatureInterpretation(
            feature_idx=feature_idx,
            description="[Dead feature - never activates]",
            detection_score=0.0,
            top_activating_texts=[],
            activation_threshold=0.0,
        )

    # Generate description
    description = generate_feature_description(activating_texts, llm_fn)

    # Score description on held-out samples
    # Use a mix of high and low activating texts for balanced evaluation
    test_indices = np.random.choice(len(texts), min(n_test_samples, len(texts)), replace=False)
    test_texts_sample = [texts[i] for i in test_indices]

    # Get activations for test texts
    all_acts = collect_feature_activations(
        sae, get_activations_fn, test_texts_sample, feature_idx, top_k=len(test_texts_sample)
    )
    test_activations = [act for _, act in all_acts]

    # Compute threshold from training examples
    activation_threshold = np.median([act for _, act in activating_texts])

    # Score
    detection_score = score_feature_description(
        description, test_texts_sample, test_activations, llm_fn, activation_threshold
    )

    return FeatureInterpretation(
        feature_idx=feature_idx,
        description=description,
        detection_score=detection_score,
        top_activating_texts=activating_texts[:5],
        activation_threshold=activation_threshold,
    )


def compute_interpretability_scores(
    sae: torch.nn.Module,
    get_activations_fn: Callable[[list[str]], torch.Tensor],
    texts: list[str],
    llm_fn: Callable[[str], str],
    n_features: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute aggregate interpretability scores across sampled features.

    Args:
        sae: Sparse autoencoder model
        get_activations_fn: Function to get model activations
        texts: Corpus of texts
        llm_fn: LLM function
        n_features: Number of features to sample
        seed: Random seed for feature sampling

    Returns:
        Dictionary with:
            - mean_detection_score: Average F1 across features
            - feature_coverage: Fraction with score > 0.5
            - interpretations: List of FeatureInterpretation objects
    """
    np.random.seed(seed)

    # Sample features (excluding dead ones)
    d_sae = sae.d_sae

    # First pass: identify active features
    print("Identifying active features...")
    sample_acts = collect_feature_activations(
        sae, get_activations_fn, texts[:100], feature_idx=0, top_k=100
    )

    # Get feature activation counts
    active_features = []
    with torch.no_grad():
        for i in range(0, min(1000, len(texts)), 32):
            batch_texts = texts[i:i+32]
            activations = get_activations_fn(batch_texts)
            latents = sae.encode(activations)
            active_mask = (latents > 0).any(dim=0)
            active_features.append(active_mask.cpu())

    active_mask = torch.stack(active_features).any(dim=0)
    active_indices = active_mask.nonzero().squeeze(-1).tolist()

    if len(active_indices) == 0:
        return {
            "mean_detection_score": 0.0,
            "feature_coverage": 0.0,
            "interpretations": [],
        }

    # Sample from active features
    sampled_indices = np.random.choice(
        active_indices,
        min(n_features, len(active_indices)),
        replace=False
    ).tolist()

    # Interpret sampled features
    interpretations = []
    for idx in tqdm(sampled_indices, desc="Interpreting features"):
        interp = interpret_feature(
            sae, get_activations_fn, texts, idx, llm_fn
        )
        interpretations.append(interp)

    # Compute aggregate metrics
    scores = [i.detection_score for i in interpretations]
    mean_score = np.mean(scores) if scores else 0.0
    coverage = np.mean([1 if s > 0.5 else 0 for s in scores]) if scores else 0.0

    return {
        "mean_detection_score": float(mean_score),
        "feature_coverage": float(coverage),
        "interpretations": interpretations,
    }


# Convenience functions for common LLM providers

def create_openai_llm_fn(model: str = "gpt-4o-mini", api_key: str | None = None) -> Callable[[str], str]:
    """Create LLM function using OpenAI API.

    Args:
        model: OpenAI model name
        api_key: API key (uses OPENAI_API_KEY env var if not provided)

    Returns:
        Function that takes prompt and returns response
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package required. Install with: pip install openai")

    client = OpenAI(api_key=api_key)

    def llm_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        return response.choices[0].message.content

    return llm_fn


def create_anthropic_llm_fn(model: str = "claude-3-haiku-20240307", api_key: str | None = None) -> Callable[[str], str]:
    """Create LLM function using Anthropic API.

    Args:
        model: Anthropic model name
        api_key: API key (uses ANTHROPIC_API_KEY env var if not provided)

    Returns:
        Function that takes prompt and returns response
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic package required. Install with: pip install anthropic")

    client = Anthropic(api_key=api_key)

    def llm_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return llm_fn
