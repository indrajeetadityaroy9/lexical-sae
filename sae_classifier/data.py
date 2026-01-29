"""Data loading from HuggingFace datasets."""

import random
from collections import Counter

from datasets import load_dataset


def load_classification_data(
    dataset: str,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
    subset: str | None = None,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[int], int]:
    """Load any HuggingFace text classification dataset.

    Args:
        dataset: HuggingFace dataset identifier (e.g., 'imdb', 'ag_news')
        split: Data split ('train', 'test', 'validation')
        max_samples: Maximum samples to return (None for all)
        seed: Random seed for shuffling
        subset: Dataset subset/config name (e.g., 'sst2' for glue)
        text_column: Name of the text column
        label_column: Name of the label column

    Returns:
        Tuple of (texts, labels, num_labels)
    """
    ds = load_dataset(dataset, subset, split=split) if subset else load_dataset(dataset, split=split)

    texts = list(ds[text_column])
    labels = [int(x) for x in ds[label_column]]
    num_labels = len(set(labels))

    random.seed(seed)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]

    return texts, labels, num_labels


def load_hatexplain(
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
    rationale_threshold: float = 0.5,
) -> tuple[list[str], list[int], list[list[str]], int]:
    """Load HateXplain dataset with token-level rationale annotations.

    Args:
        split: Data split ('train', 'validation', 'test')
        max_samples: Maximum samples to return
        seed: Random seed for shuffling
        rationale_threshold: Fraction of annotators needed to include token

    Returns:
        Tuple of (texts, labels, rationale_tokens, num_labels)
    """
    ds = load_dataset("hatexplain", split=split)

    texts = []
    labels = []
    rationale_tokens_list = []

    for example in ds:
        tokens = example["post_tokens"]
        text = " ".join(tokens)
        texts.append(text)

        annotator_labels = example["annotators"]["label"]
        label_counts = Counter(annotator_labels)
        majority_label = label_counts.most_common(1)[0][0]
        labels.append(majority_label)

        rationales = example["rationales"]
        if rationales:
            token_votes = [0] * len(tokens)
            for annotator_rationale in rationales:
                for i, is_rationale in enumerate(annotator_rationale):
                    if i < len(token_votes):
                        token_votes[i] += is_rationale

            threshold_count = len(rationales) * rationale_threshold
            rationale_tokens = [
                tokens[i].lower()
                for i, votes in enumerate(token_votes)
                if votes >= threshold_count and tokens[i].strip()
            ]
        else:
            rationale_tokens = []

        rationale_tokens_list.append(rationale_tokens)

    random.seed(seed)
    combined = list(zip(texts, labels, rationale_tokens_list))
    random.shuffle(combined)
    texts, labels, rationale_tokens_list = zip(*combined) if combined else ([], [], [])
    texts, labels, rationale_tokens_list = list(texts), list(labels), list(rationale_tokens_list)

    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        rationale_tokens_list = rationale_tokens_list[:max_samples]

    return texts, labels, rationale_tokens_list, 3


def rationale_agreement(
    model_attributions: list[list[tuple[str, float]]],
    human_rationales: list[list[str]],
    k: int | None = None,
) -> dict[str, float]:
    """Compute agreement between model attributions and human rationales.

    Args:
        model_attributions: List of (token, weight) lists from model explanations
        human_rationales: List of rationale token lists from human annotations
        k: Only consider top-k model attributions (None = use all positive weights)

    Returns:
        Dictionary with precision, recall, f1, iou
    """
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    n_valid = 0

    for attrib, human_tokens in zip(model_attributions, human_rationales):
        if not attrib or not human_tokens:
            continue

        if k is not None:
            model_tokens = set(t.lower() for t, w in attrib[:k] if w > 0)
        else:
            model_tokens = set(t.lower() for t, w in attrib if w > 0)

        human_set = set(t.lower() for t in human_tokens if t.strip())

        if not model_tokens or not human_set:
            continue

        intersection = model_tokens & human_set
        union = model_tokens | human_set

        precision = len(intersection) / len(model_tokens) if model_tokens else 0
        recall = len(intersection) / len(human_set) if human_set else 0
        iou = len(intersection) / len(union) if union else 0

        total_precision += precision
        total_recall += recall
        total_iou += iou
        n_valid += 1

    if n_valid == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    avg_precision = total_precision / n_valid
    avg_recall = total_recall / n_valid
    avg_iou = total_iou / n_valid

    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0.0

    return {"precision": avg_precision, "recall": avg_recall, "f1": f1, "iou": avg_iou}
