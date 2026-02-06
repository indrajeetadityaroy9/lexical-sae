"""Data loading and evaluation metrics."""

import random
from collections import Counter

from datasets import load_dataset

_DATASET_CONFIGS: dict[str, dict[str, str | None]] = {
    "imdb": {"hf_dataset": "imdb", "subset": None, "text_column": "text", "test_split": "test"},
    "sst2": {"hf_dataset": "glue", "subset": "sst2", "text_column": "sentence", "test_split": "validation"},
    "ag_news": {"hf_dataset": "ag_news", "subset": None, "text_column": "text", "test_split": "test"},
}


def _load_split(hf_dataset: str, subset: str | None, split: str, text_column: str, max_samples: int | None) -> tuple[list[str], list[int], int]:
    ds = load_dataset(hf_dataset, subset, split=split)
    texts = list(ds[text_column])
    labels = [int(x) for x in ds["label"]]
    num_labels = len(set(labels))
    random.seed(42)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts)[:max_samples], list(labels)[:max_samples], num_labels


def load_benchmark_data(
    dataset: str, train_samples: int, test_samples: int,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    """Load train and test splits for a benchmark dataset."""
    cfg = _DATASET_CONFIGS[dataset]
    train_texts, train_labels, num_labels = _load_split(cfg["hf_dataset"], cfg["subset"], "train", cfg["text_column"], train_samples)
    test_texts, test_labels, _ = _load_split(cfg["hf_dataset"], cfg["subset"], cfg["test_split"], cfg["text_column"], test_samples)
    return train_texts, train_labels, test_texts, test_labels, num_labels


def load_hatexplain(
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 42,
    rationale_threshold: float = 0.5,
) -> tuple[list[str], list[int], list[list[str]], int]:
    """Load HateXplain with majority-vote labels and token rationales."""
    ds = load_dataset("hatexplain", split=split)

    texts, labels, rationale_tokens_list = [], [], []

    for example in ds:
        tokens = example["post_tokens"]
        texts.append(" ".join(tokens))

        label_counts = Counter(example["annotators"]["label"])
        labels.append(label_counts.most_common(1)[0][0])

        rationales = example["rationales"]
        token_votes = [0] * len(tokens)
        for r in rationales:
            for i, val in enumerate(r):
                token_votes[i] += val

        threshold = len(rationales) * rationale_threshold
        rationale_tokens_list.append([
            tokens[i].lower() for i, v in enumerate(token_votes)
            if v >= threshold and tokens[i].strip()
        ])

    random.seed(seed)
    combined = list(zip(texts, labels, rationale_tokens_list))
    random.shuffle(combined)
    texts, labels, rationale_tokens_list = zip(*combined)

    return list(texts)[:max_samples], list(labels)[:max_samples], list(rationale_tokens_list)[:max_samples], 3


def compute_rationale_agreement(
    model_attributions: list[list[tuple[str, float]]],
    human_rationales: list[list[str]],
    k: int,
) -> float:
    """Compute token-level F1 agreement between model and human rationales."""
    total_precision, total_recall = 0.0, 0.0

    for attrib, human_tokens in zip(model_attributions, human_rationales):
        model_tokens = {t.lower() for t, w in attrib[:k] if w > 0}
        human_set = {t.lower() for t in human_tokens if t.strip()}
        intersection = model_tokens & human_set
        total_precision += len(intersection) / len(model_tokens) if model_tokens else 0.0
        total_recall += len(intersection) / len(human_set) if human_set else 0.0

    n = len(model_attributions)
    p, r = total_precision / n, total_recall / n
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
