import random

import numpy
from datasets import load_dataset


_DATASETS = {
    "sst2": {"path": "glue", "name": "sst2", "text_col": "sentence", "test_split": "validation", "num_labels": 2},
    "ag_news": {"path": "ag_news", "name": None, "text_col": "text", "test_split": "test", "num_labels": 4},
    "imdb": {"path": "imdb", "name": None, "text_col": "text", "test_split": "test", "num_labels": 2},
    "yelp": {"path": "yelp_polarity", "name": None, "text_col": "text", "test_split": "test", "num_labels": 2},
    "esnli": {"path": "esnli", "name": None, "text_cols": ["premise", "hypothesis"], "test_split": "test", "num_labels": 3},
    "civilcomments": {"path": "google/civil_comments", "name": None, "text_col": "text", "label_col": "toxicity", "label_threshold": 0.5, "test_split": "test", "num_labels": 2},
}

# Identity columns in CivilComments for bias analysis
CIVILCOMMENTS_IDENTITY_COLS = [
    "male", "female", "transgender", "other_gender",
    "heterosexual", "homosexual_gay_or_lesbian", "bisexual", "other_sexual_orientation",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
    "black", "white", "asian", "latino", "other_race_or_ethnicity",
    "physical_disability", "intellectual_or_learning_disability",
    "psychiatric_or_mental_illness", "other_disability",
]


def _shuffle_and_truncate(
    texts: list[str], labels: list[int], max_samples: int, seed: int
) -> tuple[list[str], list[int]]:
    combined = list(zip(texts, labels))
    random.Random(seed).shuffle(combined)
    shuffled_texts, shuffled_labels = zip(*combined)
    if max_samples <= 0:
        return list(shuffled_texts), list(shuffled_labels)
    return list(shuffled_texts)[:max_samples], list(shuffled_labels)[:max_samples]


def infer_max_length(texts: list[str], tokenizer, model_name: str | None = None) -> int:
    sample = texts[:500]
    lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in sample]
    p99 = int(numpy.percentile(lengths, 99))
    aligned = ((p99 + 7) // 8) * 8

    if model_name is not None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        max_pos = getattr(config, "max_position_embeddings", 512)
    else:
        max_pos = 512

    return max(64, min(max_pos, aligned))


def _load_split(cfg: dict, split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset(cfg["path"], cfg["name"], split=split)
    if "text_cols" in cfg:
        cols = cfg["text_cols"]
        texts = [f"{row[cols[0]]} [SEP] {row[cols[1]]}" for row in dataset]
    else:
        texts = list(dataset[cfg["text_col"]])
    label_col = cfg.get("label_col", "label")
    threshold = cfg.get("label_threshold")
    if threshold is not None:
        labels = [int(float(row[label_col]) >= threshold) for row in dataset]
    else:
        labels = [int(label) for label in dataset[label_col]]
    return _shuffle_and_truncate(texts, labels, max_samples, seed)


def load_dataset_by_name(
    name: str,
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    cfg = _DATASETS[name]
    train_texts, train_labels = _load_split(cfg, "train", train_samples, seed)
    test_texts, test_labels = _load_split(cfg, cfg["test_split"], test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, cfg["num_labels"]


def load_civilcomments_with_identity(
    train_samples: int = -1,
    test_samples: int = -1,
    seed: int = 42,
    identity_threshold: float = 0.5,
) -> tuple[list[str], list[int], list[str], list[int], int, list[dict[str, bool]], list[dict[str, bool]]]:
    """Load CivilComments with per-example identity annotations for bias analysis.

    Returns:
        train_texts, train_labels, test_texts, test_labels, num_labels,
        train_identities, test_identities
        where each identity dict maps identity name -> bool (present in comment).
    """
    cfg = _DATASETS["civilcomments"]

    def _load_with_identity(split, max_samples):
        dataset = load_dataset(cfg["path"], cfg["name"], split=split)
        texts = list(dataset[cfg["text_col"]])
        threshold = cfg["label_threshold"]
        labels = [int(float(row[cfg["label_col"]]) >= threshold) for row in dataset]
        identities = []
        for row in dataset:
            ident = {}
            for col in CIVILCOMMENTS_IDENTITY_COLS:
                val = row.get(col)
                ident[col] = val is not None and float(val) >= identity_threshold
            identities.append(ident)
        combined = list(zip(texts, labels, identities))
        random.Random(seed).shuffle(combined)
        if max_samples > 0:
            combined = combined[:max_samples]
        t, l, i = zip(*combined) if combined else ([], [], [])
        return list(t), list(l), list(i)

    train_texts, train_labels, train_ids = _load_with_identity("train", train_samples)
    test_texts, test_labels, test_ids = _load_with_identity(cfg["test_split"], test_samples)
    return train_texts, train_labels, test_texts, test_labels, cfg["num_labels"], train_ids, test_ids
