"""Dataset loading for classification benchmarks."""

import random

from datasets import load_dataset


def _shuffle_and_truncate(
    texts: list[str], labels: list[int], max_samples: int, seed: int
) -> tuple[list[str], list[int]]:
    combined = list(zip(texts, labels))
    random.Random(seed).shuffle(combined)
    shuffled_texts, shuffled_labels = zip(*combined)
    return list(shuffled_texts)[:max_samples], list(shuffled_labels)[:max_samples]


# --- SST-2 (2-class sentiment) ---

def _load_sst2_split(split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset("glue", "sst2", split=split)
    texts = list(dataset["sentence"])
    labels = [int(label) for label in dataset["label"]]
    return _shuffle_and_truncate(texts, labels, max_samples, seed)


def load_sst2_data(
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    train_texts, train_labels = _load_sst2_split("train", train_samples, seed)
    test_texts, test_labels = _load_sst2_split("validation", test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, 2


# --- AG News (4-class topic classification) ---

def _load_ag_news_split(split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset("ag_news", split=split)
    texts = list(dataset["text"])
    labels = [int(label) for label in dataset["label"]]
    return _shuffle_and_truncate(texts, labels, max_samples, seed)


def load_ag_news_data(
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    train_texts, train_labels = _load_ag_news_split("train", train_samples, seed)
    test_texts, test_labels = _load_ag_news_split("test", test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, 4


# --- IMDB (2-class sentiment) ---

def _load_imdb_split(split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset("imdb", split=split)
    texts = list(dataset["text"])
    labels = [int(label) for label in dataset["label"]]
    return _shuffle_and_truncate(texts, labels, max_samples, seed)


def load_imdb_data(
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    train_texts, train_labels = _load_imdb_split("train", train_samples, seed)
    test_texts, test_labels = _load_imdb_split("test", test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, 2


# --- Yelp Polarity (2-class sentiment) ---

def _load_yelp_split(split: str, max_samples: int, seed: int) -> tuple[list[str], list[int]]:
    dataset = load_dataset("yelp_polarity", split=split)
    texts = list(dataset["text"])
    labels = [int(label) for label in dataset["label"]]
    return _shuffle_and_truncate(texts, labels, max_samples, seed)


def load_yelp_data(
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    train_texts, train_labels = _load_yelp_split("train", train_samples, seed)
    test_texts, test_labels = _load_yelp_split("test", test_samples, seed)
    return train_texts, train_labels, test_texts, test_labels, 2


# --- Dispatch ---

def load_dataset_by_name(
    name: str,
    train_samples: int,
    test_samples: int,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    """Dispatch to dataset-specific loader by name."""
    loaders = {
        "sst2": load_sst2_data,
        "ag_news": load_ag_news_data,
        "imdb": load_imdb_data,
        "yelp": load_yelp_data,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(loaders.keys())}")
    return loaders[name](train_samples, test_samples, seed)
