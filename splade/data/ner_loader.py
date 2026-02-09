"""CoNLL-2003 NER data loading with subword alignment.

Loads CoNLL-2003 from HuggingFace datasets and aligns BIO tags with
subword tokenization. Requires a FastTokenizer (Rust-based) for reliable
word_ids() — works identically for WordPiece (BERT) and BPE (ModernBERT).
"""

import random

import numpy
from datasets import load_dataset

CONLL2003_LABEL_NAMES = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]
CONLL2003_NUM_LABELS = len(CONLL2003_LABEL_NAMES)

IGNORE_INDEX = -100


def align_labels_with_tokens(
    word_ids: list[int | None],
    ner_tags: list[int],
) -> list[int]:
    """Align NER tags with subword tokens.

    Follows HuggingFace token-classification guide:
    - Special tokens (word_id=None) get IGNORE_INDEX
    - First subword of a word gets the word's NER tag
    - Continuation subwords get IGNORE_INDEX

    Args:
        word_ids: Per-token word indices from tokenizer (None for special tokens).
        ner_tags: Per-word NER tag indices.

    Returns:
        Per-token label list (same length as word_ids).
    """
    aligned = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(IGNORE_INDEX)
        elif word_idx != previous_word_idx:
            aligned.append(ner_tags[word_idx])
        else:
            aligned.append(IGNORE_INDEX)
        previous_word_idx = word_idx
    return aligned


def tokenize_and_align_dataset(
    tokens_list: list[list[str]],
    tags_list: list[list[int]],
    tokenizer,
    max_length: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Tokenize word-level NER data and align labels.

    Args:
        tokens_list: List of sentences, each a list of word strings.
        tags_list: List of sentences, each a list of NER tag indices.
        tokenizer: HuggingFace tokenizer (must be fast).
        max_length: Maximum sequence length.

    Returns:
        (all_input_ids, all_attention_masks, all_labels) — each a list of
        lists with length max_length.

    Raises:
        ValueError: If tokenizer is not a FastTokenizer.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "NER loader requires a FastTokenizer (Rust-based) for precise "
            "word_ids alignment. Use AutoTokenizer which returns fast "
            "tokenizers by default for most models."
        )

    encoding = tokenizer(
        tokens_list,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for i in range(len(tokens_list)):
        word_ids = encoding.word_ids(batch_index=i)
        labels = align_labels_with_tokens(word_ids, tags_list[i])
        all_input_ids.append(encoding["input_ids"][i])
        all_attention_masks.append(encoding["attention_mask"][i])
        all_labels.append(labels)

    return all_input_ids, all_attention_masks, all_labels


def infer_ner_max_length(
    tokens_list: list[list[str]],
    tokenizer,
    model_name: str | None = None,
) -> int:
    """Infer max sequence length for word-tokenized NER input.

    Args:
        tokens_list: List of sentences, each a list of word strings.
        tokenizer: HuggingFace tokenizer.
        model_name: Model name for max_position_embeddings lookup.

    Returns:
        Padded max_length capped by model's max_position_embeddings.
    """
    sample = tokens_list[:500]
    lengths = [
        len(tokenizer(words, is_split_into_words=True)["input_ids"])
        for words in sample
    ]
    p99 = int(numpy.percentile(lengths, 99))
    aligned = ((p99 + 7) // 8) * 8

    if model_name is not None:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        max_pos = getattr(config, "max_position_embeddings", 512)
    else:
        max_pos = 512

    return max(64, min(max_pos, aligned))


def load_conll2003(
    train_samples: int = -1,
    test_samples: int = -1,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[int]], list[list[str]], list[list[int]]]:
    """Load CoNLL-2003 NER dataset.

    Returns:
        (train_tokens, train_tags, test_tokens, test_tags)
        where each tokens entry is a list of word strings and each tags
        entry is a list of BIO tag indices (0-8).
    """
    dataset = load_dataset("conll2003")

    def _extract(split):
        tokens_list = [row["tokens"] for row in dataset[split]]
        tags_list = [row["ner_tags"] for row in dataset[split]]
        combined = list(zip(tokens_list, tags_list))
        random.Random(seed).shuffle(combined)
        if split == "train" and train_samples > 0:
            combined = combined[:train_samples]
        elif split != "train" and test_samples > 0:
            combined = combined[:test_samples]
        if combined:
            tokens, tags = zip(*combined)
            return list(tokens), list(tags)
        return [], []

    train_tokens, train_tags = _extract("train")
    test_tokens, test_tags = _extract("test")
    return train_tokens, train_tags, test_tokens, test_tags
