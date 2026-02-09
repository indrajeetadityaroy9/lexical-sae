"""Shared training pipeline.

Consolidates the ~30-line training setup duplicated across entry scripts
into a single function.
"""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from splade.config.schema import Config
from splade.circuits.losses import AttributionCentroidTracker
from splade.data.loader import infer_max_length, load_dataset_by_name
from splade.inference import score_model
from splade.models.lexical_sae import LexicalSAE
from splade.training.loop import train_model
from splade.training.optim import _infer_batch_size
from splade.utils.cuda import DEVICE, set_seed


@dataclass
class TrainedExperiment:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    num_labels: int
    max_length: int
    batch_size: int
    accuracy: float
    seed: int
    centroid_tracker: AttributionCentroidTracker | None = None


def setup_and_train(config: Config, seed: int) -> TrainedExperiment:
    """Load data, create model, train with CIS, and evaluate accuracy."""
    set_seed(seed)

    train_texts, train_labels, test_texts, test_labels, num_labels = load_dataset_by_name(
        config.data.dataset_name,
        config.data.train_samples,
        config.data.test_samples,
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    max_length = infer_max_length(train_texts, tokenizer, model_name=config.model.name)
    batch_size = _infer_batch_size(config.model.name, max_length)

    val_size = min(200, len(train_texts) // 5)
    val_texts = train_texts[-val_size:]
    val_labels = train_labels[-val_size:]
    train_texts_split = train_texts[:-val_size]
    train_labels_split = train_labels[:-val_size]

    model = LexicalSAE(config.model.name, num_labels).to(DEVICE)
    # torch.compile disabled: bf16 autocast dtype mismatch on PyTorch 2.7
    # (addmm gets Float vs BFloat16). Batched eval loops provide the main
    # throughput improvement.
    # model = torch.compile(model, dynamic=True)

    centroid_tracker = train_model(
        model, tokenizer, train_texts_split, train_labels_split,
        model_name=config.model.name, num_labels=num_labels,
        val_texts=val_texts, val_labels=val_labels,
        max_length=max_length, batch_size=batch_size,
        target_accuracy=config.training.target_accuracy,
        sparsity_target=config.training.sparsity_target,
        warmup_fraction=config.training.warmup_fraction,
    )

    accuracy = score_model(
        model, tokenizer, test_texts, test_labels,
        max_length, batch_size, num_labels,
    )

    return TrainedExperiment(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        num_labels=num_labels,
        max_length=max_length,
        batch_size=batch_size,
        accuracy=accuracy,
        seed=seed,
        centroid_tracker=centroid_tracker,
    )


def prepare_mechanistic_inputs(
    tokenizer,
    texts: list[str],
    max_length: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Tokenize texts into per-sample tensor lists for mechanistic evaluation."""
    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids_list = [
        encoding["input_ids"][i:i+1].to(DEVICE) for i in range(len(texts))
    ]
    attention_mask_list = [
        encoding["attention_mask"][i:i+1].to(DEVICE) for i in range(len(texts))
    ]
    return input_ids_list, attention_mask_list
