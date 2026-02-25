"""Shared training pipeline.

Consolidates the ~30-line training setup duplicated across entry scripts
into a single function.
"""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from cajt.config import Config
from cajt.training.losses import AttributionCentroidTracker
from cajt.data import infer_max_length, load_dataset_by_name
from cajt.evaluation.collect import score_model
from cajt.core.model import LexicalSAE
from cajt.training.loop import train_model
from cajt.training.optim import _infer_batch_size
from cajt.runtime import DEVICE, set_seed


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
    val_texts: list[str] | None = None
    val_labels: list[int] | None = None


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
    batch_size = config.training.batch_size or _infer_batch_size(config.model.name, max_length)

    tc = config.training
    val_size = max(
        tc.min_val_per_class * num_labels,
        min(len(train_texts) // 5, len(train_texts) // tc.val_fraction),
    )
    val_size = min(val_size, len(train_texts) // 2)  # never exceed half the data
    val_texts = train_texts[-val_size:]
    val_labels = train_labels[-val_size:]
    train_texts_split = train_texts[:-val_size]
    train_labels_split = train_labels[:-val_size]

    model = LexicalSAE(
        config.model.name, num_labels,
        vpe_config=config.vpe,
        pooling=config.training.pooling,
    ).to(DEVICE)

    centroid_tracker = train_model(
        model, tokenizer, train_texts_split, train_labels_split,
        model_name=config.model.name, num_labels=num_labels,
        val_texts=val_texts, val_labels=val_labels,
        max_length=max_length, batch_size=batch_size,
        sparsity_target=tc.sparsity_target,
        warmup_fraction=tc.warmup_fraction,
        learning_rate=tc.learning_rate,
        max_epochs=tc.max_epochs,
        early_stop_patience=tc.early_stop_patience,
        label_smoothing=tc.label_smoothing,
        num_workers=tc.num_workers,
        prefetch_factor=tc.prefetch_factor,
        seed=seed,
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
        val_texts=val_texts,
        val_labels=val_labels,
    )
