"""Inference utilities."""

import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset
from splade.utils.cuda import DEVICE

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"}

def _run_inference_loop(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    extract_sparse: bool = False,
    batch_size: int = 32, 
) -> tuple[torch.Tensor, torch.Tensor | None]:
    loader = DataLoader(
        TensorDataset(input_ids, attention_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_logits = []
    all_sparse = [] if extract_sparse else None

    with torch.inference_mode():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            logits, sparse = model(batch_ids, batch_mask)
            all_logits.append(logits.cpu())
            if extract_sparse:
                all_sparse.append(sparse.cpu())

    logits = torch.cat(all_logits, dim=0)
    sparse = torch.cat(all_sparse, dim=0) if extract_sparse else None
    return logits, sparse

def predict_model(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32, num_labels: int = 2) -> list[int]:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, _ = _run_inference_loop(model, encoding["input_ids"], encoding["attention_mask"], batch_size=batch_size)
    if num_labels == 1:
        return (logits.squeeze(-1) > 0).int().tolist()
    return logits.argmax(dim=1).tolist()

def predict_proba_model(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32, num_labels: int = 2) -> list[list[float]]:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    logits, _ = _run_inference_loop(model, encoding["input_ids"], encoding["attention_mask"], batch_size=batch_size)
    if num_labels == 1:
        positive = torch.sigmoid(logits).squeeze(-1)
        probabilities = torch.stack([1 - positive, positive], dim=1)
    else:
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities.tolist()

def score_model(model, tokenizer, texts: list[str], labels: list[int], max_length: int, batch_size: int = 32, num_labels: int = 2) -> float:
    preds = predict_model(model, tokenizer, texts, max_length, batch_size, num_labels)
    return sum(p == t for p, t in zip(preds, labels)) / len(labels)

def transform_model(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32) -> numpy.ndarray:
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    _, sparse = _run_inference_loop(model, encoding["input_ids"], encoding["attention_mask"], extract_sparse=True, batch_size=batch_size)
    return sparse.numpy()

def explain_model(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int,
    top_k: int = 10,
    target_class: int | None = None,
    input_only: bool = False,
) -> list[tuple[str, float]]:
    sparse_vector = transform_model(model, tokenizer, [text], max_length)[0]

    if target_class is None:
        probabilities = predict_proba_model(model, tokenizer, [text], max_length)[0]
        target_class = int(numpy.argmax(probabilities))

    # Access classifier weight. If compiled, check _orig_mod
    if hasattr(model, "_orig_mod"):
        classifier_layer = model._orig_mod.classifier
    else:
        classifier_layer = model.classifier

    with torch.inference_mode():
        weights = classifier_layer.weight[target_class].cpu().numpy()

    contributions = sparse_vector * weights
    nonzero_indices = numpy.nonzero(contributions)[0]

    if input_only:
        input_ids = set(tokenizer.encode(text, add_special_tokens=False))
        nonzero_indices = numpy.array([i for i in nonzero_indices if i in input_ids])

    positive_mask = contributions[nonzero_indices] > 0
    positive_indices = nonzero_indices[positive_mask]
    negative_indices = nonzero_indices[~positive_mask]
    positive_indices = positive_indices[numpy.argsort(contributions[positive_indices])[::-1]]
    negative_indices = negative_indices[numpy.argsort(numpy.abs(contributions[negative_indices]))[::-1]]
    ranked_indices = numpy.concatenate([positive_indices, negative_indices])

    explanations = []
    for index in ranked_indices:
        token = tokenizer.convert_ids_to_tokens(int(index))
        if token not in SPECIAL_TOKENS:
            explanations.append((token, float(contributions[index])))
        if len(explanations) >= top_k:
            break

    return explanations
