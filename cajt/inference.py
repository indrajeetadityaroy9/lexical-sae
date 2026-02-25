import torch
from torch.utils.data import DataLoader, TensorDataset

from cajt.runtime import autocast, DEVICE


def predict(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 64, num_labels: int = 2) -> list[int]:
    """Tokenize texts, run batched inference, return predicted class indices."""
    encoding = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    all_logits = []

    with torch.inference_mode(), autocast():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            sparse_seq, *_ = model(batch_ids, batch_mask)
            logits = model.classify(sparse_seq, batch_mask).logits
            all_logits.append(logits)

    logits = torch.cat(all_logits, dim=0)
    if num_labels == 1:
        return (logits.squeeze(-1) > 0).int().tolist()
    return logits.argmax(dim=1).tolist()


def score_model(model, tokenizer, texts: list[str], labels: list[int], max_length: int, batch_size: int = 64, num_labels: int = 2) -> float:
    """Run inference and return accuracy."""
    preds = predict(model, tokenizer, texts, max_length, batch_size, num_labels)
    return sum(p == t for p, t in zip(preds, labels)) / len(labels)
