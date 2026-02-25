"""Dense baseline: standard fine-tuned ModernBERT for accuracy ceiling comparison.

Uses AutoModelForSequenceClassification with standard HF fine-tuning protocol
(AdamW, lr=2e-5, early stopping) to establish the accuracy upper bound that
Lexical-SAE trades for interpretability.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import accuracy_score

from cajt.runtime import DEVICE, autocast


_DENSE_LR = 2e-5
_DENSE_MAX_EPOCHS = 10
_DENSE_PATIENCE = 3


def train_dense_baseline(
    model_name: str,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    num_labels: int,
    max_length: int,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple:
    """Train a dense classifier baseline.

    Uses the same model backbone, seed, and data split as Lexical-SAE
    to ensure a fair comparison.

    Args:
        model_name: HuggingFace model name (same as Lexical-SAE backbone).
        train_texts: Training texts (same split as Lexical-SAE).
        train_labels: Training labels.
        val_texts: Validation texts.
        val_labels: Validation labels.
        num_labels: Number of output classes.
        max_length: Tokenizer max length.
        batch_size: Training batch size.
        seed: Random seed for reproducibility.

    Returns:
        (model, tokenizer, best_val_accuracy)
    """
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    ).to(DEVICE)

    train_enc = tokenizer(
        train_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )

    train_loader = DataLoader(
        TensorDataset(
            train_enc["input_ids"], train_enc["attention_mask"],
            torch.tensor(train_labels, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True, pin_memory=True,
    )
    val_ids = val_enc["input_ids"].to(DEVICE)
    val_mask = val_enc["attention_mask"].to(DEVICE)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=_DENSE_LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(_DENSE_MAX_EPOCHS):
        model.train()
        for batch_ids, batch_mask, batch_labels in train_loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
                loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.inference_mode(), autocast():
            val_outputs = model(input_ids=val_ids, attention_mask=val_mask)
            val_preds = val_outputs.logits.argmax(dim=-1)
            val_acc = (val_preds == val_labels_t).float().mean().item()

        print(f"  Dense epoch {epoch + 1}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= _DENSE_PATIENCE:
            print(f"  Dense baseline: early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    return model, tokenizer, best_val_acc


def score_dense_baseline(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    max_length: int,
    batch_size: int = 64,
) -> float:
    """Score a dense baseline model on test data.

    Returns accuracy.
    """
    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    all_preds = []
    model.eval()
    with torch.inference_mode(), autocast():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())

    return accuracy_score(labels, all_preds) if labels else 0.0
