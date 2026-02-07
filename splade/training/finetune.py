"""F-Fidelity fine-tuning helpers."""

import copy
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from splade.training.optim import _adaptive_gradient_clip
from splade.utils.cuda import AUTOCAST_DEVICE_TYPE, AUTOCAST_ENABLED, COMPUTE_DTYPE, DEVICE

def _randomly_mask_text(text: str, beta: float, mask_token: str, rng: random.Random) -> str:
    """Mask each word independently with probability beta (per-position Bernoulli)."""
    words = text.split()
    masked = [mask_token if rng.random() < beta else word for word in words]
    return " ".join(masked)

def finetune_splade_for_ffidelity(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    beta: float,
    ft_epochs: int,
    ft_lr: float,
    batch_size: int,
    mask_token: str,
    seed: int,
    max_length: int = 128
):
    """Return a fine-tuned copy trained on randomly masked inputs."""
    if hasattr(model, "_orig_mod"):
        orig_model = model._orig_mod
    else:
        orig_model = model
    
    fine_tuned_model = copy.deepcopy(orig_model)
    fine_tuned_model.train()
    rng = random.Random(seed)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.AdamW(fine_tuned_model.parameters(), lr=ft_lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(ft_epochs):
        masked_texts = [_randomly_mask_text(text, beta, mask_token, rng) for text in texts]
        encoding = tokenizer(masked_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        loader = DataLoader(
            TensorDataset(encoding["input_ids"], encoding["attention_mask"], labels_tensor),
            batch_size=batch_size,
            shuffle=True,
        )
        for input_ids, attention_mask, batch_labels in loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(AUTOCAST_DEVICE_TYPE, dtype=COMPUTE_DTYPE, enabled=AUTOCAST_ENABLED):
                logits, _ = fine_tuned_model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
            loss.backward()
            _adaptive_gradient_clip(fine_tuned_model, clip_factor=0.01)
            optimizer.step()

    fine_tuned_model.eval()
    if hasattr(model, "_orig_mod"):
        try:
            fine_tuned_model = torch.compile(fine_tuned_model)
        except Exception:
            pass
    return fine_tuned_model