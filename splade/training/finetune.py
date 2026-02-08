import copy
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from splade.evaluation.constants import FFIDELITY_FT_EPOCHS, FFIDELITY_FT_LR
from splade.training.optim import _adaptive_gradient_clip
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE


def _randomly_mask_text(text: str, beta: float, mask_token: str, rng: random.Random) -> str:
    words = text.split()
    masked = [mask_token if rng.random() < beta else word for word in words]
    return " ".join(masked)


def finetune_splade_for_ffidelity(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    beta: float,
    batch_size: int,
    mask_token: str,
    seed: int,
    max_length: int,
):
    fine_tuned_model = copy.deepcopy(model._orig_mod)
    fine_tuned_model.train()
    rng = random.Random(seed)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.AdamW(fine_tuned_model.parameters(), lr=FFIDELITY_FT_LR, fused=True)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(FFIDELITY_FT_EPOCHS):
        masked_texts = [_randomly_mask_text(text, beta, mask_token, rng) for text in texts]
        encoding = tokenizer(masked_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

        loader = DataLoader(
            TensorDataset(encoding["input_ids"], encoding["attention_mask"], labels_tensor),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        for input_ids, attention_mask, batch_labels in loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=COMPUTE_DTYPE):
                logits, _ = fine_tuned_model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
            loss.backward()
            _adaptive_gradient_clip(fine_tuned_model)
            optimizer.step()

    fine_tuned_model.eval()
    return torch.compile(fine_tuned_model, mode="reduce-overhead")
