"""F-Fidelity fine-tuning (arXiv:2410.02970)."""

import copy
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.cuda import COMPUTE_DTYPE, DEVICE


def _randomly_mask_text(text: str, beta: float, mask_token: str, rng: random.Random) -> str:
    """Randomly mask up to beta fraction of words."""
    words = text.split()
    n_mask = max(1, int(len(words) * beta * rng.random()))
    positions = rng.sample(range(len(words)), min(n_mask, len(words)))
    masked = list(words)
    for pos in positions:
        masked[pos] = mask_token
    return ' '.join(masked)


def finetune_splade_for_ffidelity(
    clf, texts: list[str], labels: list[int],
    beta: float, ft_epochs: int, ft_lr: float,
    batch_size: int, mask_token: str, seed: int,
):
    """Fine-tune a SPLADEClassifier on randomly masked inputs per F-Fidelity (arXiv:2410.02970).

    Returns a fine-tuned copy. The original classifier is not modified.
    """
    ft_clf = copy.deepcopy(clf)
    ft_clf.model.train()
    rng = random.Random(seed)

    masked_texts = [_randomly_mask_text(t, beta, mask_token, rng) for t in texts]
    enc = ft_clf._tokenize(masked_texts)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(enc["input_ids"], enc["attention_mask"], labels_tensor),
        batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(ft_clf.model.parameters(), lr=ft_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(ft_epochs):
        for input_ids, attention_mask, batch_labels in loader:
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            batch_labels = batch_labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=COMPUTE_DTYPE):
                logits, _ = ft_clf.model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    ft_clf.model.eval()
    return ft_clf
