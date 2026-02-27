"""Downstream loss evaluation: CE loss increase and KL divergence with SAE patching."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.activation_store import ActivationStore
from src.data.patching import run_patched_forward
from src.model.sae import StratifiedSAE
from src.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def evaluate_downstream_loss(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
    n_batches: int = 50,
) -> dict[str, float]:
    """Evaluate CE loss increase and KL divergence under SAE patching."""
    device = next(sae.parameters()).device

    total_ce_orig = 0.0
    total_ce_patched = 0.0
    total_kl = 0.0
    total_tokens = 0

    token_iter = store._token_generator()

    for _ in range(n_batches):
        tokens = next(token_iter).to(device)
        B, S = tokens.shape
        labels = tokens[:, 1:]

        orig_logits, patched_logits, _, _, _, _ = run_patched_forward(
            store, sae, whitener, tokens
        )

        ce_orig = F.cross_entropy(
            orig_logits[:, :-1].reshape(-1, orig_logits.shape[-1]).float(),
            labels.reshape(-1),
        )
        ce_patched = F.cross_entropy(
            patched_logits[:, :-1].reshape(-1, patched_logits.shape[-1]).float(),
            labels.reshape(-1),
        )

        log_p = F.log_softmax(orig_logits[:, :-1].float(), dim=-1)
        log_q = F.log_softmax(patched_logits[:, :-1].float(), dim=-1)
        kl = F.kl_div(log_q, log_p.exp(), reduction="batchmean")

        n_tok = B * (S - 1)
        total_ce_orig += ce_orig.item() * n_tok
        total_ce_patched += ce_patched.item() * n_tok
        total_kl += kl.item() * n_tok
        total_tokens += n_tok

    avg_ce_orig = total_ce_orig / total_tokens
    avg_ce_patched = total_ce_patched / total_tokens
    avg_kl = total_kl / total_tokens

    results = {
        "ce_loss_orig": avg_ce_orig,
        "ce_loss_patched": avg_ce_patched,
        "ce_loss_increase": avg_ce_patched - avg_ce_orig,
        "kl_div": avg_kl,
    }
    print(f"Downstream loss results: {results}")
    return results
