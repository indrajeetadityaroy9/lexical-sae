"""Dead latent counting: features with zero activations across the evaluation set."""


import torch

from spalf.data.activation_store import ActivationStore
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def count_dead_latents(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
) -> dict[str, float]:
    """Count features that never activate, split by anchored and free strata.

    Self-terminating: checks dead count stability every 10 batches, stops after
    3 consecutive unchanged checks past 500K tokens (hard cap 10M tokens).
    """
    device = next(sae.parameters()).device
    tokens_per_batch = store.batch_size * store.seq_len

    ever_active = torch.zeros(sae.F, dtype=torch.bool, device=device)

    total_tokens = 0
    batch_count = 0
    prev_dead_count = -1
    stable_checks = 0

    while True:
        x = store.next_batch().to(device)
        x_tilde = whitener.forward(x)
        _, _, gate_mask, _, _ = sae(x_tilde)
        ever_active |= gate_mask.bool().any(dim=0)

        total_tokens += tokens_per_batch
        batch_count += 1

        if batch_count % 10 == 0:
            current_dead = int((~ever_active).sum().item())
            if current_dead == prev_dead_count:
                stable_checks += 1
            else:
                stable_checks = 0
            prev_dead_count = current_dead

            if total_tokens >= 500_000 and stable_checks >= 3:
                break

        if total_tokens >= 10_000_000:
            break

    dead = ~ever_active
    dead_anchored = dead[: sae.V]
    dead_free = dead[sae.V :]

    return {
        "n_dead": dead.sum().item(),
        "frac_dead": dead.float().mean().item(),
        "n_dead_anchored": dead_anchored.sum().item(),
        "frac_dead_anchored": dead_anchored.float().mean().item(),
        "n_dead_free": dead_free.sum().item(),
        "frac_dead_free": dead_free.float().mean().item(),
        "eval_tokens_used": total_tokens,
    }
