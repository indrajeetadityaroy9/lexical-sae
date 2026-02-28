"""Downstream loss evaluation: CE loss increase, KL divergence, and Loss Recovered."""


import torch
import torch.nn.functional as F

from spalf.data.activation_store import ActivationStore
from spalf.data.patching import run_patched_forward, run_zero_ablation_forward
from spalf.evaluation.convergence import WelfordMean
from spalf.model import StratifiedSAE
from spalf.whitening.whitener import SoftZCAWhitener


@torch.no_grad()
def evaluate_downstream_loss(
    sae: StratifiedSAE,
    whitener: SoftZCAWhitener,
    store: ActivationStore,
) -> dict[str, float]:
    """Evaluate CE loss increase, KL divergence, and Loss Recovered under SAE patching.

    Self-terminating: runs until the CE loss increase estimate reaches 1% relative
    standard error (minimum 500K tokens, hard cap 20M tokens).
    """
    device = next(sae.parameters()).device
    tokens_per_batch = store.batch_size * store.seq_len

    total_ce_orig = 0.0
    total_ce_patched = 0.0
    total_ce_zero = 0.0
    total_kl = 0.0
    total_tokens = 0

    tracker = WelfordMean()
    token_iter = store._token_generator()

    while True:
        tokens = next(token_iter).to(device)
        B, S = tokens.shape
        labels = tokens[:, 1:]
        n_tok = B * (S - 1)

        orig_logits, patched_logits, _, _, _, _ = run_patched_forward(
            store, sae, whitener, tokens
        )
        zero_logits = run_zero_ablation_forward(store, tokens)

        orig_flat = orig_logits[:, :-1].reshape(-1, orig_logits.shape[-1]).float()
        patched_flat = patched_logits[:, :-1].reshape(-1, patched_logits.shape[-1]).float()
        zero_flat = zero_logits[:, :-1].reshape(-1, zero_logits.shape[-1]).float()
        labels_flat = labels.reshape(-1)

        ce_orig = F.cross_entropy(orig_flat, labels_flat)
        ce_patched = F.cross_entropy(patched_flat, labels_flat)
        ce_zero = F.cross_entropy(zero_flat, labels_flat)

        V_vocab = orig_logits.shape[-1]
        log_p = F.log_softmax(orig_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
        log_q = F.log_softmax(patched_logits[:, :-1].float(), dim=-1).reshape(-1, V_vocab)
        kl = F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)

        total_ce_orig += ce_orig.item() * n_tok
        total_ce_patched += ce_patched.item() * n_tok
        total_ce_zero += ce_zero.item() * n_tok
        total_kl += kl.item() * n_tok
        total_tokens += n_tok

        tracker.update(ce_patched.item() - ce_orig.item())

        if total_tokens >= 500_000 and tracker.relative_se < 0.01:
            break
        if total_tokens >= 20_000_000:
            break

    avg_ce_orig = total_ce_orig / total_tokens
    avg_ce_patched = total_ce_patched / total_tokens
    avg_ce_zero = total_ce_zero / total_tokens
    avg_kl = total_kl / total_tokens

    loss_recovered = (avg_ce_zero - avg_ce_patched) / (avg_ce_zero - avg_ce_orig)

    return {
        "ce_loss_orig": avg_ce_orig,
        "ce_loss_patched": avg_ce_patched,
        "ce_loss_zero_ablation": avg_ce_zero,
        "ce_loss_increase": avg_ce_patched - avg_ce_orig,
        "kl_div": avg_kl,
        "loss_recovered": loss_recovered,
        "eval_tokens_used": total_tokens,
    }
