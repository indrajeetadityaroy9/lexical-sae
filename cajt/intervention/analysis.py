"""Bias evaluation and W_eff sign analysis for surgical intervention."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cajt.runtime import autocast, DEVICE


def evaluate_bias(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    identities: list[dict[str, bool]],
    max_length: int,
    batch_size: int = 64,
) -> dict:
    """Compute accuracy, FPR, and collateral damage broken down by identity group.

    Returns dict with:
        overall_accuracy, overall_fpr,
        per_identity: {name: {accuracy, fpr, count, fpr_gap}},
        nontoxic_identity_accuracy, nontoxic_noidentity_accuracy, collateral_gap
    """
    from cajt.inference import predict

    preds = predict(model, tokenizer, texts, max_length, batch_size, num_labels=2)

    # Overall metrics
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    overall_acc = correct / len(labels) if labels else 0.0
    neg_count = sum(1 for l in labels if l == 0)
    fp_count = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    overall_fpr = fp_count / neg_count if neg_count > 0 else 0.0

    # Collateral damage: accuracy on non-toxic samples with/without identity mentions
    nontoxic_identity_correct = 0
    nontoxic_identity_total = 0
    nontoxic_noidentity_correct = 0
    nontoxic_noidentity_total = 0

    # Per-identity metrics
    identity_names = set()
    for ident in identities:
        identity_names.update(k for k, v in ident.items() if v)

    per_identity = {}
    for name in sorted(identity_names):
        group_indices = [i for i, ident in enumerate(identities) if ident.get(name, False)]
        if len(group_indices) < 10:
            continue
        g_preds = [preds[i] for i in group_indices]
        g_labels = [labels[i] for i in group_indices]
        g_correct = sum(1 for p, l in zip(g_preds, g_labels) if p == l)
        g_neg = sum(1 for l in g_labels if l == 0)
        g_fp = sum(1 for p, l in zip(g_preds, g_labels) if p == 1 and l == 0)
        g_fpr = g_fp / g_neg if g_neg > 0 else 0.0
        per_identity[name] = {
            "accuracy": g_correct / len(g_labels),
            "fpr": g_fpr,
            "count": len(group_indices),
            "fpr_gap": g_fpr - overall_fpr,
        }

    # Collateral damage computation
    for i, (pred, label) in enumerate(zip(preds, labels)):
        if label != 0:  # only non-toxic/safe samples
            continue
        has_identity = any(identities[i].get(name, False) for name in identity_names)
        if has_identity:
            nontoxic_identity_total += 1
            if pred == label:
                nontoxic_identity_correct += 1
        else:
            nontoxic_noidentity_total += 1
            if pred == label:
                nontoxic_noidentity_correct += 1

    nontoxic_identity_acc = (
        nontoxic_identity_correct / nontoxic_identity_total
        if nontoxic_identity_total > 0 else 0.0
    )
    nontoxic_noidentity_acc = (
        nontoxic_noidentity_correct / nontoxic_noidentity_total
        if nontoxic_noidentity_total > 0 else 0.0
    )
    collateral_gap = nontoxic_identity_acc - nontoxic_noidentity_acc

    return {
        "overall_accuracy": overall_acc,
        "overall_fpr": overall_fpr,
        "per_identity": per_identity,
        "nontoxic_identity_accuracy": nontoxic_identity_acc,
        "nontoxic_noidentity_accuracy": nontoxic_noidentity_acc,
        "collateral_gap": collateral_gap,
    }


def analyze_weff_sign_flips(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    target_tokens: list[str],
    max_length: int,
    batch_size: int = 64,
    target_class: int = 1,
) -> dict:
    """Analyze W_eff sign flips for target tokens across samples.

    For each target token, extracts W_eff[target_class, token_id] across all
    samples where the token is active. Reports fraction with positive vs
    negative sign. Sign flips prove the model is context-dependent.

    Memory-efficient: computes W_eff externally via einsum on sliced weights,
    never materializing the full [B, C, V] tensor.

    Returns:
        {token: {positive_frac, negative_frac, n_active, mean_magnitude}}
    """


    token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    valid = [(name, tid) for name, tid in zip(target_tokens, token_ids)
             if tid != tokenizer.unk_token_id]
    if not valid:
        return {}

    token_names, token_id_list = zip(*valid)
    token_id_tensor = torch.tensor(token_id_list, device=DEVICE, dtype=torch.long)

    encoding = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoding["input_ids"], encoding["attention_mask"]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    # Accumulators per token: count positive, negative, total magnitude
    n_tokens = len(token_id_list)
    pos_counts = torch.zeros(n_tokens, device=DEVICE)
    neg_counts = torch.zeros(n_tokens, device=DEVICE)
    mag_sums = torch.zeros(n_tokens, device=DEVICE)
    active_counts = torch.zeros(n_tokens, device=DEVICE)

    w1 = model.classifier_fc1.weight  # [H, V]
    w2 = model.classifier_fc2.weight  # [C, H]
    b1 = model.classifier_fc1.bias    # [H]

    w1_slice = w1[:, token_id_tensor]  # [H, n_tokens]

    with torch.inference_mode(), autocast():
        for batch_ids, batch_mask in loader:
            batch_ids = batch_ids.to(DEVICE, non_blocking=True)
            batch_mask = batch_mask.to(DEVICE, non_blocking=True)
            sparse_seq, *_ = model(batch_ids, batch_mask)
            sparse_vector = model.to_pooled(sparse_seq, batch_mask)  # [B, V]

            # Compute ReLU mask from classifier hidden layer
            h = model.classifier_fc1(sparse_vector)  # [B, H]
            mask = (h > 0).float()  # [B, H]

            # W_eff_slice[b, c, t] = sum_h W2[c,h] * mask[b,h] * W1[h,t]
            weff_slice = torch.einsum(
                'ch,bh,ht->bct', w2.float(), mask.float(), w1_slice.float(),
            )  # [B, C, n_tokens]

            # Extract target class row
            weff_target = weff_slice[:, target_class, :]  # [B, n_tokens]

            # Check which tokens are active in each sample
            sparse_slice = sparse_vector[:, token_id_tensor]  # [B, n_tokens]
            is_active = sparse_slice > 0  # [B, n_tokens]

            # Accumulate sign statistics only for active tokens
            pos_counts += ((weff_target > 0) & is_active).float().sum(dim=0)
            neg_counts += ((weff_target < 0) & is_active).float().sum(dim=0)
            mag_sums += (weff_target.abs() * is_active.float()).sum(dim=0)
            active_counts += is_active.float().sum(dim=0)

    results = {}
    for i, name in enumerate(token_names):
        n_active = int(active_counts[i].item())
        if n_active == 0:
            continue
        results[name] = {
            "positive_frac": pos_counts[i].item() / n_active,
            "negative_frac": neg_counts[i].item() / n_active,
            "n_active": n_active,
            "mean_magnitude": mag_sums[i].item() / n_active,
        }

    return results
