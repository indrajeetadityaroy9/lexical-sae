"""Surgical intervention API for Lexical-SAE.

Provides verifiable concept removal at the sparse bottleneck level.
Since logit[c] = sum_j s[j] * W_eff[c,j] + b_eff[c], zeroing s[j]
removes token j's contribution to ALL classes with mathematical certainty.

Two mechanisms:
  1. Global suppression via weight surgery (permanent, modifies output embeddings)
  2. Inference-time suppression via SuppressedModel wrapper (reversible)

Note: If the backbone uses tied weights (most MLMs do), global suppression
also zeroes the input embedding, effectively removing the concept entirely
from the model's universe — consistent with "lobotomy" semantics.
"""

import torch
import torch.nn as nn

from splade.circuits.core import CircuitState
from splade.utils.cuda import COMPUTE_DTYPE, DEVICE, unwrap_compiled


def get_clean_vocab_mask(tokenizer) -> torch.Tensor:
    """Return a boolean mask selecting whole-word, non-special tokens.

    Filters out special tokens, subword continuations, and single
    alphanumeric characters. Handles both WordPiece (## prefix for
    BERT/DistilBERT) and byte-level BPE (Ġ prefix for ModernBERT/
    RoBERTa) tokenizers.
    """
    vocab_size = tokenizer.vocab_size
    mask = torch.ones(vocab_size, dtype=torch.bool)

    special_ids = set(tokenizer.all_special_ids)
    vocab = tokenizer.get_vocab()

    # Detect tokenizer type: WordPiece uses ##, BPE uses Ġ (U+0120) prefix
    sample_tokens = list(vocab.keys())[:2000]
    has_wordpiece = any(t.startswith("##") for t in sample_tokens)

    for token, tid in vocab.items():
        if tid >= vocab_size:
            continue
        if tid in special_ids:
            mask[tid] = False
        elif has_wordpiece and token.startswith("##"):
            # WordPiece subword continuation
            mask[tid] = False
        elif not has_wordpiece and token.isalpha() and not token.startswith("\u0120"):
            # BPE continuation token (alphabetic, no space prefix)
            mask[tid] = False
        elif token.startswith("[unused"):
            mask[tid] = False
        elif len(token.lstrip("\u0120")) <= 1 and token.lstrip("\u0120").isalnum():
            # Single character (stripping Ġ prefix if present)
            mask[tid] = False

    return mask


def get_top_tokens(
    model: nn.Module,
    tokenizer,
    class_idx: int,
    centroid_tracker=None,
    top_k: int = 20,
    clean_vocab: bool = True,
) -> list[tuple[str, float]]:
    """Return top-k attributed vocabulary tokens for a class.

    Uses training centroids if available, otherwise requires a dataset pass.
    Returns list of (token_name, attribution_score) sorted by score descending.

    If clean_vocab=True, filters out subwords, special tokens, and single
    characters so results show only human-readable whole-word concepts.
    """
    _model = unwrap_compiled(model)

    if centroid_tracker is not None and centroid_tracker._initialized[class_idx]:
        attr = centroid_tracker.centroids[class_idx].clone()
    else:
        raise ValueError(
            "centroid_tracker required with initialized centroids for class "
            f"{class_idx}. Train the model first."
        )

    if clean_vocab:
        vocab_mask = get_clean_vocab_mask(tokenizer).to(attr.device)
        # Zero out non-clean tokens so they never appear in topk
        attr[:vocab_mask.shape[0]] *= vocab_mask.float()

    scores, indices = attr.topk(top_k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(tokens, scores.tolist()))


def suppress_token_globally(model: nn.Module, token_id: int) -> None:
    """Permanently remove a token from the model's vocabulary.

    Zeros the output embedding weights for this token, guaranteeing that
    s[token_id] = 0 for all inputs. This is an irreversible, verifiable
    safety guarantee: the concept cannot influence any class.

    Mathematical guarantee: If the output projection weight[token_id, :] = 0
    and bias[token_id] = 0, then the MLM logit for this token is always 0,
    so after DReLU (which has threshold >= 0), s[token_id] = 0 for all inputs.
    """
    _model = unwrap_compiled(model)
    output_emb = _model.backbone.get_output_embeddings()
    with torch.no_grad():
        output_emb.weight[token_id, :] = 0
        if output_emb.bias is not None:
            output_emb.bias[token_id] = 0
        _model.activation.theta[token_id] = 1e6  # ensure DReLU blocks it


def suppress_tokens_by_name(
    model: nn.Module,
    tokenizer,
    token_names: list[str],
) -> list[int]:
    """Suppress multiple tokens by name. Returns list of suppressed token IDs."""
    suppressed = []
    for name in token_names:
        ids = tokenizer.convert_tokens_to_ids([name])
        if ids and ids[0] != tokenizer.unk_token_id:
            suppress_token_globally(model, ids[0])
            suppressed.append(ids[0])
    return suppressed


class SuppressedModel(nn.Module):
    """Inference-time wrapper that masks tokens in the sparse sequence.

    Unlike global suppression, this is reversible and does not modify
    the underlying model weights. Useful for experimentation.

    The suppression mask zeros specified dimensions of the sparse sequence
    before any task-specific processing. Since logit[c] = sum_j s[j] * W_eff[c,j]
    + b_eff[c], this cleanly removes the masked tokens' contributions while
    preserving the exact DLA identity for remaining tokens.
    """

    def __init__(self, model: nn.Module, suppressed_token_ids: list[int]):
        super().__init__()
        self.model = model
        _orig = unwrap_compiled(model)
        mask = torch.ones(_orig.vocab_size, device=DEVICE)
        for tid in suppressed_token_ids:
            mask[tid] = 0.0
        self.register_buffer("keep_mask", mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, L, V] sparse sequence with suppressed tokens zeroed."""
        sparse_seq = self.model(input_ids, attention_mask)
        return sparse_seq * self.keep_mask

    def classify(
        self,
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CircuitState:
        """Max-pool and classify (delegates to underlying model)."""
        _orig = unwrap_compiled(self.model)
        return _orig.classify(sparse_sequence, attention_mask)

    def tag(self, sparse_sequence: torch.Tensor) -> torch.Tensor:
        """Per-position logits [B, L, C] (delegates to underlying model)."""
        _orig = unwrap_compiled(self.model)
        return _orig.tag(sparse_sequence)

    @staticmethod
    def to_pooled(
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from splade.models.lexical_sae import LexicalSAE
        return LexicalSAE.to_pooled(sparse_sequence, attention_mask)

    @property
    def vocab_size(self):
        return unwrap_compiled(self.model).vocab_size

    def classifier_logits_only(self, sparse_vector: torch.Tensor) -> torch.Tensor:
        return unwrap_compiled(self.model).classifier_logits_only(sparse_vector)

    def classifier_forward(self, sparse_vector: torch.Tensor):
        return unwrap_compiled(self.model).classifier_forward(sparse_vector)

    def classifier_parameters(self):
        return unwrap_compiled(self.model).classifier_parameters()


def evaluate_bias(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    labels: list[int],
    identities: list[dict[str, bool]],
    max_length: int,
    batch_size: int = 64,
) -> dict:
    """Compute accuracy and false positive rate broken down by identity group.

    Returns dict with:
        overall_accuracy, overall_fpr,
        per_identity: {name: {accuracy, fpr, count, fpr_gap}}
    """
    from splade.inference import _predict_model

    preds = _predict_model(model, tokenizer, texts, max_length, batch_size, num_labels=2)

    # Overall metrics
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    overall_acc = correct / len(labels) if labels else 0.0
    neg_count = sum(1 for l in labels if l == 0)
    fp_count = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    overall_fpr = fp_count / neg_count if neg_count > 0 else 0.0

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

    return {
        "overall_accuracy": overall_acc,
        "overall_fpr": overall_fpr,
        "per_identity": per_identity,
    }
