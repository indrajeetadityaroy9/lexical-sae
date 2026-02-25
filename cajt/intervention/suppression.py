"""Surgical suppression at the sparse bottleneck level.

Provides SuppressedModel (reversible inference-time wrapper) and
vocabulary inspection utilities.
"""

import torch
import torch.nn as nn

from cajt.core.types import CircuitState
from cajt.runtime import DEVICE


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
    centroid_tracker,
    top_k: int = 20,
    clean_vocab: bool = True,
) -> list[tuple[str, float]]:
    """Return top-k attributed vocabulary tokens for a class.

    Uses training centroids from centroid_tracker.
    Returns list of (token_name, attribution_score) sorted by score descending.

    If clean_vocab=True, filters out subwords, special tokens, and single
    characters so results show only human-readable whole-word concepts.
    """
    attr = centroid_tracker.centroids[class_idx].clone()

    if clean_vocab:
        vocab_mask = get_clean_vocab_mask(tokenizer).to(attr.device)
        # Zero out non-clean tokens so they never appear in topk
        attr[:vocab_mask.shape[0]] *= vocab_mask.float()

    scores, indices = attr.topk(top_k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(tokens, scores.tolist()))


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
        mask = torch.ones(model.vocab_size_expanded, device=DEVICE)
        for tid in suppressed_token_ids:
            mask[tid] = 0.0
        # Also suppress virtual sense slots for polysemous tokens
        if model.virtual_expander is not None:
            V = model.vocab_size
            vpe = model.virtual_expander
            for tid in suppressed_token_ids:
                if tid in vpe._token_to_idx:
                    idx = vpe._token_to_idx[tid]
                    M = vpe.num_senses
                    for m in range(1, M):
                        virtual_idx = V + idx * (M - 1) + (m - 1)
                        mask[virtual_idx] = 0.0
        self.register_buffer("keep_mask", mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns [B, L, V] sparse sequence with suppressed tokens zeroed."""
        sparse_seq, gate_mask, l0_probs = self.model(input_ids, attention_mask)
        return sparse_seq * self.keep_mask, gate_mask, l0_probs

    def classify(
        self,
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CircuitState:
        """Max-pool and classify (delegates to underlying model)."""
        return self.model.classify(sparse_sequence, attention_mask)
