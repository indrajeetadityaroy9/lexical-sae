"""Tests for sequence model, NER data loading, and sequence circuit losses."""

import math

import pytest
import torch

from splade.circuits.sequence_core import SequenceCircuitState, sequence_circuit_mask
from splade.data.ner_loader import (
    CONLL2003_LABEL_NAMES,
    CONLL2003_NUM_LABELS,
    IGNORE_INDEX,
    align_labels_with_tokens,
)


class TestSequenceCircuitState:

    def test_namedtuple_fields(self):
        token_logits = torch.randn(2, 10, 9)
        sparse_sequence = torch.randn(2, 10, 100)
        attention_mask = torch.ones(2, 10)
        state = SequenceCircuitState(token_logits, sparse_sequence, attention_mask)
        assert state.token_logits.shape == (2, 10, 9)
        assert state.sparse_sequence.shape == (2, 10, 100)
        assert state.attention_mask.shape == (2, 10)

    def test_unpacking(self):
        state = SequenceCircuitState(
            torch.randn(1, 5, 3),
            torch.randn(1, 5, 50),
            torch.ones(1, 5),
        )
        logits, sparse, mask = state
        assert logits.shape == (1, 5, 3)


class TestSequenceCircuitMask:

    def test_output_shape(self):
        attr = torch.rand(20, 100)
        mask = sequence_circuit_mask(attr, circuit_fraction=0.1, temperature=10.0)
        assert mask.shape == (20, 100)

    def test_values_in_range(self):
        attr = torch.rand(10, 50)
        mask = sequence_circuit_mask(attr, circuit_fraction=0.1, temperature=1e6)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_hard_mask_sparsity(self):
        """At high temperature, mask should be ~binary with correct sparsity."""
        torch.manual_seed(42)
        attr = torch.rand(100, 200)
        mask = sequence_circuit_mask(attr, circuit_fraction=0.1, temperature=1e6)
        # Each row should have ~20 dims active (10% of 200)
        active_per_row = (mask > 0.5).sum(dim=-1).float()
        assert active_per_row.mean().item() == pytest.approx(20.0, abs=2.0)


class TestNERLabelAlignment:

    def test_basic_alignment(self):
        word_ids = [None, 0, 1, 1, 2, None]  # [CLS] word0 word1 word1_cont word2 [SEP]
        ner_tags = [0, 3, 1]  # O, B-ORG, B-PER
        aligned = align_labels_with_tokens(word_ids, ner_tags)
        assert aligned == [IGNORE_INDEX, 0, 3, IGNORE_INDEX, 1, IGNORE_INDEX]

    def test_special_tokens_get_ignore(self):
        word_ids = [None, 0, None]
        ner_tags = [5]
        aligned = align_labels_with_tokens(word_ids, ner_tags)
        assert aligned[0] == IGNORE_INDEX
        assert aligned[2] == IGNORE_INDEX
        assert aligned[1] == 5

    def test_subword_continuations_get_ignore(self):
        """All continuations of a word get IGNORE_INDEX."""
        word_ids = [None, 0, 0, 0, 1, None]
        ner_tags = [1, 2]  # B-PER, I-PER
        aligned = align_labels_with_tokens(word_ids, ner_tags)
        assert aligned == [IGNORE_INDEX, 1, IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX]

    def test_edge_case_o_to_i_transition(self):
        """O followed by I-PER (unusual but valid in data)."""
        word_ids = [None, 0, 1, None]
        ner_tags = [0, 2]  # O, I-PER
        aligned = align_labels_with_tokens(word_ids, ner_tags)
        assert aligned == [IGNORE_INDEX, 0, 2, IGNORE_INDEX]

    def test_consecutive_b_tags(self):
        """Consecutive B-PER B-LOC (adjacent entities)."""
        word_ids = [None, 0, 1, None]
        ner_tags = [1, 5]  # B-PER, B-LOC
        aligned = align_labels_with_tokens(word_ids, ner_tags)
        assert aligned == [IGNORE_INDEX, 1, 5, IGNORE_INDEX]


class TestConll2003Constants:

    def test_label_count(self):
        assert CONLL2003_NUM_LABELS == 9

    def test_label_names(self):
        assert CONLL2003_LABEL_NAMES[0] == "O"
        assert "B-PER" in CONLL2003_LABEL_NAMES
        assert "I-PER" in CONLL2003_LABEL_NAMES
        assert "B-LOC" in CONLL2003_LABEL_NAMES
        assert "B-ORG" in CONLL2003_LABEL_NAMES
        assert "B-MISC" in CONLL2003_LABEL_NAMES


class TestGatherValidPositions:

    def test_filters_padding_and_ignore(self):
        from splade.circuits.sequence_losses import _gather_valid_positions

        V = 10
        sparse = torch.randn(2, 4, V)
        labels = torch.tensor([
            [1, IGNORE_INDEX, 0, 3],  # 3 valid (positions 0, 2, 3)
            [0, 2, IGNORE_INDEX, IGNORE_INDEX],  # 2 valid (positions 0, 1)
        ])
        mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],  # last position is padding
        ])

        flat, lab = _gather_valid_positions(sparse, labels, mask)
        assert flat.shape[0] == 5  # 3 + 2 valid positions
        assert lab.shape[0] == 5
        assert (lab >= 0).all()  # No IGNORE_INDEX

    def test_empty_batch(self):
        from splade.circuits.sequence_losses import _gather_valid_positions

        sparse = torch.randn(1, 3, 5)
        labels = torch.full((1, 3), IGNORE_INDEX)
        mask = torch.ones(1, 3, dtype=torch.long)

        flat, lab = _gather_valid_positions(sparse, labels, mask)
        assert flat.shape[0] == 0


class TestTokenSharpnessLoss:

    def test_output_scalar(self):
        from splade.circuits.sequence_losses import compute_token_sharpness_loss

        N, V, C = 20, 100, 9
        sparse = torch.rand(N, V)
        W_eff = torch.randn(N, C, V)
        labels = torch.randint(0, C, (N,))

        loss = compute_token_sharpness_loss(sparse, W_eff, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0.0

    def test_loss_magnitude_order_one(self):
        """GECO scaling: loss should be ~O(1), not ~O(L)."""
        from splade.circuits.sequence_losses import compute_token_sharpness_loss

        N, V, C = 200, 100, 9
        sparse = torch.rand(N, V)
        W_eff = torch.randn(N, C, V)
        labels = torch.randint(0, C, (N,))

        loss = compute_token_sharpness_loss(sparse, W_eff, labels)
        # Should be between 0 and 1 (1 - Hoyer sparsity)
        assert loss.item() < 5.0, f"Loss {loss.item()} too large for GECO"


class TestTokenSeparationLoss:

    def test_no_initialized_centroids(self):
        from splade.circuits.sequence_losses import (
            TokenAttributionCentroidTracker,
            compute_token_separation_loss,
        )

        tracker = TokenAttributionCentroidTracker(num_tags=9, vocab_size=100)
        loss = compute_token_separation_loss(tracker)
        assert loss.item() == 0.0

    def test_single_initialized_centroid(self):
        from splade.circuits.sequence_losses import (
            TokenAttributionCentroidTracker,
            compute_token_separation_loss,
        )

        tracker = TokenAttributionCentroidTracker(num_tags=9, vocab_size=100)
        tracker._initialized[0] = True
        tracker.centroids[0] = torch.randn(100)
        loss = compute_token_separation_loss(tracker)
        assert loss.item() == 0.0  # Need >= 2 for pairwise similarity


class TestSeqevalIntegration:

    def test_seqeval_import(self):
        """Verify seqeval is installed and importable."""
        from seqeval.metrics import f1_score
        pred = [["B-PER", "I-PER", "O"]]
        gold = [["B-PER", "I-PER", "O"]]
        f1 = f1_score(gold, pred)
        assert f1 == 1.0

    def test_seqeval_partial_match(self):
        from seqeval.metrics import f1_score
        pred = [["B-PER", "O", "O"]]
        gold = [["B-PER", "I-PER", "O"]]
        f1 = f1_score(gold, pred)
        assert f1 < 1.0
