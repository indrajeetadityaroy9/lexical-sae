"""LexicalSAE: unified sparse autoencoder for classification and sequence labeling.

Single architecture producing per-position sparse representations [B, L, V].
Callers choose the task-specific head: classify() for document-level classification
(max-pool over positions) or tag() for per-position sequence labeling (NER).

Uses AutoModelForMaskedLM as a black-box backbone, delegating architecture
compatibility (BERT, DistilBERT, RoBERTa, ModernBERT, etc.) to HuggingFace.

DLA identity (per position or per document):
    logit[c] = sum_j(s[j] * W_eff[c,j]) + b_eff[c]

where W_eff depends on the ReLU activation mask at the given position/document.
"""

import inspect

import torch
import torch.nn
from transformers import AutoModelForMaskedLM

from splade.circuits.core import CircuitState
from splade.models.layers.activation import DReLU
from splade.training.constants import CLASSIFIER_HIDDEN


class LexicalSAE(torch.nn.Module):
    """Unified Lexical Sparse Autoencoder.

    Args:
        model_name: HuggingFace model name (e.g. "answerdotai/ModernBERT-base").
        num_labels: Number of output classes/tags.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Black-box backbone: encoder + MLM head in one module
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="sdpa",
        )
        self.vocab_size = self.backbone.config.vocab_size

        # Discover supported forward params (e.g. ModernBERT lacks token_type_ids)
        self._backbone_params = set(
            inspect.signature(self.backbone.forward).parameters.keys()
        )

        # DReLU activation gate
        self.activation = DReLU(self.vocab_size)

        # Shared ReLU MLP classifier (weight-tied across positions for NER)
        self.classifier_fc1 = torch.nn.Linear(self.vocab_size, CLASSIFIER_HIDDEN)
        self.classifier_fc2 = torch.nn.Linear(CLASSIFIER_HIDDEN, num_labels)

    @property
    def encoder(self) -> torch.nn.Module:
        """The base encoder (e.g. BertModel, ModernBertModel) within the backbone."""
        return getattr(self.backbone, self.backbone.base_model_prefix)

    def _backbone_forward(self, attention_mask, *, input_ids=None, inputs_embeds=None):
        """Run backbone with cleaned kwargs."""
        kwargs = {"attention_mask": attention_mask}
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        kwargs = {k: v for k, v in kwargs.items() if k in self._backbone_params}
        return self.backbone(**kwargs)

    def _compute_sparse_sequence(
        self,
        attention_mask: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SPLADE head: backbone MLM logits -> DReLU -> log1p -> mask.

        Runs the full backbone (encoder + MLM head) as a black box.

        Returns:
            [B, L, V] per-position sparse representations (masked at padding).
        """
        mlm_logits = self._backbone_forward(
            attention_mask, input_ids=input_ids, inputs_embeds=inputs_embeds,
        ).logits  # [B, L, V]

        activated = self.activation(mlm_logits)
        log_activations = torch.log1p(activated)

        # Zero out padding positions
        return log_activations.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )

    def classify(
        self,
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CircuitState:
        """Max-pool sparse sequence and classify.

        Args:
            sparse_sequence: [B, L, V] per-position sparse representations.
            attention_mask: [B, L] attention mask.

        Returns:
            CircuitState(logits, sparse_vector, W_eff, b_eff).
        """
        sparse_vector = self.to_pooled(sparse_sequence, attention_mask)
        logits, W_eff, b_eff = self.classifier_forward(sparse_vector)
        return CircuitState(logits, sparse_vector, W_eff, b_eff)

    def tag(self, sparse_sequence: torch.Tensor) -> torch.Tensor:
        """Per-position classification logits.

        Args:
            sparse_sequence: [B, L, V] per-position sparse representations.

        Returns:
            [B, L, C] per-position logits.
        """
        B, L, V = sparse_sequence.shape
        flat = sparse_sequence.view(B * L, V)
        return self.classifier_logits_only(flat).view(B, L, self.num_labels)

    def classifier_forward(
        self, sparse_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ReLU MLP classifier returning logits and exact DLA weights.

        Computes fc1 -> ReLU -> fc2 for logits, and derives W_eff from the
        same activation mask (single fc1 computation):
            W_eff(s) = W2 @ diag(D(s)) @ W1
            logit_c = sum_j s_j * W_eff[c,j] + b_eff_c

        Returns:
            logits: [B, C] classification logits.
            W_eff: [B, C, V] effective weight matrix for exact DLA.
            b_eff: [B, C] effective bias vector.
        """
        pre_relu = self.classifier_fc1(sparse_vector)
        activation_mask = (pre_relu > 0).float()
        hidden = pre_relu * activation_mask
        logits = self.classifier_fc2(hidden)

        W1 = self.classifier_fc1.weight  # [H, V]
        W2 = self.classifier_fc2.weight  # [C, H]
        b1 = self.classifier_fc1.bias    # [H]
        b2 = self.classifier_fc2.bias    # [C]

        masked_W1 = activation_mask.unsqueeze(-1) * W1.unsqueeze(0)
        W_eff = torch.matmul(W2.unsqueeze(0), masked_W1)
        b_eff = torch.matmul(activation_mask * b1, W2.T) + b2

        return logits, W_eff, b_eff

    def classifier_logits_only(self, sparse_vector: torch.Tensor) -> torch.Tensor:
        """ReLU MLP classifier returning only logits (for masked/patched evaluation)."""
        return self.classifier_fc2(torch.relu(self.classifier_fc1(sparse_vector)))

    def classifier_parameters(self) -> list[torch.nn.Parameter]:
        """Return classifier head parameters (for optimizer param groups)."""
        return list(self.classifier_fc1.parameters()) + list(self.classifier_fc2.parameters())

    def compute_weff_for_positions(
        self,
        sparse_sequence: torch.Tensor,
        position_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute W_eff only for selected positions (memory-efficient).

        Args:
            sparse_sequence: [B, L, V] full sparse sequence.
            position_mask: [B, L] bool mask selecting positions.

        Returns:
            sparse_selected: [N, V] selected position vectors.
            W_eff: [N, C, V] effective weight matrix for selected positions.
            b_eff: [N, C] effective bias for selected positions.
        """
        sparse_selected = sparse_sequence[position_mask]  # [N, V]
        return self.classifier_forward(sparse_selected)

    @staticmethod
    def to_pooled(
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool [B, L, V] to [B, V]."""
        masked = sparse_sequence.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        return masked.max(dim=1).values

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings (architecture-agnostic)."""
        return self.backbone.get_input_embeddings()(input_ids)

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass from pre-computed embeddings (for gradient-based attribution).

        Returns:
            [B, L, V] sparse sequence.
        """
        return self._compute_sparse_sequence(attention_mask, inputs_embeds=embeddings)

    def _get_mlm_head_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Capture transformed hidden state (input to output projection).

        Architecture-agnostic: registers a pre-hook on get_output_embeddings()
        to capture the intermediate representation that feeds the final
        vocabulary projection. Used by SAE comparison.

        Returns:
            [B, L, H] transformed hidden state before output projection.
        """
        captured = {}

        def _hook(module, args):
            captured["hidden"] = args[0].detach()

        handle = self.backbone.get_output_embeddings().register_forward_pre_hook(_hook)
        try:
            self._backbone_forward(attention_mask, input_ids=input_ids)
        finally:
            handle.remove()
        return captured["hidden"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input to per-position sparse representations.

        Returns:
            [B, L, V] sparse sequence. Use classify() or tag() for task-specific output.
        """
        return self._compute_sparse_sequence(attention_mask, input_ids=input_ids)
