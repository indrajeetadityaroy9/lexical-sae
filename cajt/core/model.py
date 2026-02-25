"""LexicalSAE: sparse autoencoder for interpretable text classification.

Produces per-position sparse representations [B, L, V] from a pretrained MLM
backbone, then pools and classifies via a ReLU MLP head with exact DLA:

    logit[c] = sum_j(s[j] * W_eff[c,j]) + b_eff[c]

Uses AutoModelForMaskedLM as a black-box backbone, delegating architecture
compatibility (BERT, DistilBERT, RoBERTa, ModernBERT, etc.) to HuggingFace.
"""

import torch
import torch.nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from cajt.core.types import CircuitState
from cajt.core.constants import CLASSIFIER_HIDDEN, JUMPRELU_BANDWIDTH



# JumpReLU gate (Rajamanoharan et al. 2024)


class _JumpReLUSTE(torch.autograd.Function):
    """JumpReLU with sigmoid straight-through estimator.

    Forward:  z · H(z - θ)           (exact binary gate, DLA-compatible)
    Backward: Uses σ'((z - θ) / ε)   (smooth gradient through θ)
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float):
        theta = log_threshold.exp()
        gate = (z > theta).float()
        ctx.save_for_backward(z, theta, gate)
        ctx.bandwidth = bandwidth
        return z * gate, gate

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_gate: torch.Tensor):
        z, theta, gate = ctx.saved_tensors
        eps = ctx.bandwidth

        grad_z = grad_output * gate

        scaled = (z - theta) / eps
        sigmoid_deriv = torch.sigmoid(scaled) * (1 - torch.sigmoid(scaled))
        upstream = grad_output * theta + grad_gate
        grad_log_theta = -(upstream * sigmoid_deriv * theta / eps).sum(
            dim=list(range(len(z.shape) - 1))
        )

        return grad_z, grad_log_theta, None


class JumpReLUGate(torch.nn.Module):
    """JumpReLU gate producing exact binary gates for DLA identity.

    Returns 3-tuple: (output, gate_mask, l0_probs)
      - gate_mask: binary {0,1} tensor
      - l0_probs: σ((z - θ) / ε), differentiable proxy for L0 sparsity loss
    """

    def __init__(self, dim: int, init_log_threshold: float = 0.0):
        super().__init__()
        self.dim = dim
        self.log_threshold = torch.nn.Parameter(
            torch.full((dim,), init_log_threshold)
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.relu(x)
        theta = self.log_threshold.exp()

        if self.training:
            output, gate = _JumpReLUSTE.apply(
                z, self.log_threshold, JUMPRELU_BANDWIDTH,
            )
        else:
            gate = (z > theta).float()
            output = z * gate

        l0_probs = torch.sigmoid(
            (z - theta) / JUMPRELU_BANDWIDTH
        )

        return output, gate, l0_probs



# Virtual Polysemy Expansion (VPE)


class VirtualExpander(torch.nn.Module):
    """Expands polysemous tokens into multiple virtual sense slots.

    For K polysemous tokens with M senses each, adds K*(M-1) virtual
    dimensions to the sparse vector. Winner-take-all hard assignment
    with straight-through estimator for gradients.
    """

    def __init__(
        self,
        backbone_hidden_dim: int,
        polysemous_token_ids: list[int],
        num_senses: int = 4,
    ):
        super().__init__()
        self.token_ids = polysemous_token_ids
        self.num_senses = num_senses
        K = len(polysemous_token_ids)
        self.num_virtual_slots = K * (num_senses - 1)

        self.sense_proj = torch.nn.Linear(backbone_hidden_dim, K * num_senses, bias=False)
        self._token_to_idx = {tid: i for i, tid in enumerate(polysemous_token_ids)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlm_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Expand MLM logits with virtual sense slots.

        Args:
            hidden_states: [B, L, H] backbone hidden states.
            mlm_logits: [B, L, V] original MLM logits.

        Returns:
            [B, L, V + K*(M-1)] expanded logits.
        """
        B, L, V = mlm_logits.shape
        K = len(self.token_ids)
        M = self.num_senses

        sense_scores = self.sense_proj(hidden_states).view(B, L, K, M)

        sense_hard = F.one_hot(sense_scores.argmax(dim=-1), M).float()
        sense_soft = F.softmax(sense_scores, dim=-1)
        sense_gate = sense_hard.detach() - sense_soft.detach() + sense_soft

        token_ids_t = torch.tensor(self.token_ids, device=mlm_logits.device)
        poly_logits = mlm_logits[:, :, token_ids_t]  # [B, L, K]

        sense_logits = poly_logits.unsqueeze(-1) * sense_gate  # [B, L, K, M]

        mlm_logits_expanded = mlm_logits.clone()
        mlm_logits_expanded[:, :, token_ids_t] = sense_logits[:, :, :, 0]

        virtual_logits = sense_logits[:, :, :, 1:].reshape(B, L, K * (M - 1))

        return torch.cat([mlm_logits_expanded, virtual_logits], dim=-1)



# Pooling


class AttentionPool(torch.nn.Module):
    """Learned attention-weighted pooling over sequence positions."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = torch.nn.Linear(dim, 1, bias=False)

    def forward(
        self, sparse_seq: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.query(sparse_seq).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = F.softmax(scores, dim=1)  # [B, L]
        return (sparse_seq * weights.unsqueeze(-1)).sum(dim=1)  # [B, V]



# Main model


class LexicalSAE(torch.nn.Module):
    """Lexical Sparse Autoencoder for interpretable classification.

    Args:
        model_name: HuggingFace model name (e.g. "answerdotai/ModernBERT-base").
        num_labels: Number of output classes.
        vpe_config: Optional VPE configuration for polysemy expansion.
        pooling: Pooling strategy ("max" or "attention").
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        vpe_config=None,
        pooling: str = "max",
    ):
        super().__init__()
        self.num_labels = num_labels
        self._pooling_mode = pooling

        # Black-box backbone: encoder + MLM head in one module
        self.backbone = AutoModelForMaskedLM.from_pretrained(
            model_name, attn_implementation="sdpa",
        )
        self.vocab_size = self.backbone.config.vocab_size

        # Virtual Polysemy Expansion
        self.virtual_expander = None
        self._captured_hidden = None
        expanded_dim = self.vocab_size
        if vpe_config and vpe_config.enabled and vpe_config.token_ids:
            self.virtual_expander = VirtualExpander(
                backbone_hidden_dim=self.backbone.config.hidden_size,
                polysemous_token_ids=vpe_config.token_ids,
                num_senses=vpe_config.num_senses,
            )
            expanded_dim = self.vocab_size + self.virtual_expander.num_virtual_slots
            # Persistent hook to capture hidden states for VPE
            self.backbone.get_output_embeddings().register_forward_pre_hook(
                self._capture_hook
            )

        # JumpReLU gate (exact binary gates for DLA identity)
        self.activation = JumpReLUGate(expanded_dim)

        # Attention-weighted pooling (optional, default: max-pool)
        self.attention_pool = (
            AttentionPool(expanded_dim) if pooling == "attention" else None
        )

        # ReLU MLP classifier head
        self.classifier_fc1 = torch.nn.Linear(expanded_dim, CLASSIFIER_HIDDEN)
        self.classifier_fc2 = torch.nn.Linear(CLASSIFIER_HIDDEN, num_labels)

    @property
    def vocab_size_expanded(self) -> int:
        """Effective sparse vector dimensionality (V + virtual slots if VPE active)."""
        if self.virtual_expander:
            return self.vocab_size + self.virtual_expander.num_virtual_slots
        return self.vocab_size

    @property
    def encoder(self) -> torch.nn.Module:
        """The base encoder (e.g. BertModel, ModernBertModel) within the backbone."""
        return getattr(self.backbone, self.backbone.base_model_prefix)

    def _capture_hook(self, module, args):
        """Persistent hook capturing hidden states before output projection."""
        self._captured_hidden = args[0]

    def backbone_forward(self, input_ids, attention_mask):
        """Run backbone forward pass."""
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask)

    def compute_sparse_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SPLADE head: backbone MLM logits -> [VPE expansion] -> JumpReLU gate.

        Returns:
            sparse_seq: [B, L, V_expanded] per-position sparse representations.
            gate_mask: [B, L, V_expanded] binary gate mask {0,1}.
            l0_probs: [B, L, V_expanded] differentiable P(z > θ) for L0 loss.
        """
        mlm_logits = self.backbone_forward(
            input_ids, attention_mask,
        ).logits  # [B, L, V]

        if self.virtual_expander is not None and self._captured_hidden is not None:
            mlm_logits = self.virtual_expander(self._captured_hidden, mlm_logits)

        activated, gate_mask, l0_probs = self.activation(mlm_logits)

        # Zero out padding positions
        sparse_seq = activated.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        return sparse_seq, gate_mask, l0_probs

    def classify(
        self,
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> CircuitState:
        """Pool sparse sequence and classify.

        Returns:
            CircuitState(logits, sparse_vector, W_eff, b_eff).
        """
        if self.attention_pool is not None:
            sparse_vector = self.attention_pool(sparse_sequence, attention_mask)
        else:
            sparse_vector = self.to_pooled(sparse_sequence, attention_mask)
        logits, W_eff, b_eff = self.classifier_forward(sparse_vector)
        return CircuitState(logits, sparse_vector, W_eff, b_eff)

    def classifier_forward(
        self, sparse_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ReLU MLP classifier returning logits and exact DLA weights.

        W_eff(s) = W2 @ diag(D(s)) @ W1
        logit_c = sum_j s_j * W_eff[c,j] + b_eff_c

        Returns:
            logits: [B, C] classification logits.
            W_eff: [B, C, V_expanded] effective weight matrix for exact DLA.
            b_eff: [B, C] effective bias vector.
        """
        pre_relu = self.classifier_fc1(sparse_vector)
        activation_mask = (pre_relu > 0).float()
        hidden = pre_relu * activation_mask
        logits = self.classifier_fc2(hidden)

        W1 = self.classifier_fc1.weight  # [H, V_expanded]
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

    @staticmethod
    def to_pooled(
        sparse_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool [B, L, V_expanded] to [B, V_expanded]."""
        masked = sparse_sequence.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 0.0
        )
        return masked.max(dim=1).values

    def get_mlm_head_input(
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
            self.backbone_forward(input_ids, attention_mask)
        finally:
            handle.remove()
        return captured["hidden"]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to per-position sparse representations.

        Returns:
            sparse_seq: [B, L, V_expanded] sparse sequence.
            gate_mask: [B, L, V_expanded] binary gate mask {0,1}.
            l0_probs: [B, L, V_expanded] differentiable P(z > θ) for L0 loss.
        """
        return self.compute_sparse_sequence(input_ids, attention_mask)
