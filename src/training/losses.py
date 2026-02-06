"""Training loss functions: DF-FLOPS regularization and KL distillation."""

import torch
import torch.nn.functional as F

from src.models.components import _EPS


class DocumentFrequencyTracker:
    """Track document frequency for DF-FLOPS regularization (arXiv:2505.15070).

    Penalizes high-DF terms to reduce posting list lengths and latency.
    """

    def __init__(self, vocab_size: int, device: str | torch.device = 'cuda'):
        self.vocab_size = vocab_size
        self.device = device
        self.df_counts = torch.zeros(vocab_size, device=device)
        self.doc_count = 0

    def update(self, sparse_vectors: torch.Tensor) -> None:
        term_presence = (sparse_vectors.detach() > 0).float()
        self.df_counts += term_presence.sum(dim=0)
        self.doc_count += sparse_vectors.shape[0]

    def reset(self) -> None:
        """Reset DF counts for periodic re-estimation (arXiv:2505.15070)."""
        self.df_counts.zero_()
        self.doc_count = 0

    def get_weights(self, alpha: float = 0.1, beta: float = 10.0) -> torch.Tensor:
        """Compute DF-based penalty weights: w_t = 1 / (1 + (x^(log_alpha(2)) - 1)^beta)."""
        df_ratio = self.df_counts / self.doc_count
        eps = _EPS[df_ratio.dtype]['log']
        log_alpha = torch.log(torch.tensor(alpha, device=self.device))
        log_alpha = torch.where(log_alpha.abs() < eps, torch.tensor(eps, device=self.device), log_alpha)
        log_alpha_2 = torch.log(torch.tensor(2.0, device=self.device)) / log_alpha

        x_clamped = df_ratio.clamp(min=1e-8)
        x_pow = x_clamped.pow(log_alpha_2)
        inner = (x_pow - 1.0).clamp(min=0.0)
        return 1.0 / (1.0 + inner.pow(beta))

    def get_stats(self) -> dict:
        df_ratio = self.df_counts / self.doc_count
        return {
            "doc_count": self.doc_count,
            "top1_df_pct": df_ratio.max().item() * 100,
            "mean_df_pct": df_ratio.mean().item() * 100,
        }


class DFFlopsRegFunction(torch.autograd.Function):
    """DF-FLOPS regularization with autograd support."""

    @staticmethod
    def forward(ctx, activations: torch.Tensor, df_weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(activations, df_weights)
        weighted_mean = (df_weights.unsqueeze(0) * activations.abs()).mean(dim=0)
        return (weighted_mean ** 2).sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        act, df_weights = ctx.saved_tensors
        B = act.shape[0]
        weighted_mean = (df_weights.unsqueeze(0) * act.abs()).mean(dim=0)
        sign = torch.where(act != 0, torch.sign(act), torch.zeros_like(act))
        grad_act = grad_output * 2.0 * df_weights.unsqueeze(0) * weighted_mean.unsqueeze(0) * sign / B
        return grad_act, None


def _kl_divergence_sparse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """KL divergence for sparse SPLADE vectors (arXiv:2109.10086).

    Normalizes activations to probability distributions then computes KL(teacher || student).
    """
    eps = 1e-8  # standard numerical guard
    s_dist = F.softmax(student, dim=-1) + eps
    t_dist = F.softmax(teacher, dim=-1) + eps
    return F.kl_div(s_dist.log(), t_dist, reduction='batchmean')


def sparse_contrastive_loss(
    sparse_vecs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss on sparse SPLADE vectors (arXiv:2004.11362).

    Pulls same-class sparse representations closer, improving concept
    coherence for better attribution consistency.
    """
    B = sparse_vecs.shape[0]
    if B <= 1:
        return sparse_vecs.new_tensor(0.0)

    normed = F.normalize(sparse_vecs, p=2, dim=-1)
    sim = normed @ normed.T / temperature  # [B, B]

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_mask.fill_diagonal_(0)

    if pos_mask.sum() == 0:
        return sparse_vecs.new_tensor(0.0)

    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max)

    self_mask = 1.0 - torch.eye(B, device=sparse_vecs.device)
    denom = (exp_sim * self_mask).sum(dim=1, keepdim=True)

    log_prob = (sim - sim_max) - torch.log(denom + 1e-8)

    loss = -(pos_mask * log_prob).sum() / pos_mask.sum()
    return loss
