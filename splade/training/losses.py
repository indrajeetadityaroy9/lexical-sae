"""Training losses and DF statistics."""

import torch

class DocumentFrequencyTracker:
    """Track per-term document frequency for DF-weighted regularization."""

    def __init__(self, vocab_size: int, device: str | torch.device = "cuda"):
        self.vocab_size = vocab_size
        self.device = device
        self.df_counts = torch.zeros(vocab_size, device=device)
        self.doc_count = 0

    def update(self, sparse_vectors: torch.Tensor) -> None:
        term_presence = (sparse_vectors.detach() > 0).float()
        self.df_counts += term_presence.sum(dim=0)
        self.doc_count += sparse_vectors.shape[0]

    def get_weights(self, alpha: float = 0.1, beta: float = 10.0) -> torch.Tensor:
        df_ratio = self.df_counts / self.doc_count
        # EPS for log stability in weight calculation (hardcoded for SOTA stability)
        eps = 1e-7
        
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

    def reset(self) -> None:
        self.df_counts.zero_()
        self.doc_count = 0

    def soft_reset(self, momentum: float = 0.9) -> None:
        """Decay counts instead of zeroing — stabilizes early-epoch DF estimates."""
        self.df_counts *= momentum
        self.doc_count = int(self.doc_count * momentum)

class DFFlopsRegFunction(torch.autograd.Function):
    """Autograd function for DF-weighted FLOPS regularization."""

    @staticmethod
    def forward(ctx, activations: torch.Tensor, df_weights: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(activations, df_weights)
        weighted_mean = (df_weights.unsqueeze(0) * activations.abs()).mean(dim=0)
        return (weighted_mean ** 2).sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        activations, df_weights = ctx.saved_tensors
        batch_size = activations.shape[0]
        weighted_mean = (df_weights.unsqueeze(0) * activations.abs()).mean(dim=0)
        sign = torch.where(activations != 0, torch.sign(activations), torch.zeros_like(activations))
        grad_activations = (
            grad_output
            * 2.0
            * df_weights.unsqueeze(0)
            * weighted_mean.unsqueeze(0)
            * sign
            / batch_size
        )
        return grad_activations, None