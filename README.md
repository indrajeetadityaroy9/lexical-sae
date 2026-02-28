# SPALF: Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds

SPALF trains sparse autoencoders (SAEs) for mechanistic interpretability by formulating sparsity as a constrained optimization problem. Rather than manually weighting reconstruction against sparsity, SPALF minimizes L0 sparsity subject to faithfulness, lexical drift, and orthogonality constraints, solved via an augmented Lagrangian with adaptive dual dynamics.

### Stratified Decoder Architecture

The decoder is partitioned into two:
- **Anchored columns** `W_A ∈ R^{d×V}`: initialized from the model's unembedding matrix and constrained to remain close, providing a lexically grounded basis.
- **Free columns** `W_B ∈ R^{d×(F-V)}`: learned from scratch with unit-norm projection, capturing residual structure orthogonal to the vocabulary.

The encoder is initialized via matched filters: `W_enc[:V] = Σ^{-1} W_vocab` (whitened vocabulary directions), with the free encoder block QR-orthogonalized against the anchored block.

### JumpReLU Gating with Moreau Envelope STE

Gating uses JumpReLU with learnable log-thresholds. The non-differentiable Heaviside step is handled via a Moreau envelope proximal gradient, which provides a linear ramp STE in the transition zone `[-√(2γ_j), 0]` around each threshold. Per-feature bandwidth `γ_j = (c_ε · IQR_j)² / 2` is calibrated from local pre-activation statistics and periodically recalibrated during training.

Both the L0 sparsity objective and the reconstruction loss route gradients to the threshold parameters through the Moreau envelope kernel, implemented as fused Triton kernels. This follows Eq. 11 from [Rajamanoharan et al. (2024)](#references), generalized from the rectangle kernel to the Moreau envelope.

### Augmented Lagrangian Optimization

The constrained problem is solved via an augmented Lagrangian with the AL-CoLe smooth penalty:

```
L(θ, λ, ρ) = E[||z||_0] + Σ_i ρ_i · Ψ(g_i(θ), λ_i/ρ_i)
```

where `Ψ(g, y) = (max(0, 2g + y)² - y²) / 4` is the C¹-smooth penalty from [Hoeltgen et al. (2025)](#references) that eliminates the non-smooth `max(0, ·)²` transition of the classical augmented Lagrangian.

The dual variables `λ` and penalty coefficients `ρ` operate on a slow timescale separated from the primal SGD updates:

- **Dual update (nuPI controller)**: `λ ← max(λ + ρ·e + κ_p·ρ·(e - e_prev), 0)`, where `e` is the slow-EMA-filtered constraint violation. The proportional term damps oscillations around constraint boundaries. The gain `κ_p = 0.5` is structurally derived from the sampling-to-timescale ratio (not a tunable parameter).

- **Penalty update (Monotone CAPU)**: `ρ_i ← max(ρ_i, η_i / √v̄_i)`, a monotone-increasing per-constraint adaptive penalty. The initial scale `η_i` is calibrated from initial violation magnitudes.

- **Constraint filtering (dual-rate EMA)**: Raw per-batch violations are filtered through fast (β ≈ 0.9) and slow (β = 0.99) exponential moving averages with Adam-style bias correction. The fast channel drives penalty adaptation; the slow channel drives dual updates.