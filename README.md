# SPALF  
### Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds

SPALF trains sparse autoencoders (SAEs) for mechanistic interpretability by formulating sparsity as a **constrained optimization problem**.

Instead of manually weighting reconstruction against sparsity, SPALF minimizes $\ell_0$ sparsity subject to faithfulness, lexical drift, and orthogonality constraints. Optimization is performed using an **augmented Lagrangian with adaptive dual dynamics**.

---

## Core Objective

We solve:

```math
\min_\theta \; \mathbb{E}\left[\|z\|_0\right]
\quad
\text{s.t.}
\quad
g_i(\theta) \le 0
```

Using the augmented Lagrangian:

```math
\mathcal{L}(\theta, \lambda, \rho)
=
\mathbb{E}\left[\|z\|_0\right]
+
\sum_i
\rho_i \,
\Psi\!\left(
g_i(\theta),
\frac{\lambda_i}{\rho_i}
\right)
```

Where the **AL-CoLe smooth penalty** is:

```math
\Psi(g, y)
=
\frac{\max(0,\, 2g + y)^2 - y^2}{4}
```

This provides a $C^1$-smooth transition that removes the classical non-smooth $\max(0,\cdot)^2$ kink.

---

# Stratified Decoder Architecture

The decoder is partitioned into two blocks.

## Anchored Columns

```math
W_A \in \mathbb{R}^{d \times V}
```

- Initialized from the model unembedding matrix  
- Constrained to remain close to preserve lexical grounding  
- Provides a vocabulary-aligned basis  

---

## Free Columns

```math
W_B \in \mathbb{R}^{d \times (F - V)}
```

- Learned from scratch  
- Unit-norm projected  
- Captures residual structure orthogonal to vocabulary directions  

---

## Encoder Initialization

Matched-filter initialization:

```math
W_{\text{enc}}^{[:V]}
=
\Sigma^{-1} W_{\text{vocab}}
```

- $\Sigma$ = covariance matrix (whitening transform)  
- Anchored directions are spectrally preconditioned  
- Free encoder block is QR-orthogonalized against anchored block  

---

# JumpReLU Gating with Moreau Envelope STE

SPALF uses **JumpReLU with learnable log-thresholds**.

The non-differentiable Heaviside step function $H(x)$ is replaced during backpropagation by a **Moreau envelope proximal gradient surrogate**.

The straight-through estimator (STE) becomes a linear ramp inside:

```math
\left[-\sqrt{2\gamma_j}, \; 0 \right]
```

around each feature threshold.

Per-feature bandwidth:

```math
\gamma_j
=
\frac{\left(c_\varepsilon \cdot \operatorname{IQR}_j\right)^2}{2}
```

- $\operatorname{IQR}_j$ = interquartile range of pre-activations  
- $c_\varepsilon$ = calibration constant  
- Recalibrated periodically during training  

Both the $\ell_0$ objective and reconstruction loss propagate gradients through this kernel.

---

# Augmented Lagrangian Optimization

SPALF separates primal and dual timescales.

---

## Dual Update (nuPI Controller)

```math
\lambda
\leftarrow
\max\!\left(
\lambda
+
\rho e
+
\kappa_p \rho (e - e_{\text{prev}}),
\; 0
\right)
```

- $e$ = slow-EMA filtered violation  
- $\kappa_p = 0.5$ derived from sampling-to-timescale ratio  
- Not a tunable hyperparameter  

The proportional term damps oscillations near constraint boundaries.

---

## Penalty Update (Monotone CAPU)

```math
\rho_i
\leftarrow
\max\!\left(
\rho_i,
\frac{\eta_i}{\sqrt{\bar{v}_i}}
\right)
```

- $\eta_i$ calibrated from initial violation magnitudes  
- $\bar{v}_i$ = running variance estimate  
- Monotone increasing by construction  

---

## Constraint Filtering (Dual-Rate EMA)

Per-batch violations are filtered via two EMAs.

Fast channel:

```math
\beta_f \approx 0.9
```

Slow channel:

```math
\beta_s = 0.99
```

With Adam-style bias correction:

```math
\hat{m}_t
=
\frac{m_t}{1 - \beta^t}
```

- Fast EMA drives penalty adaptation  
- Slow EMA drives dual updates  

---

# Design Principles

- No manual sparsity weighting  
- Lexically grounded decoder geometry  
- Spectral preconditioning  
- Smooth augmented Lagrangian constraints  
- Adaptive dual control  
- Minimal hyperparameter surface  

---
SPALF replaces heuristic sparsity trade-offs with a principled constrained optimization framework.

It combines:

- Spectrally preconditioned encoder initialization  
- Stratified decoder manifolds  
- Moreau-envelope STE gating  
- Smooth AL-CoLe augmented Lagrangian  
- nuPI dual dynamics  
- Monotone adaptive penalties  

The result is a **structurally stable, lexically grounded, constraint-satisfying sparse autoencoder** optimized via principled dual control rather than manual coefficient tuning.