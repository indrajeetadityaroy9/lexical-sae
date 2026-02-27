# SPALF: Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds

Training interpretable sparse autoencoders via constrained optimization with control-theoretic dual dynamics.

## Abstract

Sparse autoencoders (SAEs) decompose transformer activations into interpretable features, but standard training treats reconstruction and sparsity as competing objectives with manually tuned trade-offs. SPALF recasts SAE training as a single constrained optimization problem: **minimize the expected activation rate of a JumpReLU dictionary over a spectrally-preconditioned activation space, subject to faithfulness, anchoring, and diversity constraints, solved via an augmented Lagrangian whose dual dynamics use active disturbance rejection.**

Six components — Soft-ZCA whitening, a stratified decoder, CAGE-inspired discretization correction, ADRC-controlled dual ascent, modified CAPU penalty adaptation, and two-phase training — are unified under this formulation. The system requires **4 user-facing hyperparameters**; all remaining parameters (18 values) are self-calibrated from data.

## Method Overview

SPALF solves the following augmented Lagrangian:

```
L(theta, lambda, rho) = l0_corr(theta) + sum_i rho_i * Psi(g_i(theta), lambda_i / rho_i)
```

where `theta` are primal SAE parameters, `lambda` are dual variables governed by ADRC, `rho` are per-constraint adaptive penalties, `l0_corr` is a discretization-corrected sparsity objective, and `Psi` is the AL-CoLe smooth inequality penalty.

### Components

| Component | Role | Reference |
|:---|:---|:---|
| **Soft-ZCA whitening** | Isotropic preconditioning of residual stream activations | [Kiani et al., 2024](https://arxiv.org/abs/2411.17538) |
| **Stratified decoder** | *V* anchored features (tied to unembedding) + *F-V* free features | [Dinesh & Nanda, 2025](https://arxiv.org/abs/2512.05534) |
| **Discretization correction** | Scheduled penalty reducing STE bias in JumpReLU thresholds | [Defossez et al., 2025](https://arxiv.org/abs/2510.18784) |
| **ADRC dual dynamics** | ESO + PI controller for Lagrange multiplier updates | [Wang & Yang, 2026](https://arxiv.org/abs/2601.18142) |
| **Modified CAPU** | Non-monotone per-constraint adaptive penalty weights | [Chen et al., 2025](https://arxiv.org/abs/2508.15695) |
| **Two-phase training** | Phase 1: constrained sparse opt; Phase 2: end-to-end KL calibration | [Bricken et al., 2025](https://arxiv.org/abs/2503.17272) |

### Architecture

**Encoder.** A single matrix `W_enc` in R^{F x d} operates on whitened activations `x_tilde = W_white(x - mu)`. JumpReLU gating with per-feature learnable thresholds produces sparse codes `z`.

**Decoder.** Partitioned into two strata:
- **Stratum A (Anchored):** `W_dec_A` in R^{d x V}, initialized to the model's unembedding matrix. Subject to a drift constraint that permits controlled departure from vocabulary directions.
- **Stratum B (Free):** `W_dec_B` in R^{d x (F-V)}, random unit-norm columns. Subject to a co-activation orthogonality constraint.

This stratification is motivated by the SDL non-identifiability theorem ([Dinesh & Nanda, 2025](https://arxiv.org/abs/2512.05534), Theorem 3.4), which proves that unconstrained SAEs can converge to zero-loss solutions recovering no ground-truth features.

### Constraints

Three inequality constraints `g_i(theta) <= 0` encode structural priors:

| Constraint | Definition | Threshold |
|:---|:---|:---|
| **C1 — Faithfulness** | MSE in whitened space (Mahalanobis norm) | `tau_faith = (1 - R2_target) * d` |
| **C2 — Vocabulary drift** | Frobenius norm of anchored decoder departure | `tau_drift = delta_drift^2 * \|\|W_vocab\|\|_F^2` |
| **C3 — Co-activation orthogonality** | Mean pairwise cos^2 of active decoder columns | `tau_ortho = c_ortho / d` |

### Control System

The dual variables `lambda_i` are updated via an ADRC controller that combines PI feedback with disturbance rejection:

```
u_i = k_ap * (v_fast_i - v_fast_prev_i) + k_i * v_slow_i - f_hat_i
lambda_i <- max(0, lambda_i + u_i)
```

where `k_ap = 2*omega_o`, `k_i = omega_o^2` yield critically damped dynamics, and `f_hat_i` is the Extended State Observer's disturbance estimate. The penalty weights `rho_i` adapt via a non-monotone variant of CAPU that allows relaxation when violations decrease. Both operate on a slow timescale (every 100 steps), maintaining two-timescale separation from primal Adam updates.

### Training Pipeline

```
1. Calibration     Stream tokens, compute covariance, build whitener, set thresholds
2. Initialization  Matched-filter encoder, vocabulary-anchored decoder, threshold calibration
3. Phase 1         Constrained sparse optimization (convergence-gated)
4. Phase 2         End-to-end causal calibration with KL divergence (~3% of budget)
```

## Installation

Requires Python >= 3.10 and CUDA.

```bash
pip install -e .
```

**Dependencies:** PyTorch >= 2.4, TransformerLens >= 2.0, Transformers >= 4.40, Triton >= 3.0, safetensors >= 0.4, Accelerate >= 1.0

## Usage

### Training

```bash
spalf-train experiments/pythia_1b.yaml
```

Example configuration (`experiments/pythia_1b.yaml`):

```yaml
model_name: "EleutherAI/pythia-1.4b"
hook_point: "blocks.6.hook_resid_post"
dataset: "monology/pile-uncopyrighted"
total_tokens: 1_000_000_000
batch_size: 4096
seq_len: 128
R2_target: 0.97
lr: 3e-4
seed: 42
output_dir: "runs/pythia_1b"
```

### Evaluation

```bash
spalf-eval experiments/pythia_1b.yaml
```

The config must specify a `checkpoint` field pointing to a checkpoint directory. Four evaluation suites are available:

| Suite | Measures |
|:---|:---|
| `downstream_loss` | KL divergence under SAE-patched forward passes |
| `sparsity_frontier` | L0 vs. cross-entropy Pareto trade-off |
| `drift_fidelity` | Cosine similarity between anchored decoder and vocabulary |
| `feature_absorption` | Free feature alignment with vocabulary directions ([Engels et al., 2024](https://arxiv.org/abs/2409.14507)) |

### Configuration

SPALF requires 4 hyperparameters. All other values are self-calibrated:

| Parameter | Default | Description |
|:---|:---|:---|
| `F` | `32 * d_model` | Dictionary size (0 = auto) |
| `L0_target` | `ceil(F / 400)` | Target active features per input (None = auto) |
| `R2_target` | 0.97 | Reconstruction explained variance |
| `lr` | 3e-4 | Adam learning rate |

## Checkpoints

Checkpoints use safetensors and are saved as directories:

```
spalf_phase2_step244140/
  model.safetensors        SAE weights
  tensors.safetensors      Whitener state + W_vocab
  control.safetensors      ADRC/CAPU/EMA state
  optimizer.safetensors    Adam state
  metadata.json            Scalars, config, calibration parameters
```

## Theoretical Grounding

SPALF draws on and integrates results from several lines of work. The table below maps each component to its primary theoretical reference and the nature of the transfer:

| Component | Reference | Transfer Status |
|:---|:---|:---|
| Soft-ZCA whitening | [Kiani et al., 2024](https://arxiv.org/abs/2411.17538); [Kessy et al., 2018](https://arxiv.org/abs/1804.08450) | Direct application |
| Stratified anchoring | [Dinesh & Nanda, 2025](https://arxiv.org/abs/2512.05534), Theorem 3.4 | Motivates architecture; partial anchoring validated empirically |
| JumpReLU STE | [Rajamanoharan et al., 2024](https://arxiv.org/abs/2407.14435) | Direct application |
| AL-CoLe smooth penalty | [Hounie et al., 2025](https://arxiv.org/abs/2510.20995), Theorem 2.1 | Penalty function adopted; strong duality theorem does not transfer (per-constraint rho, ADRC updates) |
| PI = ALM equivalence | [Gemp et al., 2025](https://arxiv.org/abs/2509.22500), Theorem 1 | Conceptual motivation; strict equivalence broken by Adam, EMA, ESO |
| ADRC + ESO | [Wang & Yang, 2026](https://arxiv.org/abs/2601.18142), Theorem C.7 | ISS tracking bound conjectured for first-order setting; proven for second-order |
| CAGE discretization | [Defossez et al., 2025](https://arxiv.org/abs/2510.18784), Theorem 1 | Empirically motivated; convergence guarantee does not transfer to discontinuous JumpReLU |
| Modified CAPU | [Chen et al., 2025](https://arxiv.org/abs/2508.15695), Theorem 1 | Non-monotone variant loses monotone convergence guarantee; boundedness preserved |
| Two-timescale separation | [Doan et al., 2021](https://arxiv.org/abs/2112.03515) | Timescale ratio r >= 10 satisfied by EMA rates |

## References

### Core SAE Methods

- [Rajamanoharan et al., 2024](https://arxiv.org/abs/2407.14435) — JumpReLU Sparse Autoencoders. *ICLR 2025.*
- [Gao et al., 2024](https://arxiv.org/abs/2406.04093) — Scaling and Evaluating Sparse Autoencoders (TopK). *OpenAI, 2024.*
- [Bussmann et al., 2024](https://arxiv.org/abs/2412.06410) — BatchTopK Sparse Autoencoders. *2024.*
- [Koppel et al., 2025](https://arxiv.org/abs/2502.12892) — Archetypal SAE (A-SAE / RA-SAE). *ICML 2025.*
- [Bricken et al., 2025](https://arxiv.org/abs/2503.17272) — Revisiting End-to-End SAE Training. *2025.*

### Feature Geometry and Interpretability

- [Dinesh & Nanda, 2025](https://arxiv.org/abs/2512.05534) — Unified Theory of Sparse Dictionary Learning. *ICLR 2026.*
- [Engels et al., 2024](https://arxiv.org/abs/2409.14507) — Feature Absorption in SAEs. *NeurIPS 2025.*
- [Park et al., 2025](https://arxiv.org/abs/2501.17727) — Not All Language Model Features Are Linear. *ICLR 2025.*
- [Elhage et al., 2026](https://arxiv.org/abs/2602.02385) — Transformers Learn Factored Representations. *2026.*
- [Chanin et al., 2026](https://arxiv.org/abs/2602.14111) — SAE Sanity Checks: Random Baselines. *2026.*

### Whitening and Preconditioning

- [Kiani et al., 2024](https://arxiv.org/abs/2411.17538) — Soft-ZCA Whitening. *2024.*
- [Kessy et al., 2018](https://arxiv.org/abs/1804.08450) — Optimal Whitening and Decorrelation. *2018.*
- [Balagansky & Gavves, 2025](https://arxiv.org/abs/2511.13981) — Data Whitening Improves SAE Learning. *AAAI 2026.*

### Control Theory and Constrained Optimization

- [Gemp et al., 2025](https://arxiv.org/abs/2509.22500) — PI Control = Augmented Lagrangian. *ICLR 2026.*
- [Wang & Yang, 2026](https://arxiv.org/abs/2601.18142) — ADRC-Lagrangian Methods for Safety. *2026.*
- [Stooke et al., 2024](https://arxiv.org/abs/2406.04558) — nuPI: PI Controllers for Constrained Optimization. *ICML 2024.*
- [Hounie et al., 2025](https://arxiv.org/abs/2510.20995) — AL-CoLe: Augmented Lagrangian for Constrained Learning. *NeurIPS 2025.*
- [Chen et al., 2025](https://arxiv.org/abs/2508.15695) — CAPU: Conditionally Adaptive Penalty Updates. *2025.*

### Non-Smooth Optimization

- [Defossez et al., 2025](https://arxiv.org/abs/2510.18784) — CAGE: Curvature-Aware Gradient Estimation. *2025.*
- [Doan et al., 2021](https://arxiv.org/abs/2112.03515) — Multi-Timescale Stochastic Approximation. *2025.*

### Evaluation and Benchmarks

- [Karvonen et al., 2025](https://arxiv.org/abs/2503.09532) — SAEBench. *2025.*
- [Lindsey et al., 2026](https://arxiv.org/abs/2602.14687) — SynthSAEBench. *2026.*
- [Balagansky et al., 2025](https://arxiv.org/abs/2505.20254) — Feature Consistency (PW-MCC). *2025.*
