# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPALF (Spectrally-Preconditioned Augmented Lagrangian on Stratified Feature Manifolds) trains interpretable sparse autoencoders (SAEs) on transformer activations via constrained optimization with control-theoretic dual dynamics. It requires only 4 user hyperparameters (F, L0_target, R2_target, lr); 18+ parameters are self-calibrated from data.

## Commands

### Install
```bash
pip install -e .
```

### Train
```bash
spalf-train experiments/pythia_1b.yaml
```

### Evaluate
```bash
spalf-eval experiments/pythia_1b.yaml
```
Config must include a `checkpoint` field pointing to a saved checkpoint directory.

## Architecture

### Source Layout (`src/`)

- **`config.py`** — `SPALFConfig` dataclass, YAML serialization
- **`constants.py`** — 7 fixed structural constants (not user-tunable)
- **`constraints.py`** — Three inequality constraints (faithfulness, drift, orthogonality) and AL-CoLe smooth penalty
- **`runtime.py`** — CUDA/TF32/flash-attention setup, seeding
- **`checkpoint.py`** — Safetensors checkpoint save/load

### Model (`src/model/`)
- **`sae.py`** — `StratifiedSAE`: encoder `W_enc` with JumpReLU gating, stratified decoder split into anchored stratum `W_dec_A` (tied to unembedding) and free stratum `W_dec_B` (random unit-norm)
- **`jumprelu.py`** — JumpReLU activation with Rectangle STE, learnable per-feature thresholds and adaptive bandwidth
- **`initialization.py`** — Matched-filter encoder init, vocabulary-anchored decoder init, Gram-Schmidt orthogonalization for free columns, threshold calibration

### Control System (`src/control/`)
- **`adrc.py`** — `ExtendedStateObserver` + `ADRCController` for dual multiplier updates (PI + disturbance rejection)
- **`capu.py`** — `ModifiedCAPU`: non-monotone per-constraint adaptive penalty weights
- **`ema_filter.py`** — Dual-rate EMA (fast β=0.9, slow β=0.99) providing two-timescale constraint violation signals

### Whitening (`src/whitening/`)
- **`whitener.py`** — `SoftZCAWhitener`: full-rank or low-rank (d > 4096) spectral preconditioning with self-calibrating regularization α
- **`covariance.py`** — `OnlineCovariance`: streaming Welford's algorithm, convergence-gated

### Data (`src/data/`)
- **`activation_store.py`** — Streams activations from TransformerLens or HuggingFace hooks
- **`buffer.py`** — In-memory activation buffer (2^20 tokens) with half-refill strategy
- **`patching.py`** — Causal intervention: replaces activations with SAE reconstruction, computes KL/CE

### Training (`src/training/`)
- **`calibration.py`** — Pre-training: covariance estimation, whitener construction, threshold computation
- **`initialization.py`** — SAE weight initialization from calibration results
- **`phase1.py`** — Constrained sparse optimization (convergence-gated transition)
- **`phase2.py`** — End-to-end KL calibration (~3% of budget), freezes CAPU penalties
- **`trainer.py`** — `SPALFTrainer`: orchestrates calibration → init → Phase 1 → Phase 2
- **`logging.py`** — Metrics tracking

### GPU Kernels (`src/kernels/`)
- **`jumprelu_kernel.py`** — Fused Triton kernel for JumpReLU + discretization correction + Moreau envelope STE backward

### Evaluation (`src/evaluation/`)
- **`downstream_loss.py`** — KL divergence and CE loss under SAE-patched forward passes
- **`sparsity_frontier.py`** — L0 vs. CE Pareto frontier, drift-fidelity, feature absorption metrics

### Entry Points (`src/scripts/`)
- **`train.py`** — `spalf-train` CLI
- **`evaluate.py`** — `spalf-eval` CLI

## Key Design Patterns

- **Two-phase training**: Phase 1 minimizes sparsity subject to constraints; Phase 2 switches faithfulness constraint to a 50/50 MSE+KL blend for causal validity. Phase transition is convergence-gated (all constraints satisfied for 100 consecutive steps).
- **Control-theoretic duals**: Lagrange multipliers are updated by ADRC controllers (not simple gradient ascent). The ESO estimates stochastic disturbance, and CAPU adapts per-constraint penalty weights. Both update every 100 primal steps.
- **Stratified decoder**: Anchored columns track the unembedding matrix (drift-constrained); free columns are orthogonality-constrained. This addresses SDL non-identifiability.
- **Self-calibration**: Constraint thresholds, JumpReLU parameters, whitening regularization, and controller gains are all derived from activation statistics and user knobs — not manually tuned.

## Configuration

YAML configs in `experiments/` specify model, dataset, and the 4 user knobs. Example fields: `model_name`, `hook_point`, `dataset`, `total_tokens`, `batch_size`, `seq_len`, `R2_target`, `lr`, `seed`, `output_dir`.

## Checkpoints

Saved as directories containing:
- `model.safetensors` — SAE weights
- `tensors.safetensors` — Whitener state + W_vocab
- `control.safetensors` — ADRC/CAPU/EMA state
- `optimizer.safetensors` — Adam state
- `metadata.json` — Scalars, config, calibration parameters

## Dependencies

Python >= 3.10, CUDA required. Key: PyTorch >= 2.4, TransformerLens >= 2.0, Triton >= 3.0, safetensors >= 0.4.
