# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lexical-SAE (Circuit-Anchored JumpReLU Transcoder) is a sparse autoencoder for interpretable text classification. It repurposes the masked language model vocabulary as a sparse feature dictionary and trains end-to-end with circuit-integrated losses. The key property is that predictions decompose exactly via Direct Linear Attribution (DLA): `logit[c] = Σⱼ s[j] · W_eff[c, j] + b_eff[c]`, verified algebraically (error < 1e-3) for every input.

## Commands

### Install
```bash
pip install -e .
```

### Run tests
```bash
pytest tests/
pytest tests/ -k "not slow"          # skip tests requiring network access
pytest tests/test_circuits.py        # run a single test file
pytest tests/test_circuits.py -k "test_geco"  # run a single test
```

### Run experiments
```bash
python -m splade.scripts.run_experiment --config experiments/paper/imdb.yaml
python -m splade.scripts.run_surgery --config experiments/surgery.yaml
python -m splade.scripts.run_ablation --config experiments/ablation/ablation.yaml
python -m splade.scripts.run_faithfulness --config experiments/verify.yaml
```

## Architecture

### Data Flow
```
Input → Backbone (AutoModelForMaskedLM) → MLM Logits [B,L,V]
  → Optional VPE (Virtual Polysemy Expansion)
  → JumpReLU Gate → Sparse Activations
  → Pooling (max or attention) → [B, V_expanded]
  → ReLU MLP Classifier (fc1 → ReLU → fc2) → Logits + W_eff + b_eff
  → DLA Attribution: attr[j] = s[j] * W_eff[c,j]
```

### Key Modules (under `splade/`)

- **`models/lexical_sae.py`** — `LexicalSAE`: main model combining backbone, JumpReLU gate, optional VPE, and ReLU MLP classifier. `classifier_forward()` derives `W_eff` and `b_eff` for exact DLA.
- **`models/layers/activation.py`** — `JumpReLUGate`: forward uses exact Heaviside gate `z · H(z - θ)`, backward uses sigmoid STE. Thresholds stored as `log θ` for positivity.
- **`models/layers/virtual_expander.py`** — Optional polysemy expansion via learned sense vectors.
- **`circuits/core.py`** — `CircuitState` (NamedTuple: logits, sparse_vector, W_eff, b_eff) and `circuit_mask()` with temperature-parameterized sigmoid.
- **`circuits/losses.py`** — KL completeness, contrastive separation, gate sparsity (L0), feature frequency penalty. Also `AttributionCentroidTracker` (EMA per-class centroids) and `FeatureFrequencyTracker`.
- **`circuits/geco.py`** — `GECOController`: constrained optimization. τ set from 25th percentile of warmup CE; λ adapted via EMA-smoothed dual ascent.
- **`training/loop.py`** — Two-phase training: Phase 1 (warmup, CE only) → Phase 2 (GECO-constrained circuit optimization). Uses Schedule-Free AdamW with gradient centralization.
- **`training/constants.py`** — Key hyperparameters: `CLASSIFIER_HIDDEN=256`, `MAX_EPOCHS=50`, `EARLY_STOP_PATIENCE=10`, `LABEL_SMOOTHING=0.1`, `JUMPRELU_BANDWIDTH=0.001`.
- **`mechanistic/attribution.py`** — `compute_attribution_tensor()`: core DLA computation.
- **`evaluation/mechanistic.py`** — Full evaluation pipeline: DLA verification, completeness, separation, ERASER metrics, explainer comparisons.
- **`intervene.py`** — `SuppressedModel` for reversible inference-time concept removal.
- **`pipelines.py`** — `setup_and_train()` shared pipeline used by all scripts.
- **`config/schema.py`** — Dataclass configs (`DataConfig`, `ModelConfig`, `TrainingConfig`, `VPEConfig`) loaded from YAML via `config/load.py`.
- **`data/loader.py`** — Dataset loading for banking77, imdb, yelp, civilcomments, beavertails.

### Training Flow

YAML config → `setup_and_train()` → loads dataset, creates `LexicalSAE`, initializes Schedule-Free AdamW + GECOController + trackers → Phase 1 warmup (CE only, records values for GECO τ) → Phase 2 circuit optimization (GECO constrains CE ≤ τ while minimizing circuit losses) → early stopping → mechanistic evaluation.

### Critical Invariant

Both JumpReLU and the classifier ReLU are piecewise-linear, so `W_eff` is locally constant per input. This makes the DLA decomposition exact (not approximate). Any change to the activation functions or classifier architecture must preserve this property.
