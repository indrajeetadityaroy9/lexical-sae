# Circuit-Anchored JumpReLU Transcoder (CAJT)

A circuit-anchored sparse transcoder that repurposes the masked language model vocabulary as a sparse feature dictionary. The architecture provides exact per-prediction attributions via a Direct Linear Attribution (DLA) identity, and enables surgical concept removal with algebraic guarantees.

1. **JumpReLU gating** with exact binary gates and sigmoid straight-through estimation ([Rajamanoharan et al., 2024](https://arxiv.org/abs/2407.14435))
2. **KL-divergence completeness loss** ensuring sparse circuits preserve the full model's output distribution
3. **Contrastive separation loss** with hard negative mining and learned margin for orthogonal per-class circuits
4. **Frequency-based feature targeting** penalizing under-active features via continuous EMA frequency tracking
5. **GECO constrained optimization** ([Rezende & Viola, 2018](https://arxiv.org/abs/1810.00597)) balancing accuracy against circuit interpretability

## Method

**JumpReLU** ([Rajamanoharan et al., 2024](https://arxiv.org/abs/2407.14435)) uses a learnable per-dimension threshold θ (stored as log θ for positivity) with a hard binary Heaviside gate in the forward pass and a sigmoid straight-through estimator in the backward pass. Active features retain their full `relu(x)` magnitude. The binary gate ensures the DLA identity holds exactly during both training and evaluation, not just at inference time.

**DLA Identity.** For the ReLU MLP classifier `fc1 → ReLU → fc2`, the effective weight matrix is `W_eff(x) = W₂ · diag(D(x)) · W₁` where `D(x) = 𝟙[fc1(x) > 0]` is the binary ReLU activation mask. Since both the JumpReLU gate and the classifier ReLU are piecewise-linear, `W_eff` is locally constant for each input and the prediction decomposes exactly:

```
logit[c] = Σⱼ s[j] · W_eff[c, j] + b_eff[c]
```

This identity is verified algebraically (error < 1e-3) for every input at evaluation time.

### Circuit-Integrated Training

Training uses a two-phase GECO-constrained optimization with **Schedule-Free AdamW** ([Defazio & Mishchenko, 2024](https://arxiv.org/abs/2405.15682)), which subsumes LR scheduling, warmup, and model EMA into a single optimizer via Primal Averaging:

**Phase 1 — Warmup** (first ~20% of steps): Cross-entropy only. The GECO controller records CE values and sets its constraint threshold τ from the 25th percentile.

**Phase 2 — Circuit optimization** (remaining steps): GECO constrains CE ≤ τ while minimizing the circuit objective:

| Loss | Purpose | Formulation |
|------|---------|-------------|
| **KL Completeness** | Top-k% sparse dims (by attribution magnitude) preserve the full model's output distribution | `KL(p_full ‖ p_masked)` with soft circuit masking |
| **Contrastive Separation** | Per-class attribution circuits are orthogonal | Triplet margin loss with learned margin against class centroid hard negatives |
| **Gate Sparsity (L0)** | Penalize number of open gates | `mean(σ((z − θ) / ε))`, differentiable L0 proxy |
| **Feature Frequency Penalty** | Resurrect under-active features | Penalize high `log θ` for features with EMA frequency below `sparsity_target × 0.1` |

The combined Lagrangian is:

```
L = (cc_loss + sep_loss + gate_loss + freq_loss) + λ_ce · CE_loss
```

where λ_ce is adapted via EMA-smoothed dual ascent with step size η = 2 / (steps_per_epoch + 1).

Additional training components: Schedule-Free AdamW with gradient centralization ([Yong et al., 2020](https://arxiv.org/abs/2004.01461)), derived weight decay (0.1 × lr, Loshchilov-Hutter scaling), early stopping, and label smoothing.

### Evaluation

The mechanistic evaluation pipeline verifies:
- **DLA verification error** — algebraic identity holds per sample
- **Circuit completeness** — top-k% of active dims at multiple fractions (1%, 5%, 10%, 20%, 50%)
- **Circuit separation** — cosine similarity and Jaccard overlap between per-class circuits
- **ERASER metrics** ([DeYoung et al., 2020](https://arxiv.org/abs/1911.03429)) — comprehensiveness, sufficiency, AOPC
- **Explainer comparison** — vs. LIME, Integrated Gradients, Attention baselines
- **SAE baseline comparison** — reconstruction error and active feature counts
- **Layer-wise attribution** — distribution across backbone layers

### Surgical Concept Removal

The sparse bottleneck enables verifiable concept removal. Since `logit[c] = Σⱼ s[j] · W_eff[c,j] + b_eff[c]`, zeroing `s[j]` removes token j's contribution to all classes with algebraic certainty.

The surgery pipeline identifies identity-correlated tokens in class attribution centroids and compares:
1. **Baseline** — unmodified FPR gap per identity group
2. **Surgical suppression** — zero sparse activations for target tokens via `SuppressedModel` (reversible, inference-time)
3. **LEACE erasure** ([Belrose et al., 2023](https://arxiv.org/abs/2306.03819)) — covariance-based concept projection as a rigorous baseline

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.4.0, and a CUDA-enabled GPU.

## Usage

### Classification (train + mechanistic evaluation)

```bash
python -m splade.scripts.run_experiment --config experiments/paper/imdb.yaml
```

### Surgical Concept Removal

```bash
python -m splade.scripts.run_surgery --config experiments/surgery.yaml
```

### Ablation Studies

```bash
python -m splade.scripts.run_ablation --config experiments/ablation/ablation.yaml
```

### Faithfulness Benchmark

```bash
python -m splade.scripts.run_faithfulness --config experiments/verify.yaml
```

### Configuration

Experiments are defined by YAML files loaded into dataclass schemas:

```yaml
experiment_name: "paper_imdb"
output_dir: "results/paper/imdb"

data:
  dataset_name: "imdb"
  train_samples: -1   # full split
  test_samples: -1

model:
  name: "answerdotai/ModernBERT-base"

training:
  sparsity_target: 0.1
  warmup_fraction: 0.2
  pooling: "max"          # "max" or "attention"
  learning_rate: 3e-4     # Schedule-Free AdamW base LR
```

Supported datasets: `banking77`, `imdb`, `yelp`, `civilcomments`, `beavertails`.

## Repository Structure

```
splade/
├── models/
│   ├── lexical_sae.py                 # LexicalSAE: backbone + JumpReLU + task head
│   └── layers/
│       ├── activation.py              # JumpReLUGate with sigmoid STE
│       └── virtual_expander.py        # Virtual Polysemy Expansion (VPE)
├── circuits/
│   ├── core.py                        # CircuitState, circuit_mask()
│   ├── losses.py                      # KL completeness, contrastive separation,
│   │                                  #   gate sparsity, feature frequency penalty,
│   │                                  #   AttributionCentroidTracker (learned margin),
│   │                                  #   FeatureFrequencyTracker
│   ├── geco.py                        # GECO constrained optimization controller
│   └── metrics.py                     # Circuit extraction, completeness measurement
├── training/
│   ├── loop.py                        # Training loop (two-phase GECO)
│   ├── optim.py                       # Schedule-Free AdamW, param groups,
│   │                                  #   gradient centralization
│   └── constants.py                   # Training hyperparameters
├── mechanistic/
│   ├── attribution.py                 # compute_attribution_tensor() — core DLA
│   ├── sae.py                         # Post-hoc SAE baseline
│   └── layerwise.py                   # Per-layer attribution analysis
├── evaluation/
│   ├── mechanistic.py                 # Full mechanistic evaluation pipeline
│   ├── eraser.py                      # ERASER faithfulness metrics
│   ├── leace.py                       # LEACE concept erasure baseline
│   ├── compare_explainers.py          # Comparison with LIME, IG, Attention
│   ├── baselines.py                   # Baseline accuracy scoring
│   ├── dense_baseline.py              # Dense SAE comparison
│   └── polysemy.py                    # VPE polysemy analysis
├── data/
│   └── loader.py                      # Dataset loading and preparation
├── config/
│   ├── schema.py                      # Dataclass configs (Data, Model, Training, VPE)
│   └── load.py                        # YAML config loader
├── scripts/
│   ├── run_experiment.py              # Train + mechanistic evaluation
│   ├── run_surgery.py                 # Surgical concept removal (bias)
│   ├── run_ablation.py                # Circuit loss ablation studies
│   ├── run_faithfulness.py            # ERASER faithfulness benchmark
│   ├── run_multi_dataset.py           # Cross-dataset evaluation
│   ├── run_long_context.py            # Extended sequence evaluation
│   └── run_sota_comparison.py         # SOTA baseline comparison
├── inference.py                       # Predict, explain, score utilities
├── intervene.py                       # SuppressedModel, bias evaluation
├── pipelines.py                       # Shared setup_and_train() pipeline
└── utils/cuda.py                      # Device, dtype, seed management
experiments/                           # YAML configs for all experiments
tests/                                 # Unit tests
```

## Comparison with Post-Hoc Sparse Autoencoders

|  | Post-Hoc SAE | Lexical-SAE (CAJT) |
|---|---|---|
| **Training** | After model (frozen activations) | End-to-end (circuit losses co-trained) |
| **Reconstruction** | Lossy (x ≈ x̂) | Exact (DLA identity, error < 1e-3) |
| **Feature dictionary** | Learned latent directions | Vocabulary tokens (human-readable) |
| **Intervention** | Approximate steering | Exact zeroing (s_j = 0, algebraic guarantee) |
| **Activation** | TopK / ReLU | JumpReLU (binary gate + sigmoid STE) |
| **Dead features** | TopK avoids by design | Frequency-based feature targeting via EMA |
| **Circuit losses** | None (post-hoc analysis only) | KL completeness + contrastive separation |
| **Optimization** | Standard AdamW + scheduler | Schedule-Free AdamW (no scheduler, no EMA) |

## Known Limitations

- **Vocabulary-level granularity.** Attributions map to individual vocabulary tokens (subwords), not input spans. A clean-vocab filter masks subword continuations and special tokens for display.
- **Input-dependent W_eff.** The effective weight matrix varies per input due to the ReLU activation mask. Explanations are per-sample, not global.
- **Encoder family scope.** Evaluated on ModernBERT-base. Compatible with any `AutoModelForMaskedLM` backbone but other encoders are not yet benchmarked.

## References

### Core Method

- Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. [`arXiv:2407.14435`](https://arxiv.org/abs/2407.14435)
  *JumpReLU activation with binary Heaviside gate (forward) and sigmoid STE (backward). Provides the gating mechanism that preserves the exact DLA identity.*

- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR*. [`arXiv:2107.05720`](https://arxiv.org/abs/2107.05720)
  *MLM-logit sparse representation architecture. Lexical-SAE repurposes this from information retrieval to interpretable classification.*

- Balestriero, R. & Baraniuk, R. (2018). A Spline Theory of Deep Networks. *ICML*. [`arXiv:1802.09210`](https://arxiv.org/abs/1802.09210)
  *Piecewise-linear structure of ReLU networks. Each input maps to a locally linear region with a fixed effective weight matrix, enabling the DLA decomposition.*

- Rezende, D. J. & Viola, F. (2018). Taming VAEs. [`arXiv:1810.00597`](https://arxiv.org/abs/1810.00597)
  *GECO constrained optimization. Provides the Lagrangian framework balancing classification accuracy against circuit interpretability losses.*

### Optimization

- Defazio, A. & Mishchenko, K. (2024). The Road Less Scheduled. [`arXiv:2405.15682`](https://arxiv.org/abs/2405.15682)
  *Schedule-Free AdamW optimizer. Subsumes LR scheduling, warmup, and model EMA via Primal Averaging, replacing three separate mechanisms with a single optimizer.*

- Yong, H., et al. (2020). Gradient Centralization: A New Optimization Technique for Deep Neural Networks. [`arXiv:2004.01461`](https://arxiv.org/abs/2004.01461)
  *Parameter-free gradient conditioning. Constrains gradients to the zero-mean hyperplane, improving Lipschitz smoothness without tunable hyperparameters.*

### Sparsity and Dead Feature Prevention

- Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. [`arXiv:2406.04093`](https://arxiv.org/abs/2406.04093)
  *SAE evaluation methodology (downstream loss, probe loss, ablation sparsity). Informs the frequency-based feature targeting approach adapted here.*

- Lassance, C., et al. (2024). SPLADE-v3: New Baselines for SPLADE. [`arXiv:2403.06789`](https://arxiv.org/abs/2403.06789)
  *FLOPS-regularized sparse representation training informing the sparsity control approach.*

- Lei, T., et al. (2025). Sparse Attention Post-Training. [`arXiv:2512.05865`](https://arxiv.org/abs/2512.05865)
  *Cross-layer transcoder work informing sparse representation design and attribution across layers.*

### Evaluation and Interpretability

- DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL*. [`arXiv:1911.03429`](https://arxiv.org/abs/1911.03429)
  *Comprehensiveness, sufficiency, and AOPC metrics for attribution faithfulness evaluation.*

- Belrose, N., et al. (2023). LEACE: Perfect Linear Concept Erasure in Closed Form. *NeurIPS*. [`arXiv:2306.03819`](https://arxiv.org/abs/2306.03819)
  *Covariance-based concept erasure baseline for surgical removal experiments.*

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*. [`arXiv:1703.01365`](https://arxiv.org/abs/1703.01365)
  *Integrated Gradients baseline for explainer comparison.*

### Mechanistic Interpretability

- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.
  *Foundational framework for understanding transformer computations as circuits.*

- Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS*. [`arXiv:2304.14997`](https://arxiv.org/abs/2304.14997)
  *Circuit discovery methodology contextualizing the circuit extraction evaluation.*

- Marks, S., et al. (2024). Sparse Feature Circuits. *ICLR*. [`arXiv:2403.19647`](https://arxiv.org/abs/2403.19647)
  *Sparse circuit analysis connecting SAE features to model behavior.*

- Chen, J., et al. (2025). Rethinking Circuit Completeness. [`arXiv:2505.10039`](https://arxiv.org/abs/2505.10039)
  *Motivates the completeness loss design and evaluation methodology.*
