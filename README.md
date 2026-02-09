# Circuit-Integrated SPLADE (CIS)

**Training Text Classifiers with Mechanistic Interpretability Objectives**

CIS repurposes SPLADE's sparse lexical bottleneck for classification and exploits the resulting piecewise-linear structure to obtain **exact** per-token attribution from a single forward pass. Three differentiable circuit losses---computed by the *same function* used for evaluation---are optimized during training via constrained Lagrangian optimization, producing models with cleaner internal circuits without sacrificing accuracy.

---

## Method

The ReLU activation mask `D(s) = diag(1[W1*s + b1 > 0])` yields an exact per-input effective weight matrix:

```
W_eff(s) = W2 @ diag(D(s)) @ W1

logit_c  = sum_j [ s_j * W_eff(s)[c,j] ] + b_eff(s)_c      (algebraic identity, not an approximation)
```

This **Direct Logit Attribution (DLA)** decomposes every prediction into per-token contributions at zero cost. Verification error is ~0.001 (machine precision for BF16).

### Training

CIS minimizes circuit structure losses subject to a classification performance constraint:

```
minimize     L_completeness + L_separation + L_sharpness + L_DF-FLOPS
subject to   L_CE <= tau_ce
```

| Loss | Objective | Mechanism |
|------|-----------|-----------|
| **Completeness** | Circuit-masked predictions match full | DLA &rarr; soft top-k mask &rarr; reclassify &rarr; CE |
| **Separation** | Per-class circuits use distinct tokens | EMA centroids &rarr; mean pairwise cosine |
| **Sharpness** | Attributions concentrate on few dims | Hoyer sparsity of attribution magnitudes |
| **DF-FLOPS** | Sparse bottleneck stays sparse | Scale-invariant L1/L2 document frequency reg |

The constraint threshold `tau_ce` is set automatically from the 25th percentile of warmup CE. A single GECO Lagrangian multiplier ([Rezende & Viola, 2018](https://arxiv.org/abs/1810.00597)) replaces manual loss weighting. Loss weights within the circuit objective are learned via uncertainty weighting ([Kendall et al., 2018](https://arxiv.org/abs/1705.07115)).

**Key invariant**: `compute_attribution_tensor()` is a single function called by both training losses and evaluation metrics. There is no separate training-time vs. evaluation-time attribution.

### Evaluation

An 8-step mechanistic evaluation pipeline runs automatically after training:

| Step | Metric | What it measures |
|------|--------|-----------------|
| 1 | DLA verification | Algebraic identity holds to machine precision |
| 2 | Circuit extraction | Top-k% of *active* vocab dims by attribution |
| 3 | Completeness | Per-class accuracy retention under circuit ablation |
| 4 | Separation | Cosine + Jaccard separation between class circuits |
| 5 | ERASER faithfulness | Comprehensiveness / sufficiency / AOPC at sparse bottleneck |
| 6 | Explainer comparison | DLA vs gradient vs IG vs attention (ERASER metrics) |
| 7 | Layerwise attribution | Per-BERT-layer contribution decomposition |
| 8 | SAE comparison | Optional sparse autoencoder baseline |

ERASER metrics ([DeYoung et al., 2020](https://arxiv.org/abs/1911.03429)) operate at the sparse bottleneck level, not input tokens. Zeroing `s_j` entries causes no distribution shift---a **Faithfulness Measurable Model** property ([Madsen et al., 2024](https://arxiv.org/abs/2310.01538)).

---

## Installation

```bash
git clone https://github.com/<repo>/interpretable-splade-classifier.git
cd interpretable-splade-classifier
pip install -e .
```

**Requirements**: Python >= 3.10, PyTorch >= 2.1, CUDA GPU.

---

## Reproducing Results

All paper experiments use full dataset splits (`train_samples: -1, test_samples: -1`).

```bash
# Main experiment (SST-2, full split, single seed)
python -m splade.scripts.run_experiment --config experiments/paper/sst2.yaml

# Ablation: Baseline (no circuit losses) vs Full CIS
python -m splade.scripts.run_ablation --config experiments/paper/sst2_ablation.yaml

# Multi-dataset benchmark (SST-2, AG News, IMDB)
python -m splade.scripts.run_multi_dataset --config experiments/paper/multi_dataset.yaml
```

Additional datasets and configurations:

```bash
# AG News (4-class)
python -m splade.scripts.run_experiment --config experiments/paper/ag_news.yaml

# IMDB (binary)
python -m splade.scripts.run_experiment --config experiments/paper/imdb.yaml

# B-cos architecture variant (exact DLA at arbitrary depth)
python -m splade.scripts.run_bcos_ablation --config experiments/paper/sst2.yaml
```

Quick verification runs (toy-sized, for development):

```bash
python -m splade.scripts.run_experiment --config experiments/verify_full.yaml
```

---

## Datasets

All loaded automatically from HuggingFace `datasets`:

| Dataset | Task | Classes | Train | Test |
|---------|------|---------|-------|------|
| SST-2 | Sentiment | 2 | 67,349 | 872 |
| AG News | Topic | 4 | 120,000 | 7,600 |
| IMDB | Sentiment | 2 | 25,000 | 25,000 |
| Yelp | Polarity | 2 | 560,000 | 38,000 |

---

## Project Structure

```
splade/
  models/
    splade.py              # SpladeModel: BERT + sparse bottleneck + ReLU MLP -> CircuitState
    bcos.py                # B-cos linear layers (exact DLA at arbitrary depth)
    layers/activation.py   # DReLU with learnable thresholds
  circuits/
    core.py                # CircuitState NamedTuple, circuit_mask()
    geco.py                # GECOController: dataset-size-invariant Lagrangian optimization
    losses.py              # Completeness, separation, sharpness + uncertainty weighting
    metrics.py             # Circuit extraction, completeness, cosine/Jaccard separation
  mechanistic/
    attribution.py         # compute_attribution_tensor(): canonical DLA for all consumers
    layerwise.py           # Per-BERT-layer contribution decomposition
    sae.py                 # Sparse autoencoder baseline
  evaluation/
    eraser.py              # ERASER comprehensiveness, sufficiency, AOPC
    baselines.py           # Gradient, IG, attention attribution methods
    compare_explainers.py  # Side-by-side ERASER comparison across explainers
    mechanistic.py         # 8-step evaluation pipeline orchestrator
  training/
    loop.py                # GECO-integrated training with EMA and early stopping
    losses.py              # DF-FLOPS regularization
    optim.py               # LR range test, gradient centralization
    constants.py           # All hyperparameters (hardwired, not configurable)
  data/loader.py           # HuggingFace dataset loading
  scripts/
    run_experiment.py      # Train + evaluate pipeline
    run_ablation.py        # Baseline vs Full CIS comparison
    run_multi_dataset.py   # Cross-dataset benchmark
    run_bcos_ablation.py   # B-cos architecture comparison
experiments/
  paper/                   # Publication configs (full splits, single seed)
  verify*.yaml             # Quick verification configs (toy-sized)
tests/                     # 143 unit tests
```

---

## Design Decisions

**No configurable hyperparameters.** All training hyperparameters (learning rate schedule, loss weights, circuit fraction, GECO dynamics) are hardwired constants or derived automatically. YAML configs specify only the experimental setup: dataset, model, and seeds.

**Single attribution function.** `compute_attribution_tensor()` is called by training losses, evaluation metrics, circuit extraction, and centroid tracking. This eliminates the common failure mode where training optimizes a proxy that diverges from the evaluation metric.

**GECO over fixed loss weights.** A single Lagrangian multiplier adapts automatically to balance classification and circuit objectives. The dual step size scales with `1/steps_per_epoch` for dataset-size-invariant behavior.

**Sparse bottleneck as FMM.** The vocabulary-sized sparse vector `s` has ~100-200 non-zero dimensions out of ~30K. Zeroing entries for faithfulness evaluation causes no distribution shift, unlike input-space token erasure.

---

## Limitations

- **Text classification only.** The architecture requires a sparse vocabulary bottleneck; not applicable to generation or retrieval.
- **BERT-family encoders.** Tested with DistilBERT and BERT-base. Other encoder families may require adaptation.
- **Vocabulary-level granularity.** Attributions identify vocabulary tokens, not input spans. Subword tokenization may split meaningful units.
- **Input-dependent W_eff.** The effective weight matrix varies per input due to the ReLU activation mask; explanations are per-sample.

---

## References

### Core Method

- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR*. [`arXiv:2107.05720`](https://arxiv.org/abs/2107.05720)
- Balestriero, R. & Baraniuk, R. (2018). A Spline Theory of Deep Networks. *ICML*. [`arXiv:1802.09210`](https://arxiv.org/abs/1802.09210)
- Rezende, D. J. & Viola, F. (2018). Taming VAEs. [`arXiv:1810.00597`](https://arxiv.org/abs/1810.00597)
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. *CVPR*. [`arXiv:1705.07115`](https://arxiv.org/abs/1705.07115)

### Evaluation

- DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL*. [`arXiv:1911.03429`](https://arxiv.org/abs/1911.03429)
- Madsen, A., et al. (2024). Are Faithfulness Measures Faithful? [`arXiv:2310.01538`](https://arxiv.org/abs/2310.01538)
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*. [`arXiv:1703.01365`](https://arxiv.org/abs/1703.01365)

### Related Work

- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.
- Bohle, M., Fritz, M., & Schiele, B. (2024). B-cos Networks: Alignment is All We Need for Interpretability. *CVPR*. [`arXiv:2205.10268`](https://arxiv.org/abs/2205.10268)
- Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS*. [`arXiv:2304.14997`](https://arxiv.org/abs/2304.14997)
- Marks, S., et al. (2024). Sparse Feature Circuits. *ICLR*. [`arXiv:2403.19647`](https://arxiv.org/abs/2403.19647)
- Lei, T., et al. (2025). Sparse Attention Post-Training. [`arXiv:2512.05865`](https://arxiv.org/abs/2512.05865)
- Lieberum, T., et al. (2025). Open Problems in Mechanistic Interpretability. [`arXiv:2501.16496`](https://arxiv.org/abs/2501.16496)
- Chen, J., et al. (2025). Rethinking Circuit Completeness. [`arXiv:2505.10039`](https://arxiv.org/abs/2505.10039)
