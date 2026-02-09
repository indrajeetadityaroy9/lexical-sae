# Lexical-SAE

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

**Circuit-Integrated SPLADE: A Supervised Exact Sparse Autoencoder for Interpretable NLP**

> **TL;DR**: We repurpose SPLADE's sparse lexical bottleneck as an intrinsically interpretable model with **(1)** zero reconstruction error via algebraic DLA identity, **(2)** a single task-agnostic architecture for both classification and sequence labeling, **(3)** intrinsic alignment via training-time circuit constraints, and **(4)** surgical concept removal with mathematical guarantees. The feature dictionary is the tokenizer vocabulary itself --- every sparse dimension is a known word, not an opaque learned feature.

---

## Key Results

### Text Classification

| Dataset | Accuracy | DLA Error | Active Dims | Sparsity |
|---------|----------|-----------|-------------|----------|
| SST-2 | 92.3% | ~0.001 | ~130 | 99.7% |
| AG News | 93.7% | ~0.001 | ~150 | 99.7% |
| IMDB | 91.8% | ~0.001 | ~120 | 99.8% |

### Named Entity Recognition (CoNLL-2003)

| Metric | Value |
|--------|-------|
| Token Accuracy | 97.0% |
| Entity F1 (micro) | 84.0% |
| DLA Verification Error | 2.1 x 10<sup>-4</sup> |
| Mean Active Dims | 2,095 / 50,257 (4.2%) |

Per-entity breakdown:

| Entity | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| PER | 0.93 | 0.92 | 0.93 |
| LOC | 0.86 | 0.89 | 0.88 |
| ORG | 0.75 | 0.80 | 0.77 |
| MISC | 0.70 | 0.75 | 0.72 |

---

## Contributions

1. **Exact per-token attribution at zero cost.** The piecewise-linear ReLU MLP yields an algebraic identity `logit_c = sum_j s_j * W_eff[c,j] + b_eff_c` that holds to BF16 machine precision. No approximation, no sampling, no gradient computation.

2. **Unified task-agnostic architecture.** A single `LexicalSAE` model produces per-position sparse representations `[B, L, V]`. Callers choose the task head: `classify()` for document-level classification (max-pool) or `tag()` for per-position sequence labeling (NER). No task-specific parameters, no architecture divergence --- the same model instance can serve both tasks.

3. **Sparsity-aware constrained optimization.** A dynamic GECO controller with L1/L0 sparsity pressure and adaptive DReLU gate boosting breaks the optimization deadlock that traps dense-to-sparse training in local minima (26K → 2K active dims on CoNLL-2003).

4. **Surgical concept removal with mathematical guarantees.** Zeroing `s[j]` in the sparse bottleneck removes token j's contribution to all classes with certifiable effect --- no retraining, no approximation.

---

## Method

### Direct Logit Attribution (DLA)

The ReLU activation mask `D(s) = diag(1[W1*s + b1 > 0])` yields an exact per-input effective weight matrix:

```
W_eff(s) = W2 @ D(s) @ W1

logit_c = sum_j [ s_j * W_eff(s)[c,j] ] + b_eff(s)_c     (algebraic identity)
```

This decomposes every prediction into per-token contributions. Verification error is ~0.001 (BF16 machine precision).

### Architecture

The backbone is any HuggingFace `AutoModelForMaskedLM` (BERT, DistilBERT, RoBERTa, ModernBERT, etc.), used as a black box. Architecture compatibility is delegated entirely to HuggingFace --- no model-specific branching.

**Unified forward pass** produces per-position sparse representations:

```
Input -> AutoModelForMaskedLM backbone -> MLM Logits -> DReLU
      -> Sparse Sequence [B, L, V] (per-position sparse representations)
```

Callers choose the task-specific head on the same output:

```python
sparse_seq = model(input_ids, attention_mask)       # [B, L, V] universal representation

state = model.classify(sparse_seq, attention_mask)   # Classification: max-pool -> ReLU MLP -> CircuitState
# state.logits [B, C], state.sparse_vector [B, V], state.W_eff [B, C, V], state.b_eff [B, C]

token_logits = model.tag(sparse_seq)                 # NER: per-position ReLU MLP -> [B, L, C]
```

The ReLU MLP classifier is weight-tied across positions, so `classify()` and `tag()` share all parameters. The DLA identity `logit_c = sum_j s_j * W_eff[c,j] + b_eff_c` holds at both the document level (after max-pooling) and the position level.

The sparse bottleneck is a **Faithfulness Measurable Model** ([Madsen et al., 2024](https://arxiv.org/abs/2310.01538)): zeroing `s_j` entries causes no distribution shift, unlike input-space token erasure.

### Training Objective

Three circuit losses optimized via GECO constrained optimization ([Rezende & Viola, 2018](https://arxiv.org/abs/1810.00597)):

```
minimize     L_completeness + L_separation + L_sparsity
subject to   L_CE <= tau_ce
```

| Loss | Objective | Mechanism |
|------|-----------|-----------|
| **Completeness** | Circuit-masked predictions match full | DLA -> soft top-k mask -> reclassify -> CE |
| **Separation** | Per-class circuits use distinct tokens | EMA centroids -> mean pairwise cosine similarity |
| **Sparsity** | Activations concentrate on few dims | L1 (FLOPS) + L0 sigmoid proxy on sparse vector |

The sparsity loss uses **L1 regularization** with constant-gradient pressure (gradient = +/-1 regardless of magnitude), replacing the Hoyer metric which suffers from vanishing gradients in the dense regime. An L0 sigmoid proxy drives small activations to exactly zero.

**GECO dynamics.** The constraint threshold `tau_ce` is set automatically from the 25th percentile of warmup CE. A sparsity-aware extension dynamically relaxes tau when the model is dense (active dims >> target), creating a "safe harbor" for the optimizer to sacrifice accuracy temporarily while finding sparse solutions. As sparsity improves, tau tightens back to the baseline.

**DReLU gate boosting.** The learnable sparsity thresholds (`activation.theta`) receive a 5x learning rate multiplier, providing sufficient "kinetic energy" to overcome CE gradient inertia during the dense-to-sparse transition.

**Key invariant**: `compute_attribution_tensor()` is a single function called by both training losses and evaluation metrics. There is no divergence between the training proxy and the evaluation metric.

### Surgical Intervention

Two mechanisms for verifiable concept removal at the sparse bottleneck:

| Mechanism | Scope | Reversible | Guarantee |
|-----------|-------|------------|-----------|
| **Global suppression** (`suppress_token_globally`) | Zeros output embedding weights + DReLU threshold | No | `s[token_id] = 0` for all inputs |
| **Inference-time masking** (`SuppressedModel`) | Masks sparse sequence before task head | Yes | DLA identity preserved for remaining dims |

Since `logit[c] = sum_j s[j] * W_eff[c,j] + b_eff[c]`, zeroing `s[j]` removes token j's contribution to ALL classes with mathematical certainty.

---

## Installation

```bash
git clone https://github.com/<repo>/interpretable-splade-classifier.git
cd interpretable-splade-classifier
pip install -e .
```

**Requirements**: Python >= 3.10, PyTorch >= 2.1, CUDA GPU.

---

## Quick Start

```bash
# Smoke test (~2 min, toy-sized data)
python -m splade.scripts.run_experiment --config experiments/verify_full.yaml
```

---

## Reproducing Results

### Text Classification

```bash
# SST-2 (full split)
python -m splade.scripts.run_experiment --config experiments/paper/sst2.yaml

# Ablation: baseline (no circuit losses) vs full Lexical-SAE
python -m splade.scripts.run_ablation --config experiments/paper/sst2_ablation.yaml

# Multi-dataset benchmark (SST-2, AG News, IMDB)
python -m splade.scripts.run_multi_dataset --config experiments/paper/multi_dataset.yaml
```

### Named Entity Recognition

```bash
# CoNLL-2003 NER (full split, ~20 min on H100)
python -m splade.scripts.run_ner --config experiments/ner/conll2003.yaml

# Quick NER verification (200 train samples)
python -m splade.scripts.run_ner --config experiments/ner/conll2003_verify.yaml
```

### Faithfulness and Intervention Experiments

```bash
# Experiment A: Faithfulness stress test
# Compares DLA removability vs gradient/IG/attention baselines
python -m splade.scripts.run_faithfulness --config experiments/civilcomments.yaml

# Experiment B: Surgical bias removal ("lobotomy")
# Suppresses identity-correlated tokens, measures FPR gap reduction
python -m splade.scripts.run_surgery --config experiments/surgery.yaml

# Experiment C: Long-context needle in haystack
# Tests sparse bottleneck signal preservation at increasing document lengths
python -m splade.scripts.run_long_context --config experiments/long_context.yaml
```

### Tests

```bash
pytest tests/
pytest tests/ -k "not slow"              # skip network-dependent tests
pytest tests/test_sequence_model.py      # NER-specific tests
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
| CivilComments | Toxicity | 2 | 1,804,874 | 97,320 |
| CoNLL-2003 | NER | 9 (BIO) | 14,041 | 3,453 |

CivilComments includes 24 identity group annotations for bias analysis. CoNLL-2003 uses BIO tagging with 4 entity types (PER, LOC, ORG, MISC).

---

## Configuration

### Classification

Three training knobs via YAML `training:` section:

| Knob | Default | Effect |
|------|---------|--------|
| `target_accuracy` | `null` (auto) | GECO tau override. Null = auto from warmup 25th percentile |
| `sparsity_target` | `0.1` | Circuit fraction: top k% of active sparse dims |
| `warmup_fraction` | `0.2` | Fraction of training for CE-only warmup |

### NER

Additional knobs for sequence labeling (`NERTrainingConfig`):

| Knob | Default | Effect |
|------|---------|--------|
| `batch_size` | `null` (auto) | Explicit batch size override. Null = GPU-memory-aware auto-inference |
| `gradient_accumulation_steps` | `1` | Micro-batch accumulation for memory-constrained GPUs |

All other hyperparameters (LR schedule, EMA decay, GECO dynamics, sparsity gain, gate LR multiplier) are derived automatically.

---

## Project Structure

```
splade/
  models/
    lexical_sae.py           # LexicalSAE: task-agnostic encoder with classify()/tag() heads
    layers/activation.py     # DReLU: learnable thresholds for sparsity gating
  circuits/
    core.py                  # CircuitState, circuit_mask()
    sequence_core.py         # SequenceCircuitState for per-position models
    geco.py                  # GECOController: Lagrangian optimization + sparsity-aware tau
    losses.py                # Circuit losses for classification (completeness, separation, sharpness)
    sequence_losses.py       # Token-level circuit losses for NER (L1 sparsity, centroid tracking)
    metrics.py               # Circuit extraction and evaluation metrics
  mechanistic/
    attribution.py           # compute_attribution_tensor(): canonical DLA for all consumers
    layerwise.py             # Per-layer contribution decomposition
    sae.py                   # Post-hoc sparse autoencoder baseline
  evaluation/
    mechanistic.py           # Tiered evaluation for classification
    sequence_mechanistic.py  # Tiered evaluation for NER (seqeval integration)
    faithfulness.py          # Removability metric: flip rate under DLA-guided ablation
    baselines.py             # Gradient, IG, attention attribution baselines
    eraser.py                # ERASER comprehensiveness, sufficiency, AOPC
    compare_explainers.py    # Side-by-side ERASER comparison
  intervene.py               # Surgical intervention: suppress_token_globally, evaluate_bias
  training/
    loop.py                  # GECO training loop for classification
    sequence_loop.py         # GECO training loop for NER (L1 loss, gate boost, sparsity tracking)
    optim.py                 # LR range test, param groups, gradient centralization
    constants.py             # Internal hyperparameters
  data/
    loader.py                # HuggingFace dataset loading + CivilComments identity annotations
    ner_loader.py            # CoNLL-2003 loading with subword-to-word label alignment
  inference.py               # Batched inference, prediction, explanation API
  pipelines.py               # Shared setup_and_train() pipeline
  config/
    schema.py                # Classification config schema
    ner_schema.py            # NER config schema
    load.py                  # YAML -> Config
  scripts/
    run_experiment.py        # Classification: train + mechanistic evaluation
    run_ablation.py          # Baseline vs full Lexical-SAE comparison
    run_multi_dataset.py     # Cross-dataset benchmark
    run_faithfulness.py      # Experiment A: removability comparison
    run_surgery.py           # Experiment B: surgical bias removal
    run_long_context.py      # Experiment C: needle in haystack
    run_ner.py               # NER: train + sequence mechanistic evaluation
experiments/
  paper/                     # Publication configs (full splits)
  ner/                       # NER experiment configs (CoNLL-2003)
  civilcomments.yaml         # Faithfulness experiment config
  surgery.yaml               # Bias removal experiment config
  long_context.yaml          # Long-context experiment config
  verify*.yaml               # Quick verification configs (toy-sized)
tests/                       # Unit tests (classification + NER)
archive/                     # B-cos variant (alternative architecture)
```

---

## Design Decisions

**Feature dictionary = tokenizer vocabulary.** Unlike post-hoc SAEs which learn an opaque feature dictionary, Lexical-SAE operates directly in vocabulary space. Each sparse dimension corresponds to a known token. If the model relies on "Muslim" to predict toxicity, you identify it by name and remove it with a single operation.

**Zero reconstruction error.** The DLA identity holds exactly (verified to BF16 machine precision). Post-hoc SAEs incur reconstruction error that compounds through downstream layers.

**Task-agnostic sparse representation.** The model produces `[B, L, V]` per-position sparse vectors as a universal intermediate. Classification and sequence labeling are thin post-processing heads (`classify()` and `tag()`) over the same representation, not separate architectures. This eliminates task-type parameters and enables the same model instance to serve multiple tasks.

**Intrinsic alignment.** Circuit structure is optimized during training, not extracted post-hoc. The same `compute_attribution_tensor()` function drives both training losses and evaluation metrics.

**L1 over Hoyer for sparsity.** Hoyer sparsity (L1/L2 ratio) has vanishing gradients in the dense regime, causing optimization deadlocks. L1 provides constant-gradient pressure (+/-1 regardless of activation magnitude), consistent with FLOPS regularization in Mistral-SPLADE and BGE-M3.

**GECO over fixed loss weights.** A single Lagrangian multiplier adapts automatically. The sparsity-aware extension prevents deadlocks by relaxing the CE constraint when the model is far from the sparsity target.

**Sparse bottleneck as FMM.** ~100-2000 non-zero dimensions out of ~50K vocabulary. Zeroing entries for faithfulness evaluation or surgical intervention causes no distribution shift.

---

## Limitations

- **Vocabulary-level granularity.** Attributions identify vocabulary tokens, not input spans. A clean vocab filter masks subword continuations and special tokens for human-readable output.
- **Input-dependent W_eff.** The effective weight matrix varies per input due to the ReLU activation mask; explanations are per-sample, not global.
- **Encoder family scope.** Tested on ModernBERT-base; compatible with any HuggingFace `AutoModelForMaskedLM` backbone but other encoders are not yet benchmarked.
- **NER sparsity gap.** Sequence labeling achieves higher active dims (~2K) than classification (~100-200) due to the per-position sparse representation.

---

## References

### Core Method

- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR*. [`arXiv:2107.05720`](https://arxiv.org/abs/2107.05720)
- Balestriero, R. & Baraniuk, R. (2018). A Spline Theory of Deep Networks. *ICML*. [`arXiv:1802.09210`](https://arxiv.org/abs/1802.09210)
- Rezende, D. J. & Viola, F. (2018). Taming VAEs. [`arXiv:1810.00597`](https://arxiv.org/abs/1810.00597)
- Lassance, C., et al. (2024). SPLADE-v3: New Baselines for SPLADE. [`arXiv:2403.06789`](https://arxiv.org/abs/2403.06789)

### Sparsity and Optimization

- Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders. [`arXiv:2406.04093`](https://arxiv.org/abs/2406.04093)
- Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. [`arXiv:2407.14435`](https://arxiv.org/abs/2407.14435)
- Lei, T., et al. (2025). Sparse Attention Post-Training. [`arXiv:2512.05865`](https://arxiv.org/abs/2512.05865)

### Evaluation and Interpretability

- DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL*. [`arXiv:1911.03429`](https://arxiv.org/abs/1911.03429)
- Madsen, A., et al. (2024). Are Faithfulness Measures Faithful? [`arXiv:2310.01538`](https://arxiv.org/abs/2310.01538)
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*. [`arXiv:1703.01365`](https://arxiv.org/abs/1703.01365)

### Mechanistic Interpretability

- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.
- Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS*. [`arXiv:2304.14997`](https://arxiv.org/abs/2304.14997)
- Marks, S., et al. (2024). Sparse Feature Circuits. *ICLR*. [`arXiv:2403.19647`](https://arxiv.org/abs/2403.19647)
- Lieberum, T., et al. (2025). Open Problems in Mechanistic Interpretability. [`arXiv:2501.16496`](https://arxiv.org/abs/2501.16496)
- Chen, J., et al. (2025). Rethinking Circuit Completeness. [`arXiv:2505.10039`](https://arxiv.org/abs/2505.10039)

---

## Citation

```bibtex
@misc{lexicalsae2025,
  title     = {Lexical-SAE: A Supervised Exact Sparse Autoencoder for Interpretable NLP},
  year      = {2025},
  url       = {https://github.com/<repo>/interpretable-splade-classifier}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
