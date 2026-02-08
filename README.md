# Interpretable SPLADE Classifier

**Sparse Lexical Representations for Faithful Text Classification**

---

## Overview

Deep learning classifiers achieve strong performance but lack inherent interpretability. Post-hoc explainers (LIME, SHAP, Integrated Gradients) approximate model behavior but offer no guarantee that their explanations reflect the true decision process. This creates a fundamental tension between accuracy and trustworthiness.

This repository adapts the **SPLADE v2** architecture &mdash; originally designed for sparse information retrieval &mdash; into an **interpretable-by-design text classifier**. The model produces sparse vocabulary-level activations where each nonzero dimension corresponds to a recognizable term, and the linear classifier head operates directly on these activations. Explanations are therefore not approximations: they are the model's actual internal representation, decomposable into per-token contributions via the classifier weight matrix.

To rigorously evaluate explanation quality, we implement faithfulness metrics from **six recent papers**, including the F-Fidelity protocol that addresses out-of-distribution artifacts in standard masking-based evaluation.

### Key Contributions

- **Inherent interpretability via sparse lexical bottleneck.** Classification decisions pass through a vocabulary-sized sparse vector, making token contributions directly readable without post-hoc approximation.
- **DF-FLOPS regularization.** Document-frequency-weighted sparsity penalty preserves rare, discriminative terms while suppressing common, uninformative ones &mdash; improving both sparsity and explanation quality over uniform FLOPS.
- **Shifted-ReLU (DReLU) activation.** Learnable per-dimension thresholds $f(x) = \max(0, x - \theta)$ promote exact zeros, producing sparser representations than standard ReLU.
- **Comprehensive faithfulness benchmark.** 15+ metrics spanning ERASER baselines, NAOPC normalization, soft probabilistic perturbations, F-Fidelity fine-tuning, causal counterfactuals, and adversarial stability &mdash; all rigorously aligned to their source papers.
- **H100-optimized implementation.** Batched evaluation, fused operators, Tensor Core alignment, and `torch.compile` for research-scale experiments on modern hardware.

---

## Architecture

```
Input Text
    |
    v
[BERT Encoder]  (DistilBERT / BERT-base / BERT-large, SDPA attention)
    |
    v
[MLM Head]      (Dense -> GELU -> LayerNorm -> Linear, initialized from pretrained MLM weights)
    |
    v
[DReLU]         (max(0, x - theta), learnable per-vocabulary-dim threshold)
    |
    v
[log(1 + .)]    (log saturation)
    |
    v
[Max-Pool]      (over sequence length -> sparse vector s in R^|V|)
    |
    v
[Linear]        (sparse vector -> class logits)
```

The sparse document vector is computed as:

$$s_j = \max_{i \in [1, L]} \log\bigl(1 + \max(0,\; w_{ij} - \theta_j)\bigr)$$

where $w_{ij}$ is the MLM logit for vocabulary term $j$ at position $i$, and $\theta_j$ is the learnable DReLU threshold. The classifier computes $\hat{y} = \text{softmax}(W \cdot s + b)$, so each token's contribution to a class $c$ is directly $W_{c,j} \cdot s_j$.

### Regularization

The training objective is $\mathcal{L} = \mathcal{L}_{CE} + \lambda(t) \cdot \mathcal{L}_{sparse}$ where $\lambda(t)$ follows a SAT (Sparsity-Accelerated Training) schedule ramping from 0 to $\lambda_{final}$.

| Mode | Formula | Behavior |
|------|---------|----------|
| **FLOPS** | $\sum_j \bar{a}_j^2$ | Uniform L2 penalty on mean activations |
| **DF-FLOPS** | $\sum_j (d_j \cdot \bar{a}_j)^2$ | DF-weighted: downweights high-frequency terms, preserves rare discriminative features |

DF weights use a sigmoid-shaped function of document frequency: $d_j = 1 / (1 + (\text{df}_j^{\log 2 / \log \alpha} - 1)^\beta)$ with defaults $\alpha = 0.1$, $\beta = 5.0$.

### Training

- **Optimizer:** Fused AdamW with NFNet Adaptive Gradient Clipping (AGC) &mdash; exact reproduction of [arXiv:2102.06171](https://arxiv.org/abs/2102.06171) with unit-wise clipping for matrix parameters and branchless scalar clipping for biases
- **LR schedule:** Linear warmup + cosine decay, base LR auto-scaled by hidden dimension
- **Lambda schedule:** Linear or quadratic ramp (SPLADE v2 uses quadratic)
- **Precision:** BF16 autocast for forward pass and loss computation

---

## Faithfulness Evaluation

The benchmark evaluates explanation quality across six complementary protocols. All metrics are computed with cached original predictions to eliminate redundant forward passes.

### Metrics

| Metric | Source | What it measures |
|--------|--------|------------------|
| **Comprehensiveness / Sufficiency** | ERASER ([1911.03429](https://arxiv.org/abs/1911.03429)) | Confidence change when top-k tokens are removed / retained |
| **NAOPC** | [2408.08137](https://arxiv.org/abs/2408.08137) | AOPC normalized per-example via beam-search upper/lower bounds |
| **Soft Comp. / Suff.** | [2305.10496](https://arxiv.org/abs/2305.10496) | Embedding-level Bernoulli dropout weighted by attribution, with $\max(0, \cdot)$ normalization |
| **F-Comprehensiveness / F-Sufficiency** | [2410.02970](https://arxiv.org/abs/2410.02970) | Faithfulness on a model fine-tuned on masked inputs (in-distribution evaluation) |
| **Monotonicity** | ERASER | Fraction of steps where confidence decreases monotonically under token removal |
| **Filler Comprehensiveness** | [2308.14272](https://arxiv.org/abs/2308.14272) | OOD-robust variant replacing masked tokens with corpus-sampled unigrams |
| **Causal Faithfulness** | &mdash; | Spearman correlation between attribution scores and MLM-counterfactual confidence shifts |
| **Adversarial Sensitivity** | &mdash; | Explanation stability (Kendall $\hat{\tau}$) under synonym, TextFooler, and character-level attacks |

### Explainer Baselines

The benchmark compares SPLADE's inherent explanations against:

- **Random** &mdash; uniform random attribution (sanity check)
- **LIME** &mdash; local surrogate model with 500 perturbation samples
- **Integrated Gradients** &mdash; gradient-based attribution via Captum (50 interpolation steps, PAD baseline)

### F-Fidelity Protocol

Following Algorithm 1 of [arXiv:2410.02970](https://arxiv.org/abs/2410.02970):

1. Fine-tune a copy of the trained model on randomly masked inputs (Bernoulli masking with $\beta = 0.1$, 30 epochs)
2. Evaluate faithfulness metrics on this fine-tuned model using $N = 50$ stochastic mask samples per example
3. The fine-tuned model is in-distribution with respect to masked inputs, eliminating OOD artifacts that inflate standard comprehensiveness scores

---

## Repository Structure

```
splade/
  models/
    splade.py              # SpladeModel: encoder + MLM head + sparse bottleneck + classifier
    classifier.py          # SPLADEClassifier: sklearn-compatible fit/predict/explain API
    layers/
      activation.py        # DReLU: shifted-ReLU with learnable threshold
  training/
    loop.py                # Training loop with lambda schedule, AGC, mixed precision
    losses.py              # DocumentFrequencyTracker, DFFlopsRegFunction
    optim.py               # NFNet AGC (adaptive gradient clipping)
    finetune.py            # F-Fidelity fine-tuning on masked inputs
    scheduler/
      lambda_sched.py      # SAT lambda schedule (linear / quadratic ramp)
  evaluation/
    faithfulness.py        # All faithfulness metrics (~850 lines)
    benchmark.py           # Full benchmark orchestration with multi-seed aggregation
    adversarial.py         # WordNet, TextFooler, character-level attack sensitivity
    causal.py              # MLM-driven counterfactual evaluation
    explainers.py          # LIME and Integrated Gradients baselines
    token_alignment.py     # Subword-to-word attribution mapping
  data/
    loader.py              # HuggingFace dataset loading (SST-2, AG News, IMDB, Yelp)
  config/
    schema.py              # Dataclass config schema
    load.py                # YAML config loading with validation
  inference.py             # Inference utilities, predict_proba, explain
  scripts/
    train.py               # Training entry point
    eval.py                # Benchmark entry point

experiments/
  main/
    benchmark_flops.yaml        # SST-2 + FLOPS regularization
    benchmark_df_flops.yaml     # SST-2 + DF-FLOPS regularization
  datasets/
    ag_news.yaml                # AG News (4-class topic classification)
    imdb.yaml                   # IMDB (binary sentiment, max_length=256)
    yelp.yaml                   # Yelp Polarity (binary sentiment, max_length=256)
  scaling/
    bert_base.yaml              # BERT-base-uncased backbone
    bert_large.yaml             # BERT-large-uncased backbone
  ablations/
    no_compilation.yaml         # torch.compile disabled (measures compilation speedup)
  sensitivity/
    seed_sweep.yaml             # Multi-seed run (seeds: [42, 1337, 7])
```

---

## Installation

Requires Python 3.10+ and PyTorch 2.1+.

```bash
pip install -e .
```

NLTK WordNet data is required for adversarial evaluation:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Usage

### Training

```bash
python -m splade.scripts.train --config experiments/main/benchmark_df_flops.yaml
```

Trains a SPLADE classifier on the specified dataset, saves the model checkpoint and training metrics to the configured `output_dir`.

### Benchmark Evaluation

```bash
python -m splade.scripts.eval --config experiments/main/benchmark_df_flops.yaml
```

Runs the full interpretability benchmark: trains the model, fine-tunes for F-Fidelity, computes all faithfulness metrics across all configured explainers, and reports aggregated results.

### Programmatic API

```python
from splade.models.classifier import SPLADEClassifier

clf = SPLADEClassifier(model_name="distilbert-base-uncased")
clf.fit(X_train, y_train, validation_data=(X_val, y_val))

predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Inherent explanations: top-k sparse vocabulary activations
explanation = clf.explain("This movie was absolutely brilliant!", top_k=10)
# -> [("brilliant", 3.41), ("movie", 1.87), ("absolutely", 1.23), ...]
```

### Experiment Configurations

All experiments are configured via YAML. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.regularization` | `df_flops` | Sparsity penalty: `flops` or `df_flops` |
| `training.target_lambda_ratio` | `1e-3` | Final sparsity weight (SPLADE v2 range: $[10^{-4}, 3 \times 10^{-3}]$) |
| `training.lambda_schedule_type` | `linear` | Lambda ramp: `linear` or `quadratic` |
| `training.clip_factor` | `0.01` | AGC clipping factor |
| `evaluation.explainers` | `[splade, random]` | Explainers to benchmark |
| `evaluation.seeds` | `[42]` | Seeds for multi-run aggregation |

---

## Datasets

| Dataset | Classes | Task | Max Length |
|---------|---------|------|------------|
| SST-2 | 2 | Sentiment analysis | 128 |
| AG News | 4 | Topic classification | 128 |
| IMDB | 2 | Sentiment analysis | 256 |
| Yelp Polarity | 2 | Sentiment analysis | 256 |

---

## Limitations

- **Text classification only.** The architecture is not designed for retrieval, generation, or token-level tasks.
- **English, BERT-family models.** Tested with DistilBERT, BERT-base, and BERT-large. Other encoder architectures may require MLM head path adjustments.
- **Sparse representations trade accuracy for interpretability.** The vocabulary bottleneck constrains model capacity relative to dense classifiers.
- **Evaluation is compute-intensive.** The full benchmark (15+ metrics, 4 explainers, F-Fidelity fine-tuning) requires significant GPU time per dataset.

---

## References

| Paper | How it informs this work |
|-------|--------------------------|
| **SPLADE v2** &mdash; Formal et al. ([2109.10086](https://arxiv.org/abs/2109.10086)) | Core architecture: MLM head, log1p saturation, max-pooling aggregation |
| **SPLADE v3** &mdash; Lassance et al. ([2403.06789](https://arxiv.org/abs/2403.06789)) | Updated training practices and regularization insights |
| **F-Fidelity** ([2410.02970](https://arxiv.org/abs/2410.02970)) | Fine-tuning protocol for OOD-robust faithfulness evaluation |
| **Soft Evaluation Metrics** ([2305.10496](https://arxiv.org/abs/2305.10496)) | Embedding-level Bernoulli dropout comprehensiveness and sufficiency |
| **OOD Artifacts in Faithfulness** ([2308.14272](https://arxiv.org/abs/2308.14272)) | Motivation for filler-based and distribution-aware metrics |
| **ERASER** &mdash; DeYoung et al. ([1911.03429](https://arxiv.org/abs/1911.03429)) | Baseline comprehensiveness, sufficiency, and AOPC definitions |
| **NFNet (AGC)** &mdash; Brock et al. ([2102.06171](https://arxiv.org/abs/2102.06171)) | Adaptive Gradient Clipping for stable training without batch normalization |
| **NAOPC** ([2408.08137](https://arxiv.org/abs/2408.08137)) | Per-example normalized AOPC with beam-search bounds |
| **Turbo Sparse** ([2406.05955](https://arxiv.org/abs/2406.05955)) | Inspiration for shifted-ReLU sparsity (distinct formulation) |
