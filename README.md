# Interpretable SPLADE Classifier

Sparse lexical bottlenecks for inherently interpretable text classification, with a faithfulness-oriented evaluation pipeline.

## Overview
Modern text classifiers are accurate but often opaque. This repository implements a SPLADE-style classifier where prediction is computed from a sparse vocabulary activation vector, so feature-level contributions are directly tied to model internals rather than post-hoc approximation.

The project objective is to make interpretability a first-class modeling constraint while retaining competitive supervised classification behavior and rigorous faithfulness analysis.

## Project Objectives
- Build an inherently interpretable text classifier based on SPLADE-style sparse lexical representations.
- Train with document-frequency-aware sparsity regularization to preserve discriminative rare terms.
- Evaluate faithfulness with metrics grounded in recent interpretability literature, including OOD-aware protocols.
- Provide deterministic, config-driven training/evaluation entry points for reproducible experiments.

## Main Contributions
- **Interpretable-by-design classifier**: class logits are computed from sparse vocabulary activations, enabling direct token-level contribution analysis.
- **DF-FLOPS regularization**: sparsity is weighted by empirical document frequency to reduce over-penalization of rare informative terms.
- **Shifted-ReLU lexical sparsifier (DReLU)**: learnable thresholds induce exact zeros before log-saturation and max pooling.
- **Faithfulness benchmark implementation**: includes ERASER-style masking metrics, NAOPC normalization, soft perturbation metrics, F-Fidelity, adversarial stability, and causal counterfactual correlation.
- **Research-oriented execution path**: YAML-configured experiments via module entry points (`python -m ...`), deterministic training/evaluation flow.

## Method
### Model Architecture
Input text is encoded with a BERT-family encoder, projected through MLM-style vocabulary logits, sparsified with DReLU, log-saturated, max-pooled over sequence length, and classified with a linear head.

\[
s_j = \max_i \log\left(1 + \max(0, w_{ij} - \theta_j)\right)
\]

where \(w_{ij}\) is the vocabulary logit for token position \(i\), and \(\theta_j\) is the learnable threshold for vocabulary dimension \(j\).

### Training Objective
\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda(t)\,\mathcal{L}_{DF\text{-}FLOPS}
\]

with SAT-style lambda scheduling and adaptive gradient clipping.

- Training/evaluation entry points are:
  - `python -m splade.scripts.train --config ...`
  - `python -m splade.scripts.eval --config ...`
- The benchmark script currently runs the canonical **SPLADE explainer path** in the evaluation loop.
- The repository includes baseline explainer modules (LIME / IG / GradientSHAP / Attention / Saliency / DeepLIFT) as reusable components in `splade/evaluation/explainers.py`.

For adversarial evaluation, install WordNet resources:

```python
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Configuration
Experiments are YAML-driven. Core schema fields:
- `experiment_name`
- `output_dir`
- `data`: dataset name + sample sizes
- `model`: encoder backbone
- `evaluation`: seeds and explainer list configuration

## Faithfulness Metrics Implemented
- Comprehensiveness / Sufficiency (ERASER)
- Monotonicity (ERASER-style)
- AOPC + NAOPC (beam-normalized)
- Filler Comprehensiveness (OOD-aware masking variant)
- F-Fidelity comprehensiveness/sufficiency (masked-input fine-tuned model)
- Soft comprehensiveness/sufficiency (embedding-space Bernoulli perturbation)
- Adversarial sensitivity (ranking stability under perturbations)
- Causal faithfulness (MLM counterfactual correlation)

## Limitations
- **Text classification only.** The architecture is not designed for retrieval, generation, or token-level tasks.
- **English, BERT-family models.** Tested with DistilBERT, BERT-base, and BERT-large. Other encoder architectures may require MLM head path adjustments.
- **Sparse representations trade accuracy for interpretability.** The vocabulary bottleneck constrains model capacity relative to dense classifiers.
- **Evaluation is compute-intensive.** The full benchmark (15+ metrics, 4 explainers, F-Fidelity fine-tuning) requires significant GPU time per dataset.

## References
- SPLADE v2: Formal et al., arXiv:2109.10086
- SPLADE v3: Lassance et al., arXiv:2403.06789
- ERASER: DeYoung et al., arXiv:1911.03429
- Soft evaluation metrics: arXiv:2305.10496
- OOD artifacts in faithfulness: arXiv:2308.14272
- F-Fidelity: arXiv:2410.02970
- NAOPC: arXiv:2408.08137
- NFNet / AGC: Brock et al., arXiv:2102.06171
