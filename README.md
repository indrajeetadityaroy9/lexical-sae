# Interpretable SPLADE Classifier

This project investigates whether sparse vocabulary activations from SPLADE-style architectures provide inherently interpretable explanations that outperform post-hoc attribution methods on rigorous faithfulness metrics.

## Research Objectives

### Primary Hypothesis

Sparse lexical models (SPLADE) produce explanations that are more faithful than post-hoc methods because the explanation *is* the model's computation—not an approximation of it.

### Concept Vocabulary Bottleneck (CVB)

SPLADE's 30,522-dimensional sparse vector functions as a Concept Vocabulary Bottleneck: a human-readable intermediate layer where each dimension corresponds to a vocabulary term. This connects to the Concept Bottleneck Model literature (ICLR 2025) but requires zero concept annotation—the MLM head provides the concept vocabulary for free.

Key differentiators from prior work:
- **Yang (2024)**: Used SPLADE features for classification but framed as efficiency, not interpretability
- **CB-LLMs (ICLR 2025)**: Concept-level interpretability requiring LLM annotation; SPLADE is zero-annotation
- **Mackenzie & Zhuang (2023)**: Caution about opaque vocabulary signals; DF-FLOPS regularization addresses this
- **Jacovi & Goldberg (2020)**: "Inherent interpretability is a claim" that must be verified empirically

### Key Claims Under Investigation

1. **Architectural Faithfulness**: SPLADE's sparse vocabulary activations directly determine the classification output; removing high-activation terms necessarily changes predictions.

2. **Baseline Superiority**: SPLADE explanations should outperform attention weights, LIME, and Integrated Gradients on comprehensiveness, sufficiency, and normalized AOPC.

3. **Semantic Expansion**: Unlike bag-of-words, SPLADE captures semantic relationships (e.g., "fantastic" activates "excellent", "wonderful") via the MLM head, providing richer explanations.

## Contributions

| Contribution | Source | Implementation |
|--------------|--------|----------------|
| **DF-FLOPS Regularization** | arXiv:2505.15070 | `src/kernels.py` |
| **Normalized AOPC** | arXiv:2408.08137 | `src/faithfulness.py` |
| **F-Fidelity Protocol** | arXiv:2410.02970 | `src/faithfulness.py` |
| **Adversarial Sensitivity** | arXiv:2409.17774 | `src/adversarial.py` |
| **CVB Analysis** | -- | `src/interpretability_benchmark.py` |

---

## Architecture

### SPLADE Classifier Pipeline

```
Input Text -> DistilBERT -> MLM Head -> log(1 + ReLU(logits)) -> max-pool -> sparse [B, V] -> Linear -> prediction
```

### Why Sparse Representations Enable Faithful Explanations

Post-hoc methods (LIME, SHAP, IG) approximate feature importance *after* the model computes its prediction. They perturb inputs, sample neighborhoods, or integrate gradients to estimate which tokens mattered—but these are approximations with known failure modes.

SPLADE's sparse vectors ARE the prediction mechanism:
- Each dimension corresponds to a vocabulary term
- The classifier operates directly on these activations
- Removing a high-weight term necessarily changes the input to the classifier
- No approximation, no sampling, no integration required

### DF-FLOPS Regularization (arXiv:2505.15070)

Document-frequency weighted regularization penalizes high-DF terms to reduce posting-list lengths while improving OOD generalization. Parameters `alpha=0.1` and `beta=10.0` are paper-mandated and applied internally during training:

```python
clf = SPLADEClassifier(num_labels=2)
clf.fit(train_texts, train_labels)  # DF-FLOPS applied automatically
```

### GPU Kernels

Training uses a custom Triton backward kernel (`src/kernels.py`) for the SPLADE aggregation (log1p-relu max-pool), while inference uses a PyTorch forward pass. Mixed-precision training runs in bfloat16 on CUDA with `torch.compile`.

---

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10, CUDA GPU, and Triton. All dependencies (torch, transformers, triton, captum, lime, nltk, scikit-learn, etc.) are installed automatically.

---

## Quick Start

```python
from src import SPLADEClassifier

# Initialize and train
clf = SPLADEClassifier(num_labels=2)
clf.fit(train_texts, train_labels, epochs=3)

# Predict
predictions = clf.predict(test_texts)
probabilities = clf.predict_proba(test_texts)

# Explain (these ARE the model's computation)
explanation = clf.explain("This movie was fantastic!")
# [('fantastic', 2.31), ('movie', 1.45), ('excellent', 0.89), ...]

# Access raw sparse vectors
sparse_vectors = clf.transform(texts)  # [n_samples, 30522]
```

---

## Faithfulness Evaluation

### ERASER Metrics (arXiv:1911.03429)

```python
from src.faithfulness import compute_comprehensiveness, compute_sufficiency, compute_monotonicity

attributions = [clf.explain(text, top_k=20) for text in test_texts]
mask_token = clf.tokenizer.mask_token

comp = compute_comprehensiveness(clf, test_texts, attributions, k_values=[5, 10, 20], mask_token=mask_token)
suff = compute_sufficiency(clf, test_texts, attributions, k_values=[5, 10, 20], mask_token=mask_token)
mono = compute_monotonicity(clf, test_texts, attributions, steps=10, mask_token=mask_token)
```

### Normalized AOPC (arXiv:2408.08137)

Standard AOPC bounds vary across models, making cross-model comparisons unreliable. Normalized AOPC addresses this with beam-search upper bounds and random-ordering lower bounds:

```python
from src.faithfulness import compute_normalized_aopc

result = compute_normalized_aopc(clf, test_texts, attributions, k_max=20, beam_size=15, mask_token=mask_token)
# {
#     'naopc': 0.72,          # Normalized score (0-1)
#     'aopc_lower': 0.12,     # Random ordering bound
#     'aopc_upper': 0.58,     # Optimal ordering bound (beam search)
# }
```

### F-Fidelity Protocol (arXiv:2410.02970)

Addresses out-of-distribution issues in standard faithfulness evaluation by bounding the fraction of tokens that can be masked. Pass `beta < 1.0` to `compute_comprehensiveness` or `compute_sufficiency`:

```python
f_comp = compute_comprehensiveness(clf, test_texts, attributions, k_values=[5, 10, 20], mask_token=mask_token, beta=0.5)
f_suff = compute_sufficiency(clf, test_texts, attributions, k_values=[5, 10, 20], mask_token=mask_token, beta=0.5)
```

### Adversarial Sensitivity (arXiv:2409.17774)

Tests whether explanations capture model vulnerabilities via multi-attack perturbation:

```python
from src.adversarial import compute_adversarial_sensitivity, WordNetAttack, CharacterAttack

result = compute_adversarial_sensitivity(
    model=clf,
    explainer_fn=lambda text, k: clf.explain(text, top_k=k),
    texts=test_texts,
    attacks=[WordNetAttack(max_changes=3), CharacterAttack(max_changes=2)],
    max_changes=3, mcp_threshold=0.7, top_k=20, seed=42,
)
# {'adversarial_sensitivity': 0.65, 'mean_tau': 0.30, 'per_attack': {...}}
```

---

## Benchmark

Runs SPLADE against three post-hoc baselines (Attention, LIME, Integrated Gradients) on the full faithfulness metric suite:

```bash
python -m src.interpretability_benchmark --dataset sst2 --test-samples 200 --epochs 2

# With CVB analysis
python -m src.interpretability_benchmark --dataset sst2 --cvb
```

Options: `--dataset {sst2,ag_news,imdb,hatexplain}`, `--train-samples N`, `--test-samples N`, `--epochs N`, `--batch-size N`, `--cvb`

## API Reference

### SPLADEClassifier

```python
SPLADEClassifier(
    num_labels: int = 2,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
)
```

Training hyperparameters are self-adaptive: learning rate is derived from model width via muP scaling, warmup is sqrt-proportional, gradient clipping uses AGC, and regularization weight is loss-ratio balanced. Early stopping with patience=3 is used by default.

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X, y, epochs=None)` | `self` | Train on texts and labels |
| `predict(X)` | `list[int]` | Class predictions |
| `predict_proba(X)` | `list[list[float]]` | Class probabilities |
| `score(X, y)` | `float` | Classification accuracy |
| `transform(X)` | `np.ndarray` | Sparse vectors `[n, 30522]` |
| `explain(text, top_k=10)` | `list[tuple[str, float]]` | Top-k (token, weight) pairs |

### Faithfulness Metrics

| Function | Signature | Returns |
|----------|-----------|---------|
| `compute_comprehensiveness` | `(model, texts, attributions, k_values, mask_token, beta=1.0)` | `dict[int, float]` |
| `compute_sufficiency` | `(model, texts, attributions, k_values, mask_token, beta=1.0)` | `dict[int, float]` |
| `compute_monotonicity` | `(model, texts, attributions, steps, mask_token)` | `float` |
| `compute_normalized_aopc` | `(model, texts, attributions, k_max, beam_size, mask_token)` | `dict` with `naopc`, `aopc_lower`, `aopc_upper` |
| `compute_adversarial_sensitivity` | `(model, explainer_fn, texts, attacks, max_changes, mcp_threshold, top_k, seed)` | `dict` with `adversarial_sensitivity`, `mean_tau` |
| `compute_rationale_agreement` | `(attributions, human_rationales, k)` | `float` (token-level F1) |

---

## Bibliography

### Core Methods

1. Formal, T., Piwowarski, B., & Clinchant, S. (2021). **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**. SIGIR 2021. [arXiv:2107.05720](https://arxiv.org/abs/2107.05720)

2. Lassance, C., & Clinchant, S. (2022). **An Efficiency Study for SPLADE Models**. SIGIR 2022. [arXiv:2207.03834](https://arxiv.org/abs/2207.03834)

3. DeYoung, J., et al. (2020). **ERASER: A Benchmark to Evaluate Rationalized NLP Models**. ACL 2020. [arXiv:1911.03429](https://arxiv.org/abs/1911.03429)

### Faithfulness Evaluation

4. Edin, J., et al. (2024). **Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability**. [arXiv:2408.08137](https://arxiv.org/abs/2408.08137)

5. Pal, A., et al. (2024). **Faithfulness and the Notion of Adversarial Sensitivity in NLP Explanations**. [arXiv:2409.17774](https://arxiv.org/abs/2409.17774)

6. Wei, X., et al. (2024). **F-Fidelity: A Robust Framework for Faithfulness Evaluation of Explainable AI**. [arXiv:2410.02970](https://arxiv.org/abs/2410.02970)

### Regularization

7. (2025). **An Alternative to FLOPS Regularization to Effectively Productionize SPLADE-Doc**. [arXiv:2505.15070](https://arxiv.org/abs/2505.15070)

### Interpretability Theory

8. Jacovi, A., & Goldberg, Y. (2020). **Towards Faithfully Interpretable NLP Systems: A Survey of Current Trends**. ACL 2020.

9. Koh, P. W., et al. (2020). **Concept Bottleneck Models**. ICML 2020.
