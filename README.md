# Faithful by Construction: Sparse Lexical Representations as Inherently Interpretable Text Classifiers

An interpretable text classifier based on [SPLADE](https://arxiv.org/abs/2107.05720) (Sparse Lexical and Expansion Model) that provides **faithful explanations by construction**. Unlike post-hoc explanation methods (LIME, SHAP, Attention, Integrated Gradients), SPLADE's sparse vocabulary dimensions *are* the model's computation—not approximations.

## Research Contribution

This library demonstrates that SPLADE explanations are more **faithful** than post-hoc methods applied to the same underlying DistilBERT backbone:

- **Comprehensiveness**: Removing SPLADE's top-k terms causes larger prediction drops
- **Sufficiency**: SPLADE's top-k terms alone predict better than post-hoc selections
- **Monotonicity**: SPLADE importance rankings are more consistent

## Features

- **Inherently Interpretable** - Explanations are the actual model computation, not post-hoc approximations
- **Sklearn-compatible API** - `fit()`, `predict()`, `transform()`, `explain()`
- **Faithfulness Metrics** - Built-in comprehensiveness, sufficiency, monotonicity, AOPC
- **Baseline Comparisons** - Attention, LIME, SHAP, Integrated Gradients on same backbone
- **Sparse & Fast** - 97%+ sparsity with CUDA/Triton-accelerated inference

## Installation

```bash
pip install -e .

# With interpretability baselines (LIME, SHAP, Captum)
pip install -e ".[interpretability]"

# Build CUDA kernels (optional, requires nvcc)
python setup.py build_ext --inplace
```

## Quick Start

```python
from splade_classifier import SPLADEClassifier

# Train
clf = SPLADEClassifier(num_labels=2)
clf.fit(train_texts, train_labels, epochs=3)

# Predict
predictions = clf.predict(test_texts)

# Explain - these ARE the model's computation
explanations = clf.explain("This movie was fantastic!")
# [('fantastic', 2.31), ('movie', 1.45), ('great', 0.89), ...]

# See semantic expansions (terms not in input)
expansions = clf.get_expansion_terms("This movie was fantastic!")
# [('excellent', 0.72), ('amazing', 0.65), ('wonderful', 0.58), ...]
```

## Interpretability Benchmark

Compare SPLADE's inherent explanations against post-hoc methods:

```bash
# Run interpretability benchmark
python -m splade_classifier.interpretability_benchmark --dataset sst2 --test-samples 200

# With all baselines (requires interpretability extras)
python -m splade_classifier.interpretability_benchmark --dataset sst2 --test-samples 200
```

### Faithfulness Metrics

| Metric | What it Measures | Better |
|--------|------------------|--------|
| **Comprehensiveness** | Prediction drop when removing top-k important tokens | Higher |
| **Sufficiency** | Prediction using only top-k important tokens | Lower |
| **Monotonicity** | Consistency of importance ordering under perturbation | Higher |
| **AOPC** | Area over perturbation curve (aggregate faithfulness) | Higher |

### Why SPLADE Explanations Are Faithful

Post-hoc methods approximate feature importance *after* the model computes its prediction. SPLADE's sparse vectors ARE the prediction mechanism:

```
SPLADE:     Text → DistilBERT → MLM → sparse vec → classifier
                                       ↑
                               THESE dimensions are the explanation
                               (no approximation needed)

Post-hoc:   Text → DistilBERT → classifier → prediction
                                              ↓
                               THEN approximate which inputs mattered
                               (LIME perturbs, SHAP samples, IG integrates)
```

## API Reference

### SPLADEClassifier

```python
clf = SPLADEClassifier(
    num_labels=2,                           # Number of classes
    model_name="distilbert-base-uncased",   # Backbone model
    batch_size=32,                          # Training batch size
    target_sparsity=0.95,                   # Target sparsity level
    max_length=128,                         # Max sequence length
)
```

### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y, epochs)` | Train on texts and labels |
| `predict(X)` | Get class predictions |
| `predict_proba(X)` | Get class probabilities |
| `transform(X)` | Get sparse SPLADE vectors `[n_samples, vocab_size]` |
| `explain(text, top_k)` | Get top-k `(token, weight)` pairs |
| `explain_tokens(text)` | Get `(token, weight, position)` aligned to input |
| `get_expansion_terms(text, top_k)` | Get semantic expansions not in input |
| `save(path)` / `load(path)` | Model persistence |

### Faithfulness Evaluation

```python
from splade_classifier import comprehensiveness, sufficiency, monotonicity

# Generate explanations
attributions = [clf.explain(text, top_k=20) for text in test_texts]

# Evaluate faithfulness
comp = comprehensiveness(clf, test_texts, attributions, k_values=[5, 10, 20])
suff = sufficiency(clf, test_texts, attributions, k_values=[5, 10, 20])
mono = monotonicity(clf, test_texts, attributions)
```

### Rationale Datasets

```python
from splade_classifier import load_hatexplain, rationale_agreement

# Load dataset with human rationale annotations
texts, labels, human_rationales, num_labels = load_hatexplain(split="test")

# Compare model explanations to human rationales
agreement = rationale_agreement(model_attributions, human_rationales, k=10)
# {'precision': 0.45, 'recall': 0.38, 'f1': 0.41, 'iou': 0.29}
```

## Architecture

SPLADE uses a masked language model to produce sparse vocabulary-sized vectors:

```
Text → DistilBERT → MLM logits [B, S, V] → log(1 + ReLU) → max-pool → sparse [B, V] → classifier
```

The sparse vectors are:
- **Sparse**: 97%+ zeros (efficient)
- **Interpretable**: Each dimension = vocabulary term weight
- **Expandable**: Semantically related terms get non-zero weights

## GPU Acceleration

Hybrid CUDA C++ (forward) + Triton (backward) backend:

| Backend | Latency | Speedup |
|---------|---------|---------|
| PyTorch baseline | 1.28 ms | 1.0x |
| Triton | 0.22 ms | 5.9x |
| **CUDA C++** | **0.17 ms** | **7.4x** |

## Citation

If you use this work, please cite:

```bibtex
@software{splade_classifier,
  title={Faithful by Construction: Sparse Lexical Representations as Inherently Interpretable Text Classifiers},
  year={2025},
  url={https://github.com/...}
}
```

## References

- Formal et al. (2021) [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- DeYoung et al. (2020) [ERASER: A Benchmark to Evaluate Rationalized NLP Models](https://arxiv.org/abs/1911.03429)
