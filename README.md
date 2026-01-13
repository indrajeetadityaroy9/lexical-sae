# SPLADE Classifier

A fast, interpretable sparse text classifier with an sklearn-compatible API.

Built on [SPLADE](https://arxiv.org/abs/2107.05720) (Sparse Lexical and Expansion Model), this library provides neural sparse representations that outperform traditional TF-IDF while remaining fully interpretable.

## Features

- **Sklearn-compatible API** - `fit()`, `predict()`, `transform()` just like scikit-learn
- **Interpretable** - See exactly which terms drive each prediction
- **Sparse & Fast** - 98%+ sparsity with Triton-accelerated inference (10x speedup)
- **State-of-the-art** - Outperforms TF-IDF baselines by 3-4 percentage points

## Quick Start

```python
from src.models import SPLADEClassifier

# Train
clf = SPLADEClassifier()
clf.fit(train_texts, train_labels, epochs=5)

# Predict
predictions = clf.predict(test_texts)

# Explain predictions
clf.print_explanation("This movie was fantastic!")
```

Output:
```
SPLADE PREDICTION EXPLANATION
============================================================
Text: This movie was fantastic!

Prediction: Positive
Confidence: 97.23%

Top Contributing Terms:
----------------------------------------
   1. fantastic       2.31 ██████████████████████████
   2. movie           1.89 █████████████████████
   3. great           1.76 ███████████████████
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/splade-classifier.git
cd splade-classifier

# Install dependencies
pip install torch transformers scikit-learn pandas tqdm scipy

# Optional: Triton for GPU acceleration
pip install triton
```

## Benchmark Results

| Model | Accuracy | F1 Score | Sparsity |
|-------|----------|----------|----------|
| **SPLADE (Ours)** | **82.0%** | **0.829** | 98.3% |
| TF-IDF + LogReg | 78.5% | 0.805 | 95.2% |

Run the benchmark:
```bash
python -m src.benchmark --data_dir Data --epochs 5
```

## API Reference

### SPLADEClassifier

```python
clf = SPLADEClassifier(
    model_name="distilbert-base-uncased",  # Backbone model
    max_length=128,                         # Max sequence length
    batch_size=32,                          # Training batch size
    learning_rate=2e-5,                     # Learning rate
    flops_lambda=1e-4,                      # Sparsity regularization
    random_state=42,                        # Reproducibility seed
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y, epochs=5)` | Train on texts and labels |
| `predict(X)` | Get class predictions |
| `predict_proba(X)` | Get class probabilities |
| `transform(X)` | Get sparse SPLADE vectors |
| `score(X, y)` | Compute accuracy |
| `save(path)` / `load(path)` | Model persistence |

#### Interpretability

| Method | Description |
|--------|-------------|
| `explain(text)` | Get top weighted terms |
| `explain_prediction(text)` | Full prediction breakdown |
| `compare_texts(text1, text2)` | Compare representations |
| `print_explanation(text)` | Pretty-print explanation |

## How It Works

SPLADE uses a masked language model (DistilBERT) to produce sparse vocabulary-sized vectors:

1. **Encode**: Text → DistilBERT → Token logits `[batch, seq_len, vocab_size]`
2. **Activate**: `log(1 + ReLU(logits))` for log-saturation
3. **Pool**: Max-pool over sequence → Sparse document vector
4. **Classify**: Linear layer on sparse vector

The resulting vectors are:
- **Sparse**: ~98% zeros (efficient storage/retrieval)
- **Interpretable**: Each dimension = vocabulary term weight
- **Expandable**: Semantically related terms get non-zero weights

## Project Structure

```
src/
├── models/
│   ├── sklearn_wrapper.py    # SPLADEClassifier (main API)
│   ├── splade_distilbert.py  # DistilBERT backbone
│   └── splade_mistral.py     # Mistral backbone (experimental)
├── interpretability/
│   ├── sparse_autoencoder.py # SAE for deep analysis
│   ├── feature_analysis.py   # Feature statistics
│   └── visualization.py      # Plotting utilities
├── ops/
│   ├── splade_kernels.py     # Triton-accelerated ops
│   └── sae_kernels.py        # SAE Triton kernels
├── train.py                  # Training script
├── benchmark.py              # Comparison vs baselines
└── analyze_sae.py            # SAE interpretability
```

## Advanced: Sparse Autoencoder Analysis

For deeper interpretability, train a Sparse Autoencoder on SPLADE vectors:

```bash
# Extract vectors
python -m src.extract_vectors --model_path models/model.pth

# Train SAE
python -m src.train_sae --vectors_path outputs/vectors.pt --epochs 10

# Analyze features
python -m src.analyze_sae --sae_path outputs/sae/sae_best.pt
```

This decomposes polysemantic SPLADE dimensions into monosemantic features.

## Performance Optimization

Triton kernels provide up to **10x speedup** on GPU inference:

| Operation | PyTorch | Triton | Speedup |
|-----------|---------|--------|---------|
| SPLADE Aggregation | 1.29ms | 0.12ms | **10.5x** |
| Log Saturation | 0.66ms | 0.33ms | 2.0x |

Triton is used automatically during inference when available.

## Citation

```bibtex
@article{formal2021splade,
  title={SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking},
  author={Formal, Thibault and Piwowarski, Benjamin and Clinchant, St{\'e}phane},
  journal={SIGIR},
  year={2021}
}
```

