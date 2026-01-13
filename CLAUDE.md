# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPLADE Classifier - A sparse neural text classifier with sklearn-compatible API, built on SPLADE (Sparse Lexical and Expansion Model). Uses DistilBERT to produce interpretable sparse vocabulary-sized vectors that outperform TF-IDF baselines while maintaining ~98% sparsity.

## Common Commands

### Training
```bash
# Train model with benchmark against TF-IDF baseline
python -m src.benchmark --data_dir Data --epochs 5

# Lower-level training script
python -m src.train --data_dir Data --epochs 5 --flops_lambda 1e-4 --seed 42

# Train with model persistence
python -m src.benchmark --data_dir Data --epochs 5 --model_path models/splade.pth
```

### Sparse Autoencoder Pipeline (interpretability)
```bash
# 1. Extract SPLADE vectors from trained model
python -m src.extract_vectors --model_path models/model.pth --data_dir Data --output_path outputs/vectors.pt

# 2. Train SAE on extracted vectors
python -m src.train_sae --vectors_path outputs/vectors.pt --epochs 10

# 3. Analyze learned features
python -m src.analyze_sae --sae_path outputs/sae/sae_best.pt
```

### Profiling Triton kernels
```bash
python -m src.profile_ops
python -m src.profile_comparison
```

### Testing
```bash
pytest
python -m src.ops.test_kernels  # Test Triton/PyTorch parity
```

## Architecture

### Core Flow
1. **Text → Tokens**: DistilBERT tokenizer
2. **Tokens → MLM Logits**: DistilBERT produces `[batch, seq_len, vocab_size]` logits
3. **Log-Saturation**: `log(1 + ReLU(logits))` - converts to sparse weights
4. **Max-Pool over sequence**: Produces document vector `[batch, vocab_size]`
5. **Linear Classifier**: Binary classification head on sparse vector

### Key Modules

**`src/models/`**
- `sklearn_wrapper.py` → `SPLADEClassifier`: Main user-facing API with `fit()`, `predict()`, `transform()`, `explain()`
- `splade_distilbert.py` → `DistilBERTVectorizer`, `DistilBERTSparseClassifier`: Core SPLADE implementation
- `splade_mistral.py`: Experimental Mistral-7B decoder backbone

**`src/ops/`**
- Triton-accelerated GPU kernels for SPLADE aggregation (10x speedup)
- Automatic fallback to PyTorch when Triton unavailable or during training (gradients require PyTorch)
- `splade_aggregate()`: Fused log1p + relu + mask + max_pool
- `TRITON_AVAILABLE` flag for runtime backend detection

**`src/regularizers.py`**
- `flops_regularization()`: SPLADE paper's sparsity loss: `sum_j(mean_i(|w_ij|))^2`
- `block_l1_loss()`: Group Lasso for structured sparsity

**`src/interpretability/`**
- `sparse_autoencoder.py`: TopK SAE for decomposing polysemantic SPLADE dimensions
- `feature_analysis.py`, `visualization.py`: Analysis utilities

### Data Format
Training data: TSV files with columns `[id, review, label]` in `Data/` directory:
- `movie_reviews_train.txt`
- `movie_reviews_test.txt`

### Device Handling
- Models auto-detect CUDA vs CPU
- Triton kernels only used during inference (not training) on CUDA with contiguous tensors
- `torch.is_grad_enabled()` check prevents Triton use during backprop

## API Usage Pattern

```python
from src.models import SPLADEClassifier

clf = SPLADEClassifier(flops_lambda=1e-4)
clf.fit(train_texts, train_labels, epochs=5)

preds = clf.predict(test_texts)
sparse_vectors = clf.transform(test_texts)  # scipy.sparse.csr_matrix
clf.print_explanation("This movie was great!")  # Interpretability
```
