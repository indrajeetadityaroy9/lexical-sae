# Lexical-SAE

**Intrinsic, Exact Sequence Autoencoders for Interpretable NLP**

Lexical-SAE is a general-purpose interpretability architecture that repurposes the tokenizer vocabulary as a sparse feature dictionary. Unlike post-hoc Sparse Autoencoders (SAEs) which suffer from reconstruction error and feature opacity, Lexical-SAE provides **(1)** zero reconstruction error via algebraic DLA identity, **(2)** a single task-agnostic architecture for classification and sequence labeling, and **(3)** surgical concept removal with mathematical guarantees.

---

## Key Results

### Named Entity Recognition (CoNLL)
*Backbone: ModernBERT-base | Context: 128 tokens*

| Metric | Value | SOTA Context |
|--------|-------|--------------|
| **Token Accuracy** | **97.0%** | Matches dense baselines |
| **Entity F1 (micro)** | **84.0%** | Competitive with BERT-base (dense) |
| **Sparsity** | **95.8%** | Only ~2k active dims out of 50k |
| **DLA Error** | **2.1e-4** | Exact attribution (Machine Precision) |

**Per-Entity Breakdown:**
PER: **0.93** F1 | LOC: **0.88** F1 | ORG: **0.77** F1 | MISC: **0.72** F1

### Text Classification
*Backbone: DistilBERT / ModernBERT*

| Dataset | Accuracy | DLA Error | Active Dims | Sparsity |
|---------|----------|-----------|-------------|----------|
| SST-2 | 92.3% | ~0.001 | ~130 | 99.7% |
| AG News | 93.7% | ~0.001 | ~150 | 99.7% |
| IMDB | 91.8% | ~0.001 | ~120 | 99.8% |

---

## Comparison: Lexical-SAE vs. Post-Hoc SAE

| Feature | Post-Hoc SAE | **Lexical-SAE** |
|:---|:---|:---|
| **Training** | Trained *after* model (Frozen) | Trained *during* model (Intrinsic) |
| **Reconstruction** | Lossy ($x \approx \hat{x}$) | **Exact** ($y \equiv \sum s_i w_i$) |
| **Dictionary** | Learned Latents (Feature #1405) | **Vocabulary**|
| **Granularity** | Layer-wise or Residual Stream | **Token-wise** (Sequence Modeling) |
| **Intervention** | Approximate Steering | **Exact Removal** (Guaranteed $s_i=0$) |

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
