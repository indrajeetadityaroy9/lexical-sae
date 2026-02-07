# Interpretable SPLADE Classifier: Sparse Lexical Representations for Faithful Text Classification

Deep learning classifiers often suffer from a lack of inherent interpretability, relying on post-hoc explainers that may fail to capture the true model logic. By adapting the SPLADE v2 architecture—originally designed for information retrieval—this repository implements a classifier that is **interpretable by design**. Through sparse vocabulary activations and document-frequency (DF) weighted regularization, the model produces human-readable lexical explanations that are directly tied to the classification decision. To validate these explanations, the **F-Fidelity protocol** is employed, ensuring that faithfulness metrics are measured on a model distribution resilient to OOD artifacts.

---

1.  **SPLADE v2 with Shifted-ReLU**: Adapts the SPLADE v2 architecture for classification with a learnable-threshold activation $f(x) = \max(0, x - \theta)$ that promotes exact sparsity in vocabulary representations.
2.  **DF-FLOPS Regularization**: Introduction of Document-Frequency weighted FLOPS regularization to control representation sparsity while preserving informative, rare terms.
3.  **F-Fidelity Protocol**: Implementation of the F-Fidelity fine-tuning procedure ([arXiv:2410.02970](https://arxiv.org/abs/2410.02970)) to mitigate masking bias in faithfulness evaluation.

---

### SPLADE Aggregation
The model computes sparse document vectors $\mathbf{s} \in \mathbb{R}^{|V|}$ by aggregating BERT-derived token logits over the sequence length $L$:
$$s_j = \max_{i \in [1, L]} \log(1 + \max(0, w_{ij} - \theta_j))$$
where $w_{ij}$ is the logit for the $j$-th vocabulary term at the $i$-th sequence position and $\theta_j$ is a learnable per-dimension threshold (shifted-ReLU) that promotes exact zeros in the sparse representation.

### Regularization Objectives
The training process minimizes a joint objective $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{sparse}$:
- **FLOPS**: $\mathcal{L}_{FLOPS} = \sum_{j} \bar{a}_j^2$, where $\bar{a}_j$ is the mean activation of term $j$ across the batch.
- **DF-FLOPS**: Weights the penalty by document frequency to encourage the selection of discriminative lexical features.

Implements evaluation protocol to measure explanation **Faithfulness**:

- **Normalized AOPC (NAOPC)**: Per-example normalized Area Over the Perturbation Curve using beam-search bounds ([arXiv:2408.08137](https://arxiv.org/abs/2408.08137)).
- **Soft Metrics**: Embedding-level Bernoulli dropout comprehensiveness and sufficiency with zero-tensor baseline ([arXiv:2305.10496](https://arxiv.org/abs/2305.10496)).
- **Adversarial Sensitivity**: Stability of explanation rankings under synonym substitution and character-level perturbations using Kendall-Tau-$\hat{h}$.
- **Subword-to-Word Alignment**: A robust normalization layer that maps subword attributions to whitespace-delimited words for fair comparison.

## References

- **SPLADE v2**: Formal et al. (2021). [arXiv:2109.10086](https://arxiv.org/abs/2109.10086)
- **SPLADE v3**: Lassance et al. (2024). [arXiv:2403.06789](https://arxiv.org/abs/2403.06789)
- **F-Fidelity**: [arXiv:2410.02970](https://arxiv.org/abs/2410.02970)
- **Soft Evaluation Metrics**: [arXiv:2305.10496](https://arxiv.org/abs/2305.10496)
- **OOD Artifacts in Faithfulness**: [arXiv:2308.14272](https://arxiv.org/abs/2308.14272)
- **ERASER Benchmark**: [arXiv:1911.03429](https://arxiv.org/abs/1911.03429)
- **NFNet (AGC)**: [arXiv:2102.06171](https://arxiv.org/abs/2102.06171)
- **NAOPC**: [arXiv:2408.08137](https://arxiv.org/abs/2408.08137)
