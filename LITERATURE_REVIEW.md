# Neural Sparse Representations (2022–2025)

### The SPLADE Family (Sparse Lexical and Expansion Models)
*   **SPLADE v2 (Formal et al., 2021-2022):** Introduces a framework where a BERT model predicts weights for the *entire* vocabulary (30k+ tokens) for each input text. It uses a **Log-Saturation** activation (similar to the project's `torch.log1p`) and **FLOPS-regularized** sparsity.
    *   *Key Insight:* Instead of just weighting existing words (like the current project), it performs **Term Expansion**—predicting relevant words *not* present in the text (e.g., adding "computer" to a text containing "PC").
    *   *Relevance:* This directly addresses the "Vocabulary Mismatch" problem of TF-IDF while maintaining sparse, interpretable vectors.

### Decoder-Only Sparse Retrievers
*   **Mistral-SPLADE (Doshi et al., Aug 2024):** Demonstrates that **Decoder-only LLMs** (like Mistral) significantly outperform Encoder-only models (BERT) for generating sparse representations.
    *   *Key Insight:* Larger, generative models learn better semantic expansions, effectively "hallucinating" the perfect TF-IDF vector for a document.

### Sparsified Late Interaction
*   **SLIM (Li et al., 2023):** Proposes mapping contextualized token embeddings into a **high-dimensional sparse lexical space**.
    *   *Key Insight:* This allows "Late Interaction" models (like ColBERT) to be indexed by standard inverted indexes (Lucene/Elasticsearch), bridging the gap between high-performance neural matching and low-latency keyword search.

## 2. Theoretical Guarantees & Foundations
*   **Transformers as n-gram Models (Svete & Cotterell, Apr 2024):**
    *   *Finding:* Proves that Transformers with hard or sparse attention can **exactly represent** any n-gram language model.
    *   *Implication:* This provides a theoretical "existence proof" that the project's Neural Vectorizer *can* strictly subsume the baseline TF-IDF/n-gram model given the right sparsity pattern.

*   **Non-Vacuous Generalization Bounds (Lotfi et al., Dec 2023):**
    *   Derives compression-based generalization bounds for large models, suggesting that **highly sparse** (compressible) models should generalize better to unseen data, validating the project's hypothesis that sparsity aids robustness.

## 3. Scaling & Hardware Efficiency

*   **Massive Scale Analysis (Bruch et al., Jan 2025):**
    *   *Study:* "Investigating the Scalability of Approximate Sparse Retrieval Algorithms to Massive Datasets" (138M documents).
    *   *Key Finding:* Naive sparse implementations fail at scale. **Block-based** storage and **quantized** weights are essential for maintaining low latency in inverted indexes.

*   **Chase: Hardware-Efficient Sparse Training (Nov 2023):**
    *   *Technique:* "Channel-wise Sparsity". Instead of random (unstructured) sparsity—which GPUs hate—it enforces sparsity on entire **channels**.
    *   *Result:* Achieved **1.2x–1.7x inference speedups** on standard commodity GPUs (without custom hardware).
    *   *Relevance:* The project's `block_l1_loss` is a step in this direction, but *channel-wise* sparsity might offer better real-world speedups than generic *block* sparsity.

*   **Rethinking Sparse Optimization (Kuznedelev et al., Aug 2023):**
    *   *Finding:* Standard training recipes (AdamW, constant pruning) lead to under-trained sparse models. They propose modified schedules specifically for high-sparsity regimes.
