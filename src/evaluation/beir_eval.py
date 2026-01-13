"""
BEIR Benchmark Evaluation for SPLADE models.

BEIR (Benchmarking IR) provides heterogeneous evaluation across
18 diverse retrieval datasets for zero-shot evaluation.

Reference:
    BEIR: A Heterogenous Benchmark for Zero-shot Evaluation
    https://arxiv.org/abs/2104.08663
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

# BEIR imports (optional)
try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    HAS_BEIR = True
except ImportError:
    HAS_BEIR = False


# Available BEIR datasets
BEIR_DATASETS = [
    "trec-covid",
    "nfcorpus",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
    "msmarco",
]

# Smaller subset for quick evaluation
BEIR_QUICK_DATASETS = ["nfcorpus", "scifact", "fiqa"]


@dataclass
class BEIRResults:
    """Results from BEIR evaluation."""
    dataset: str
    ndcg_10: float
    ndcg_100: float
    recall_10: float
    recall_100: float
    mrr_10: float
    map_10: float
    num_queries: int
    num_corpus: int


def check_beir():
    """Check if BEIR is installed."""
    if not HAS_BEIR:
        raise ImportError(
            "beir is required for BEIR evaluation. "
            "Install with: pip install beir"
        )


class BEIRDataset:
    """
    Wrapper for loading BEIR datasets.

    Downloads and caches datasets automatically.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "datasets/beir",
        split: str = "test"
    ):
        check_beir()

        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.split = split

        # Download if not exists
        self._download_if_needed()

        # Load dataset
        self.corpus, self.queries, self.qrels = self._load()

    def _download_if_needed(self):
        """Download dataset if not already present."""
        dataset_path = self.data_dir / self.dataset_name

        if not dataset_path.exists():
            print(f"Downloading {self.dataset_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            util.download_and_unzip(url, str(self.data_dir))

    def _load(self) -> Tuple[Dict, Dict, Dict]:
        """Load corpus, queries, and qrels."""
        data_path = self.data_dir / self.dataset_name

        loader = GenericDataLoader(data_folder=str(data_path))
        corpus, queries, qrels = loader.load(split=self.split)

        return corpus, queries, qrels

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_corpus(self) -> int:
        return len(self.corpus)


class SPLADERetriever:
    """
    Retriever using SPLADE sparse vectors.

    Encodes queries and documents into sparse vectors,
    then retrieves using sparse dot product.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 256,
        batch_size: int = 32
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = max_length
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode list of texts into sparse vectors.

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            vectors: Sparse vectors [num_texts, vocab_size]
        """
        all_vectors = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                # Get sparse vectors
                if hasattr(self.model, "vectorizer"):
                    # DistilBERTSparseClassifier
                    vectors = self.model.vectorizer(input_ids, attention_mask)
                else:
                    # Direct vectorizer
                    vectors = self.model(input_ids, attention_mask)

                all_vectors.append(vectors.cpu())

        return torch.cat(all_vectors, dim=0)

    def encode_corpus(
        self,
        corpus: Dict[str, Dict[str, str]]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Encode BEIR corpus.

        Args:
            corpus: BEIR corpus dict {doc_id: {"title": ..., "text": ...}}

        Returns:
            vectors: Document vectors [num_docs, vocab_size]
            doc_ids: List of document IDs
        """
        doc_ids = list(corpus.keys())
        texts = []

        for doc_id in doc_ids:
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            full_text = f"{title} {text}".strip()
            texts.append(full_text)

        vectors = self.encode_texts(texts, show_progress=True)
        return vectors, doc_ids

    def encode_queries(
        self,
        queries: Dict[str, str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Encode BEIR queries.

        Args:
            queries: BEIR queries dict {query_id: query_text}

        Returns:
            vectors: Query vectors [num_queries, vocab_size]
            query_ids: List of query IDs
        """
        query_ids = list(queries.keys())
        texts = [queries[qid] for qid in query_ids]

        vectors = self.encode_texts(texts, show_progress=True)
        return vectors, query_ids

    def retrieve(
        self,
        query_vectors: torch.Tensor,
        corpus_vectors: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
        top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve top-k documents for each query.

        Uses sparse dot product for scoring.

        Args:
            query_vectors: Query vectors [num_queries, vocab_size]
            corpus_vectors: Document vectors [num_docs, vocab_size]
            query_ids: List of query IDs
            doc_ids: List of document IDs
            top_k: Number of documents to retrieve

        Returns:
            results: Dict mapping query_id to {doc_id: score}
        """
        results = {}

        print(f"Retrieving top-{top_k} documents...")

        for i, qid in enumerate(tqdm(query_ids, desc="Retrieval")):
            query_vec = query_vectors[i:i+1]  # [1, vocab_size]

            # Sparse dot product
            scores = torch.mm(query_vec, corpus_vectors.t()).squeeze(0)  # [num_docs]

            # Get top-k
            top_values, top_indices = torch.topk(scores, min(top_k, len(doc_ids)))

            results[qid] = {
                doc_ids[idx]: float(score)
                for idx, score in zip(top_indices.tolist(), top_values.tolist())
            }

        return results


def evaluate_on_beir(
    model: nn.Module,
    tokenizer,
    dataset_name: str,
    data_dir: str = "datasets/beir",
    batch_size: int = 32,
    max_length: int = 256,
    top_k: int = 100,
    device: Optional[torch.device] = None
) -> BEIRResults:
    """
    Evaluate SPLADE model on a BEIR dataset.

    Args:
        model: SPLADE model (vectorizer or classifier)
        tokenizer: HuggingFace tokenizer
        dataset_name: BEIR dataset name
        data_dir: Directory to store BEIR data
        batch_size: Encoding batch size
        max_length: Maximum sequence length
        top_k: Number of documents to retrieve
        device: Device for computation

    Returns:
        BEIRResults with all metrics
    """
    check_beir()

    print(f"\n{'='*60}")
    print(f"Evaluating on BEIR: {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    beir_data = BEIRDataset(dataset_name, data_dir)

    print(f"Corpus size: {beir_data.num_corpus}")
    print(f"Queries: {beir_data.num_queries}")

    # Create retriever
    retriever = SPLADERetriever(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size
    )

    # Encode corpus
    print("\nEncoding corpus...")
    corpus_vectors, doc_ids = retriever.encode_corpus(beir_data.corpus)

    # Encode queries
    print("\nEncoding queries...")
    query_vectors, query_ids = retriever.encode_queries(beir_data.queries)

    # Retrieve
    results = retriever.retrieve(
        query_vectors=query_vectors,
        corpus_vectors=corpus_vectors,
        query_ids=query_ids,
        doc_ids=doc_ids,
        top_k=top_k
    )

    # Evaluate
    print("\nComputing metrics...")
    evaluator = EvaluateRetrieval()
    ndcg, map_score, recall, precision = evaluator.evaluate(
        beir_data.qrels,
        results,
        k_values=[10, 100]
    )

    # Calculate MRR manually
    mrr_10 = _calculate_mrr(results, beir_data.qrels, k=10)

    beir_results = BEIRResults(
        dataset=dataset_name,
        ndcg_10=ndcg.get("NDCG@10", 0.0),
        ndcg_100=ndcg.get("NDCG@100", 0.0),
        recall_10=recall.get("Recall@10", 0.0),
        recall_100=recall.get("Recall@100", 0.0),
        mrr_10=mrr_10,
        map_10=map_score.get("MAP@10", 0.0),
        num_queries=beir_data.num_queries,
        num_corpus=beir_data.num_corpus
    )

    print(f"\nResults for {dataset_name}:")
    print(f"  NDCG@10:   {beir_results.ndcg_10:.4f}")
    print(f"  NDCG@100:  {beir_results.ndcg_100:.4f}")
    print(f"  Recall@10: {beir_results.recall_10:.4f}")
    print(f"  Recall@100:{beir_results.recall_100:.4f}")
    print(f"  MRR@10:    {beir_results.mrr_10:.4f}")

    return beir_results


def _calculate_mrr(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> float:
    """Calculate Mean Reciprocal Rank."""
    mrr_scores = []

    for qid, doc_scores in results.items():
        if qid not in qrels:
            continue

        relevant_docs = set(
            doc_id for doc_id, rel in qrels[qid].items() if rel > 0
        )

        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        for rank, (doc_id, _) in enumerate(sorted_docs, 1):
            if doc_id in relevant_docs:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    return np.mean(mrr_scores) if mrr_scores else 0.0


def evaluate_on_multiple_datasets(
    model: nn.Module,
    tokenizer,
    datasets: List[str] = None,
    **kwargs
) -> Dict[str, BEIRResults]:
    """
    Evaluate on multiple BEIR datasets.

    Args:
        model: SPLADE model
        tokenizer: HuggingFace tokenizer
        datasets: List of dataset names (default: BEIR_QUICK_DATASETS)
        **kwargs: Additional arguments for evaluate_on_beir

    Returns:
        Dictionary mapping dataset names to results
    """
    if datasets is None:
        datasets = BEIR_QUICK_DATASETS

    all_results = {}

    for dataset_name in datasets:
        try:
            results = evaluate_on_beir(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                **kwargs
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("BEIR EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<20} {'NDCG@10':<10} {'Recall@100':<12} {'MRR@10':<10}")
    print("-" * 60)

    avg_ndcg = []
    for name, res in all_results.items():
        print(f"{name:<20} {res.ndcg_10:<10.4f} {res.recall_100:<12.4f} {res.mrr_10:<10.4f}")
        avg_ndcg.append(res.ndcg_10)

    if avg_ndcg:
        print("-" * 60)
        print(f"{'Average':<20} {np.mean(avg_ndcg):<10.4f}")

    return all_results
