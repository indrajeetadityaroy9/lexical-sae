"""
Run BEIR benchmark evaluation on trained SPLADE model.

Usage:
    python -m src.run_beir_eval --model_path models/model.pth --dataset nfcorpus
"""

import argparse
import os

from transformers import AutoTokenizer

from src.models import SPLADEClassifier
from src.evaluation.beir_eval import evaluate_on_beir, BEIR_QUICK_DATASETS


def main(args):
    # Load model using SPLADEClassifier
    print(f"Loading model from {args.model_path}...")
    clf = SPLADEClassifier(verbose=False)
    clf.load(args.model_path)
    print(f"Using device: {clf.device}")

    # Get internal PyTorch model for BEIR eval
    model = clf.model
    model.eval()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    print(f"\nEvaluating on BEIR dataset: {args.dataset}")
    print("Note: This model was trained for classification, not retrieval.")
    print("      Retrieval metrics serve as a baseline/demonstration.\n")

    try:
        results = evaluate_on_beir(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_k=args.top_k,
            device=clf.device
        )

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, f'beir_{args.dataset}_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"BEIR Evaluation Results: {args.dataset}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Corpus size: {results.num_corpus}\n")
            f.write(f"Queries: {results.num_queries}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  NDCG@10:    {results.ndcg_10:.4f}\n")
            f.write(f"  NDCG@100:   {results.ndcg_100:.4f}\n")
            f.write(f"  Recall@10:  {results.recall_10:.4f}\n")
            f.write(f"  Recall@100: {results.recall_100:.4f}\n")
            f.write(f"  MRR@10:     {results.mrr_10:.4f}\n")
            f.write(f"  MAP@10:     {results.map_10:.4f}\n")

        print(f"\nResults saved to {results_path}")

    except ImportError as e:
        print(f"Error: {e}")
        print("The BEIR package is required for this evaluation.")
        print("Install with: pip install beir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/model.pth')
    parser.add_argument('--dataset', type=str, default='nfcorpus',
                       choices=BEIR_QUICK_DATASETS + ['scifact', 'arguana'])
    parser.add_argument('--data_dir', type=str, default='datasets/beir')
    parser.add_argument('--output_dir', type=str, default='outputs/beir')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--top_k', type=int, default=100)

    args = parser.parse_args()
    main(args)
