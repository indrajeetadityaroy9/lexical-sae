"""Module entrypoint for python -m splade_classifier."""

import sys

from splade_classifier import benchmark


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        sys.argv = sys.argv[1:]
        benchmark.main()
    else:
        print("SPLADE Classifier - Sparse lexical text classification")
        print()
        print("Usage:")
        print("  python -m splade_classifier benchmark [OPTIONS]")
        print()
        print("Benchmark options:")
        print("  --dataset {ag_news,sst2,imdb,all}  Dataset to benchmark (default: all)")
        print("  --train-samples N                  Training samples (default: 10000)")
        print("  --test-samples N                   Test samples (default: 2000)")
        print("  --epochs N                         Training epochs (default: 3)")
        print("  --batch-size N                     Batch size (default: 64)")
        print()
        print("Python API:")
        print("  from splade_classifier import SPLADEClassifier, load_classification_data")
        print("  clf = SPLADEClassifier(num_labels=2)")
        print("  clf.fit(texts, labels, epochs=3)")
        print("  predictions = clf.predict(new_texts)")


if __name__ == "__main__":
    main()
