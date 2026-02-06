#!/usr/bin/env bash
# Reproduce all paper results.
#
# Usage:
#   bash scripts/reproduce_results.sh
#
# Outputs:
#   results/logs/   — per-experiment logs
#   checkpoints/    — trained model weights

set -euo pipefail

RESULTS_DIR="results/logs"
mkdir -p "$RESULTS_DIR"

echo "=== Main Results (SST-2) ==="
python -m scripts.eval \
    --dataset sst2 \
    --train-samples 2000 \
    --test-samples 200 \
    --epochs 2 \
    --batch-size 32 \
    --cvb \
    2>&1 | tee "$RESULTS_DIR/main_sst2.log"

echo ""
echo "=== Main Results (AG News) ==="
python -m scripts.eval \
    --dataset ag_news \
    --train-samples 2000 \
    --test-samples 200 \
    --epochs 2 \
    --batch-size 32 \
    2>&1 | tee "$RESULTS_DIR/main_ag_news.log"

echo ""
echo "=== Multi-Seed Benchmark (SST-2) ==="
python -m scripts.eval \
    --dataset sst2 \
    --train-samples 2000 \
    --test-samples 200 \
    --epochs 2 \
    --batch-size 32 \
    --multi-seed \
    2>&1 | tee "$RESULTS_DIR/multi_seed_sst2.log"

echo ""
echo "=== Done ==="
echo "Results saved to $RESULTS_DIR/"
