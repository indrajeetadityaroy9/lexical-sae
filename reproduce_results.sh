#!/bin/bash
set -e

echo "============================================"
echo "Circuit-Integrated SPLADE (CIS) — Reproduction Pipeline"
echo "============================================"

# 1. Main Benchmarks
echo "Running CIS Main Benchmarks..."
for config in experiments/main/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.run_experiment --config "$config"
done

# 2. Dataset Experiments
echo "Running CIS Dataset Experiments..."
for config in experiments/datasets/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.run_experiment --config "$config"
done

# 3. Ablations
echo "Running CIS Ablations..."
for config in experiments/ablation/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.run_ablation --config "$config"
done

# 4. Mechanistic Evaluation
echo "Running Mechanistic Evaluation..."
for config in experiments/mechanistic/*.yaml; do
    echo "Running $config"
    python -m splade.scripts.run_experiment --config "$config"
done

echo "Reproduction Complete. Results are in results/"
