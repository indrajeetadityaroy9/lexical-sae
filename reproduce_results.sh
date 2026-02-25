#!/bin/bash
set -e

echo "============================================"
echo "Lexical-SAE — Reproduction Pipeline"
echo "============================================"

RUNNER="python -m cajt.scripts.run"

# Core experiments
$RUNNER --config experiments/core/banking77.yaml
$RUNNER --config experiments/core/imdb.yaml
$RUNNER --config experiments/core/yelp.yaml

# Ablations
$RUNNER --config experiments/ablation/banking77_ablation.yaml
$RUNNER --config experiments/ablation/imdb_ablation.yaml

# SOTA comparison (experiment + dense baseline)
$RUNNER --config experiments/analysis/banking77_sota.yaml

# Surgical bias removal
$RUNNER --config experiments/surgery/civilcomments_surgery.yaml
$RUNNER --config experiments/surgery/beavertails_surgery.yaml

# Long-context needle-in-haystack
$RUNNER --config experiments/analysis/imdb_long_context.yaml

echo "Done."
