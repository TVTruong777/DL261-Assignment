#!/usr/bin/env bash
# Trains all five mandatory models (same dataset, same split seed) so the
# main comparison table can be filled in from results/experiment_log.csv.
#
# Usage:
#   bash scripts/train_all.sh

set -euo pipefail

CONFIGS=(
  "configs/linear.yaml"
  "configs/mlp.yaml"
  "configs/cnn.yaml"
  "configs/rnn.yaml"
  "configs/transformer.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=================================================="
  echo "Training with $cfg"
  echo "=================================================="
  python -m src.train --config "$cfg"
done

echo "All five mandatory models trained. See results/experiment_log.csv for traceability."
