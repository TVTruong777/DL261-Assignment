#!/usr/bin/env bash
# Runs standalone evaluation (confusion matrix + example grids) for each
# trained model. Run scripts/train_all.sh first.
#
# Usage:
#   bash scripts/evaluate_all.sh

set -euo pipefail

declare -A CKPT_FOR_CONFIG=(
  ["configs/linear.yaml"]="checkpoints/linear_fashion_mnist_best.pt"
  ["configs/mlp.yaml"]="checkpoints/mlp_fashion_mnist_best.pt"
  ["configs/cnn.yaml"]="checkpoints/cnn_fashion_mnist_best.pt"
  ["configs/rnn.yaml"]="checkpoints/rnn_lstm_rows_fashion_mnist_best.pt"
  ["configs/transformer.yaml"]="checkpoints/transformer_patch4_fashion_mnist_best.pt"
)

for cfg in "${!CKPT_FOR_CONFIG[@]}"; do
  ckpt="${CKPT_FOR_CONFIG[$cfg]}"
  echo "=================================================="
  echo "Evaluating $cfg -> $ckpt"
  echo "=================================================="
  python -m src.evaluate --config "$cfg" --checkpoint "$ckpt"
done
