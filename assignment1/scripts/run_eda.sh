#!/usr/bin/env bash
# Runs EDA on Fashion-MNIST (primary) and MNIST (debug dataset, for completeness).
#
# Usage:
#   bash scripts/run_eda.sh

set -euo pipefail

python -m src.eda --dataset fashion_mnist --out results/eda/fashion_mnist
python -m src.eda --dataset mnist --out results/eda/mnist

echo "EDA artifacts written under results/eda/"
