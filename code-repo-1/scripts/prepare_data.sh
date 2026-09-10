#!/usr/bin/env bash
# Prepare the dataset: download (if applicable), preprocess, and create
# train/val/test splits. Replace the body below with the real steps.
#
# Usage:
#   bash scripts/prepare_data.sh

set -euo pipefail

echo "[TODO] Download raw data to: [raw_dir]"
# wget/curl or dataset-API download commands go here

echo "[TODO] Preprocess and write splits to: [processed_dir]"
# python -m src.data.preprocess --config configs/config.example.yaml

echo "Dataset preparation complete."
