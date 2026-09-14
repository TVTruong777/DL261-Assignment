#!/usr/bin/env bash
# Downloads MNIST, Fashion-MNIST (and optionally CIFAR-10) into ./data,
# and runs a 2-epoch debug run to sanity-check the whole pipeline end to end.
#
# Usage:
#   bash scripts/prepare_data.sh            # download Fashion-MNIST + MNIST, run debug
#   bash scripts/prepare_data.sh --cifar10   # also download CIFAR-10

set -euo pipefail

python - <<'PY'
from torchvision import datasets
print("Downloading MNIST...")
datasets.MNIST(root="./data", train=True, download=True)
datasets.MNIST(root="./data", train=False, download=True)
print("Downloading Fashion-MNIST...")
datasets.FashionMNIST(root="./data", train=True, download=True)
datasets.FashionMNIST(root="./data", train=False, download=True)
PY

if [[ "${1:-}" == "--cifar10" ]]; then
  python - <<'PY'
from torchvision import datasets
print("Downloading CIFAR-10 (optional extension)...")
datasets.CIFAR10(root="./data", train=True, download=True)
datasets.CIFAR10(root="./data", train=False, download=True)
PY
fi

echo "Running a 2-epoch debug smoke test (configs/debug_mnist.yaml)..."
python -m src.train --config configs/debug_mnist.yaml

echo "Data ready and pipeline smoke-tested successfully."
