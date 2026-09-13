"""Exploratory Data Analysis (EDA) — required before/alongside the main comparison.

    python -m src.eda --dataset fashion_mnist --out results/eda

Produces, under the given --out directory:
  - class_distribution.png / class_distribution.csv  (per split)
  - sample_grid.png                                    (representative samples per class)
  - eda_summary.json                                    (input size, split sizes, imbalance ratio)

Run this once per dataset actually used (at minimum Fashion-MNIST; also MNIST
if you want to document the debug dataset, and CIFAR-10 if attempting the
optional extension).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from src.data import DATASET_INFO
from src.utils import ensure_dir


def load_raw(dataset_name: str, root: str):
    info = DATASET_INFO[dataset_name]
    to_tensor = transforms.ToTensor()
    train = info["cls"](root=root, train=True, download=True, transform=to_tensor)
    test = info["cls"](root=root, train=False, download=True, transform=to_tensor)
    return info, train, test


def class_distribution(targets: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        counts[c] = int((targets == c).sum())
    return counts


def plot_distribution(counts: np.ndarray, class_names: list[str], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(counts)), counts)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(dataset, class_names: list[str], out_path: Path, per_class: int = 3):
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, per_class, figsize=(per_class * 1.6, num_classes * 1.6))

    targets = np.array(dataset.targets)
    for c in range(num_classes):
        idxs = np.where(targets == c)[0][:per_class]
        for j, idx in enumerate(idxs):
            img, _ = dataset[idx]
            ax = axes[c, j] if num_classes > 1 else axes[j]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            img_np = img.numpy()
            if img_np.shape[0] == 1:
                ax.imshow(img_np[0], cmap="gray")
            else:
                ax.imshow(np.transpose(img_np, (1, 2, 0)))
            if j == 0:
                ax.set_ylabel(class_names[c], fontsize=7, rotation=0, ha="right", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eda(dataset_name: str, root: str, out_dir: str):
    out = ensure_dir(out_dir)
    info, train, test = load_raw(dataset_name, root)
    class_names = info["class_names"]
    num_classes = len(class_names)

    train_targets = np.array(train.targets)
    test_targets = np.array(test.targets)

    train_counts = class_distribution(train_targets, num_classes)
    test_counts = class_distribution(test_targets, num_classes)

    plot_distribution(train_counts, class_names, f"{dataset_name} — train class distribution", out / "class_distribution_train.png")
    plot_distribution(test_counts, class_names, f"{dataset_name} — test class distribution", out / "class_distribution_test.png")
    plot_sample_grid(train, class_names, out / "sample_grid.png")

    img0, _ = train[0]
    imbalance_ratio = float(train_counts.max() / max(train_counts.min(), 1))

    summary = {
        "dataset": dataset_name,
        "input_shape": list(img0.shape),  # (C, H, W)
        "num_classes": num_classes,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_class_counts": train_counts.tolist(),
        "test_class_counts": test_counts.tolist(),
        "class_imbalance_ratio_train": round(imbalance_ratio, 3),
        "note": info["note"],
    }
    with open(out / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"EDA artifacts written to {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_INFO))
    parser.add_argument("--root", default="./data")
    parser.add_argument("--out", default="results/eda")
    args = parser.parse_args()
    run_eda(args.dataset, args.root, args.out)


if __name__ == "__main__":
    main()
