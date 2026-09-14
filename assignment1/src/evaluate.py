"""Standalone evaluation from a saved checkpoint.

    python -m src.evaluate --config configs/cnn.yaml --checkpoint checkpoints/cnn_best.pt

Produces, under results/runs/<experiment_name>/eval/:
  - metrics.json           (accuracy, macro-F1, confusion matrix)
  - confusion_matrix.png
  - correct_examples.png
  - incorrect_examples.png

This is the script referenced by the assignment page's "Results" and
"Error analysis" sections — the qualitative figures should be regenerated
from here, not hand-picked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import get_dataloaders
from src.metrics import compute_metrics
from src.models import build_model
from src.train import load_config
from src.utils import ensure_dir, get_device, load_checkpoint, set_seed


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_x, all_true, all_pred, all_prob = [], [], [], []
    model.eval()
    for x, y in loader:
        x_dev = x.to(device)
        logits = model(x_dev)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_x.append(x.cpu())
        all_true.append(y.cpu())
        all_pred.append(preds.cpu())
        all_prob.append(probs.cpu())

    return (
        torch.cat(all_x),
        torch.cat(all_true).numpy(),
        torch.cat(all_pred).numpy(),
        torch.cat(all_prob),
    )


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_grid(images_for_display: np.ndarray, titles: list[str], out_path: Path, n: int = 12):
    n = min(n, len(images_for_display))
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img = images_for_display[i]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray")
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_title(titles[i], fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def to_display_image(x: torch.Tensor) -> np.ndarray:
    """De-normalize a (C,H,W) tensor (mean=0.5,std=0.5) back to [0,1] for plotting."""
    img = x.numpy() * 0.5 + 0.5
    return np.clip(img, 0.0, 1.0)


def evaluate(config: dict, checkpoint_path: str):
    set_seed(config["seed"])
    device = get_device(config.get("hardware", {}).get("device", "auto"))

    # NOTE: for correct/incorrect example plots we need the raw image, so we
    # temporarily force representation="image" regardless of the model's
    # normal representation. Metrics still use the model's own representation.
    data_bundle = get_dataloaders(config, data_root=config["data"].get("root", "./data"))
    model = build_model(config["model"], data_bundle.input_shape, data_bundle.num_classes).to(device)

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    x, y_true, y_pred, probs = collect_predictions(model, data_bundle.test_loader, device)
    metrics = compute_metrics(y_true, y_pred, num_classes=data_bundle.num_classes)

    out_dir = ensure_dir(
        Path(config.get("logging", {}).get("log_dir", "results/runs"))
        / config["experiment_name"] / "eval"
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), data_bundle.class_names, out_dir / "confusion_matrix.png")

    # NOTE: `x` here is in the model's actual representation (e.g. flattened
    # or patches), which isn't directly plottable as an image unless
    # representation == "image". If you need example grids for a non-CNN
    # model, re-run the test loader with representation="image" separately
    # and reuse the same y_true/y_pred to select indices.
    if config["data"]["representation"] == "image":
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        correct_idx = np.where(correct_mask)[0][:12]
        incorrect_idx = np.where(incorrect_mask)[0][:12]

        correct_titles = [
            f"true={data_bundle.class_names[y_true[i]]}" for i in correct_idx
        ]
        incorrect_titles = [
            f"true={data_bundle.class_names[y_true[i]]}\npred={data_bundle.class_names[y_pred[i]]}"
            for i in incorrect_idx
        ]

        plot_example_grid(
            [to_display_image(x[i]) for i in correct_idx], correct_titles, out_dir / "correct_examples.png"
        )
        plot_example_grid(
            [to_display_image(x[i]) for i in incorrect_idx], incorrect_titles, out_dir / "incorrect_examples.png"
        )

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"Artifacts written to {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
