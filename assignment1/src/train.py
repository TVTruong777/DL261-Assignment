"""Training entrypoint.

    python -m src.train --config configs/cnn.yaml

Trains exactly one model as specified by the config. To reproduce the full
model comparison, run this once per config in `configs/` (see
`scripts/train_all.sh`). Every run:
  1. fixes the seed from the config,
  2. uses the same stratified split (same seed) across all configs, so the
     five mandatory models are compared fairly on identical data,
  3. saves the best-val-metric checkpoint,
  4. writes per-epoch train/val curves to results/runs/<experiment_name>/,
  5. appends one row to results/experiment_log.csv linking the result to its
     config file, dataset split, checkpoint, experiment id, and git commit.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.data import get_dataloaders
from src.metrics import compute_metrics
from src.models import build_model
from src.utils import (
    Checkpoint,
    append_experiment_log,
    count_parameters,
    ensure_dir,
    get_device,
    get_git_commit,
    save_checkpoint,
    set_seed,
    timer,
)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 0.0)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=train_cfg.get("momentum", 0.9)
        )
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, train_cfg: dict):
    name = (train_cfg.get("lr_scheduler") or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.get("step_size", 10), gamma=0.1)
    raise ValueError(f"Unknown lr_scheduler '{name}'")


def run_epoch(model, loader, device, criterion, optimizer=None) -> tuple[float, dict]:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_seen = 0.0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / max(n_seen, 1)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred, num_classes=int(max(y_true.max(), y_pred.max())) + 1)
    return avg_loss, metrics


def train(config: dict) -> dict:
    exp_name = config["experiment_name"]
    seed = config["seed"]
    set_seed(seed, deterministic=config.get("deterministic", True))
    device = get_device(config.get("hardware", {}).get("device", "auto"))

    data_bundle = get_dataloaders(config, data_root=config["data"].get("root", "./data"))
    model = build_model(config["model"], data_bundle.input_shape, data_bundle.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    run_dir = ensure_dir(Path(config.get("logging", {}).get("log_dir", "results/runs")) / exp_name)
    curves_path = run_dir / "curves.csv"

    best_metric_name = config["checkpoint"].get("selection_metric", "accuracy")
    best_metric_value = -float("inf")
    ckpt_path = Path(config["checkpoint"].get("save_dir", "checkpoints")) / f"{exp_name}_best.pt"

    n_params = count_parameters(model)
    epochs = config["training"]["epochs"]

    with open(curves_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"])

    with timer() as train_timer:
        for epoch in tqdm(range(1, epochs + 1), desc=f"[{exp_name}] training"):
            train_loss, train_metrics = run_epoch(
                model, data_bundle.train_loader, device, criterion, optimizer
            )
            val_loss, val_metrics = run_epoch(model, data_bundle.val_loader, device, criterion)

            if scheduler is not None:
                scheduler.step()

            with open(curves_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, train_loss, train_metrics["accuracy"], val_loss,
                     val_metrics["accuracy"], val_metrics["macro_f1"]]
                )

            current = val_metrics.get(best_metric_name, val_metrics["accuracy"])

            if current > best_metric_value:
                best_metric_value = current
                save_checkpoint(
                    ckpt_path,
                    Checkpoint(
                        epoch=epoch,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        best_metric=best_metric_value,
                        config=config,
                    ),
                )

    training_time_sec = train_timer.elapsed

    # Final test-set evaluation using the best checkpoint, and inference-time measurement.
    from src.utils import load_checkpoint

    best = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])

    with timer() as infer_timer:
        test_loss, test_metrics = run_epoch(model, data_bundle.test_loader, device, criterion)
    inference_time_sec = infer_timer.elapsed

    summary = {
        "experiment_name": exp_name,
        "model": config["model"]["name"],
        "dataset": config["data"]["dataset_name"],
        "seed": seed,
        "num_params": n_params,
        "epochs": epochs,
        "training_time_sec": round(training_time_sec, 2),
        "inference_time_sec_full_test_set": round(inference_time_sec, 2),
        "best_epoch": int(best["epoch"]),
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "split_sizes": data_bundle.split_sizes,
        "checkpoint_path": str(ckpt_path),
        "git_commit": get_git_commit(),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump({**summary, "confusion_matrix": test_metrics["confusion_matrix"]}, f, indent=2)

    append_experiment_log(
        config.get("logging", {}).get("experiment_log", "results/experiment_log.csv"),
        {
            "result_name": f"{exp_name}_test_accuracy",
            "metric": "accuracy",
            "value": round(test_metrics["accuracy"], 4),
            "config_file": config.get("_config_path", "unknown"),
            "dataset_split": f"test (n={data_bundle.split_sizes['test']})",
            "checkpoint": str(ckpt_path),
            "log_experiment_id": exp_name,
            "git_commit_or_tag": summary["git_commit"],
            "seed": seed,
            "date": time.strftime("%Y-%m-%d"),
        },
    )
    append_experiment_log(
        config.get("logging", {}).get("experiment_log", "results/experiment_log.csv"),
        {
            "result_name": f"{exp_name}_test_macro_f1",
            "metric": "macro_f1",
            "value": round(test_metrics["macro_f1"], 4),
            "config_file": config.get("_config_path", "unknown"),
            "dataset_split": f"test (n={data_bundle.split_sizes['test']})",
            "checkpoint": str(ckpt_path),
            "log_experiment_id": exp_name,
            "git_commit_or_tag": summary["git_commit"],
            "seed": seed,
            "date": time.strftime("%Y-%m-%d"),
        },
    )

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = args.config
    train(config)


if __name__ == "__main__":
    main()
