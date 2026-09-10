"""
Training entrypoint.

Usage:
    python scripts/train.py --config configs/config_expA.yaml

Replace the body with the actual training loop. Keep this contract:
- reads all hyperparameters from the given config file (no hidden defaults
  that silently diverge from the config),
- fixes the seed from the config before any random operation,
- writes checkpoints to `checkpoint.save_dir` from the config,
- logs an experiment/run ID that gets recorded in results/experiment_log.csv.
"""
import argparse
# import yaml, torch, random, numpy as np  # etc.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    # [TODO] load config, set seed, build model/data/optimizer, train loop,
    # save checkpoint(s), log metrics + experiment id.
    raise NotImplementedError("Fill in the training loop for this assignment.")


if __name__ == "__main__":
    main()
