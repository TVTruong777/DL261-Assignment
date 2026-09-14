"""
Evaluation entrypoint.

Usage:
    python scripts/evaluate.py --config configs/config_expA.yaml --checkpoint checkpoints/expA_best.pt

Replace the body with the actual evaluation logic. Should print/save metrics
in a form that can be copied into results/experiment_log.csv and the report's
results table.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    # [TODO] load config + checkpoint, run on the test split, report metrics.
    raise NotImplementedError("Fill in evaluation logic for this assignment.")


if __name__ == "__main__":
    main()
