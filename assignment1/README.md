# Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

CO3133 — Deep Learning and Its Applications — Semester 261
Group [GROUP_ID] — Ho Chi Minh City University of Technology – VNU-HCM

From Linear Models to Modern Sequence Models: A Comparative Study for Image
Classification. Implements and fairly compares five architecture families
(Linear/Softmax, MLP, CNN, LSTM/GRU, Transformer) on a shared data split.

## Links

| Deliverable | Link |
|---|---|
| Assignment page (GitHub Pages) | `[https://<user>.github.io/<repo>/assignment1.html]` |
| Report | `[link]` |
| Slides | `[link]` |
| YouTube presentation | `[link — Public or Unlisted]` |
| AI usage disclosure | [`AI_USAGE.md`](./AI_USAGE.md) |

## 1. Installation

```bash
git clone [repo url]
cd [repo folder]
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate dl-a1
```

**Hardware used for the reported results:** fill in after running —
see `configs/*.yaml -> hardware.gpu` and `checkpoints/README.md`.

## 2. Dataset preparation

```bash
bash scripts/prepare_data.sh          # downloads MNIST + Fashion-MNIST, runs a 2-epoch smoke test
bash scripts/prepare_data.sh --cifar10  # also downloads CIFAR-10 (optional extension)
```

- **MNIST** — development/debugging only (`configs/debug_mnist.yaml`). Not used for main results.
- **Fashion-MNIST** — primary dataset for the main model comparison.
- **CIFAR-10** — optional extension only.

All five mandatory models use the **same stratified train/val split**
(`data.split_seed: 42` in every `configs/*.yaml`) and the **same official
test set** from torchvision, so the comparison is apples-to-apples. See
`src/data.py::stratified_split_indices`.

## 3. Exploratory Data Analysis (EDA)

```bash
bash scripts/run_eda.sh
```

Writes class distribution plots/CSVs, a representative sample grid, and
`eda_summary.json` (input size, split sizes, class-imbalance ratio) to
`results/eda/`. Include these in the report's dataset/EDA section.

## 4. Train

Train one model:

```bash
python -m src.train --config configs/cnn.yaml
```

Train all five mandatory models:

```bash
bash scripts/train_all.sh
```

Each run fixes its seed, trains, saves the best-val-accuracy checkpoint to
`checkpoints/`, writes per-epoch train/val curves to
`results/runs/<experiment_name>/curves.csv`, and appends a row to
[`results/experiment_log.csv`](./results/experiment_log.csv) linking the
result to its config, split, checkpoint, and git commit.

## 5. Evaluate

```bash
python -m src.evaluate --config configs/cnn.yaml --checkpoint checkpoints/cnn_fashion_mnist_best.pt
# or, for all five:
bash scripts/evaluate_all.sh
```

Writes `metrics.json`, `confusion_matrix.png`, and (for the CNN, which uses
the raw-image representation) `correct_examples.png` /
`incorrect_examples.png` to `results/runs/<experiment_name>/eval/`.

## 6. Configuration

| File | Model | Representation |
|---|---|---|
| `configs/linear.yaml` | Linear / softmax | flattened pixels |
| `configs/mlp.yaml` | MLP | flattened pixels |
| `configs/cnn.yaml` | CNN | raw image (C,H,W) |
| `configs/rnn.yaml` | LSTM (or GRU — see comment in file) | sequence of rows |
| `configs/transformer.yaml` | Transformer | 4×4 patch tokens |
| `configs/debug_mnist.yaml` | CNN, 2 epochs | debug only, on MNIST |

Copy a config before changing hyperparameters for a new experiment (e.g.
`cp configs/rnn.yaml configs/rnn_gru.yaml`) rather than editing and losing
the config that produced a reported result.

## 7. Seed settings

Every config sets `seed: 42` and `deterministic: true`, fixing Python,
NumPy, and PyTorch RNGs (`src/utils.py::set_seed`) plus the dataset split
seed (`data.split_seed: 42`). Record the actual seed used for any reported
number in the results table below (all mandatory-model configs currently
use 42).

## 8. Dependency versions

Pinned in [`requirements.txt`](./requirements.txt) /
[`environment.yml`](./environment.yml). After finalizing your environment,
regenerate a fully locked file with `pip freeze > requirements.lock.txt` and
record key versions here:

| Package | Version |
|---|---|
| Python | `[fill in]` |
| torch | `[fill in]` |
| torchvision | `[fill in]` |
| CUDA (if used) | `[fill in]` |

## 9. Checkpoints

See [`checkpoints/README.md`](./checkpoints/README.md) for download links
and/or exact reconstruction commands.

## 10. Results traceability

Every number in the report's results table must trace back to a row in
[`results/experiment_log.csv`](./results/experiment_log.csv):

| Result | Config | Split | Checkpoint | Log/experiment ID | Commit |
|---|---|---|---|---|---|
| Linear test accuracy | `configs/linear.yaml` | test | `checkpoints/linear_fashion_mnist_best.pt` | `linear_fashion_mnist` | `[commit]` |
| MLP test accuracy | `configs/mlp.yaml` | test | `checkpoints/mlp_fashion_mnist_best.pt` | `mlp_fashion_mnist` | `[commit]` |
| CNN test accuracy | `configs/cnn.yaml` | test | `checkpoints/cnn_fashion_mnist_best.pt` | `cnn_fashion_mnist` | `[commit]` |
| LSTM test accuracy | `configs/rnn.yaml` | test | `checkpoints/rnn_lstm_rows_fashion_mnist_best.pt` | `rnn_lstm_rows_fashion_mnist` | `[commit]` |
| Transformer test accuracy | `configs/transformer.yaml` | test | `checkpoints/transformer_patch4_fashion_mnist_best.pt` | `transformer_patch4_fashion_mnist` | `[commit]` |

> This table is auto-populated (accuracy + macro-F1 rows) by every
> `python -m src.train` run — pull the actual values from
> `results/experiment_log.csv` rather than retyping them by hand.

A pre-run notebook with results baked in but no way to reproduce them is
**not** an acceptable substitute for the above.

## 11. Sanity tests

Fast, no dataset download required — checks that all five models build and
produce correctly shaped outputs, and that the stratified split is correct
and reproducible:

```bash
pytest
```

## 12. Repository structure

```
.
├── README.md                 # this file
├── AI_USAGE.md
├── requirements.txt
├── environment.yml
├── pytest.ini
├── conftest.py
├── configs/
│   ├── linear.yaml
│   ├── mlp.yaml
│   ├── cnn.yaml
│   ├── rnn.yaml
│   ├── transformer.yaml
│   └── debug_mnist.yaml
├── src/
│   ├── data.py               # datasets, stratified split, representations
│   ├── models/
│   │   ├── linear.py
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── rnn.py
│   │   └── transformer.py
│   ├── metrics.py             # accuracy, macro-F1, confusion matrix
│   ├── train.py                # training loop + checkpointing + logging
│   ├── evaluate.py             # confusion matrix / example-grid generation
│   ├── eda.py                  # class distribution, sample grid, imbalance
│   └── utils.py                 # seed, device, checkpoint I/O, experiment log
├── scripts/
│   ├── prepare_data.sh
│   ├── run_eda.sh
│   ├── train_all.sh
│   └── evaluate_all.sh
├── tests/
│   ├── test_models.py
│   └── test_data.py
├── checkpoints/
│   └── README.md
├── results/
│   ├── experiment_log.csv
│   ├── eda/                    # created by scripts/run_eda.sh
│   └── runs/                   # created by src.train / src.evaluate
└── data/                       # created by scripts/prepare_data.sh (gitignored)
```

## 13. Notes on fairness across models

- All five configs use `data.split_seed: 42` — identical train/val/test
  partitions.
- All five report `accuracy`, `macro_f1`, `num_params`
  (`src.utils.count_parameters`), `training_time_sec`, and
  `inference_time_sec_full_test_set` — see each run's `summary.json`.
  **Do not conclude one model is "better" from accuracy alone**; compare
  macro-F1, params, and training/inference time together, and back
  conclusions with the confusion matrices and error analysis in
  `results/runs/*/eval/`.
- The five models differ in **data representation** (flattened vector /
  image / row-sequence / patch-sequence) by design — this is exactly the
  inductive-bias comparison the assignment asks for.
