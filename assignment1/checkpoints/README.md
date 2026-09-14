# Checkpoints

Checkpoints are **not** committed to git (see `.gitignore`) — they're large
binaries. Choose Option A or B below.

## Option A — Hosted download

| Checkpoint | Model | Link | Size |
|---|---|---|---|
| `linear_fashion_mnist_best.pt` | Linear | `[Google Drive / HF Hub / release link]` | `[size]` |
| `mlp_fashion_mnist_best.pt` | MLP | `[link]` | `[size]` |
| `cnn_fashion_mnist_best.pt` | CNN | `[link]` | `[size]` |
| `rnn_lstm_rows_fashion_mnist_best.pt` | LSTM | `[link]` | `[size]` |
| `transformer_patch4_fashion_mnist_best.pt` | Transformer | `[link]` | `[size]` |

> If hosting on Google Drive, set sharing to "Anyone with the link" and
> verify each link in an incognito window before submitting.

Download into `checkpoints/` with matching filenames, then evaluate directly:

```bash
python -m src.evaluate --config configs/cnn.yaml --checkpoint checkpoints/cnn_fashion_mnist_best.pt
```

## Option B — Reconstruction from scratch

Every checkpoint above can be regenerated exactly from its config (same
seed, same stratified split):

```bash
pip install -r requirements.txt
bash scripts/prepare_data.sh
python -m src.train --config configs/cnn.yaml
# -> writes checkpoints/cnn_fashion_mnist_best.pt
```

Or reconstruct all five at once:

```bash
bash scripts/train_all.sh
```

| Model | Seed | Approx. training time | Hardware used for reported numbers |
|---|---|---|---|
| Linear | 42 | `[fill in]` | `[fill in]` |
| MLP | 42 | `[fill in]` | `[fill in]` |
| CNN | 42 | `[fill in]` | `[fill in]` |
| LSTM | 42 | `[fill in]` | `[fill in]` |
| Transformer | 42 | `[fill in]` | `[fill in]` |
