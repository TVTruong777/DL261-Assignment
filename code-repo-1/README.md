# Assignment 1 — Foundations of Deep Learning Pipelines and Architectures

CO3133 — Deep Learning and Its Applications — Semester 261
Group [GROUP_ID] — Ho Chi Minh City University of Technology – VNU-HCM

## Links

| Deliverable | Link |
|---|---|
| Assignment page (GitHub Pages) | `[https://<user>.github.io/<repo>/assignmentN.html]` |
| Report | `[link to report PDF/doc]` |
| Slides | `[link to slides]` |
| YouTube presentation | `[link to video, Public or Unlisted]` |
| Dataset Proposal (A2/A3 only) | `[link + approval status, e.g. "Approved on YYYY-MM-DD"]` |
| AI usage disclosure | [`AI_USAGE.md`](./AI_USAGE.md) |

## 1. Installation

```bash
git clone [repo url]
cd [repo folder]

# Option A: pip
pip install -r environment/requirements.txt

# Option B: conda
conda env create -f environment/environment.yml
conda activate [env-name]
```

**Hardware used for the reported results:**
- GPU: `[e.g., 1x NVIDIA T4, 16GB]`
- CPU / RAM: `[e.g., 8 vCPU, 32GB RAM]`
- Approx. training time: `[e.g., 3.5 hours]`

## 2. Dataset preparation

```bash
# [Commands / script to download and prepare the dataset]
python scripts/prepare_data.sh   # or .py — replace with the actual entrypoint
```

- Dataset source: `[name + URL / citation]`
- Expected raw data location: `[path]`
- Output of preparation (splits, format): `[path + description]`
- Train / val / test split sizes and how the split was made: `[describe / seed used]`

## 3. Configuration

All run configs live in [`configs/`](./configs). Each experiment referenced in the
report/results table must correspond to a specific config file (not a config edited
in place and lost).

```bash
configs/
└── config.example.yaml   # copy and edit per experiment, e.g. config_expA.yaml
```

## 4. Seed settings

All reported runs must fix and record seeds for reproducibility:

```yaml
seed: 42          # python, numpy
torch_seed: 42    # or tensorflow_seed
cudnn_deterministic: true
```

State the seed(s) actually used for each reported result in the results table below.

## 5. Train

```bash
python scripts/train.py --config configs/config_expA.yaml
```

## 6. Evaluate

```bash
python scripts/evaluate.py --config configs/config_expA.yaml --checkpoint checkpoints/[name].pt
```

## 7. Dependency versions

See [`environment/requirements.txt`](./environment/requirements.txt) /
[`environment/environment.yml`](./environment/environment.yml) for exact pinned
versions. Key versions to also state here for quick reference:

| Package | Version |
|---|---|
| Python | `[e.g., 3.10.13]` |
| PyTorch / TensorFlow | `[version]` |
| CUDA | `[version]` |
| Other key libs | `[version]` |

## 8. Checkpoints

See [`checkpoints/README.md`](./checkpoints/README.md) for download links or
step-by-step reconstruction instructions (i.e., exact commands to retrain the
checkpoint from scratch if it cannot be hosted).

## 9. Results traceability

Every main reported number must be traceable to a specific run. Fill in
[`results/experiment_log.csv`](./results/experiment_log.csv) and/or the table
below so a grader can go from "the number in the report" back to "the exact run
that produced it."

| Result | Config file | Dataset split | Checkpoint | Log / experiment ID | Commit / tag |
|---|---|---|---|---|---|
| `[e.g., Test accuracy 91.2%]` | `configs/config_expA.yaml` | `[test split name/version]` | `checkpoints/expA_best.pt` | `[e.g., wandb run id / log file]` | `[git commit hash or tag]` |

> A pre-run notebook with results baked in, but no way to reproduce them, is
> **not** an acceptable substitute for the steps above.

## 10. Repository structure

```
.
├── README.md              # this file
├── AI_USAGE.md
├── configs/                # experiment configuration files
├── environment/
│   ├── requirements.txt
│   └── environment.yml
├── src/                    # model / data / training code
├── scripts/
│   ├── prepare_data.sh
│   ├── train.py
│   └── evaluate.py
├── checkpoints/
│   └── README.md           # download links or reconstruction instructions
└── results/
    └── experiment_log.csv  # traceability log for all reported results
```
