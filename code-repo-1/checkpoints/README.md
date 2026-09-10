# Checkpoints

## Option A — Hosted download

| Checkpoint | Description | Link | Size |
|---|---|---|---|
| `[expA_best.pt]` | `[e.g., best model by val accuracy, exp A]` | `[Google Drive / HF Hub / release asset link]` | `[e.g., 480MB]` |

> If hosting on Google Drive, set sharing to "Anyone with the link" and verify
> the link works in an incognito/private window before submitting.

## Option B — Reconstruction instructions

If a checkpoint cannot be hosted (too large, storage limits, etc.), give exact
steps to regenerate it from scratch:

```bash
pip install -r ../environment/requirements.txt
bash ../scripts/prepare_data.sh
python ../scripts/train.py --config ../configs/config_expA.yaml
# Expected output: checkpoints/expA_best.pt
# Expected training time on [hardware]: [e.g., ~3.5 hours]
```

Seed used: `[42]` — same run should reproduce reported results within
`[expected variance, e.g., ±0.3%]`.
