"""Datasets, stratified splitting, and input representations.

Supports MNIST (debug only, per assignment rules), Fashion-MNIST (primary
dataset for the main comparison), and CIFAR-10 (optional extension).

All five mandatory models consume the *same* underlying (image, label) pairs
and the *same* train/val/test split for a given seed — only the final
`representation` transform differs (flat vector / image / sequence / patches),
which is what the assignment asks us to analyze (inductive bias / data
representation effects), not a difference in the underlying data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

Representation = Literal["flatten", "image", "seq_rows", "seq_cols", "patches"]


# ---------------------------------------------------------------------------
# Representation transforms
# ---------------------------------------------------------------------------

class ToRepresentation:
    """Turns a (C, H, W) image tensor into the representation a model needs.

    - "flatten":  (C*H*W,)                     -> linear / softmax classifier
    - "image":    (C, H, W)                     -> CNN
    - "seq_rows": (H, C*W)                       -> LSTM/GRU, timestep = row
    - "seq_cols": (W, C*H)                       -> LSTM/GRU, timestep = column
    - "patches":  (num_patches, C*patch*patch)   -> Transformer tokens
    """

    def __init__(self, mode: Representation, patch_size: int = 4):
        self.mode = mode
        self.patch_size = patch_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape

        if self.mode == "flatten":
            return img.reshape(-1)

        if self.mode == "image":
            return img

        if self.mode == "seq_rows":
            # (C,H,W) -> (H, C*W): each timestep is one row across channels
            return img.permute(1, 0, 2).reshape(h, c * w)

        if self.mode == "seq_cols":
            # (C,H,W) -> (W, C*H): each timestep is one column across channels
            return img.permute(2, 0, 1).reshape(w, c * h)

        if self.mode == "patches":
            p = self.patch_size
            assert h % p == 0 and w % p == 0, (
                f"Image size {h}x{w} must be divisible by patch_size={p}"
            )
            # unfold into non-overlapping patches, flatten each patch
            patches = img.unfold(1, p, p).unfold(2, p, p)  # (C, H/p, W/p, p, p)
            patches = patches.contiguous().view(c, -1, p, p)  # (C, n_patches, p, p)
            patches = patches.permute(1, 0, 2, 3).reshape(-1, c * p * p)  # (n_patches, C*p*p)
            return patches

        raise ValueError(f"Unknown representation mode: {self.mode}")


def representation_input_dim(mode: Representation, image_size: int, channels: int, patch_size: int = 4) -> dict:
    """Shapes each model constructor needs, derived once from config."""
    if mode == "flatten":
        return {"in_dim": channels * image_size * image_size}
    if mode == "image":
        return {"in_channels": channels, "image_size": image_size}
    if mode == "seq_rows":
        return {"seq_len": image_size, "feature_dim": channels * image_size}
    if mode == "seq_cols":
        return {"seq_len": image_size, "feature_dim": channels * image_size}
    if mode == "patches":
        n_patches = (image_size // patch_size) ** 2
        return {"seq_len": n_patches, "feature_dim": channels * patch_size * patch_size}
    raise ValueError(f"Unknown representation mode: {mode}")


# ---------------------------------------------------------------------------
# Dataset wrapper applying a representation on top of a torchvision dataset
# ---------------------------------------------------------------------------

class RepresentedDataset(Dataset):
    def __init__(self, base: Dataset, representation: ToRepresentation):
        self.base = base
        self.representation = representation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.representation(img), label


# ---------------------------------------------------------------------------
# Stratified train/val split (test split comes from torchvision's official split)
# ---------------------------------------------------------------------------

def stratified_split_indices(targets: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train/val, preserving per-class proportions.

    Using a fixed seed here (from the run config) is what makes the split
    reproducible and identical across all five mandatory models.
    """
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []

    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_fraction))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])

    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "mnist": {
        "cls": datasets.MNIST,
        "channels": 1,
        "image_size": 28,
        "class_names": [str(i) for i in range(10)],
        "note": "Debug/development only — do not report main results on MNIST.",
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "channels": 1,
        "image_size": 28,
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "note": "Primary dataset for the main model comparison.",
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "channels": 3,
        "image_size": 32,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "note": "Optional extension only.",
    },
}


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: list[str]
    input_shape: dict  # from representation_input_dim(), used to construct the model
    split_sizes: dict  # {"train": n, "val": n, "test": n} — report this in the EDA/report


def get_dataloaders(config: dict, data_root: str = "./data") -> DataBundle:
    """Build train/val/test DataLoaders for the dataset + representation in `config`.

    Expected config keys (see configs/*.yaml):
      data.dataset_name: "mnist" | "fashion_mnist" | "cifar10"
      data.val_fraction: float, e.g. 0.1
      data.split_seed: int
      data.representation: "flatten" | "image" | "seq_rows" | "seq_cols" | "patches"
      data.patch_size: int (only used when representation == "patches")
      data.augment: bool
      training.batch_size: int
    """
    data_cfg = config["data"]
    name = data_cfg["dataset_name"]
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset_name '{name}'. Choose from {list(DATASET_INFO)}.")

    info = DATASET_INFO[name]
    normalize_mean = (0.5,) * info["channels"]
    normalize_std = (0.5,) * info["channels"]

    base_transforms = [transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)]

    if data_cfg.get("augment", False):
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(info["image_size"], padding=2),
                transforms.RandomHorizontalFlip() if name == "cifar10" else transforms.Lambda(lambda x: x),
                *base_transforms,
            ]
        )
    else:
        train_transform = transforms.Compose(base_transforms)

    eval_transform = transforms.Compose(base_transforms)

    # Official train/test partition from torchvision; val is carved out of train below.
    train_full = info["cls"](root=data_root, train=True, download=True, transform=train_transform)
    train_full_eval = info["cls"](root=data_root, train=True, download=True, transform=eval_transform)
    test_set = info["cls"](root=data_root, train=False, download=True, transform=eval_transform)

    targets = np.array(train_full.targets)
    train_idx, val_idx = stratified_split_indices(
        targets, val_fraction=data_cfg["val_fraction"], seed=data_cfg["split_seed"]
    )

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(train_full_eval, val_idx)  # no augmentation for validation

    representation = ToRepresentation(
        mode=data_cfg["representation"], patch_size=data_cfg.get("patch_size", 4)
    )

    train_ds = RepresentedDataset(train_subset, representation)
    val_ds = RepresentedDataset(val_subset, representation)
    test_ds = RepresentedDataset(test_set, representation)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    input_shape = representation_input_dim(
        mode=data_cfg["representation"],
        image_size=info["image_size"],
        channels=info["channels"],
        patch_size=data_cfg.get("patch_size", 4),
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(info["class_names"]),
        class_names=info["class_names"],
        input_shape=input_shape,
        split_sizes={"train": len(train_idx), "val": len(val_idx), "test": len(test_set)},
    )
