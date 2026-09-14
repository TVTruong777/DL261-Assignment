"""Tests for the stratified split helper — must preserve per-class proportions
and produce disjoint, seed-reproducible train/val partitions."""
import numpy as np

from src.data import stratified_split_indices


def test_split_is_disjoint_and_covers_all_indices():
    targets = np.array([0] * 100 + [1] * 50 + [2] * 25)
    train_idx, val_idx = stratified_split_indices(targets, val_fraction=0.2, seed=0)

    assert set(train_idx).isdisjoint(set(val_idx))
    assert len(set(train_idx) | set(val_idx)) == len(targets)


def test_split_preserves_class_proportions():
    targets = np.array([0] * 100 + [1] * 50 + [2] * 25)
    train_idx, val_idx = stratified_split_indices(targets, val_fraction=0.2, seed=0)

    for cls, n in [(0, 100), (1, 50), (2, 25)]:
        n_val_cls = int((targets[val_idx] == cls).sum())
        assert n_val_cls == round(n * 0.2)


def test_split_is_reproducible_given_same_seed():
    targets = np.array([0] * 40 + [1] * 40)
    t1, v1 = stratified_split_indices(targets, val_fraction=0.25, seed=7)
    t2, v2 = stratified_split_indices(targets, val_fraction=0.25, seed=7)

    assert (t1 == t2).all()
    assert (v1 == v2).all()


def test_split_differs_with_different_seed():
    targets = np.array([0] * 200 + [1] * 200)
    t1, v1 = stratified_split_indices(targets, val_fraction=0.3, seed=1)
    t2, v2 = stratified_split_indices(targets, val_fraction=0.3, seed=2)

    assert not (v1 == v2).all()
