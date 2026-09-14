"""Fast sanity tests — run before any real training:

    pytest tests/

These don't download any dataset; they just check that each of the five
mandatory models builds and produces the right output shape for a given
input representation. If a new developer's environment can't pass these,
something is wrong before they even get to training.
"""
import torch

from src.data import ToRepresentation, representation_input_dim
from src.models import build_model

IMAGE_SIZE = 28
CHANNELS = 1
NUM_CLASSES = 10
PATCH_SIZE = 4
BATCH = 4


def _random_batch(mode: str):
    imgs = torch.rand(BATCH, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    rep = ToRepresentation(mode=mode, patch_size=PATCH_SIZE)
    batch = torch.stack([rep(img) for img in imgs])
    shape = representation_input_dim(mode, IMAGE_SIZE, CHANNELS, PATCH_SIZE)
    return batch, shape


def test_linear_forward():
    x, shape = _random_batch("flatten")
    model = build_model({"name": "linear", "params": {}}, shape, NUM_CLASSES)
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_mlp_forward():
    x, shape = _random_batch("flatten")
    model = build_model(
        {"name": "mlp", "params": {"hidden_dims": [64, 32], "dropout": 0.1}}, shape, NUM_CLASSES
    )
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_cnn_forward():
    x, shape = _random_batch("image")
    model = build_model(
        {"name": "cnn", "params": {"base_channels": 8, "dropout": 0.1}}, shape, NUM_CLASSES
    )
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_rnn_forward_lstm():
    x, shape = _random_batch("seq_rows")
    model = build_model(
        {"name": "rnn", "params": {"cell_type": "lstm", "hidden_dim": 32, "num_layers": 1}},
        shape,
        NUM_CLASSES,
    )
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_rnn_forward_gru():
    x, shape = _random_batch("seq_cols")
    model = build_model(
        {"name": "rnn", "params": {"cell_type": "gru", "hidden_dim": 32, "num_layers": 1}},
        shape,
        NUM_CLASSES,
    )
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_transformer_forward():
    x, shape = _random_batch("patches")
    model = build_model(
        {
            "name": "transformer",
            "params": {"d_model": 32, "num_heads": 4, "num_layers": 2, "dropout": 0.1},
        },
        shape,
        NUM_CLASSES,
    )
    out = model(x)
    assert out.shape == (BATCH, NUM_CLASSES)


def test_cnn_forward_cifar_size():
    """The CNN's dynamic flatten-dim calculation must also work for 32x32 (CIFAR-10)."""
    imgs = torch.rand(BATCH, 3, 32, 32)
    rep = ToRepresentation(mode="image")
    batch = torch.stack([rep(img) for img in imgs])
    shape = representation_input_dim("image", 32, 3)
    model = build_model({"name": "cnn", "params": {"base_channels": 8}}, shape, NUM_CLASSES)
    out = model(batch)
    assert out.shape == (BATCH, NUM_CLASSES)
