"""Model factory: builds any of the five mandatory architectures from config."""
from __future__ import annotations

from .linear import LinearClassifier
from .mlp import MLPClassifier
from .cnn import CNNClassifier
from .rnn import RNNClassifier
from .transformer import TransformerClassifier

_REGISTRY = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "cnn": CNNClassifier,
    "rnn": RNNClassifier,
    "transformer": TransformerClassifier,
}


def build_model(model_cfg: dict, input_shape: dict, num_classes: int):
    """Instantiate a model from its config block + the data's input_shape.

    `input_shape` comes from `src.data.representation_input_dim(...)` so the
    model's input dimensions always match the representation actually
    produced by the DataLoader for this run.
    """
    name = model_cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Choose from {list(_REGISTRY)}.")
    cls = _REGISTRY[name]
    return cls(num_classes=num_classes, **input_shape, **model_cfg.get("params", {}))
