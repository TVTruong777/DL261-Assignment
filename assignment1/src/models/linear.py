"""Linear / softmax classifier.

Flattens the image and applies a single linear layer to produce logits.
Loss is CrossEntropyLoss, applied to these raw logits directly — softmax is
NOT applied here, since nn.CrossEntropyLoss already combines log-softmax and
NLL loss internally (applying softmax first would be incorrect / double
softmax).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim) — data.py's "flatten" representation already flattens per-sample
        return self.fc(x)
