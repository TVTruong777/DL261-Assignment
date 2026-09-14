"""LSTM/GRU classifier.

The image is represented as a sequence (rows, columns, or — less commonly —
patches) by `src.data.ToRepresentation`; this module just consumes whatever
(seq_len, feature_dim) sequence it's given and classifies from the final
hidden state. `cell_type` switches between LSTM and GRU (useful for the
optional LSTM-vs-GRU extension) without changing anything else.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        cell_type: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len  # kept for reference/debugging; not required at forward time
        cell_type = cell_type.lower()

        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(cell_type)
        if rnn_cls is None:
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {cell_type!r}")

        self.rnn = rnn_cls(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, feature_dim) — data.py's "seq_rows"/"seq_cols" representation
        output, state = self.rnn(x)
        # Use the last timestep's hidden representation (handles LSTM's (h,c) tuple)
        last_hidden = output[:, -1, :]  # (B, hidden_dim * num_directions)
        return self.classifier(last_hidden)
