"""Transformer classifier.

The image arrives as a sequence of patch tokens from
`src.data.ToRepresentation(mode="patches")`: (B, num_patches, patch_dim).
This module:
  1. Projects each flattened patch to the model dimension (token embedding).
  2. Prepends a learnable [CLS] token.
  3. Adds learnable positional encodings (patch order matters — without this,
     a Transformer is permutation-invariant over patches).
  4. Runs a standard Transformer encoder stack.
  5. Classifies from the final [CLS] token representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Token embedding: project each flattened patch -> d_model
        self.patch_embed = nn.Linear(feature_dim, d_model)

        # 2. Learnable [CLS] token, prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. Learnable positional encoding for [CLS] + all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # 4. Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, feature_dim) — data.py's "patches" representation
        b = x.shape[0]

        tokens = self.patch_embed(x)  # (B, num_patches, d_model)
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, num_patches + 1, d_model)
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)  # (B, num_patches + 1, d_model)
        cls_out = self.norm(encoded[:, 0, :])  # take the [CLS] token
        return self.classifier(cls_out)
