"""Custom Convolutional Neural Network (CNN) classifier.

A small from-scratch CNN (this is the mandatory CNN submission — pretrained
backbones are only allowed as an optional ablation/extension, per the
assignment rules, not as the required CNN).

Architecture: three conv blocks (conv -> BatchNorm -> ReLU -> pool), each
block roughly doubling the channel count and halving the spatial size, then
a small classifier head. Feature-map spatial size after each block is
computed dynamically so this works for both 28x28 (MNIST/Fashion-MNIST) and
32x32 (CIFAR-10) inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # halves H and W
        )

    def forward(self, x):
        return self.block(x)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels * 2)
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4)

        # 3 pooling layers of stride 2 -> spatial size / 8 (rounded down each time)
        final_size = image_size
        for _ in range(3):
            final_size = final_size // 2
        flat_dim = base_channels * 4 * final_size * final_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — data.py's "image" representation
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
