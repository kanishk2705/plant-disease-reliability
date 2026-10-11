"""Model A (scratch CNN) and Model B (ResNet-18, ImageNet-1K V1 weights, full fine-tuning)."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class ScratchCNN(nn.Module):
    """Five conv blocks (BN, ReLU, max-pool), global average pooling, dropout, linear head."""

    def __init__(self, num_classes: int, widths=(32, 64, 128, 256, 384), dropout: float = 0.3):
        super().__init__()
        layers, cin = [], 3
        for w in widths:
            layers.append(_block(cin, w)); cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.features(x)).flatten(1)))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "cnn":
        return ScratchCNN(num_classes)
    if name == "resnet18":
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"unknown model {name!r}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
