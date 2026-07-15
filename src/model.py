"""
Model architectures for the Banana Ripeness Predictor.

Three models for progressive experimentation:

1. BaselineCNN        — Simple CNN built from scratch (comparison baseline)
2. ResNetFrozen       — Pretrained ResNet-18 with frozen backbone (feature extractor)
3. ResNetFineTuned    — Pretrained ResNet-18 with last 2 blocks unfrozen (fine-tuning)

All models output a single continuous value: predicted days remaining.
This is a REGRESSION problem, not classification.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import DROPOUT_RATE, FINETUNE_UNFREEZE_BLOCKS


# ==============================================================================
# Model 1: Baseline CNN
# ==============================================================================

class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for banana ripeness regression.

    Architecture:
        Conv2d(3→16) → ReLU → MaxPool
        Conv2d(16→32) → ReLU → MaxPool
        Conv2d(32→64) → ReLU → MaxPool
        AdaptiveAvgPool → Flatten
        Linear(64→32) → ReLU → Dropout
        Linear(32→1)

    This provides a lower-bound performance to compare against ResNet.
    """

    def __init__(self, dropout_rate: float = DROPOUT_RATE):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224×224×3 → 112×112×16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112×112×16 → 56×56×32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56×56×32 → 28×28×64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global average pooling → 1×1×64
            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)  # Output shape: (batch_size,)


# ==============================================================================
# Model 2: ResNet Feature Extractor (Frozen Backbone)
# ==============================================================================

class ResNetFrozen(nn.Module):
    """
    Pretrained ResNet-18 with frozen backbone (feature extractor mode).

    All ResNet layers are frozen — only the custom regression head is trained.
    This is faster to train and works well when the dataset is small.

    Architecture:
        ResNet-18 pretrained (all layers frozen)
        → Replace fc with:
          Linear(512→128) → ReLU → Dropout
          Linear(128→1)
    """

    def __init__(self, dropout_rate: float = DROPOUT_RATE):
        super().__init__()

        # Load pretrained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the classification head with a regression head
        num_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x.squeeze(-1)  # Output shape: (batch_size,)


# ==============================================================================
# Model 3: ResNet Fine-Tuned
# ==============================================================================

class ResNetFineTuned(nn.Module):
    """
    Pretrained ResNet-18 with the last N residual blocks unfrozen.

    This allows the model to adapt the learned features to banana-specific
    patterns while keeping early layers (edges, textures) frozen.

    Architecture:
        ResNet-18 pretrained
        → Freeze all layers except last 2 residual blocks
        → Replace fc with regression head (same as frozen)
    """

    def __init__(
        self,
        dropout_rate: float = DROPOUT_RATE,
        unfreeze_blocks: int = FINETUNE_UNFREEZE_BLOCKS,
    ):
        super().__init__()

        # Load pretrained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze ALL parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N residual blocks
        # ResNet-18 has layer1, layer2, layer3, layer4
        layers = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

        # Unfreeze from the end
        for layer in layers[-unfreeze_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Replace the classification head with a regression head
        num_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

        # The new fc layer params are trainable by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x.squeeze(-1)  # Output shape: (batch_size,)


# ==============================================================================
# Model Factory
# ==============================================================================

def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_type: One of "baseline", "frozen", "finetuned"

    Returns:
        Instantiated model (nn.Module)
    """
    model_map = {
        "baseline": BaselineCNN,
        "frozen": ResNetFrozen,
        "finetuned": ResNetFineTuned,
    }

    if model_type not in model_map:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {list(model_map.keys())}"
        )

    return model_map[model_type](**kwargs)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }
