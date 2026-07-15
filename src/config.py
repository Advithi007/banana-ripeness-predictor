"""
Central configuration for the Banana Ripeness Predictor.

All hyperparameters, paths, and settings are defined here
so they can be easily modified without touching other code.
"""

import os
from pathlib import Path

# ==============================================================================
# Paths
# ==============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LABELS_CSV = DATA_DIR / "labels.csv"
SPLITS_JSON = DATA_DIR / "splits.json"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Image Settings
# ==============================================================================

IMAGE_SIZE = 224  # ResNet input size
IMAGE_CHANNELS = 3  # RGB

# ImageNet normalization (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ==============================================================================
# Data Augmentation
# ==============================================================================

# Training augmentation parameters
AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 15,          # ±15 degrees (mild, won't distort banana)
    "brightness_jitter": 0.2,        # ±20% brightness variation
    "contrast_jitter": 0.2,          # ±20% contrast variation
    "saturation_jitter": 0.1,        # ±10% saturation variation
    "random_crop_scale": (0.85, 1.0),  # Mild cropping
}

# ==============================================================================
# Dataset Split
# ==============================================================================

# Banana-level split ratios (NOT image-level!)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# Training Hyperparameters
# ==============================================================================

RANDOM_SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 2  # DataLoader workers (set to 0 on Windows if issues occur)

# Baseline CNN
BASELINE_LR = 1e-3
BASELINE_EPOCHS = 100

# ResNet (frozen backbone)
FROZEN_LR = 1e-3  # Higher LR since only training the head
FROZEN_EPOCHS = 100

# ResNet (fine-tuned)
FINETUNE_LR = 1e-4  # Lower LR for fine-tuning pretrained weights
FINETUNE_EPOCHS = 100

# Learning rate scheduler
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Weight decay (regularization)
WEIGHT_DECAY = 1e-4

# ==============================================================================
# Model Architecture
# ==============================================================================

# Dropout rate for regression head
DROPOUT_RATE = 0.3

# Number of unfrozen residual blocks for fine-tuning (from the end)
FINETUNE_UNFREEZE_BLOCKS = 2

# ==============================================================================
# Model Names
# ==============================================================================

MODEL_NAMES = {
    "baseline": "baseline_cnn",
    "frozen": "resnet_frozen",
    "finetuned": "resnet_finetuned",
}

# ==============================================================================
# Device
# ==============================================================================

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
