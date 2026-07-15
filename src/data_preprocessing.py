"""
Image transforms for the Banana Ripeness Predictor.

Provides separate transform pipelines for training (with augmentation)
and evaluation (resize + normalize only). No processed/ copies are created —
transforms are applied on-the-fly during data loading.
"""

from torchvision import transforms

from src.config import (
    AUGMENTATION,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transforms() -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Augmentation is kept mild and realistic:
    - Small rotations (banana orientation may vary)
    - Horizontal flips (bananas can face either direction)
    - Slight brightness/contrast changes (lighting variation)
    - Mild random crops (framing variation)

    We do NOT apply extreme augmentation (e.g., vertical flips,
    large color shifts) that would make bananas look unrealistic.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=AUGMENTATION["random_crop_scale"],
            ratio=(0.9, 1.1),
        ),
        transforms.RandomHorizontalFlip(
            p=AUGMENTATION["horizontal_flip_prob"]
        ),
        transforms.RandomRotation(
            degrees=AUGMENTATION["rotation_degrees"]
        ),
        transforms.ColorJitter(
            brightness=AUGMENTATION["brightness_jitter"],
            contrast=AUGMENTATION["contrast_jitter"],
            saturation=AUGMENTATION["saturation_jitter"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Evaluation transforms (validation and test sets).

    No augmentation — just resize, center crop, and normalize.
    This ensures consistent, deterministic evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Inference transforms for single-image prediction (web app).

    Same as eval transforms — no augmentation.
    """
    return get_eval_transforms()
