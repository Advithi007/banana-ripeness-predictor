"""
PyTorch Dataset class for the Banana Ripeness Predictor.

Loads banana images and their days_remaining labels from labels.csv.
Transforms are applied on-the-fly (no separate processed/ directory).
"""

import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import PROJECT_ROOT


class BananaDataset(Dataset):
    """
    Custom Dataset for banana ripeness images.

    Each sample returns:
        image: Tensor of shape (3, 224, 224)
        target: float scalar — days remaining until overripe

    Args:
        labels_df: DataFrame with columns [image_path, banana_id,
                   day_after_purchase, total_lifespan_days, days_remaining]
        transform: torchvision transforms to apply to each image
        root_dir: root directory to resolve relative image paths
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        transform: transforms.Compose | None = None,
        root_dir: Path | None = None,
    ):
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.root_dir = root_dir or PROJECT_ROOT

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels_df.iloc[idx]

        # Load image
        image_path = self.root_dir / row["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Target: days remaining as a float tensor
        target = torch.tensor(row["days_remaining"], dtype=torch.float32)

        return image, target

    def get_banana_ids(self) -> list[str]:
        """Return unique banana IDs in this dataset."""
        return self.labels_df["banana_id"].unique().tolist()

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        return {
            "num_images": len(self),
            "num_bananas": self.labels_df["banana_id"].nunique(),
            "days_remaining_mean": self.labels_df["days_remaining"].mean(),
            "days_remaining_std": self.labels_df["days_remaining"].std(),
            "days_remaining_min": self.labels_df["days_remaining"].min(),
            "days_remaining_max": self.labels_df["days_remaining"].max(),
        }
