"""
Banana-level train/validation/test split.

CRITICAL: Split is done at the BANANA level, not image level.
This prevents the same banana from appearing in both training and testing,
which would be data leakage (the model would have already "seen" that banana).

Example with 30 bananas:
    Bananas 1–21  → Training  (70%)
    Bananas 22–25 → Validation (15%)
    Bananas 26–30 → Test       (15%)

Usage:
    python -m src.split
    python -m src.split --labels data/labels.csv --output data/splits.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LABELS_CSV,
    RANDOM_SEED,
    SPLITS_JSON,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def create_banana_level_split(
    labels_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """
    Split banana IDs into train/validation/test sets.

    Args:
        labels_df: DataFrame with a 'banana_id' column
        train_ratio: fraction of bananas for training
        val_ratio: fraction of bananas for validation
        test_ratio: fraction of bananas for testing
        seed: random seed for reproducibility

    Returns:
        Dictionary mapping split names to lists of banana IDs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Get unique banana IDs and shuffle
    banana_ids = sorted(labels_df["banana_id"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(banana_ids)

    n_total = len(banana_ids)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    # Test gets the remainder to ensure no IDs are lost
    n_test = n_total - n_train - n_val

    if n_test < 1:
        # If very few bananas, ensure at least 1 in each split
        n_train = max(1, n_total - 2)
        n_val = 1
        n_test = max(1, n_total - n_train - n_val)

    splits = {
        "train": banana_ids[:n_train],
        "val": banana_ids[n_train:n_train + n_val],
        "test": banana_ids[n_train + n_val:],
    }

    return splits


def save_splits(splits: dict[str, list[str]], output_path: Path) -> None:
    """Save splits to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Load splits from a JSON file."""
    with open(splits_path, "r") as f:
        return json.load(f)


def get_split_dataframes(
    labels_df: pd.DataFrame,
    splits: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """
    Split the labels DataFrame according to banana-level splits.

    Returns:
        Dictionary mapping split names to DataFrames.
    """
    split_dfs = {}
    for split_name, banana_ids in splits.items():
        mask = labels_df["banana_id"].isin(banana_ids)
        split_dfs[split_name] = labels_df[mask].reset_index(drop=True)
    return split_dfs


def print_split_summary(
    split_dfs: dict[str, pd.DataFrame],
    splits: dict[str, list[str]],
) -> None:
    """Print a summary of the splits."""
    print(f"\n{'=' * 60}")
    print(f"Banana-Level Split Summary")
    print(f"{'=' * 60}")

    for split_name in ["train", "val", "test"]:
        df = split_dfs[split_name]
        ids = splits[split_name]
        print(f"\n  {split_name.upper()}:")
        print(f"    Bananas: {len(ids)} ({', '.join(ids[:5])}{'...' if len(ids) > 5 else ''})")
        print(f"    Images:  {len(df)}")
        print(f"    Days remaining: {df['days_remaining'].min():.0f} – "
              f"{df['days_remaining'].max():.0f} "
              f"(mean: {df['days_remaining'].mean():.1f})")

    # Verify no overlap
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert train_set.isdisjoint(val_set), "Train/Val overlap detected!"
    assert train_set.isdisjoint(test_set), "Train/Test overlap detected!"
    assert val_set.isdisjoint(test_set), "Val/Test overlap detected!"
    print(f"\n  ✅ No banana overlap between splits (data leakage check passed)")


def main():
    parser = argparse.ArgumentParser(
        description="Create banana-level train/val/test splits"
    )
    parser.add_argument("--labels", type=Path, default=LABELS_CSV)
    parser.add_argument("--output", type=Path, default=SPLITS_JSON)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # Load labels
    if not args.labels.exists():
        print(f"[ERROR] Labels file not found: {args.labels}")
        print(f"Run 'python scripts/create_labels.py' first.")
        sys.exit(1)

    labels_df = pd.read_csv(args.labels)
    print(f"🍌 Loaded {len(labels_df)} images from {args.labels}")

    # Create splits
    splits = create_banana_level_split(labels_df, seed=args.seed)
    split_dfs = get_split_dataframes(labels_df, splits)
    print_split_summary(split_dfs, splits)

    # Save
    save_splits(splits, args.output)
    print(f"\n  Splits saved to: {args.output}")


if __name__ == "__main__":
    main()
