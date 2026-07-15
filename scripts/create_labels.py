"""
Labeling script for the Banana Ripeness Predictor dataset.

Scans data/raw/ for banana folders, extracts day information from filenames,
and generates labels.csv with the target variable (days_remaining).

Usage:
    python scripts/create_labels.py
    python scripts/create_labels.py --data-dir path/to/data/raw
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


def find_banana_folders(raw_dir: Path) -> list[Path]:
    """Find all banana_XXX folders in the raw data directory."""
    folders = sorted([
        f for f in raw_dir.iterdir()
        if f.is_dir() and re.match(r"banana_\d+", f.name)
    ])
    return folders


def find_images_in_folder(folder: Path) -> list[tuple[Path, int]]:
    """
    Find all banana images in a folder and extract day numbers.

    Returns list of (image_path, day_number) tuples.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []

    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in image_extensions:
            continue

        # Try to extract day number from filename: banana_XXX_dayYY
        match = re.search(r"day(\d+)", f.name, re.IGNORECASE)
        if match:
            day_num = int(match.group(1))
            images.append((f, day_num))
        else:
            print(f"  [WARNING] Could not extract day number from: {f.name}")
            print(f"            Expected format: banana_XXX_dayYY.jpg")

    return images


def create_labels(raw_dir: Path, output_csv: Path) -> pd.DataFrame:
    """
    Scan raw data directory and create labels.csv.

    For each banana, the final day photographed is assumed to be the
    overripe day (day of "death"). Days remaining is calculated as:

        days_remaining = final_day - current_day

    The final day image gets days_remaining = 0.
    """
    banana_folders = find_banana_folders(raw_dir)

    if not banana_folders:
        print(f"\n[ERROR] No banana folders found in {raw_dir}")
        print(f"Expected folders named: banana_001, banana_002, ...")
        print(f"\nPlease collect banana images first. See data/README.md")
        sys.exit(1)

    print(f"\nFound {len(banana_folders)} banana folder(s) in {raw_dir}\n")

    all_records = []

    for folder in banana_folders:
        banana_id = folder.name  # e.g., "banana_001"
        images = find_images_in_folder(folder)

        if not images:
            print(f"  [WARNING] No valid images in {folder.name}, skipping")
            continue

        # Find the final day (assumed overripe / "death" day)
        final_day = max(day for _, day in images)

        print(f"  {banana_id}: {len(images)} images, "
              f"days 0–{final_day}, lifespan = {final_day} days")

        for image_path, day_num in images:
            days_remaining = final_day - day_num

            # Use relative path from project root for portability
            rel_path = image_path.relative_to(raw_dir.parent.parent)

            all_records.append({
                "image_path": str(rel_path).replace("\\", "/"),
                "banana_id": banana_id,
                "day_after_purchase": day_num,
                "total_lifespan_days": final_day,
                "days_remaining": days_remaining,
            })

    if not all_records:
        print("\n[ERROR] No valid images found in any banana folder.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    df = df.sort_values(["banana_id", "day_after_purchase"]).reset_index(drop=True)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Dataset Summary")
    print(f"{'=' * 60}")
    print(f"Total images:       {len(df)}")
    print(f"Total bananas:      {df['banana_id'].nunique()}")
    print(f"Days remaining range: {df['days_remaining'].min():.0f} – "
          f"{df['days_remaining'].max():.0f}")
    print(f"Mean lifespan:      {df.groupby('banana_id')['total_lifespan_days'].first().mean():.1f} days")
    print(f"\nLabels saved to: {output_csv}")

    # Print distribution
    print(f"\nDays remaining distribution:")
    for day in sorted(df["days_remaining"].unique()):
        count = (df["days_remaining"] == day).sum()
        bar = "█" * count
        print(f"  Day {day:>2}: {bar} ({count})")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate labels.csv from banana image folders"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to raw data directory (default: data/raw/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/labels.csv)"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = args.data_dir or (project_root / "data" / "raw")
    output_csv = args.output or (project_root / "data" / "labels.csv")

    print("🍌 Banana Ripeness Predictor — Label Generator")
    print(f"   Raw data dir: {raw_dir}")
    print(f"   Output CSV:   {output_csv}")

    create_labels(raw_dir, output_csv)


if __name__ == "__main__":
    main()
