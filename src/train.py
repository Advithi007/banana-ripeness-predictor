"""
Training pipeline for the Banana Ripeness Predictor.

Supports training all three model types with:
- MSE loss (regression)
- Adam optimizer with weight decay
- ReduceLROnPlateau scheduler
- Early stopping on validation MAE
- Checkpoint saving (best model)
- Training history logging

Usage:
    python -m src.train --model baseline
    python -m src.train --model frozen
    python -m src.train --model finetuned
    python -m src.train --model finetuned --epochs 50 --lr 1e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BASELINE_EPOCHS,
    BASELINE_LR,
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FROZEN_EPOCHS,
    FROZEN_LR,
    LABELS_CSV,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MODEL_NAMES,
    MODELS_DIR,
    NUM_WORKERS,
    RANDOM_SEED,
    RESULTS_DIR,
    SPLITS_JSON,
    WEIGHT_DECAY,
)
from src.data_preprocessing import get_eval_transforms, get_train_transforms
from src.dataset import BananaDataset
from src.model import count_parameters, create_model
from src.split import (
    create_banana_level_split,
    get_split_dataframes,
    load_splits,
    print_split_summary,
    save_splits,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(predictions - targets).sum().item()
        num_samples += batch_size

    return {
        "loss": total_loss / num_samples,
        "mae": total_mae / num_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = criterion(predictions, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(predictions - targets).sum().item()
        num_samples += batch_size

    return {
        "loss": total_loss / num_samples,
        "mae": total_mae / num_samples,
    }


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_metric: validation MAE (lower is better)

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_metric
            return False

        if val_metric < self.best_score - self.min_delta:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def train(
    model_type: str,
    labels_csv: Path = LABELS_CSV,
    splits_json: Path = SPLITS_JSON,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    device: torch.device = DEVICE,
) -> dict:
    """
    Full training pipeline.

    Args:
        model_type: "baseline", "frozen", or "finetuned"
        labels_csv: path to labels CSV
        splits_json: path to splits JSON (created if not exists)
        epochs: number of training epochs (uses defaults if None)
        lr: learning rate (uses defaults if None)
        batch_size: batch size for data loaders
        seed: random seed
        device: torch device

    Returns:
        Dictionary with training history and final metrics.
    """
    set_seed(seed)

    # --- Resolve hyperparameters ---
    defaults = {
        "baseline": (BASELINE_EPOCHS, BASELINE_LR),
        "frozen": (FROZEN_EPOCHS, FROZEN_LR),
        "finetuned": (FINETUNE_EPOCHS, FINETUNE_LR),
    }
    default_epochs, default_lr = defaults[model_type]
    epochs = epochs or default_epochs
    lr = lr or default_lr
    model_name = MODEL_NAMES[model_type]

    print(f"\n{'=' * 60}")
    print(f"🍌 Training: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {epochs}")
    print(f"  LR:           {lr}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Seed:         {seed}")

    # --- Load data ---
    labels_df = pd.read_csv(labels_csv)

    # Create or load splits
    if splits_json.exists():
        splits = load_splits(splits_json)
        print(f"\n  Loaded existing splits from {splits_json}")
    else:
        splits = create_banana_level_split(labels_df, seed=seed)
        save_splits(splits, splits_json)
        print(f"\n  Created new splits → {splits_json}")

    split_dfs = get_split_dataframes(labels_df, splits)
    print_split_summary(split_dfs, splits)

    # --- Create datasets and loaders ---
    train_dataset = BananaDataset(split_dfs["train"], transform=get_train_transforms())
    val_dataset = BananaDataset(split_dfs["val"], transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"\n  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    # --- Create model ---
    model = create_model(model_type).to(device)
    param_counts = count_parameters(model)
    print(f"\n  Model parameters:")
    print(f"    Total:     {param_counts['total']:,}")
    print(f"    Trainable: {param_counts['trainable']:,}")
    print(f"    Frozen:    {param_counts['frozen']:,}")

    # --- Training setup ---
    criterion = nn.MSELoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=LR_SCHEDULER_PATIENCE,
        factor=LR_SCHEDULER_FACTOR,
        verbose=True,
    )
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # --- Training loop ---
    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "lr": [],
    }
    best_val_mae = float("inf")
    best_epoch = 0
    checkpoint_path = MODELS_DIR / f"{model_name}.pth"

    print(f"\n{'─' * 60}")
    print(f"{'Epoch':>6} │ {'Train Loss':>11} │ {'Train MAE':>10} │ "
          f"{'Val Loss':>9} │ {'Val MAE':>8} │ {'LR':>10}")
    print(f"{'─' * 60}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Get current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Log history
        history["train_loss"].append(train_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["lr"].append(current_lr)

        # Print progress
        marker = ""
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            marker = " ★"
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": best_val_mae,
                "model_type": model_type,
            }, checkpoint_path)

        print(f"{epoch:>6} │ {train_metrics['loss']:>11.4f} │ "
              f"{train_metrics['mae']:>10.4f} │ {val_metrics['loss']:>9.4f} │ "
              f"{val_metrics['mae']:>8.4f} │ {current_lr:>10.2e}{marker}")

        # Learning rate scheduling
        scheduler.step(val_metrics["mae"])

        # Early stopping
        if early_stopping(val_metrics["mae"]):
            print(f"\n  ⏹ Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

    elapsed = time.time() - start_time

    # --- Save training history ---
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"Training Complete: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Best epoch:    {best_epoch}")
    print(f"  Best val MAE:  {best_val_mae:.4f} days")
    print(f"  Training time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"  Checkpoint:    {checkpoint_path}")
    print(f"  History:       {history_path}")

    return {
        "model_type": model_type,
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "training_time_s": elapsed,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a banana ripeness prediction model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["baseline", "frozen", "finetuned"],
        help="Model type to train"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--labels", type=Path, default=LABELS_CSV)
    parser.add_argument("--splits", type=Path, default=SPLITS_JSON)

    args = parser.parse_args()

    train(
        model_type=args.model,
        labels_csv=args.labels,
        splits_json=args.splits,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
