"""
Model evaluation for the Banana Ripeness Predictor.

Computes regression metrics (MAE, RMSE, R²) on the test set and
generates visualization plots:
- Actual vs Predicted scatter plot
- Residual (error) distribution plot
- Training curves (loss and MAE over epochs)

Usage:
    python -m src.evaluate --model baseline
    python -m src.evaluate --model frozen
    python -m src.evaluate --model finetuned
    python -m src.evaluate --all  (evaluate all trained models + comparison table)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DEVICE,
    LABELS_CSV,
    MODEL_NAMES,
    MODELS_DIR,
    NUM_WORKERS,
    PLOTS_DIR,
    RESULTS_DIR,
    SPLITS_JSON,
)
from src.data_preprocessing import get_eval_transforms
from src.dataset import BananaDataset
from src.model import create_model
from src.split import get_split_dataframes, load_splits


# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


@torch.no_grad()
def get_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a dataset.

    Returns:
        (actual_values, predicted_values) as numpy arrays
    """
    model.eval()
    all_actuals = []
    all_predictions = []

    for images, targets in dataloader:
        images = images.to(device)
        predictions = model(images)

        all_actuals.extend(targets.numpy())
        all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_actuals), np.array(all_predictions)


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def plot_actual_vs_predicted(
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Create actual vs predicted scatter plot with perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actuals, predictions, alpha=0.6, s=60, edgecolors="w", linewidth=0.5)

    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    margin = (max_val - min_val) * 0.1
    line_range = [min_val - margin, max_val + margin]
    ax.plot(line_range, line_range, "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Days Remaining")
    ax.set_ylabel("Predicted Days Remaining")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_residuals(
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Create residual (error) distribution plot."""
    residuals = predictions - actuals

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs Actual
    axes[0].scatter(actuals, residuals, alpha=0.6, s=60, edgecolors="w", linewidth=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Actual Days Remaining")
    axes[0].set_ylabel("Prediction Error (days)")
    axes[0].set_title(f"Residuals vs Actual — {model_name}")

    # Error distribution histogram
    axes[1].hist(residuals, bins=20, edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Prediction Error (days)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Error Distribution — {model_name}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(
    history: dict,
    model_name: str,
    save_path: Path,
) -> None:
    """Plot training and validation loss/MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"Loss Curves — {model_name}")
    axes[0].legend()

    # MAE curves
    axes[1].plot(epochs, history["train_mae"], label="Train MAE", linewidth=2)
    axes[1].plot(epochs, history["val_mae"], label="Val MAE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (days)")
    axes[1].set_title(f"MAE Curves — {model_name}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(
    comparison: list[dict],
    save_path: Path,
) -> None:
    """Create a bar chart comparing all models."""
    names = [r["model_name"] for r in comparison]
    maes = [r["mae"] for r in comparison]
    rmses = [r["rmse"] for r in comparison]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, maes, width, label="MAE", color="#2196F3")
    bars2 = ax.bar(x + width / 2, rmses, width, label="RMSE", color="#FF9800")

    ax.set_xlabel("Model")
    ax.set_ylabel("Error (days)")
    ax.set_title("Model Comparison — Test Set Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def evaluate_model(
    model_type: str,
    device: torch.device = DEVICE,
    generate_plots: bool = True,
) -> dict:
    """
    Evaluate a trained model on the test set.

    Args:
        model_type: "baseline", "frozen", or "finetuned"
        device: torch device
        generate_plots: whether to generate visualization plots

    Returns:
        Dictionary with metrics and model info.
    """
    model_name = MODEL_NAMES[model_type]
    checkpoint_path = MODELS_DIR / f"{model_name}.pth"
    history_path = RESULTS_DIR / f"{model_name}_history.json"

    if not checkpoint_path.exists():
        print(f"  [SKIP] No checkpoint found: {checkpoint_path}")
        return None

    print(f"\n{'─' * 60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'─' * 60}")

    # Load model
    model = create_model(model_type).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test data
    labels_df = pd.read_csv(LABELS_CSV)
    splits = load_splits(SPLITS_JSON)
    split_dfs = get_split_dataframes(labels_df, splits)

    test_dataset = BananaDataset(split_dfs["test"], transform=get_eval_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"  Test set: {len(test_dataset)} images "
          f"({len(splits['test'])} bananas)")

    # Get predictions
    actuals, predictions = get_predictions(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(actuals, predictions)
    print(f"\n  MAE:  {metrics['mae']:.4f} days")
    print(f"  RMSE: {metrics['rmse']:.4f} days")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Generate plots
    if generate_plots:
        plot_actual_vs_predicted(
            actuals, predictions, model_name,
            PLOTS_DIR / f"{model_name}_actual_vs_predicted.png"
        )
        plot_residuals(
            actuals, predictions, model_name,
            PLOTS_DIR / f"{model_name}_residuals.png"
        )

        # Training curves (if history exists)
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(
                history, model_name,
                PLOTS_DIR / f"{model_name}_training_curves.png"
            )

    result = {
        "model_type": model_type,
        "model_name": model_name,
        "best_epoch": checkpoint.get("epoch", "?"),
        **metrics,
    }

    # Save metrics
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    return result


def evaluate_all(device: torch.device = DEVICE) -> None:
    """Evaluate all trained models and create comparison table + plot."""
    print(f"\n{'=' * 60}")
    print(f"🍌 Evaluating All Models")
    print(f"{'=' * 60}")

    comparison = []
    for model_type in ["baseline", "frozen", "finetuned"]:
        result = evaluate_model(model_type, device=device)
        if result:
            comparison.append(result)

    if not comparison:
        print("\n  [ERROR] No trained models found. Run training first.")
        return

    # Print comparison table
    print(f"\n{'=' * 60}")
    print(f"Model Comparison — Test Set")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} │ {'MAE':>8} │ {'RMSE':>8} │ {'R²':>8} │ {'Best Epoch':>10}")
    print(f"{'─' * 70}")
    for r in comparison:
        print(f"{r['model_name']:<25} │ {r['mae']:>8.4f} │ "
              f"{r['rmse']:>8.4f} │ {r['r2']:>8.4f} │ {r['best_epoch']:>10}")

    # Best model
    best = min(comparison, key=lambda x: x["mae"])
    print(f"\n  🏆 Best model: {best['model_name']} (MAE = {best['mae']:.4f} days)")

    # Save comparison
    comparison_path = RESULTS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Comparison plot
    plot_model_comparison(comparison, PLOTS_DIR / "model_comparison.png")

    print(f"\n  Comparison saved: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate banana ripeness prediction models"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["baseline", "frozen", "finetuned"],
        help="Specific model to evaluate"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all trained models and create comparison"
    )
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    if args.all:
        evaluate_all()
    elif args.model:
        evaluate_model(args.model, generate_plots=not args.no_plots)
    else:
        print("Specify --model <type> or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
