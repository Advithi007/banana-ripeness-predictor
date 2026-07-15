"""
Single-image prediction for the Banana Ripeness Predictor.

Loads a trained model and predicts days remaining from a single banana image.
Used by the web app and for command-line inference.

Usage:
    python -m src.predict --image path/to/banana.jpg
    python -m src.predict --image path/to/banana.jpg --model finetuned
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.config import DEVICE, MODEL_NAMES, MODELS_DIR
from src.data_preprocessing import get_inference_transforms
from src.model import create_model


class BananaPredictor:
    """
    Loads a trained model and predicts days remaining for a banana image.

    Usage:
        predictor = BananaPredictor(model_type="finetuned")
        result = predictor.predict("path/to/banana.jpg")
        print(result["days_remaining"])
    """

    def __init__(
        self,
        model_type: str = "finetuned",
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.model_type = model_type
        self.model_name = MODEL_NAMES[model_type]
        self.transform = get_inference_transforms()

        # Load model
        checkpoint_path = MODELS_DIR / f"{self.model_name}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {checkpoint_path}. "
                f"Run 'python -m src.train --model {model_type}' first."
            )

        self.model = create_model(model_type).to(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Loaded model: {self.model_name} "
              f"(trained epoch {checkpoint.get('epoch', '?')}, "
              f"val MAE: {checkpoint.get('val_mae', '?'):.4f})")

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """
        Predict days remaining for a single banana image.

        Args:
            image_path: path to the banana image

        Returns:
            Dictionary with prediction results:
            - days_remaining: float (continuous prediction)
            - days_remaining_rounded: int (rounded for display)
            - model_used: str
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        prediction = self.model(image_tensor).item()

        # Clamp to non-negative (can't have negative days remaining)
        prediction = max(0.0, prediction)

        return {
            "days_remaining": round(prediction, 1),
            "days_remaining_rounded": max(0, round(prediction)),
            "model_used": self.model_name,
        }

    @torch.no_grad()
    def predict_from_pil(self, pil_image: Image.Image) -> dict:
        """
        Predict from a PIL Image object (used by the web app).

        Args:
            pil_image: PIL Image in RGB mode

        Returns:
            Same as predict()
        """
        image = pil_image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        prediction = self.model(image_tensor).item()
        prediction = max(0.0, prediction)

        return {
            "days_remaining": round(prediction, 1),
            "days_remaining_rounded": max(0, round(prediction)),
            "model_used": self.model_name,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Predict days remaining for a banana image"
    )
    parser.add_argument(
        "--image", type=Path, required=True,
        help="Path to banana image"
    )
    parser.add_argument(
        "--model", type=str, default="finetuned",
        choices=["baseline", "frozen", "finetuned"],
        help="Model to use for prediction"
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        return

    predictor = BananaPredictor(model_type=args.model)
    result = predictor.predict(args.image)

    print(f"\n🍌 Banana Ripeness Prediction")
    print(f"   Image: {args.image}")
    print(f"   Estimated remaining life: ~{result['days_remaining']} days")
    print(f"   Model: {result['model_used']}")


if __name__ == "__main__":
    main()
