"""
Flask web application for the Banana Ripeness Predictor.

Routes:
    GET  /        → Landing page with image upload
    POST /predict → Process uploaded image, return prediction
    GET  /about   → Project info and model details

Usage:
    python -m app.app
    python -m app.app --port 5000 --model finetuned
"""

import argparse
import io
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.predict import BananaPredictor

app = Flask(__name__)

# Global predictor (loaded once at startup)
predictor = None


def get_predictor(model_type: str = "finetuned") -> BananaPredictor:
    """Lazy-load the predictor."""
    global predictor
    if predictor is None:
        predictor = BananaPredictor(model_type=model_type)
    return predictor


@app.route("/")
def index():
    """Landing page with upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Process uploaded banana image and return prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({
            "error": f"Invalid file type: {ext}. "
                     f"Allowed: {', '.join(allowed_extensions)}"
        }), 400

    try:
        # Read image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get prediction
        pred = get_predictor()
        result = pred.predict_from_pil(pil_image)

        return jsonify({
            "success": True,
            "days_remaining": result["days_remaining"],
            "days_remaining_rounded": result["days_remaining_rounded"],
            "model_used": result["model_used"],
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/about")
def about():
    """Project information page."""
    return render_template("about.html")


def main():
    parser = argparse.ArgumentParser(description="Run the Banana Ripeness Predictor web app")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--model", type=str, default="finetuned",
        choices=["baseline", "frozen", "finetuned"],
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Pre-load the model
    global predictor
    predictor = BananaPredictor(model_type=args.model)

    print(f"\n🍌 Banana Ripeness Predictor")
    print(f"   http://{args.host}:{args.port}")
    print(f"   Model: {args.model}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
