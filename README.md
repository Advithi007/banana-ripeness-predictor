# 🍌 Days to Banana Death

**Image-Based Banana Ripeness Prediction** — A deep learning regression model that predicts how many days remain before a banana becomes overripe.

> Upload a banana photo → Get an estimated remaining lifespan in days.

---

## ✨ Highlights

- **Custom dataset** — Bananas photographed daily from purchase to overripe, with observed (not assumed) lifespans
- **Transfer learning** — ResNet-18 fine-tuned for visual ripeness regression
- **Banana-level data splits** — No data leakage between train/val/test sets
- **Model comparison** — Baseline CNN vs. frozen ResNet vs. fine-tuned ResNet with ablation experiments
- **Regression evaluation** — MAE, RMSE, R² with actual-vs-predicted and residual plots
- **Web application** — Upload an image and get a prediction with a visual ripeness timeline

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ▼
   ResNet-18 Backbone (pretrained on ImageNet)
   Last 2 residual blocks unfrozen
        │
        ▼
   Global Average Pooling
        │
        ▼
   FC(512 → 128) → ReLU → Dropout(0.3)
        │
        ▼
   FC(128 → 1)
        │
        ▼
   Predicted Days Remaining (continuous)
```

---

## 📁 Project Structure

```
banana-ripeness-predictor/
│
├── data/
│   ├── raw/               # Banana image folders (not in git)
│   │   ├── banana_001/
│   │   ├── banana_002/
│   │   └── ...
│   ├── labels.csv          # Image paths + days_remaining targets
│   ├── splits.json         # Banana-level train/val/test assignment
│   └── README.md           # Dataset collection guide
│
├── src/
│   ├── config.py           # Central configuration & hyperparameters
│   ├── data_preprocessing.py  # Image transforms (on-the-fly, no copies)
│   ├── dataset.py          # PyTorch Dataset class
│   ├── split.py            # Banana-level train/val/test splitting
│   ├── model.py            # 3 architectures: Baseline CNN, ResNet frozen, ResNet fine-tuned
│   ├── train.py            # Training loop with early stopping & LR scheduling
│   ├── evaluate.py         # Regression metrics + visualization plots
│   └── predict.py          # Single-image inference
│
├── app/
│   ├── app.py              # Flask web application
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS assets
│
├── scripts/
│   └── create_labels.py    # Dataset labeling helper
│
├── models/                 # Saved model checkpoints (not in git)
├── results/                # Training histories, metrics, plots
├── notebooks/              # EDA and experiment notebooks
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/banana-ripeness-predictor.git
cd banana-ripeness-predictor

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Dataset

See [`data/README.md`](data/README.md) for the photography guide.

```bash
# After collecting images, generate labels
python scripts/create_labels.py
```

### 3. Create Train/Val/Test Splits

```bash
python -m src.split
```

### 4. Train Models

```bash
# Train all three models for comparison
python -m src.train --model baseline
python -m src.train --model frozen
python -m src.train --model finetuned
```

### 5. Evaluate & Compare

```bash
# Evaluate all models and generate comparison
python -m src.evaluate --all
```

### 6. Run Web App

```bash
python -m app.app --model finetuned
# Open http://127.0.0.1:5000
```

---

## 📊 Model Comparison

| Model | MAE (days) | RMSE (days) | R² | Description |
|---|---|---|---|---|
| Baseline CNN | — | — | — | Simple 3-layer CNN from scratch |
| ResNet Frozen | — | — | — | Pretrained ResNet-18, only head trained |
| ResNet Fine-tuned | — | — | — | Last 2 blocks unfrozen + head |

> Results will be populated after training on your dataset.

---

## 📸 Dataset

- **Source:** Self-collected (bananas photographed daily)
- **Size:** 30–40 bananas × 6–10 days each
- **Target:** Days remaining until overripe (continuous regression)
- **Split:** Banana-level (no data leakage)
- **Augmentation:** Horizontal flip, ±15° rotation, brightness/contrast jitter

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Model | ResNet-18 (transfer learning) |
| Data | pandas, Pillow, NumPy |
| Visualization | matplotlib, seaborn |
| Metrics | scikit-learn |
| Web App | Flask |
| Language | Python 3.10+ |

---

## 🎯 Key Design Decisions

1. **Regression, not classification** — Days remaining is a continuous value, so we use MSE loss and regression metrics
2. **Banana-level splitting** — Prevents the same banana appearing in both training and testing (data leakage)
3. **Observed lifespans** — Each banana's "death" day is based on actual observation, not a fixed assumption
4. **Mild augmentation** — Only realistic transforms that don't distort how bananas actually look
5. **Progressive model comparison** — Baseline → frozen → fine-tuned, to justify the architecture choice

---

## 📄 License

MIT License
