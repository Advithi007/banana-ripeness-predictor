# 📸 Banana Dataset Collection Guide

## Goal

Photograph bananas daily from purchase until they reach the **overripe** condition.
Each banana's lifespan is different — don't assume a fixed number of days.

---

## Definition of "Overripe" (Death)

A banana is considered **overripe** when:
- Brown/black patches cover **more than 50%** of the skin surface
- The banana feels **very soft** when gently squeezed
- The flesh has started to become mushy

> **Important:** Once a banana reaches this state, stop photographing it and record the final day number. That banana is "dead."

---

## Dataset Target

| Metric | Target |
|---|---|
| Number of bananas | **30–40** (minimum 20) |
| Days per banana | **6–10** (varies per banana) |
| Total images | **200–400+** |

More bananas is better than more photos of the same banana.

---

## Photography Guidelines

### Setup
1. Use a **plain, consistent background** (white paper, cutting board, table)
2. Use **natural lighting** or a well-lit room — avoid harsh shadows
3. Photograph from **the same angle** each day (top-down works well)
4. Keep the banana roughly **centered** in the frame
5. Include the **whole banana** in the frame with some margin

### Daily Routine
1. Photograph each banana **once per day**, at roughly the same time
2. Take the photo from **the same position/angle** as previous days
3. Optionally take **1–2 extra angles** for dataset variety
4. Check if the banana has reached the overripe condition

### Naming Convention

```
data/raw/banana_XXX/banana_XXX_dayYY.jpg
```

Where:
- `XXX` = banana ID (001, 002, 003, ...)
- `YY` = day number (00 = day of purchase, 01 = next day, ...)

**Examples:**
```
data/raw/banana_001/banana_001_day00.jpg   ← purchased today (green/yellow)
data/raw/banana_001/banana_001_day01.jpg   ← day 1
data/raw/banana_001/banana_001_day02.jpg   ← day 2
...
data/raw/banana_001/banana_001_day07.jpg   ← overripe → this banana is done
```

---

## Tips for Better Data

- **Start with different ripeness levels.** Buy some green ones and some already-yellow ones
- **Vary the lighting slightly** between batches (but keep it consistent within a banana's series)
- **Don't move the banana** between photos — leave it in the same spot
- **Buy bananas in batches** over several weeks so you're not stuck photographing 30 bananas simultaneously
- **Take photos with your phone** — high-end camera is unnecessary

---

## After Collection

Run the labeling script to generate `labels.csv`:

```bash
python scripts/create_labels.py
```

This will scan your `data/raw/` folder and help you fill in the metadata.
