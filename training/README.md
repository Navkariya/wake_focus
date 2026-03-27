# Wake Focus — YOLO26 Training Pipeline

## Overview

This directory contains everything needed to train and fine-tune a YOLO26 model
for detecting distraction objects in driver monitoring scenarios.

## Detection Classes

| # | Class Name | Description |
|---|-----------|-------------|
| 0 | cell_phone | Handheld mobile phone |
| 1 | paper | Paper document, receipt, note |
| 2 | tablet | Tablet device (iPad-size) |
| 3 | food_drink | Food items, cups, bottles |
| 4 | cigarette | Cigarette, vape, smoking device |
| 5 | book | Book, magazine, reading material |
| 6 | handheld_device | Any other handheld electronic |
| 7 | makeup_tool | Makeup brush, mirror, grooming item |
| 8 | wallet | Wallet, purse, card interaction |
| 9 | headphones | Over-ear or in-ear headphones |

## Dataset Strategy

### Source Data
1. **COCO subset**: Extract `cell phone` (class 67) and `book` (class 73) annotations
2. **Open Images V7**: Search for relevant classes and download
3. **Custom collection**: In-cab camera footage (with consent), dashcam datasets
4. **Synthetic augmentation**: Paste distracting objects onto driving scene backgrounds

### Privacy
- All training data must be collected with informed consent
- Faces should be blurred in stored training data
- No personally identifiable information in annotations

### Splits
- Train: 70% | Val: 15% | Test: 15%
- Minimum 500 images per class recommended
- At least 1000 negative examples (no distractions)

## Quick Start

```bash
# 1. Install dependencies
pip install ultralytics

# 2. Prepare dataset (see layout below)

# 3. Train
python training/train.py

# 4. Evaluate
python training/evaluate.py

# 5. Export for edge
python training/export_onnx.py
```

## Default Inference Filter Used In App

The runtime app currently filters YOLO26 COCO classes to these handheld-distraction
targets:

- `cell phone`
- `book` as a proxy for `paper` / `document`
- `laptop`
- `mouse`
- `remote`
- `keyboard`

## Files

| File | Purpose |
|------|---------|
| `dataset.yaml` | YOLO dataset configuration |
| `train.py` | Training script |
| `evaluate.py` | Evaluation with metrics |
| `export_onnx.py` | ONNX export for edge deployment |
| `labeling_guide.md` | Annotation guidelines |
| `class_list.txt` | Class definitions |
