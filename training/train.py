"""
Wake Focus - YOLO26 Training Script

Trains a YOLO26 model for distraction object detection.
Compatible with both Linux and Windows (uses __main__ guard).

Usage:
    python training/train.py [--data dataset.yaml] [--model yolo26n.pt] [--epochs 100]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 for Wake Focus distraction detection")
    parser.add_argument("--data", type=str, default="training/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained model to fine-tune")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device: '', '0', 'cpu'")
    parser.add_argument("--project", type=str, default="runs/train", help="Project save directory")
    parser.add_argument("--name", type=str, default="wake_focus_v1", help="Experiment name")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    args = parser.parse_args()

    from ultralytics import YOLO

    # Load pretrained model
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        # Augmentation settings for in-cab scenarios
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Save settings
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\nTraining complete. Results saved to: {args.project}/{args.name}")
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")

    return results


# IMPORTANT: Required for Windows multiprocessing compatibility
if __name__ == "__main__":
    main()
