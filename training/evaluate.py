"""
Wake Focus - YOLO11 Model Evaluation

Evaluates trained model with detailed metrics.

Usage:
    python training/evaluate.py --model runs/train/wake_focus_v1/weights/best.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wake Focus YOLO11 model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, default="training/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])

    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    # Run validation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        plots=True,
        save_json=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print("=" * 60)

    # Per-class results
    if hasattr(metrics.box, "ap_class_index"):
        names = model.names
        print("\nPer-class AP@50:")
        for i, ap in enumerate(metrics.box.ap50):
            class_name = names.get(metrics.box.ap_class_index[i], f"class_{i}")
            print(f"  {class_name:20s}: {ap:.4f}")

    print(f"\nPlots saved to: {model.predictor.save_dir if hasattr(model, 'predictor') else 'runs/'}")


if __name__ == "__main__":
    main()
