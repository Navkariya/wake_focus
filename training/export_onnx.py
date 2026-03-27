"""
Wake Focus - ONNX Export for Edge Deployment

Exports trained YOLO26 model to ONNX format for Orange Pi Zero 2W.

Usage:
    python training/export_onnx.py --model runs/train/wake_focus_v1/weights/best.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26 to ONNX for edge deployment")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--imgsz", type=int, default=320, help="Export image size (smaller for edge)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph")
    parser.add_argument("--half", action="store_true", default=False, help="FP16 quantization")
    parser.add_argument("--dynamic", action="store_true", default=False, help="Dynamic batch size")

    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    # Export to ONNX
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        half=args.half,
        dynamic=args.dynamic,
        opset=12,
    )

    print(f"\nONNX model exported to: {export_path}")
    print(f"  Image size: {args.imgsz}x{args.imgsz}")
    print(f"  Simplified: {args.simplify}")
    print(f"  FP16: {args.half}")
    print("\nUsage on Orange Pi:")
    print(f"  Copy {export_path} to the device")
    print("  Set in edge_config.yaml:")
    print(f"    perception.object_detection.model_path: '{export_path}'")
    print(f"    perception.object_detection.imgsz: {args.imgsz}")


if __name__ == "__main__":
    main()
