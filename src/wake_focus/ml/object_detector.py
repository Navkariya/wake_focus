"""
Wake Focus - YOLO26 Object Detector

Wraps a local Ultralytics YOLO26 model for detecting distraction objects,
especially phone/paper proxy/handheld electronic gadget classes.

Supports both PyTorch (.pt) and ONNX (.onnx) model formats
for desktop and edge deployment respectively.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wake_focus.constants import YOLO_HANDHELD_CLASSES, YOLO_HANDHELD_TARGET_ALIASES

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    class_name: str
    class_id: int
    confidence: float


class ObjectDetector:
    """YOLO26 object detection wrapper."""

    # COCO / YOLO26 classes relevant to distraction detection.
    DISTRACTION_COCO_CLASSES = {
        67: "cell phone",
        73: "book",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
    }
    TARGET_CLASS_ALIASES = {
        "cell phone": {"cell phone", "phone", "mobile phone", "smartphone"},
        "book": {"book", "paper", "document", "notebook"},
        "laptop": {"laptop", "computer"},
        "mouse": {"mouse"},
        "remote": {"remote", "remote control"},
        "keyboard": {"keyboard"},
        "electronic gadget": {"cell phone", "laptop", "mouse", "remote", "keyboard"},
    }

    def __init__(
        self,
        model_path: str = "models/yolo26n.onnx",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        target_classes: list[str] | None = None,
        device: str = "cpu",
    ):
        """Initialize YOLO11 detector.

        Args:
            model_path: Path to YOLO model file (.pt or .onnx).
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: NMS IoU threshold.
            imgsz: Inference input image size.
            target_classes: List of class names/aliases to filter. None = use defaults.
            device: Device for inference ('', 'cpu', 'cuda:0').
        """
        self._confidence = confidence_threshold
        self._iou = iou_threshold
        self._imgsz = imgsz
        self._target_classes = target_classes or YOLO_HANDHELD_TARGET_ALIASES
        self._normalized_target_classes = self._expand_target_classes(self._target_classes)
        self._device = device
        self._model = None
        self._model_path = model_path

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            path = Path(model_path)
            if not path.exists():
                logger.warning(
                    "YOLO model file not found at %s. Object detection will stay disabled "
                    "until a local model is provided.",
                    path,
                )
                self._model = None
                return

            self._model = YOLO(str(path), task="detect")
            logger.info(
                "YOLO model loaded: %s (conf=%.2f, iou=%.2f, imgsz=%d)",
                path,
                self._confidence,
                self._iou,
                self._imgsz,
            )
        except ImportError:
            logger.error("ultralytics package not installed. Object detection disabled.")
            self._model = None
        except Exception as e:
            logger.error("Failed to load YOLO model '%s': %s", model_path, e)
            self._model = None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run object detection on a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image.

        Returns:
            List of Detection objects for distraction-relevant classes.
        """
        if self._model is None:
            return []

        try:
            results = self._model.predict(
                frame_bgr,
                conf=self._confidence,
                iou=self._iou,
                imgsz=self._imgsz,
                verbose=False,
                device=self._device if self._device else None,
            )
        except Exception as e:
            logger.error("YOLO inference failed: %s", e)
            return []

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                class_name = result.names.get(class_id, f"class_{class_id}")

                # Filter to target classes
                if not self._is_target_class(class_name, class_id):
                    continue

                # Extract bounding box (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                    )
                )

        return detections

    def _is_target_class(self, class_name: str, class_id: int) -> bool:
        """Check if a detected class is in the configured distraction list."""
        name_lower = class_name.lower().replace("_", " ")
        for target in self._normalized_target_classes:
            if target in name_lower or name_lower in target:
                return True

        # Check by known handheld-distraction COCO IDs.
        if class_id in self.DISTRACTION_COCO_CLASSES:
            return True

        return False

    def _expand_target_classes(self, target_classes: list[str]) -> set[str]:
        """Expand user-facing aliases to actual YOLO26 class names."""
        expanded: set[str] = set()
        for target in target_classes:
            normalized = target.lower().replace("_", " ")
            expanded.add(normalized)
            expanded.update(self.TARGET_CLASS_ALIASES.get(normalized, {normalized}))

        # Keep the canonical class list available even if aliases were minimal.
        expanded.update(YOLO_HANDHELD_CLASSES)
        return expanded

    @property
    def is_available(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._model is not None

    def reload_model(self, model_path: str) -> None:
        """Reload with a different model file."""
        self._model_path = model_path
        self._load_model(model_path)
