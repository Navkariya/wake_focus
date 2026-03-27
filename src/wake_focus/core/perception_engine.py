"""
Wake Focus - Perception Engine

Orchestrates all ML/CV models per frame:
1. MediaPipe Face Mesh → landmarks
2. Drowsiness detector → EAR, blink, drowsiness state
3. Head pose estimator → yaw, pitch, roll, road-facing
4. Gaze analyzer → gaze direction, road attention
5. YOLO26 object detector → distraction objects

Outputs a consolidated PerceptionResult that feeds the Alert State Machine.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Signal

from wake_focus.ml.drowsiness_detector import DrowsinessDetector, DrowsinessState
from wake_focus.ml.face_mesh import FaceMeshDetector
from wake_focus.ml.gaze_analyzer import GazeAnalyzer, GazeResult
from wake_focus.ml.head_pose import HeadPose, HeadPoseEstimator
from wake_focus.ml.object_detector import Detection, ObjectDetector

logger = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    """Consolidated result from all perception models."""

    timestamp: float = 0.0
    process_time_ms: float = 0.0

    # Face mesh
    face_detected: bool = False
    landmarks_px: np.ndarray | None = None

    # Drowsiness
    drowsiness: DrowsinessState = field(default_factory=DrowsinessState)

    # Head pose
    head_pose: HeadPose = field(default_factory=HeadPose)

    # Gaze
    gaze: GazeResult = field(default_factory=GazeResult)

    # Object detection
    detections: list[Detection] = field(default_factory=list)
    has_distraction_object: bool = False

    # Composite flags
    is_attending_road: bool = True
    is_holding_distraction: bool = False


class PerceptionEngine(QObject):
    """Orchestrates ML/CV inference pipeline."""

    result_ready = Signal(object)  # PerceptionResult

    def __init__(
        self,
        face_mesh_config: dict | None = None,
        drowsiness_config: dict | None = None,
        head_pose_config: dict | None = None,
        gaze_config: dict | None = None,
        object_detection_config: dict | None = None,
        frame_skip: int = 1,
        object_detection_interval_frames: int = 2,
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        face_mesh_config = face_mesh_config or {}
        drowsiness_config = drowsiness_config or {}
        head_pose_config = head_pose_config or {}
        gaze_config = gaze_config or {}
        object_detection_config = object_detection_config or {}

        # Initialize models
        self._face_mesh = FaceMeshDetector(**face_mesh_config)
        self._drowsiness = DrowsinessDetector(**drowsiness_config)
        self._head_pose = HeadPoseEstimator(**head_pose_config)
        self._gaze = GazeAnalyzer(**gaze_config)
        self._object_detector = ObjectDetector(**object_detection_config)

        self._frame_skip = max(1, frame_skip)
        self._object_detection_interval_frames = max(1, object_detection_interval_frames)
        self._frame_count = 0
        self._processed_frame_count = 0
        self._last_result = PerceptionResult()
        self._last_detections: list[Detection] = []

        logger.info(
            "PerceptionEngine initialized (frame_skip=%d, object_detection_interval=%d)",
            self._frame_skip,
            self._object_detection_interval_frames,
        )

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> PerceptionResult:
        """Process a single frame through all perception models.

        Args:
            frame_bgr: OpenCV BGR image.
            timestamp: Monotonic timestamp of the frame.

        Returns:
            PerceptionResult with all model outputs.
        """
        self._frame_count += 1

        # Frame skip for edge devices
        if self._frame_count % self._frame_skip != 0:
            # Return the last result with updated timestamp
            self._last_result.timestamp = timestamp
            return self._last_result

        self._processed_frame_count += 1

        t_start = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        result = PerceptionResult(timestamp=timestamp)

        # 1. Face Mesh
        mesh_result = self._face_mesh.process(frame_bgr)
        result.face_detected = mesh_result.face_detected
        result.landmarks_px = mesh_result.landmarks_px

        if mesh_result.face_detected and mesh_result.landmarks_px is not None:
            # 2. Drowsiness detection
            result.drowsiness = self._drowsiness.update(mesh_result.landmarks_px)

            # 3. Head pose
            result.head_pose = self._head_pose.estimate(mesh_result.landmarks_px, w, h)

            # 4. Gaze analysis
            result.gaze = self._gaze.analyze(
                mesh_result.landmarks_px,
                head_is_road_facing=result.head_pose.is_road_facing,
            )

        # 5. Object detection is intentionally decimated so the display path can
        # keep up near 30 FPS even on CPU-bound systems.
        if self._processed_frame_count % self._object_detection_interval_frames == 0:
            self._last_detections = self._object_detector.detect(frame_bgr)
        result.detections = list(self._last_detections)
        result.has_distraction_object = len(result.detections) > 0

        # Composite flags
        result.is_attending_road = (
            result.gaze.is_attending_road if result.face_detected else True
        )
        result.is_holding_distraction = (
            result.has_distraction_object and not result.is_attending_road
        )

        # Timing
        result.process_time_ms = (time.perf_counter() - t_start) * 1000

        self._last_result = result
        self.result_ready.emit(result)

        return result

    @property
    def fps_estimate(self) -> float:
        """Estimated processing FPS based on last result."""
        if self._last_result.process_time_ms > 0:
            return 1000.0 / self._last_result.process_time_ms
        return 0.0

    def close(self) -> None:
        """Release all model resources."""
        self._face_mesh.close()
        logger.info("PerceptionEngine closed")
