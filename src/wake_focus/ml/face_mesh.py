"""
Wake Focus - MediaPipe Face Mesh Wrapper

Wraps MediaPipe Face Mesh for real-time facial landmark extraction.
Outputs 468 3D face landmarks (plus 10 iris landmarks with refine_landmarks=True).
"""

import logging
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceMeshResult:
    """Result from face mesh processing."""

    landmarks: np.ndarray | None = None  # (478, 3) normalized coords if refined, else (468, 3)
    landmarks_px: np.ndarray | None = None  # (N, 2) pixel coordinates
    face_detected: bool = False
    num_faces: int = 0


class FaceMeshDetector:
    """MediaPipe Face Mesh wrapper for real-time face landmark detection."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._max_faces = max_num_faces
        self._refine = refine_landmarks
        self._mesh = None
        self._available = False

        mp_face_mesh = getattr(getattr(mp, "solutions", None), "face_mesh", None)
        if mp_face_mesh is None:
            logger.warning(
                "MediaPipe Face Mesh API is unavailable in mediapipe %s; "
                "face landmark detection is disabled until a tasks-based model "
                "asset is configured.",
                getattr(mp, "__version__", "unknown"),
            )
            return

        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._available = True
        logger.info(
            "FaceMesh initialized: max_faces=%d, refine=%s, det_conf=%.2f, track_conf=%.2f",
            max_num_faces,
            refine_landmarks,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> FaceMeshResult:
        """Process a BGR frame and return face mesh landmarks.

        Args:
            frame_bgr: OpenCV BGR image (H, W, 3).

        Returns:
            FaceMeshResult with normalized and pixel-space landmarks.
        """
        if self._mesh is None:
            return FaceMeshResult(face_detected=False, num_faces=0)

        h, w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Pass by reference optimization

        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return FaceMeshResult(face_detected=False, num_faces=0)

        # Take the first face
        face_landmarks = results.multi_face_landmarks[0]
        num_landmarks = len(face_landmarks.landmark)

        # Extract normalized coordinates (x, y, z) in [0, 1]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        # Convert to pixel coordinates (x, y)
        landmarks_px = np.zeros((num_landmarks, 2), dtype=np.int32)
        landmarks_px[:, 0] = (landmarks[:, 0] * w).astype(np.int32)
        landmarks_px[:, 1] = (landmarks[:, 1] * h).astype(np.int32)

        return FaceMeshResult(
            landmarks=landmarks,
            landmarks_px=landmarks_px,
            face_detected=True,
            num_faces=len(results.multi_face_landmarks),
        )

    def close(self):
        """Release resources."""
        if self._mesh is not None:
            self._mesh.close()
            logger.info("FaceMesh resources released")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
