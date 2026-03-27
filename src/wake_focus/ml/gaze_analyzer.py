"""
Wake Focus - Gaze Direction Analyzer

Analyzes iris/eye landmarks to determine where the driver is looking.
Combines with head pose to detect if gaze is oriented toward a held object
rather than the road.
"""

import logging
from dataclasses import dataclass

import numpy as np

from wake_focus.constants import LEFT_EYE_CONTOUR, LEFT_IRIS, RIGHT_EYE_CONTOUR, RIGHT_IRIS

logger = logging.getLogger(__name__)


@dataclass
class GazeResult:
    """Gaze analysis result."""

    # Gaze ratios: 0.0 = looking far left/up, 0.5 = center, 1.0 = looking far right/down
    horizontal_ratio: float = 0.5
    vertical_ratio: float = 0.5
    is_looking_forward: bool = True
    # Combined with head pose
    is_attending_road: bool = True
    # Gaze deviation from center (0 = centered, 1 = max deviation)
    deviation: float = 0.0


class GazeAnalyzer:
    """Analyze gaze direction from iris landmarks relative to eye contour."""

    def __init__(self, forward_threshold: float = 0.35):
        """
        Args:
            forward_threshold: Max deviation from center (0.5) to consider
                "looking forward". Default 0.35 means ratio must be in [0.15, 0.85].
        """
        self._forward_threshold = forward_threshold
        logger.info("GazeAnalyzer: forward_threshold=%.2f", forward_threshold)

    def analyze(
        self,
        landmarks_px: np.ndarray,
        head_is_road_facing: bool = True,
    ) -> GazeResult:
        """Analyze gaze direction from landmarks.

        Args:
            landmarks_px: (N, 2) pixel-space landmarks from FaceMesh (must have iris landmarks).
            head_is_road_facing: Whether head pose indicates road-facing.

        Returns:
            GazeResult with gaze ratios and attention assessment.
        """
        num_landmarks = len(landmarks_px)

        # Check if we have iris landmarks (478 total with refine_landmarks=True)
        if num_landmarks < 478:
            # No iris data; fall back to head pose only
            return GazeResult(
                is_looking_forward=head_is_road_facing,
                is_attending_road=head_is_road_facing,
            )

        # Compute gaze for each eye using iris center vs. eye bounding box
        h_ratio_l, v_ratio_l = self._compute_eye_gaze(
            landmarks_px, LEFT_EYE_CONTOUR, LEFT_IRIS
        )
        h_ratio_r, v_ratio_r = self._compute_eye_gaze(
            landmarks_px, RIGHT_EYE_CONTOUR, RIGHT_IRIS
        )

        # Average both eyes
        h_ratio = (h_ratio_l + h_ratio_r) / 2.0
        v_ratio = (v_ratio_l + v_ratio_r) / 2.0

        # Deviation from center
        h_dev = abs(h_ratio - 0.5)
        v_dev = abs(v_ratio - 0.5)
        deviation = (h_dev + v_dev) / 2.0

        is_looking_forward = deviation < self._forward_threshold

        # Combined assessment: both head and gaze should indicate road attention
        is_attending_road = is_looking_forward and head_is_road_facing

        return GazeResult(
            horizontal_ratio=h_ratio,
            vertical_ratio=v_ratio,
            is_looking_forward=is_looking_forward,
            is_attending_road=is_attending_road,
            deviation=deviation,
        )

    def _compute_eye_gaze(
        self,
        landmarks_px: np.ndarray,
        eye_contour_indices: list[int],
        iris_indices: list[int],
    ) -> tuple[float, float]:
        """Compute gaze ratio for one eye.

        Returns:
            (horizontal_ratio, vertical_ratio) in [0, 1].
        """
        # Eye contour bounding box
        eye_pts = landmarks_px[eye_contour_indices]
        x_min, y_min = eye_pts.min(axis=0)
        x_max, y_max = eye_pts.max(axis=0)

        eye_w = max(x_max - x_min, 1)
        eye_h = max(y_max - y_min, 1)

        # Iris center (average of iris landmarks)
        iris_pts = landmarks_px[iris_indices]
        iris_center = iris_pts.mean(axis=0)

        # Ratio: where iris center is within eye bounding box
        h_ratio = np.clip((iris_center[0] - x_min) / eye_w, 0.0, 1.0)
        v_ratio = np.clip((iris_center[1] - y_min) / eye_h, 0.0, 1.0)

        return float(h_ratio), float(v_ratio)
