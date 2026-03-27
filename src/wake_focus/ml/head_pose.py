"""
Wake Focus - Head Pose Estimation

Estimates head orientation (yaw, pitch, roll) from facial landmarks
using cv2.solvePnP with a generic 3D face model.

Determines if the driver's head is facing the road (within thresholds)
or turned away (looking at phone, etc.).
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from wake_focus.constants import (
    HEAD_PITCH_THRESHOLD,
    HEAD_POSE_3D_POINTS,
    HEAD_POSE_LANDMARKS,
    HEAD_YAW_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Head pose estimation result."""

    yaw: float = 0.0    # Left/right rotation (degrees). Negative = left, positive = right
    pitch: float = 0.0  # Up/down tilt (degrees). Negative = down, positive = up
    roll: float = 0.0   # Head tilt (degrees)
    is_road_facing: bool = True
    rotation_vector: np.ndarray | None = None
    translation_vector: np.ndarray | None = None


class HeadPoseEstimator:
    """Estimate head pose from facial landmarks using solvePnP."""

    def __init__(
        self,
        yaw_threshold: float = HEAD_YAW_THRESHOLD,
        pitch_threshold: float = HEAD_PITCH_THRESHOLD,
    ):
        self._yaw_threshold = yaw_threshold
        self._pitch_threshold = pitch_threshold

        # 3D model points (generic face in mm)
        self._model_points = np.array(HEAD_POSE_3D_POINTS, dtype=np.float64)

        # Camera matrix will be computed on first frame
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        logger.info(
            "HeadPoseEstimator: yaw_thresh=%.1f°, pitch_thresh=%.1f°",
            yaw_threshold,
            pitch_threshold,
        )

    def _ensure_camera_matrix(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Build approximate camera intrinsic matrix from frame dimensions."""
        if self._camera_matrix is None:
            focal_length = frame_w  # Approximate focal length
            center = (frame_w / 2.0, frame_h / 2.0)
            self._camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        return self._camera_matrix

    def estimate(
        self, landmarks_px: np.ndarray, frame_w: int, frame_h: int
    ) -> HeadPose:
        """Estimate head pose from face landmarks.

        Args:
            landmarks_px: (N, 2) pixel-space landmarks from FaceMesh.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            HeadPose with yaw, pitch, roll and road-facing status.
        """
        camera_matrix = self._ensure_camera_matrix(frame_w, frame_h)

        # Extract the 6 key landmarks for pose estimation
        image_points = np.array(
            [landmarks_px[i].astype(np.float64) for i in HEAD_POSE_LANDMARKS],
            dtype=np.float64,
        )

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self._model_points,
            image_points,
            camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            logger.debug("solvePnP failed")
            return HeadPose()

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Decompose rotation matrix to Euler angles
        # Using the projection matrix approach
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.vstack((proj_matrix, [0, 0, 0, 1]))[:3]
        )

        yaw = euler_angles[1, 0]   # Y-axis rotation
        pitch = euler_angles[0, 0]  # X-axis rotation
        roll = euler_angles[2, 0]   # Z-axis rotation

        # Normalize angles to [-180, 180]
        yaw = self._normalize_angle(yaw)
        pitch = self._normalize_angle(pitch)
        roll = self._normalize_angle(roll)

        # Check if road-facing
        is_road_facing = (
            abs(yaw) <= self._yaw_threshold and abs(pitch) <= self._pitch_threshold
        )

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            is_road_facing=is_road_facing,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-180, 180] range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
