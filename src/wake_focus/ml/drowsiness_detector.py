"""
Wake Focus - Drowsiness Detector

Computes Eye Aspect Ratio (EAR) from face landmarks and classifies
normal blinks vs. prolonged eye closure (drowsiness).

EAR Formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

Where p1..p6 are the 6 eye landmarks in order:
    p1 = outer corner, p2 = upper-outer, p3 = upper-inner,
    p4 = inner corner, p5 = lower-inner, p6 = lower-outer.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance as dist

from wake_focus.constants import (
    BLINK_CONSEC_FRAMES,
    DROWSY_CONSEC_FRAMES,
    EAR_THRESHOLD,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)

logger = logging.getLogger(__name__)


@dataclass
class DrowsinessState:
    """Current drowsiness detection state."""

    is_drowsy: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_avg: float = 0.0
    closed_frame_count: int = 0
    is_blinking: bool = False
    blink_count: int = 0


def compute_ear(eye_landmarks: np.ndarray) -> float:
    """Compute Eye Aspect Ratio for 6 landmark points.

    Args:
        eye_landmarks: (6, 2) array of (x, y) pixel coordinates.
            Order: p1(outer), p2(upper-outer), p3(upper-inner),
                   p4(inner), p5(lower-inner), p6(lower-outer).

    Returns:
        EAR value (float). ~0.3 for open eyes, ~0.05 for closed.
    """
    # Vertical distances
    v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])  # p2-p6
    v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])  # p3-p5
    # Horizontal distance
    h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])   # p1-p4

    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


class DrowsinessDetector:
    """Detect drowsiness from eye closure patterns using EAR."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        blink_consec_frames: int = BLINK_CONSEC_FRAMES,
        drowsy_consec_frames: int = DROWSY_CONSEC_FRAMES,
    ):
        self._ear_threshold = ear_threshold
        self._blink_frames = blink_consec_frames
        self._drowsy_frames = drowsy_consec_frames

        self._closed_count = 0
        self._blink_count = 0
        self._was_closed = False

        logger.info(
            "DrowsinessDetector: threshold=%.3f, blink_frames=%d, drowsy_frames=%d",
            ear_threshold,
            blink_consec_frames,
            drowsy_consec_frames,
        )

    def update(self, landmarks_px: np.ndarray) -> DrowsinessState:
        """Update drowsiness state with new face landmarks.

        Args:
            landmarks_px: (N, 2) pixel-space landmarks from FaceMesh.

        Returns:
            DrowsinessState with current readings.
        """
        # Extract 6-point eye landmarks
        left_eye = landmarks_px[LEFT_EYE_INDICES]
        right_eye = landmarks_px[RIGHT_EYE_INDICES]

        # Compute EAR for each eye
        ear_left = compute_ear(left_eye)
        ear_right = compute_ear(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0

        # Check if eyes are closed
        eyes_closed = ear_avg < self._ear_threshold

        if eyes_closed:
            self._closed_count += 1
        else:
            # Eyes are open - check if this was a blink vs. drowsiness
            if self._was_closed and self._closed_count >= self._blink_frames:
                if self._closed_count < self._drowsy_frames:
                    # Normal blink
                    self._blink_count += 1
            self._closed_count = 0

        self._was_closed = eyes_closed

        # Determine drowsiness
        is_drowsy = self._closed_count >= self._drowsy_frames
        is_blinking = (
            eyes_closed
            and self._closed_count >= self._blink_frames
            and self._closed_count < self._drowsy_frames
        )

        return DrowsinessState(
            is_drowsy=is_drowsy,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            closed_frame_count=self._closed_count,
            is_blinking=is_blinking,
            blink_count=self._blink_count,
        )

    def reset(self):
        """Reset internal counters."""
        self._closed_count = 0
        self._blink_count = 0
        self._was_closed = False
