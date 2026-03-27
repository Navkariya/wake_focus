"""
Wake Focus - Test Fixtures
"""

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_landmarks():
    """Generate fake 478-point face landmarks for testing."""
    rng = np.random.RandomState(42)
    # Generate normalized coords in [0, 1]
    landmarks = rng.rand(478, 3).astype(np.float32)
    landmarks[:, 0] *= 640  # x in pixel space
    landmarks[:, 1] *= 480  # y in pixel space
    landmarks[:, 2] *= 0.1  # z (depth) small range

    # Set realistic eye positions
    # Left eye (indices 33, 160, 158, 133, 153, 144)
    # Make eyes "open" by default (good vertical spacing)
    left_eye_center = np.array([200, 200])
    eye_radius = 15
    # p1 (outer)
    landmarks[33] = [left_eye_center[0] - eye_radius, left_eye_center[1], 0]
    # p2 (upper outer)
    landmarks[160] = [left_eye_center[0] - eye_radius/2, left_eye_center[1] - eye_radius, 0]
    # p3 (upper inner)
    landmarks[158] = [left_eye_center[0] + eye_radius/2, left_eye_center[1] - eye_radius, 0]
    # p4 (inner)
    landmarks[133] = [left_eye_center[0] + eye_radius, left_eye_center[1], 0]
    # p5 (lower inner)
    landmarks[153] = [left_eye_center[0] + eye_radius/2, left_eye_center[1] + eye_radius, 0]
    # p6 (lower outer)
    landmarks[144] = [left_eye_center[0] - eye_radius/2, left_eye_center[1] + eye_radius, 0]

    # Right eye similar
    right_eye_center = np.array([440, 200])
    landmarks[362] = [right_eye_center[0] - eye_radius, right_eye_center[1], 0]
    landmarks[385] = [right_eye_center[0] - eye_radius/2, right_eye_center[1] - eye_radius, 0]
    landmarks[387] = [right_eye_center[0] + eye_radius/2, right_eye_center[1] - eye_radius, 0]
    landmarks[263] = [right_eye_center[0] + eye_radius, right_eye_center[1], 0]
    landmarks[373] = [right_eye_center[0] + eye_radius/2, right_eye_center[1] + eye_radius, 0]
    landmarks[380] = [right_eye_center[0] - eye_radius/2, right_eye_center[1] + eye_radius, 0]

    return landmarks[:, :2].astype(np.int32)  # Return (N, 2) pixel coords


@pytest.fixture
def closed_eye_landmarks(sample_landmarks):
    """Modify landmarks so eyes appear closed (very low EAR)."""
    lm = sample_landmarks.copy()
    # Set upper and lower eye landmarks very close together
    for eye_indices in [(160, 158, 153, 144), (385, 387, 373, 380)]:
        center_y = int(np.mean([lm[i][1] for i in eye_indices]))
        for i in eye_indices:
            lm[i][1] = center_y  # Flatten vertically
    return lm


@pytest.fixture
def sample_frame():
    """Generate a fake camera frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
