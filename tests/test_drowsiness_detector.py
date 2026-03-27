"""Tests for the Drowsiness Detector."""

import numpy as np

from wake_focus.ml.drowsiness_detector import DrowsinessDetector, compute_ear


class TestEAR:
    """Test Eye Aspect Ratio computation."""

    def test_open_eye_high_ear(self, sample_landmarks):
        """Open eyes should have EAR > threshold."""
        from wake_focus.constants import LEFT_EYE_INDICES
        left_eye = sample_landmarks[LEFT_EYE_INDICES]
        ear = compute_ear(left_eye)
        assert ear > 0.2, f"Open eye EAR should be > 0.2, got {ear}"

    def test_closed_eye_low_ear(self, closed_eye_landmarks):
        """Closed eyes should have EAR < threshold."""
        from wake_focus.constants import LEFT_EYE_INDICES
        left_eye = closed_eye_landmarks[LEFT_EYE_INDICES]
        ear = compute_ear(left_eye)
        assert ear < 0.15, f"Closed eye EAR should be < 0.15, got {ear}"

    def test_ear_zero_horizontal(self):
        """EAR with zero horizontal distance should return 0."""
        pts = np.array([[0, 0], [0, 5], [0, 5], [0, 0], [0, -5], [0, -5]])
        ear = compute_ear(pts)
        assert ear == 0.0


class TestDrowsinessDetector:
    """Test drowsiness detection logic."""

    def test_normal_state(self, sample_landmarks):
        detector = DrowsinessDetector(drowsy_consec_frames=5)
        state = detector.update(sample_landmarks)
        assert not state.is_drowsy
        assert state.ear_avg > 0

    def test_blink_not_drowsy(self, sample_landmarks, closed_eye_landmarks):
        detector = DrowsinessDetector(
            blink_consec_frames=2,
            drowsy_consec_frames=10,
        )
        # Brief eye closure (3 frames < 10)
        for _ in range(3):
            state = detector.update(closed_eye_landmarks)
        assert not state.is_drowsy

        # Eyes open again
        state = detector.update(sample_landmarks)
        assert not state.is_drowsy

    def test_prolonged_closure_is_drowsy(self, closed_eye_landmarks):
        detector = DrowsinessDetector(drowsy_consec_frames=5)

        for _ in range(6):
            state = detector.update(closed_eye_landmarks)

        assert state.is_drowsy
        assert state.closed_frame_count >= 5

    def test_reset(self, closed_eye_landmarks, sample_landmarks):
        detector = DrowsinessDetector(drowsy_consec_frames=3)

        for _ in range(4):
            detector.update(closed_eye_landmarks)

        detector.reset()
        state = detector.update(sample_landmarks)
        assert not state.is_drowsy
        assert state.closed_frame_count == 0
