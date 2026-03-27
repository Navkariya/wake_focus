"""Tests for PerceptionEngine throttling behavior."""

import numpy as np

from wake_focus.core.perception_engine import Detection, PerceptionEngine


def _fake_mesh_result():
    return type("MeshResult", (), {"face_detected": False, "landmarks_px": None})()


def test_object_detection_runs_at_configured_interval():
    engine = PerceptionEngine(
        object_detection_config={"model_path": "models/does-not-exist.pt"},
        object_detection_interval_frames=2,
    )

    calls = []

    def fake_detect(_frame):
        calls.append(1)
        return [
            Detection(
                bbox=(0, 0, 10, 10),
                class_name="cell phone",
                class_id=67,
                confidence=0.9,
            )
        ]

    engine._object_detector.detect = fake_detect
    engine._face_mesh.process = lambda _frame: _fake_mesh_result()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result1 = engine.process_frame(frame, 1.0)
    result2 = engine.process_frame(frame, 2.0)
    result3 = engine.process_frame(frame, 3.0)
    result4 = engine.process_frame(frame, 4.0)

    assert len(calls) == 2
    assert result1.detections == []
    assert len(result2.detections) == 1
    assert len(result3.detections) == 1
    assert len(result4.detections) == 1


def test_frame_skip_reuses_previous_result():
    engine = PerceptionEngine(
        object_detection_config={"model_path": "models/does-not-exist.pt"},
        frame_skip=2,
        object_detection_interval_frames=1,
    )

    calls = []

    def fake_detect(_frame):
        calls.append(1)
        return []

    engine._object_detector.detect = fake_detect
    engine._face_mesh.process = lambda _frame: _fake_mesh_result()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    first = engine.process_frame(frame, 1.0)
    second = engine.process_frame(frame, 2.0)
    third = engine.process_frame(frame, 3.0)

    assert first.timestamp == 1.0
    assert third.timestamp == 3.0
    assert len(calls) == 1
    assert second is third
