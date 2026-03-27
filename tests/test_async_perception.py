"""Tests for the async latest-frame perception processor."""

import threading
import time

import numpy as np

from wake_focus.core.async_perception import AsyncPerceptionProcessor
from wake_focus.core.perception_engine import PerceptionResult


class _FakeEngine:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[float] = []
        self._object_detector = type("Detector", (), {"is_available": True})()

    def process_frame(self, _frame, timestamp):
        self.calls.append(timestamp)
        if len(self.calls) == 1:
            self.started.set()
            self.release.wait(timeout=1.0)
        else:
            time.sleep(0.01)
        return PerceptionResult(timestamp=timestamp, process_time_ms=10.0)

    @property
    def fps_estimate(self):
        return 100.0

    def close(self):
        return None


def test_async_processor_keeps_only_latest_pending_frame(qtbot):
    engine = _FakeEngine()
    processor = AsyncPerceptionProcessor(engine=engine)

    results: list[float] = []
    processor.result_ready.connect(lambda _frame, result: results.append(result.timestamp))

    processor.start()

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    processor.submit_frame(frame, 1.0)
    assert engine.started.wait(timeout=1.0)

    processor.submit_frame(frame, 2.0)
    processor.submit_frame(frame, 3.0)
    engine.release.set()

    qtbot.waitUntil(lambda: len(results) == 2, timeout=2000)

    assert results == [1.0, 3.0]
    processor.close()

