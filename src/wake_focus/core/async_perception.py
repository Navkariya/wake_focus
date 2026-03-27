"""
Wake Focus - Asynchronous Perception Processor

Runs the heavy perception pipeline off the GUI thread and always processes only
the latest submitted frame. Older queued frames are dropped intentionally so the
camera view stays smooth even when CPU inference is slower than capture FPS.
"""

import logging
import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from wake_focus.core.perception_engine import PerceptionEngine

logger = logging.getLogger(__name__)


class AsyncPerceptionProcessor(QObject):
    """Processes the latest camera frame on a dedicated Python worker thread."""

    result_ready = Signal(object, object)  # frame_bgr, PerceptionResult

    def __init__(
        self,
        engine: PerceptionEngine | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._engine = engine or PerceptionEngine()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._pending_frame: np.ndarray | None = None
        self._pending_timestamp: float = 0.0
        self._dropped_frames = 0

    def start(self) -> None:
        """Start the background processing thread once."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="WakeFocusPerception",
            daemon=True,
        )
        self._thread.start()
        logger.info("Async perception processor started")

    def submit_frame(self, frame_bgr: np.ndarray, timestamp: float) -> bool:
        """Submit the latest frame for async processing.

        If a frame is already waiting to be processed, it is replaced by the new
        one to keep latency low and visual feedback fresh.
        """
        if not self._running:
            return False

        with self._lock:
            if self._pending_frame is not None:
                self._dropped_frames += 1
            self._pending_frame = frame_bgr
            self._pending_timestamp = timestamp
            self._event.set()
        return True

    def _run_loop(self) -> None:
        while self._running:
            self._event.wait(timeout=0.1)
            if not self._running:
                break

            with self._lock:
                frame = self._pending_frame
                timestamp = self._pending_timestamp
                self._pending_frame = None
                self._event.clear()

            if frame is None:
                continue

            try:
                result = self._engine.process_frame(frame, timestamp)
            except Exception:
                logger.exception("Async perception processing failed")
                continue

            self.result_ready.emit(frame, result)

        logger.info(
            "Async perception processor stopped (dropped_frames=%d)",
            self._dropped_frames,
        )

    @property
    def is_object_detection_available(self) -> bool:
        return self._engine._object_detector.is_available

    @property
    def fps_estimate(self) -> float:
        return self._engine.fps_estimate

    def stop(self) -> None:
        """Stop processing and wait briefly for the thread to exit."""
        if not self._running:
            return
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        """Stop the worker thread and release perception resources."""
        self.stop()
        self._engine.close()
