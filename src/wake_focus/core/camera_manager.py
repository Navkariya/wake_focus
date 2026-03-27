"""
Wake Focus - Camera Manager

Captures frames from the camera in a dedicated thread to avoid blocking the GUI.
Supports OpenCV VideoCapture with cross-platform compatibility.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QObject, QThread, Signal

logger = logging.getLogger(__name__)


class CameraWorker(QObject):
    """Worker that captures frames in a background thread."""

    frame_ready = Signal(np.ndarray, float)  # frame_bgr, timestamp
    error_occurred = Signal(str)
    camera_opened = Signal(bool)

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        super().__init__()
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._mutex = QMutex()

    def start_capture(self) -> None:
        """Open camera and begin capture loop."""
        self._capture = cv2.VideoCapture(self._device_index)

        if not self._capture.isOpened():
            msg = f"Failed to open camera at index {self._device_index}"
            logger.error(msg)
            self.camera_opened.emit(False)
            self.error_occurred.emit(msg)
            return

        # Set resolution and FPS
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera opened: index=%d, requested=%dx%d@%dfps, actual=%dx%d@%.1ffps",
            self._device_index,
            self._width,
            self._height,
            self._fps,
            actual_w,
            actual_h,
            actual_fps,
        )
        self.camera_opened.emit(True)

        self._running = True
        frame_interval = 1.0 / max(self._fps, 1)

        while self._running:
            self._mutex.lock()
            if not self._running:
                self._mutex.unlock()
                break
            self._mutex.unlock()

            ret, frame = self._capture.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(frame_interval)
                continue

            timestamp = time.monotonic()
            self.frame_ready.emit(frame, timestamp)

            # Small sleep to prevent busy-waiting
            time.sleep(max(frame_interval - 0.005, 0.001))

        # Cleanup
        self._release()

    def stop_capture(self) -> None:
        """Stop the capture loop."""
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

    def _release(self) -> None:
        """Release camera resources."""
        if self._capture and self._capture.isOpened():
            self._capture.release()
            logger.info("Camera released")
        self._capture = None


class CameraManager(QObject):
    """Manages camera capture in a background thread.

    Usage:
        manager = CameraManager(config)
        manager.frame_ready.connect(on_frame)
        manager.start()
        # ... later ...
        manager.stop()
    """

    frame_ready = Signal(np.ndarray, float)  # Forwarded from worker
    camera_opened = Signal(bool)
    error_occurred = Signal(str)

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._thread = QThread()
        self._worker = CameraWorker(device_index, width, height, fps)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.start_capture)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.camera_opened.connect(self.camera_opened)
        self._worker.error_occurred.connect(self.error_occurred)

        self._is_running = False

    def start(self) -> None:
        """Start camera capture thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread.start()
        logger.info("Camera manager started")

    def stop(self) -> None:
        """Stop camera capture and clean up thread."""
        if not self._is_running:
            return
        self._worker.stop_capture()
        self._thread.quit()
        self._thread.wait(5000)  # Wait up to 5 seconds
        self._is_running = False
        logger.info("Camera manager stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    def __del__(self):
        self.stop()
