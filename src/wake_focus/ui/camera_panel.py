"""
Wake Focus - Camera Panel (500x500)

Displays the live camera feed with overlays:
- Green eye landmark dots
- Green bounding boxes + labels for detected objects
- DPI-aware 2mm alert borders (red for drowsiness, orange for distraction)
"""

import logging

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QWidget

from wake_focus.constants import (
    BORDER_FALLBACK_PX,
    BORDER_THICKNESS_MM,
    CAMERA_PANEL_H,
    CAMERA_PANEL_W,
    DETECTION_BOX_THICKNESS,
    DETECTION_LABEL_FONT_SIZE,
    EYE_LANDMARK_RADIUS,
    LEFT_EYE_CONTOUR,
    LEFT_IRIS,
    RIGHT_EYE_CONTOUR,
    RIGHT_IRIS,
)
from wake_focus.core.alert_state_machine import AlertStatus
from wake_focus.core.perception_engine import PerceptionResult

logger = logging.getLogger(__name__)


class CameraPanel(QFrame):
    """Camera feed display with perception overlays and alert borders."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("cameraPanel")
        self.setFixedSize(CAMERA_PANEL_W, CAMERA_PANEL_H)

        # Current frame and overlay data
        self._pixmap: QPixmap | None = None
        self._perception: PerceptionResult | None = None
        self._alert: AlertStatus = AlertStatus()

        # DPI-aware border thickness
        self._border_px = self._compute_border_px()

        logger.info("CameraPanel: %dx%d, border=%dpx", CAMERA_PANEL_W, CAMERA_PANEL_H, self._border_px)

    def _compute_border_px(self) -> int:
        """Convert 2mm border thickness to pixels using screen DPI."""
        try:
            screen = self.screen()
            if screen:
                dpi = screen.logicalDotsPerInchX()
                # 1 inch = 25.4 mm
                px = int(BORDER_THICKNESS_MM * dpi / 25.4)
                if px >= 2:
                    logger.info("DPI-aware border: %.1fmm → %dpx (DPI=%.1f)", BORDER_THICKNESS_MM, px, dpi)
                    return px
        except Exception:
            pass

        logger.info("Using fallback border thickness: %dpx", BORDER_FALLBACK_PX)
        return BORDER_FALLBACK_PX

    @Slot(np.ndarray)
    def update_frame(self, frame_bgr: np.ndarray, schedule_repaint: bool = True) -> None:
        """Update the displayed camera frame."""
        # Resize frame to panel size
        frame_resized = cv2.resize(frame_bgr, (CAMERA_PANEL_W, CAMERA_PANEL_H))

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Create QImage from numpy array
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)

        if schedule_repaint:
            self.update()  # Trigger repaint

    @Slot(object)
    def update_perception(self, result: PerceptionResult, schedule_repaint: bool = True) -> None:
        """Update perception overlay data."""
        self._perception = result
        if schedule_repaint:
            self.update()

    @Slot(object)
    def update_alert(self, status: AlertStatus, schedule_repaint: bool = True) -> None:
        """Update alert border state."""
        self._alert = status
        if schedule_repaint:
            self.update()

    def present(
        self,
        frame_bgr: np.ndarray,
        result: PerceptionResult,
        status: AlertStatus,
    ) -> None:
        """Update frame and overlays together with a single repaint."""
        self.update_frame(frame_bgr, schedule_repaint=False)
        self.update_perception(result, schedule_repaint=False)
        self.update_alert(status, schedule_repaint=False)
        self.update()

    def paintEvent(self, event) -> None:
        """Custom paint: frame + eye dots + detection boxes + alert border."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the camera frame
        if self._pixmap:
            painter.drawPixmap(0, 0, self._pixmap)
        else:
            # No frame yet — draw placeholder
            painter.fillRect(self.rect(), QColor(10, 14, 23))
            painter.setPen(QColor(100, 116, 139))
            painter.setFont(QFont("Inter", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "📷 Camera Feed")

        # Draw perception overlays
        if self._perception and self._perception.face_detected:
            self._draw_eye_landmarks(painter)
        if self._perception and self._perception.detections:
            self._draw_detections(painter)

        # Draw alert border (INSIDE the panel, on top of everything)
        if self._alert.show_border:
            self._draw_alert_border(painter)

        # Draw FPS indicator
        if self._perception:
            self._draw_fps(painter)

        painter.end()

    def _draw_eye_landmarks(self, painter: QPainter) -> None:
        """Draw green dots at eye contour landmarks."""
        if self._perception is None or self._perception.landmarks_px is None:
            return

        landmarks = self._perception.landmarks_px
        pen = QPen(QColor(0, 255, 0))  # Green
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 255, 0))

        # The current perception pipeline already stores pixel coordinates that match
        # the displayed frame transform closely enough for overlay rendering here.

        all_eye_indices = LEFT_EYE_CONTOUR + RIGHT_EYE_CONTOUR

        # Also add iris points if available
        if len(landmarks) >= 478:
            all_eye_indices.extend(LEFT_IRIS)
            all_eye_indices.extend(RIGHT_IRIS)

        for idx in all_eye_indices:
            if idx < len(landmarks):
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                # Scale from capture frame to panel
                # TODO: The exact scale depends on original frame dims, passed via perception
                # For now draw directly (will be correct if frame is pre-scaled to 500x500 for mesh)
                painter.drawEllipse(x, y, EYE_LANDMARK_RADIUS * 2, EYE_LANDMARK_RADIUS * 2)

    def _draw_detections(self, painter: QPainter) -> None:
        """Draw green bounding boxes and class labels for detected objects."""
        if not self._perception:
            return

        pen = QPen(QColor(0, 255, 0))  # Green
        pen.setWidth(DETECTION_BOX_THICKNESS)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        font = QFont("Inter", DETECTION_LABEL_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)

        for det in self._perception.detections:
            x1, y1, x2, y2 = det.bbox
            # Scale coordinates to panel dimensions
            # (same scaling issue as landmarks — depends on original frame size)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Label with background
            label = f"{det.class_name} {det.confidence:.0%}"
            label_rect = painter.fontMetrics().boundingRect(label)
            label_bg = QRect(x1, y1 - label_rect.height() - 4, label_rect.width() + 8, label_rect.height() + 4)

            painter.fillRect(label_bg, QColor(0, 0, 0, 180))
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(label_bg, Qt.AlignmentFlag.AlignCenter, label)

            # Restore box pen
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(DETECTION_BOX_THICKNESS)
            painter.setPen(pen)

    def _draw_alert_border(self, painter: QPainter) -> None:
        """Draw 2mm inner border for active alerts."""
        r, g, b = self._alert.border_color
        color = QColor(r, g, b)
        pen = QPen(color)
        pen.setWidth(self._border_px)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Draw inside the panel, offset by half pen width
        offset = self._border_px // 2
        painter.drawRect(
            offset,
            offset,
            self.width() - self._border_px,
            self.height() - self._border_px,
        )

    def _draw_fps(self, painter: QPainter) -> None:
        """Draw FPS indicator in the top-left corner."""
        if self._perception and self._perception.process_time_ms > 0:
            fps = 1000.0 / self._perception.process_time_ms
            text = f"FPS: {fps:.1f}"
        else:
            text = "FPS: --"

        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Inter", 10))
        bg_rect = QRect(5, 5, 80, 20)
        painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
        painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)
