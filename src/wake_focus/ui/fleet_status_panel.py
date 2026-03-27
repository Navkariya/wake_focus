"""
Wake Focus - Fleet Status & Event Log Panel (300x300)

Bottom-middle panel showing:
- List of known fleet devices with status/GPS/alerts
- Incident/event notifications
- Start/Stop monitoring control
- Debug info (FPS, model loaded, GPS lock)
"""

import logging
from datetime import datetime

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from wake_focus.constants import FLEET_STATUS_H, FLEET_STATUS_W
from wake_focus.ui.styles import (
    ACCENT_DANGER,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    TEXT_MUTED,
    TEXT_SECONDARY,
    make_panel_title_html,
)

logger = logging.getLogger(__name__)


class FleetStatusPanel(QFrame):
    """Fleet status, event log, and monitoring controls."""

    # Emitted when user clicks Start/Stop
    monitoring_toggled = Signal(bool)  # True = start, False = stop

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("fleetStatusPanel")
        self.setFixedSize(FLEET_STATUS_W, FLEET_STATUS_H)

        self._is_monitoring = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Title
        title = QLabel()
        title.setText(make_panel_title_html("📡", "Fleet & Events"))
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # ── Start/Stop button ──────────────────────────────────────
        self._start_stop_btn = QPushButton("▶  Start Monitoring")
        self._start_stop_btn.setObjectName("startStopButton")
        self._start_stop_btn.setMinimumHeight(30)
        self._start_stop_btn.clicked.connect(self._toggle_monitoring)
        layout.addWidget(self._start_stop_btn)

        # ── Status indicators ──────────────────────────────────────
        status_row = QHBoxLayout()
        status_row.setSpacing(6)

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        status_row.addWidget(self._fps_label)

        self._gps_label = QLabel("GPS: ○")
        self._gps_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        status_row.addWidget(self._gps_label)

        self._model_label = QLabel("Model: ○")
        self._model_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        status_row.addWidget(self._model_label)

        status_row.addStretch()
        layout.addLayout(status_row)

        # ── Event log list ─────────────────────────────────────────
        log_label = QLabel("Event Log")
        log_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-weight: 600;")
        layout.addWidget(log_label)

        self._event_list = QListWidget()
        self._event_list.setMaximumHeight(100)
        layout.addWidget(self._event_list)

        # ── Fleet devices list ─────────────────────────────────────
        fleet_label = QLabel("Fleet Devices")
        fleet_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-weight: 600;")
        layout.addWidget(fleet_label)

        self._fleet_list = QListWidget()
        self._fleet_list.setMaximumHeight(80)
        layout.addWidget(self._fleet_list)

        layout.addStretch()

        logger.info("FleetStatusPanel initialized: %dx%d", FLEET_STATUS_W, FLEET_STATUS_H)

    def _toggle_monitoring(self) -> None:
        self._is_monitoring = not self._is_monitoring
        if self._is_monitoring:
            self._start_stop_btn.setText("⏹  Stop Monitoring")
            self._start_stop_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(239, 68, 68, 0.2);
                    border-color: rgba(239, 68, 68, 0.4);
                    color: {ACCENT_DANGER};
                    font-weight: 600;
                    border-radius: 6px;
                    padding: 8px 16px;
                    border: 1px solid rgba(239, 68, 68, 0.4);
                }}
            """)
        else:
            self._start_stop_btn.setText("▶  Start Monitoring")
            self._start_stop_btn.setStyleSheet("")  # Reset to default from global stylesheet
            self._start_stop_btn.setObjectName("startStopButton")

        self.monitoring_toggled.emit(self._is_monitoring)

    @Slot(str)
    def add_event(self, message: str) -> None:
        """Add an event to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {message}")
        self._event_list.insertItem(0, item)

        # Keep max 50 events
        while self._event_list.count() > 50:
            self._event_list.takeItem(self._event_list.count() - 1)

    @Slot(float)
    def update_fps(self, fps: float) -> None:
        color = ACCENT_SUCCESS if fps >= 15 else ACCENT_WARNING if fps >= 5 else ACCENT_DANGER
        self._fps_label.setText(f"FPS: {fps:.0f}")
        self._fps_label.setStyleSheet(f"color: {color}; font-size: 10px;")

    @Slot(bool)
    def update_gps_status(self, has_fix: bool) -> None:
        if has_fix:
            self._gps_label.setText("GPS: ●")
            self._gps_label.setStyleSheet(f"color: {ACCENT_SUCCESS}; font-size: 10px;")
        else:
            self._gps_label.setText("GPS: ○")
            self._gps_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")

    @Slot(bool)
    def update_model_status(self, loaded: bool) -> None:
        if loaded:
            self._model_label.setText("Model: ●")
            self._model_label.setStyleSheet(f"color: {ACCENT_SUCCESS}; font-size: 10px;")
        else:
            self._model_label.setText("Model: ○")
            self._model_label.setStyleSheet(f"color: {ACCENT_DANGER}; font-size: 10px;")

    @Slot(dict)
    def update_fleet_devices(self, devices: dict) -> None:
        """Update fleet device list. devices = {id: (name, status, lat, lon)}"""
        self._fleet_list.clear()
        for device_id, (name, status, lat, lon) in devices.items():
            icon = "🟢" if status == "online" else "🔴" if status == "offline" else "🟡"
            item = QListWidgetItem(f"{icon} {name} ({lat:.3f}, {lon:.3f})")
            self._fleet_list.addItem(item)

    @property
    def is_monitoring(self) -> bool:
        return self._is_monitoring
