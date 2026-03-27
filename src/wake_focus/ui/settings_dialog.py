"""
Wake Focus - Settings Dialog

Configuration UI for alert thresholds, camera settings, fleet/MQTT,
GPS source, and other tunable parameters.
"""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from wake_focus.config import Config
from wake_focus.ui.styles import ACCENT_PRIMARY

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Application settings dialog."""

    settings_changed = Signal(dict)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wake Focus — Settings")
        self.setFixedSize(500, 520)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Tab widget
        tabs = QTabWidget()

        # Camera tab
        camera_tab = self._build_camera_tab()
        tabs.addTab(camera_tab, "📷 Camera")

        # Alerts tab
        alerts_tab = self._build_alerts_tab()
        tabs.addTab(alerts_tab, "🔔 Alerts")

        # Fleet tab
        fleet_tab = self._build_fleet_tab()
        tabs.addTab(fleet_tab, "📡 Fleet")

        # GPS tab
        gps_tab = self._build_gps_tab()
        tabs.addTab(gps_tab, "🛰️ GPS")

        layout.addWidget(tabs)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 600;
            }}
        """)
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _build_camera_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(10)

        self._cam_index = QSpinBox()
        self._cam_index.setRange(0, 10)
        self._cam_index.setValue(self._config.camera_index)
        form.addRow("Camera Index:", self._cam_index)

        self._cam_width = QSpinBox()
        self._cam_width.setRange(320, 1920)
        self._cam_width.setValue(self._config.camera_width)
        form.addRow("Width:", self._cam_width)

        self._cam_height = QSpinBox()
        self._cam_height.setRange(240, 1080)
        self._cam_height.setValue(self._config.camera_height)
        form.addRow("Height:", self._cam_height)

        self._cam_fps = QSpinBox()
        self._cam_fps.setRange(5, 60)
        self._cam_fps.setValue(self._config.camera_fps)
        form.addRow("FPS:", self._cam_fps)

        self._frame_skip = QSpinBox()
        self._frame_skip.setRange(1, 10)
        self._frame_skip.setValue(self._config.frame_skip)
        form.addRow("Frame Skip:", self._frame_skip)

        return widget

    def _build_alerts_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(10)

        # Drowsiness
        group_d = QGroupBox("Drowsiness (<<ogohlantirish>>)")
        gf = QFormLayout(group_d)

        self._ear_thresh = QDoubleSpinBox()
        self._ear_thresh.setRange(0.05, 0.50)
        self._ear_thresh.setSingleStep(0.01)
        self._ear_thresh.setDecimals(3)
        self._ear_thresh.setValue(self._config.ear_threshold)
        gf.addRow("EAR Threshold:", self._ear_thresh)

        self._drowsy_frames = QSpinBox()
        self._drowsy_frames.setRange(5, 100)
        self._drowsy_frames.setValue(self._config.drowsy_consec_frames)
        gf.addRow("Drowsy Consec. Frames:", self._drowsy_frames)

        self._drowsy_recovery = QDoubleSpinBox()
        self._drowsy_recovery.setRange(1.0, 10.0)
        self._drowsy_recovery.setValue(self._config.drowsy_recovery_seconds)
        gf.addRow("Recovery Stable (s):", self._drowsy_recovery)

        form.addRow(group_d)

        # Distraction
        group_a = QGroupBox("Distraction (<<ogohlantirish2>>)")
        af = QFormLayout(group_a)

        self._dist_onset = QDoubleSpinBox()
        self._dist_onset.setRange(5.0, 120.0)
        self._dist_onset.setValue(self._config.distraction_onset_seconds)
        af.addRow("Onset Time (s):", self._dist_onset)

        self._dist_recovery = QDoubleSpinBox()
        self._dist_recovery.setRange(1.0, 10.0)
        self._dist_recovery.setValue(self._config.distraction_recovery_seconds)
        af.addRow("Recovery Stable (s):", self._dist_recovery)

        self._yaw_thresh = QDoubleSpinBox()
        self._yaw_thresh.setRange(10.0, 90.0)
        self._yaw_thresh.setValue(self._config.yaw_threshold)
        af.addRow("Yaw Threshold (°):", self._yaw_thresh)

        form.addRow(group_a)

        return widget

    def _build_fleet_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(10)

        self._mqtt_host = QLineEdit(self._config.mqtt_host)
        form.addRow("MQTT Broker Host:", self._mqtt_host)

        self._mqtt_port = QSpinBox()
        self._mqtt_port.setRange(1, 65535)
        self._mqtt_port.setValue(self._config.mqtt_port)
        form.addRow("MQTT Broker Port:", self._mqtt_port)

        self._device_name = QLineEdit(self._config.device_name)
        form.addRow("Device Name:", self._device_name)

        self._fleet_group = QLineEdit(self._config.fleet_group)
        form.addRow("Fleet Group:", self._fleet_group)

        self._osrm_url = QLineEdit(self._config.osrm_url)
        form.addRow("OSRM Server URL:", self._osrm_url)

        return widget

    def _build_gps_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(10)

        self._gps_source = QComboBox()
        self._gps_source.addItems(["simulation", "gpsd", "serial"])
        self._gps_source.setCurrentText(self._config.gps_source)
        form.addRow("GPS Source:", self._gps_source)

        self._serial_port = QLineEdit(self._config.gps_serial_port)
        form.addRow("Serial Port:", self._serial_port)

        self._serial_baud = QSpinBox()
        self._serial_baud.setRange(4800, 115200)
        self._serial_baud.setValue(self._config.gps_serial_baud)
        form.addRow("Serial Baud:", self._serial_baud)

        self._fuel_economy = QDoubleSpinBox()
        self._fuel_economy.setRange(1.0, 50.0)
        self._fuel_economy.setValue(self._config.fuel_economy)
        form.addRow("Fuel Economy (L/100km):", self._fuel_economy)

        return widget

    def _save(self) -> None:
        """Collect all settings and emit signal."""
        settings = {
            "camera.device_index": self._cam_index.value(),
            "camera.width": self._cam_width.value(),
            "camera.height": self._cam_height.value(),
            "camera.fps": self._cam_fps.value(),
            "perception.frame_skip": self._frame_skip.value(),
            "alerts.drowsiness.ear_threshold": self._ear_thresh.value(),
            "alerts.drowsiness.drowsy_consec_frames": self._drowsy_frames.value(),
            "alerts.drowsiness.recovery_stable_seconds": self._drowsy_recovery.value(),
            "alerts.distraction.onset_seconds": self._dist_onset.value(),
            "alerts.distraction.recovery_stable_seconds": self._dist_recovery.value(),
            "alerts.distraction.yaw_threshold": self._yaw_thresh.value(),
            "fleet.mqtt.broker_host": self._mqtt_host.text(),
            "fleet.mqtt.broker_port": self._mqtt_port.value(),
            "fleet.device_name": self._device_name.text(),
            "profile.fleet_group": self._fleet_group.text(),
            "routing.osrm_url": self._osrm_url.text(),
            "gps.source": self._gps_source.currentText(),
            "gps.serial_port": self._serial_port.text(),
            "gps.serial_baud": self._serial_baud.value(),
            "vehicle.fuel_economy": self._fuel_economy.value(),
        }
        self.settings_changed.emit(settings)
        self.accept()
        logger.info("Settings saved")
