"""
Wake Focus - Profile Dialog

Driver profile management: name, vehicle ID, fleet group.
"""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from wake_focus.config import Config
from wake_focus.ui.styles import ACCENT_PURPLE, TEXT_SECONDARY

logger = logging.getLogger(__name__)


class ProfileDialog(QDialog):
    """Driver profile dialog."""

    profile_changed = Signal(dict)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wake Focus — Driver Profile")
        self.setFixedSize(400, 300)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header
        title = QLabel("👤  Driver Profile")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {ACCENT_PURPLE};")
        layout.addWidget(title)

        desc = QLabel("Configure your driver identity for fleet communication.")
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        self._name_input = QLineEdit(config.driver_name)
        self._name_input.setPlaceholderText("Enter your name")
        form.addRow("Driver Name:", self._name_input)

        self._vehicle_input = QLineEdit(config.vehicle_id)
        self._vehicle_input.setPlaceholderText("e.g., UZ-01-AB-1234")
        form.addRow("Vehicle ID:", self._vehicle_input)

        self._group_input = QLineEdit(config.fleet_group)
        self._group_input.setPlaceholderText("e.g., fleet-alpha")
        form.addRow("Fleet Group:", self._group_input)

        self._device_name = QLineEdit(config.device_name)
        self._device_name.setPlaceholderText("e.g., WakeFocus-Truck-01")
        form.addRow("Device Name:", self._device_name)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Profile")
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PURPLE};
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

    def _save(self) -> None:
        profile = {
            "profile.driver_name": self._name_input.text(),
            "profile.vehicle_id": self._vehicle_input.text(),
            "profile.fleet_group": self._group_input.text(),
            "fleet.device_name": self._device_name.text(),
        }
        self.profile_changed.emit(profile)
        self.accept()
        logger.info("Profile saved: %s", self._name_input.text())
