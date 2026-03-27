"""
Wake Focus - Button Panel (100x300)

Contains exactly: Settings, Profile, Exit buttons.
Vertical layout with premium styled buttons.
"""

import logging

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget

from wake_focus.constants import BUTTON_PANEL_H, BUTTON_PANEL_W

logger = logging.getLogger(__name__)


class ButtonPanel(QFrame):
    """Button panel with Settings, Profile, and Exit."""

    settings_clicked = Signal()
    profile_clicked = Signal()
    exit_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("buttonPanel")
        self.setFixedSize(BUTTON_PANEL_W, BUTTON_PANEL_H)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(10)

        layout.addStretch()

        # Settings button
        self._settings_btn = QPushButton("⚙️\nSettings")
        self._settings_btn.setObjectName("settingsButton")
        self._settings_btn.setMinimumHeight(70)
        self._settings_btn.clicked.connect(self.settings_clicked)
        layout.addWidget(self._settings_btn)

        # Profile button
        self._profile_btn = QPushButton("👤\nProfile")
        self._profile_btn.setObjectName("profileButton")
        self._profile_btn.setMinimumHeight(70)
        self._profile_btn.clicked.connect(self.profile_clicked)
        layout.addWidget(self._profile_btn)

        # Exit button
        self._exit_btn = QPushButton("🚪\nExit")
        self._exit_btn.setObjectName("exitButton")
        self._exit_btn.setMinimumHeight(70)
        self._exit_btn.clicked.connect(self._handle_exit)
        layout.addWidget(self._exit_btn)

        layout.addStretch()

        logger.info("ButtonPanel initialized: %dx%d", BUTTON_PANEL_W, BUTTON_PANEL_H)

    def _handle_exit(self) -> None:
        """Handle exit button click."""
        self.exit_clicked.emit()
