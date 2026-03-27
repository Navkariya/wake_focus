"""
Wake Focus - Camera Permission Dialog

Shown on app launch to request camera access from the user.
Provides Allow/Deny options with platform-specific guidance if blocked.
"""

import logging
import platform

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from wake_focus.ui.styles import ACCENT_DANGER, ACCENT_PRIMARY, TEXT_SECONDARY

logger = logging.getLogger(__name__)


class PermissionDialog(QDialog):
    """Camera permission request dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Wake Focus — Camera Permission")
        self.setFixedSize(420, 250)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Icon + Title
        title = QLabel("📷  Camera Access Required")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {ACCENT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Wake Focus needs camera access to monitor driver alertness "
            "and detect distractions in real time.\n\n"
            "Your camera feed is processed locally on this device "
            "and is never recorded or transmitted."
        )
        desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; line-height: 1.4;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        deny_btn = QPushButton("Deny")
        deny_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(239, 68, 68, 0.15);
                color: {ACCENT_DANGER};
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: rgba(239, 68, 68, 0.3);
            }}
        """)
        deny_btn.clicked.connect(self.reject)
        btn_layout.addWidget(deny_btn)

        allow_btn = QPushButton("Allow Camera Access")
        allow_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #2563eb;
            }}
        """)
        allow_btn.clicked.connect(self.accept)
        allow_btn.setDefault(True)
        btn_layout.addWidget(allow_btn)

        layout.addLayout(btn_layout)

    @staticmethod
    def get_platform_guidance() -> str:
        """Get platform-specific camera troubleshooting text."""
        system = platform.system()
        if system == "Windows":
            return (
                "On Windows, go to Settings → Privacy & Security → Camera "
                "and ensure camera access is enabled for desktop apps."
            )
        elif system == "Linux":
            return (
                "On Linux, ensure your user is in the 'video' group:\n"
                "  sudo usermod -aG video $USER\n"
                "Then log out and log back in. Also check that /dev/video0 exists."
            )
        elif system == "Darwin":
            return (
                "On macOS, go to System Preferences → Security & Privacy → Camera "
                "and grant access to Wake Focus."
            )
        return "Please check your system's camera privacy settings."


class CameraErrorDialog(QDialog):
    """Shown when camera can't be opened after permission was granted."""

    def __init__(self, guidance: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Error")
        self.setFixedSize(420, 220)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("⚠️  Camera Not Available")
        title.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {ACCENT_DANGER};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        if not guidance:
            guidance = PermissionDialog.get_platform_guidance()

        guide_label = QLabel(guidance)
        guide_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        guide_label.setWordWrap(True)
        layout.addWidget(guide_label)

        layout.addStretch()

        btn_layout = QHBoxLayout()

        retry_btn = QPushButton("Retry")
        retry_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: 600;
            }}
        """)
        retry_btn.clicked.connect(self.accept)
        btn_layout.addWidget(retry_btn)

        continue_btn = QPushButton("Continue Without Camera")
        continue_btn.clicked.connect(self.reject)
        btn_layout.addWidget(continue_btn)

        layout.addLayout(btn_layout)
