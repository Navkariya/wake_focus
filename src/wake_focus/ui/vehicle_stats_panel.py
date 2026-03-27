"""
Wake Focus - Vehicle Stats Panel (400x300)

Displays total distance traveled and fuel consumed.
Uses a modern dashboard-style layout with large stat displays.
"""

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from wake_focus.constants import VEHICLE_STATS_H, VEHICLE_STATS_W
from wake_focus.ui.styles import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    ACCENT_SUCCESS,
    ACCENT_WARNING,
    BG_CARD,
    TEXT_MUTED,
    TEXT_SECONDARY,
    make_panel_title_html,
)

logger = logging.getLogger(__name__)


class StatCard(QFrame):
    """Individual stat display card."""

    def __init__(self, icon: str, label: str, unit: str, color: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.06);
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        # Icon + label row
        header = QLabel(f"{icon} {label}")
        header.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")
        layout.addWidget(header)

        # Value
        self._value_label = QLabel("0.0")
        self._value_label.setStyleSheet(
            f"color: {color}; font-size: 28px; font-weight: 700; letter-spacing: -1px;"
        )
        layout.addWidget(self._value_label)

        # Unit
        unit_label = QLabel(unit)
        unit_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; text-transform: uppercase;")
        layout.addWidget(unit_label)

    def set_value(self, value: str) -> None:
        self._value_label.setText(value)


class VehicleStatsPanel(QFrame):
    """Vehicle stats panel showing distance and fuel consumption."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("vehicleStatsPanel")
        self.setFixedSize(VEHICLE_STATS_W, VEHICLE_STATS_H)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel()
        title.setText(make_panel_title_html("🚗", "Vehicle Stats"))
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # Stats grid
        grid = QGridLayout()
        grid.setSpacing(8)

        # Distance card
        self._distance_card = StatCard("📍", "Distance Traveled", "kilometers", ACCENT_PRIMARY)
        grid.addWidget(self._distance_card, 0, 0)

        # Fuel card
        self._fuel_card = StatCard("⛽", "Fuel Consumed", "liters", ACCENT_WARNING)
        grid.addWidget(self._fuel_card, 0, 1)

        # Speed card
        self._speed_card = StatCard("🏎️", "Current Speed", "km/h", ACCENT_SUCCESS)
        grid.addWidget(self._speed_card, 1, 0)

        # Trip time card
        self._time_card = StatCard("⏱️", "Trip Time", "minutes", ACCENT_SECONDARY)
        grid.addWidget(self._time_card, 1, 1)

        layout.addLayout(grid)
        layout.addStretch()

        # Data source indicator
        self._source_label = QLabel("📊 Source: GPS Estimation")
        self._source_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        self._source_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._source_label)

        logger.info("VehicleStatsPanel initialized: %dx%d", VEHICLE_STATS_W, VEHICLE_STATS_H)

    @Slot(float)
    def update_distance(self, km: float) -> None:
        self._distance_card.set_value(f"{km:.1f}")

    @Slot(float)
    def update_fuel(self, liters: float) -> None:
        self._fuel_card.set_value(f"{liters:.1f}")

    @Slot(float)
    def update_speed(self, kmh: float) -> None:
        self._speed_card.set_value(f"{kmh:.0f}")

    @Slot(float)
    def update_trip_time(self, minutes: float) -> None:
        self._time_card.set_value(f"{minutes:.0f}")

    def set_data_source(self, source: str) -> None:
        self._source_label.setText(f"📊 Source: {source}")
