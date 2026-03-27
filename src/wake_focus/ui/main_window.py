"""
Wake Focus - Main Window (800x800)

Pixel-precise layout:
  Top:    [Camera 500x500] [Map 300x500]
  Bottom: [Stats 400x300]  [Fleet 300x300] [Buttons 100x300]

Uses absolute positioning for exact pixel control.
"""

import logging

from PySide6.QtWidgets import QMainWindow, QWidget

from wake_focus.constants import WINDOW_HEIGHT, WINDOW_WIDTH
from wake_focus.ui.button_panel import ButtonPanel
from wake_focus.ui.camera_panel import CameraPanel
from wake_focus.ui.fleet_status_panel import FleetStatusPanel
from wake_focus.ui.map_panel import MapPanel
from wake_focus.ui.styles import GLOBAL_STYLESHEET
from wake_focus.ui.vehicle_stats_panel import VehicleStatsPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Wake Focus main application window — 800x800 fixed layout."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config

        self.setWindowTitle("Wake Focus — Driver Monitoring System")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # Central widget
        central = QWidget(self)
        self.setCentralWidget(central)

        # ── Create panels with absolute positioning ─────────────────
        # Top-left: Camera Panel (500x500)
        self.camera_panel = CameraPanel(central)
        self.camera_panel.move(0, 0)

        # Top-right: Map Panel (300x500)
        map_center = tuple(config.map_center) if config else (41.311, 69.279)
        map_zoom = config.map_zoom if config else 15
        self.map_panel = MapPanel(
            default_center=map_center,
            default_zoom=map_zoom,
            provider=config.map_provider if config else "yandex",
            yandex_api_key=config.yandex_maps_api_key if config else "",
            yandex_lang=config.yandex_maps_lang if config else "en_US",
            traffic_enabled=config.map_traffic_enabled if config else True,
            auto_follow=config.map_auto_follow if config else True,
            parent=central,
        )
        self.map_panel.move(500, 0)

        # Bottom-left: Vehicle Stats (400x300)
        self.stats_panel = VehicleStatsPanel(central)
        self.stats_panel.move(0, 500)

        # Bottom-middle: Fleet Status (300x300)
        self.fleet_panel = FleetStatusPanel(central)
        self.fleet_panel.move(400, 500)

        # Bottom-right: Button Panel (100x300)
        self.button_panel = ButtonPanel(central)
        self.button_panel.move(700, 500)

        logger.info(
            "MainWindow created: %dx%d, panels positioned",
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )

    def closeEvent(self, event):
        """Clean shutdown on window close."""
        logger.info("MainWindow closing")
        event.accept()
