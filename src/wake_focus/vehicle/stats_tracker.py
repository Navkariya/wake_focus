"""
Wake Focus - Vehicle Stats Tracker

Computes distance traveled and fuel consumed from GPS data.
Supports OBD-II for real vehicle data when available.
"""

import logging
import time
from typing import Optional

from PySide6.QtCore import QObject, Signal

from wake_focus.fleet.gps_manager import GPSPosition, haversine_km

logger = logging.getLogger(__name__)


class StatsTracker(QObject):
    """Tracks trip statistics from GPS and optionally OBD-II."""

    distance_updated = Signal(float)  # total km
    fuel_updated = Signal(float)      # total liters
    speed_updated = Signal(float)     # current km/h
    trip_time_updated = Signal(float) # minutes

    def __init__(
        self,
        fuel_economy_l_per_100km: float = 8.0,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._fuel_economy = fuel_economy_l_per_100km
        self._total_distance_km = 0.0
        self._total_fuel_liters = 0.0
        self._last_position: Optional[GPSPosition] = None
        self._trip_start_time: Optional[float] = None
        self._current_speed = 0.0

        logger.info("StatsTracker: fuel_economy=%.1f L/100km", fuel_economy_l_per_100km)

    def update(self, position: GPSPosition) -> None:
        """Update stats with a new GPS position."""
        if not position.has_fix:
            return

        if self._trip_start_time is None:
            self._trip_start_time = time.time()

        if self._last_position and self._last_position.has_fix:
            # Compute distance from last position
            dist = haversine_km(
                self._last_position.lat,
                self._last_position.lon,
                position.lat,
                position.lon,
            )

            # Filter out GPS noise (>1 km between updates is suspicious)
            if dist < 1.0:
                self._total_distance_km += dist
                # Estimate fuel from distance
                fuel = dist * self._fuel_economy / 100.0
                self._total_fuel_liters += fuel

        self._current_speed = position.speed_kmh
        self._last_position = position

        # Emit updates
        self.distance_updated.emit(self._total_distance_km)
        self.fuel_updated.emit(self._total_fuel_liters)
        self.speed_updated.emit(self._current_speed)

        if self._trip_start_time:
            minutes = (time.time() - self._trip_start_time) / 60.0
            self.trip_time_updated.emit(minutes)

    @property
    def total_distance_km(self) -> float:
        return self._total_distance_km

    @property
    def total_fuel_liters(self) -> float:
        return self._total_fuel_liters

    def reset(self) -> None:
        """Reset trip stats."""
        self._total_distance_km = 0.0
        self._total_fuel_liters = 0.0
        self._last_position = None
        self._trip_start_time = None
        self._current_speed = 0.0
