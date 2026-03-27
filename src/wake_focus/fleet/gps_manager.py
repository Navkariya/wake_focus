"""
Wake Focus - GPS Manager

Pluggable GPS input supporting:
- gpsd (Linux daemon for GPS receivers)
- Serial NMEA (direct serial port reading)
- Simulation (fake GPS data along a route for testing)
"""

import logging
import math
import socket
import time
from abc import ABC, abstractmethod
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from wake_focus.constants import EARTH_RADIUS_KM, GPS_UPDATE_RATE_HZ

logger = logging.getLogger(__name__)


class GPSPosition:
    """GPS position data."""

    __slots__ = ("lat", "lon", "speed_kmh", "heading", "accuracy_m", "timestamp", "has_fix")

    def __init__(
        self,
        lat: float = 0.0,
        lon: float = 0.0,
        speed_kmh: float = 0.0,
        heading: float = 0.0,
        accuracy_m: float = 0.0,
        timestamp: float = 0.0,
        has_fix: bool = False,
    ):
        self.lat = lat
        self.lon = lon
        self.speed_kmh = speed_kmh
        self.heading = heading
        self.accuracy_m = accuracy_m
        self.timestamp = timestamp
        self.has_fix = has_fix

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "speed_kmh": self.speed_kmh,
            "heading": self.heading,
            "accuracy_m": self.accuracy_m,
        }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance between two points in km."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


class GPSSource(ABC):
    """Abstract GPS source interface."""

    @abstractmethod
    def get_position(self) -> Optional[GPSPosition]:
        """Get current GPS position. Returns None if no fix."""
        ...

    def start(self) -> None:
        """Start the GPS source (if needed)."""
        pass

    def stop(self) -> None:
        """Stop the GPS source."""
        pass


class SimulationGPSSource(GPSSource):
    """Simulated GPS source that moves along a predefined route."""

    def __init__(self, waypoints: list[list[float]] | None = None, speed_kmh: float = 40.0):
        self._waypoints = waypoints or [
            [41.311, 69.279],
            [41.312, 69.281],
            [41.314, 69.283],
            [41.316, 69.285],
            [41.318, 69.287],
            [41.320, 69.289],
        ]
        self._speed = speed_kmh
        self._index = 0
        self._progress = 0.0  # 0.0 → 1.0 between waypoints
        self._step = 0.02  # Movement per update

    def get_position(self) -> GPSPosition:
        if len(self._waypoints) < 2:
            return GPSPosition(has_fix=False)

        idx = self._index % (len(self._waypoints) - 1)
        lat1, lon1 = self._waypoints[idx]
        lat2, lon2 = self._waypoints[idx + 1]

        # Interpolate
        lat = lat1 + (lat2 - lat1) * self._progress
        lon = lon1 + (lon2 - lon1) * self._progress

        # Advance
        self._progress += self._step
        if self._progress >= 1.0:
            self._progress = 0.0
            self._index += 1
            if self._index >= len(self._waypoints) - 1:
                self._index = 0  # Loop

        # Compute heading
        heading = math.degrees(math.atan2(lon2 - lon1, lat2 - lat1)) % 360

        return GPSPosition(
            lat=lat,
            lon=lon,
            speed_kmh=self._speed + (self._progress * 5 - 2.5),  # Small variation
            heading=heading,
            accuracy_m=3.0,
            timestamp=time.time(),
            has_fix=True,
        )


class GpsdSource(GPSSource):
    """GPS source using gpsd daemon."""

    def __init__(self, host: str = "localhost", port: int = 2947):
        self._host = host
        self._port = port
        self._socket: Optional[socket.socket] = None

    def start(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._host, self._port))
            self._socket.sendall(b'?WATCH={"enable":true,"json":true}\n')
            logger.info("Connected to gpsd at %s:%d", self._host, self._port)
        except Exception as e:
            logger.error("Failed to connect to gpsd: %s", e)
            self._socket = None

    def get_position(self) -> Optional[GPSPosition]:
        if not self._socket:
            return None

        try:
            import json

            data = self._socket.recv(4096).decode("utf-8", errors="ignore")
            for line in data.strip().split("\n"):
                try:
                    msg = json.loads(line)
                    if msg.get("class") == "TPV":
                        return GPSPosition(
                            lat=msg.get("lat", 0.0),
                            lon=msg.get("lon", 0.0),
                            speed_kmh=msg.get("speed", 0.0) * 3.6,
                            heading=msg.get("track", 0.0),
                            accuracy_m=msg.get("epx", 10.0),
                            timestamp=time.time(),
                            has_fix=msg.get("mode", 0) >= 2,
                        )
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error("gpsd read error: %s", e)

        return None

    def stop(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None


class GPSManager(QObject):
    """Manages GPS input with periodic updates via signals."""

    position_updated = Signal(object)  # GPSPosition
    fix_changed = Signal(bool)

    def __init__(
        self,
        source_type: str = "simulation",
        update_rate_hz: float = GPS_UPDATE_RATE_HZ,
        waypoints: list | None = None,
        gpsd_host: str = "localhost",
        gpsd_port: int = 2947,
        serial_port: str = "/dev/ttyUSB0",
        serial_baud: int = 9600,
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        # Create source
        if source_type == "gpsd":
            self._source = GpsdSource(gpsd_host, gpsd_port)
        elif source_type == "serial":
            # Serial NMEA source would go here
            logger.warning("Serial GPS not fully implemented, falling back to simulation")
            self._source = SimulationGPSSource(waypoints)
        else:
            self._source = SimulationGPSSource(waypoints)

        self._has_fix = False
        self._last_position: Optional[GPSPosition] = None

        # Update timer
        interval_ms = int(1000 / max(update_rate_hz, 0.1))
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._update)

        logger.info(
            "GPSManager: source=%s, update_rate=%.1fHz",
            source_type,
            update_rate_hz,
        )

    def start(self) -> None:
        self._source.start()
        self._timer.start()
        logger.info("GPS manager started")

    def stop(self) -> None:
        self._timer.stop()
        self._source.stop()
        logger.info("GPS manager stopped")

    def _update(self) -> None:
        pos = self._source.get_position()
        if pos:
            if pos.has_fix != self._has_fix:
                self._has_fix = pos.has_fix
                self.fix_changed.emit(pos.has_fix)

            self._last_position = pos
            self.position_updated.emit(pos)

    @property
    def last_position(self) -> Optional[GPSPosition]:
        return self._last_position
