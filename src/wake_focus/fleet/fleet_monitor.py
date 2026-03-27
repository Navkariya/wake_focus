"""
Wake Focus - Fleet Monitor

Tracks other fleet devices, their positions and alert states.
"""

import logging
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal

from wake_focus.fleet.telemetry_schema import TelemetryMessage

logger = logging.getLogger(__name__)

DEVICE_TIMEOUT_S = 30.0  # Consider device offline after 30s no update


@dataclass
class FleetDevice:
    """Tracked fleet device."""

    device_id: str = ""
    device_name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    speed_kmh: float = 0.0
    heading: float = 0.0
    alert_state: str = "normal"
    status: str = "online"
    last_update: float = 0.0
    fleet_group: str = "default"


class FleetMonitor(QObject):
    """Monitors other devices in the fleet."""

    fleet_updated = Signal(dict)  # {device_id: FleetDevice}

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._devices: dict[str, FleetDevice] = {}

        # Periodic cleanup timer
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setInterval(10000)  # Every 10s
        self._cleanup_timer.timeout.connect(self._cleanup_stale)
        self._cleanup_timer.start()

    def update_device(self, telemetry: TelemetryMessage) -> None:
        """Update or add a device from telemetry."""
        device = self._devices.get(telemetry.device_id)
        if device is None:
            device = FleetDevice(device_id=telemetry.device_id)
            self._devices[telemetry.device_id] = device
            logger.info("New fleet device discovered: %s (%s)", telemetry.device_name, telemetry.device_id)

        device.device_name = telemetry.device_name
        device.lat = telemetry.position.get("lat", 0.0)
        device.lon = telemetry.position.get("lon", 0.0)
        device.speed_kmh = telemetry.position.get("speed_kmh", 0.0)
        device.heading = telemetry.position.get("heading", 0.0)
        device.alert_state = telemetry.alert_state
        device.status = "online"
        device.last_update = time.time()
        device.fleet_group = telemetry.fleet_group

        self.fleet_updated.emit(self.get_devices_dict())

    def set_device_online(self, device_id: str, device_name: str) -> None:
        if device_id not in self._devices:
            self._devices[device_id] = FleetDevice(
                device_id=device_id,
                device_name=device_name,
                status="online",
                last_update=time.time(),
            )
        else:
            self._devices[device_id].status = "online"
            self._devices[device_id].last_update = time.time()
        self.fleet_updated.emit(self.get_devices_dict())

    def set_device_offline(self, device_id: str) -> None:
        if device_id in self._devices:
            self._devices[device_id].status = "offline"
            self.fleet_updated.emit(self.get_devices_dict())

    def _cleanup_stale(self) -> None:
        """Mark devices as offline if no update received."""
        now = time.time()
        for device in self._devices.values():
            if device.status == "online" and (now - device.last_update) > DEVICE_TIMEOUT_S:
                device.status = "offline"
                logger.info("Device %s marked offline (timeout)", device.device_id)
        self.fleet_updated.emit(self.get_devices_dict())

    def get_devices_dict(self) -> dict:
        """Get fleet devices as dict for UI: {id: (name, status, lat, lon)}"""
        return {
            d.device_id: (d.device_name, d.status, d.lat, d.lon)
            for d in self._devices.values()
        }

    def get_devices_map_dict(self) -> dict:
        """Get fleet devices for map panel: {id: (lat, lon, name, status)}"""
        return {
            d.device_id: (d.lat, d.lon, d.device_name, d.status)
            for d in self._devices.values()
        }

    def get_slow_devices_near(self, lat: float, lon: float, radius_km: float = 0.5) -> list[FleetDevice]:
        """Find devices moving slowly near a position (for congestion detection)."""
        from wake_focus.fleet.gps_manager import haversine_km

        slow = []
        for d in self._devices.values():
            if d.status == "online" and d.speed_kmh < 5.0:
                dist = haversine_km(lat, lon, d.lat, d.lon)
                if dist <= radius_km:
                    slow.append(d)
        return slow
