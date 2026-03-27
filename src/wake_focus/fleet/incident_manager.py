"""
Wake Focus - Incident Manager

Detects and manages traffic incidents:
- Explicit accident reports from devices
- Congestion inference from multiple slow devices
- Triggers rerouting for fleet devices
"""

import logging
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal

from wake_focus.constants import (
    CONGESTION_DEVICE_COUNT,
    CONGESTION_DURATION_S,
    CONGESTION_SPEED_THRESHOLD_KMH,
    INCIDENT_AVOIDANCE_RADIUS_M,
)
from wake_focus.fleet.fleet_monitor import FleetMonitor
from wake_focus.fleet.telemetry_schema import IncidentMessage

logger = logging.getLogger(__name__)


@dataclass
class ActiveIncident:
    """An active incident being tracked."""

    incident_id: str
    lat: float
    lon: float
    incident_type: str
    description: str
    severity: str
    reported_at: float
    resolved: bool = False


class IncidentManager(QObject):
    """Manages traffic incidents and congestion detection."""

    incident_detected = Signal(object)  # ActiveIncident
    incident_resolved = Signal(str)  # incident_id
    reroute_suggested = Signal(float, float, float)  # avoid_lat, avoid_lon, avoid_radius_km

    def __init__(
        self,
        fleet_monitor: FleetMonitor,
        congestion_speed: float = CONGESTION_SPEED_THRESHOLD_KMH,
        congestion_min_devices: int = CONGESTION_DEVICE_COUNT,
        congestion_duration_s: float = CONGESTION_DURATION_S,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._fleet = fleet_monitor
        self._congestion_speed = congestion_speed
        self._congestion_min_devices = congestion_min_devices
        self._congestion_duration_s = congestion_duration_s

        self._incidents: dict[str, ActiveIncident] = {}
        self._congestion_tracking: dict[str, float] = {}  # incident_id -> first_seen_time

        # Periodic congestion check
        self._check_timer = QTimer(self)
        self._check_timer.setInterval(15000)  # Every 15s
        self._check_timer.timeout.connect(self._check_congestion)
        self._check_timer.start()

    def handle_incident_message(self, msg: IncidentMessage) -> None:
        """Process an incoming incident message from MQTT."""
        if msg.resolved:
            self._resolve_incident(msg.incident_id)
            return

        if msg.incident_id in self._incidents:
            return  # Already tracking

        incident = ActiveIncident(
            incident_id=msg.incident_id,
            lat=msg.position.get("lat", 0.0),
            lon=msg.position.get("lon", 0.0),
            incident_type=msg.incident_type,
            description=msg.description,
            severity=msg.severity,
            reported_at=time.time(),
        )
        self._incidents[msg.incident_id] = incident
        self.incident_detected.emit(incident)

        # Suggest rerouting
        self.reroute_suggested.emit(
            incident.lat,
            incident.lon,
            INCIDENT_AVOIDANCE_RADIUS_M / 1000.0,
        )

        logger.warning(
            "Incident reported: %s at (%.4f, %.4f) - %s",
            msg.incident_type,
            incident.lat,
            incident.lon,
            msg.description,
        )

    def _resolve_incident(self, incident_id: str) -> None:
        if incident_id in self._incidents:
            self._incidents[incident_id].resolved = True
            self.incident_resolved.emit(incident_id)
            del self._incidents[incident_id]
            logger.info("Incident %s resolved", incident_id)

    def _check_congestion(self) -> None:
        """Check for congestion near known incidents."""
        for incident in list(self._incidents.values()):
            if incident.resolved:
                continue

            slow = self._fleet.get_slow_devices_near(
                incident.lat, incident.lon, radius_km=0.5
            )

            if len(slow) >= self._congestion_min_devices:
                key = incident.incident_id
                if key not in self._congestion_tracking:
                    self._congestion_tracking[key] = time.time()
                else:
                    duration = time.time() - self._congestion_tracking[key]
                    if duration >= self._congestion_duration_s:
                        logger.warning(
                            "Congestion detected near incident %s (%.1fs, %d slow devices)",
                            incident.incident_id,
                            duration,
                            len(slow),
                        )
                        # Re-emit reroute suggestion
                        self.reroute_suggested.emit(
                            incident.lat,
                            incident.lon,
                            INCIDENT_AVOIDANCE_RADIUS_M / 1000.0,
                        )
            else:
                self._congestion_tracking.pop(incident.incident_id, None)

    def report_accident(self, device_id: str, lat: float, lon: float, description: str = "") -> IncidentMessage:
        """Create and track an accident report."""
        msg = IncidentMessage.create(
            device_id=device_id,
            lat=lat,
            lon=lon,
            incident_type="accident",
            description=description or "Accident reported by driver",
            severity="high",
        )
        self.handle_incident_message(msg)
        return msg

    @property
    def active_incidents(self) -> list[ActiveIncident]:
        return [i for i in self._incidents.values() if not i.resolved]
