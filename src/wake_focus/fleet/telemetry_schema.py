"""
Wake Focus - Telemetry Schema

JSON schemas for MQTT fleet messages.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class TelemetryMessage:
    """Periodic device telemetry published via MQTT."""

    device_id: str = ""
    device_name: str = ""
    timestamp_utc: str = ""
    position: dict = field(default_factory=lambda: {
        "lat": 0.0, "lon": 0.0, "speed_kmh": 0.0, "heading": 0.0, "accuracy_m": 0.0
    })
    alert_state: str = "normal"  # "normal", "drowsy", "distracted"
    monitoring_active: bool = False
    fleet_group: str = "default"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "TelemetryMessage":
        d = json.loads(data)
        return cls(**d)

    @classmethod
    def create(
        cls,
        device_id: str,
        device_name: str,
        lat: float,
        lon: float,
        speed: float,
        heading: float,
        accuracy: float,
        alert_state: str,
        monitoring: bool,
        fleet_group: str = "default",
    ) -> "TelemetryMessage":
        return cls(
            device_id=device_id,
            device_name=device_name,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            position={
                "lat": lat,
                "lon": lon,
                "speed_kmh": speed,
                "heading": heading,
                "accuracy_m": accuracy,
            },
            alert_state=alert_state,
            monitoring_active=monitoring,
            fleet_group=fleet_group,
        )


@dataclass
class IncidentMessage:
    """Incident report published via MQTT."""

    incident_id: str = ""
    device_id: str = ""
    timestamp_utc: str = ""
    incident_type: str = "accident"  # "accident", "congestion", "hazard"
    position: dict = field(default_factory=lambda: {"lat": 0.0, "lon": 0.0})
    description: str = ""
    severity: str = "medium"  # "low", "medium", "high", "critical"
    resolved: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "IncidentMessage":
        d = json.loads(data)
        return cls(**d)

    @classmethod
    def create(
        cls,
        device_id: str,
        lat: float,
        lon: float,
        incident_type: str = "accident",
        description: str = "",
        severity: str = "medium",
    ) -> "IncidentMessage":
        return cls(
            incident_id=str(uuid.uuid4())[:8],
            device_id=device_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            incident_type=incident_type,
            position={"lat": lat, "lon": lon},
            description=description,
            severity=severity,
        )


@dataclass
class HeartbeatMessage:
    """Device heartbeat for presence detection."""

    device_id: str = ""
    device_name: str = ""
    timestamp_utc: str = ""
    status: str = "online"  # "online", "offline"
    fleet_group: str = "default"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "HeartbeatMessage":
        d = json.loads(data)
        return cls(**d)
