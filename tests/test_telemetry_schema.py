"""Tests for telemetry JSON schemas."""

import json


from wake_focus.fleet.telemetry_schema import (
    HeartbeatMessage,
    IncidentMessage,
    TelemetryMessage,
)


class TestTelemetryMessage:
    def test_create_and_serialize(self):
        msg = TelemetryMessage.create(
            device_id="test-001",
            device_name="TestDevice",
            lat=41.311,
            lon=69.279,
            speed=45.0,
            heading=180.0,
            accuracy=3.5,
            alert_state="normal",
            monitoring=True,
        )
        json_str = msg.to_json()
        data = json.loads(json_str)

        assert data["device_id"] == "test-001"
        assert data["position"]["lat"] == 41.311
        assert data["alert_state"] == "normal"
        assert data["monitoring_active"] is True

    def test_roundtrip(self):
        msg = TelemetryMessage.create(
            device_id="rt-001", device_name="RT",
            lat=10.0, lon=20.0, speed=30.0,
            heading=90.0, accuracy=5.0,
            alert_state="drowsy", monitoring=False,
        )
        restored = TelemetryMessage.from_json(msg.to_json())
        assert restored.device_id == "rt-001"
        assert restored.alert_state == "drowsy"


class TestIncidentMessage:
    def test_create(self):
        msg = IncidentMessage.create(
            device_id="dev-01",
            lat=41.5,
            lon=69.3,
            incident_type="accident",
            description="Rear-end collision",
            severity="high",
        )
        assert msg.incident_id  # Should be generated
        assert msg.severity == "high"

    def test_serialize(self):
        msg = IncidentMessage.create(
            device_id="dev-01", lat=0.0, lon=0.0,
        )
        data = json.loads(msg.to_json())
        assert "incident_id" in data
        assert data["incident_type"] == "accident"


class TestHeartbeat:
    def test_roundtrip(self):
        hb = HeartbeatMessage(
            device_id="hb-01",
            device_name="HeartbeatDevice",
            status="online",
            fleet_group="alpha",
        )
        restored = HeartbeatMessage.from_json(hb.to_json())
        assert restored.device_id == "hb-01"
        assert restored.status == "online"
