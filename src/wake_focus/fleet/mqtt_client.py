"""
Wake Focus - MQTT Client for Fleet Telemetry

Publishes device telemetry and subscribes to fleet updates.
Uses paho-mqtt with auto-reconnect.
"""

import logging
import uuid

from PySide6.QtCore import QObject, Signal

from wake_focus.constants import (
    MQTT_DEFAULT_PORT,
    MQTT_KEEPALIVE,
    TOPIC_FLEET_ROSTER,
    TOPIC_INCIDENTS,
    TOPIC_TELEMETRY,
)
from wake_focus.fleet.telemetry_schema import HeartbeatMessage, IncidentMessage, TelemetryMessage

logger = logging.getLogger(__name__)


class MQTTFleetClient(QObject):
    """MQTT client for fleet communication."""

    # Signals for received messages
    telemetry_received = Signal(object)   # TelemetryMessage from another device
    incident_received = Signal(object)    # IncidentMessage
    device_online = Signal(str, str)      # device_id, device_name
    device_offline = Signal(str)          # device_id
    connection_changed = Signal(bool)     # connected

    def __init__(
        self,
        device_id: str = "",
        device_name: str = "WakeFocus-Device",
        broker_host: str = "localhost",
        broker_port: int = MQTT_DEFAULT_PORT,
        username: str = "",
        password: str = "",
        fleet_group: str = "default",
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        self._device_id = device_id or str(uuid.uuid4())[:8]
        self._device_name = device_name
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._fleet_group = fleet_group
        self._connected = False
        self._client = None

        self._username = username
        self._password = password

        self._init_client()

        logger.info(
            "MQTTFleetClient: device=%s, broker=%s:%d",
            self._device_id,
            broker_host,
            broker_port,
        )

    def _init_client(self) -> None:
        """Initialize paho-mqtt client."""
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(
                client_id=f"wake_focus_{self._device_id}",
                protocol=mqtt.MQTTv311,
            )

            if self._username:
                self._client.username_pw_set(self._username, self._password)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            # Will message (offline notification)
            will_msg = HeartbeatMessage(
                device_id=self._device_id,
                device_name=self._device_name,
                status="offline",
                fleet_group=self._fleet_group,
            )
            self._client.will_set(
                TOPIC_FLEET_ROSTER,
                will_msg.to_json(),
                qos=1,
                retain=True,
            )

        except ImportError:
            logger.warning("paho-mqtt not installed. Fleet communication disabled.")
            self._client = None

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        if not self._client:
            return False

        try:
            self._client.connect_async(
                self._broker_host,
                self._broker_port,
                keepalive=MQTT_KEEPALIVE,
            )
            self._client.loop_start()
            return True
        except Exception as e:
            logger.error("MQTT connect failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._client:
            # Send offline heartbeat
            hb = HeartbeatMessage(
                device_id=self._device_id,
                device_name=self._device_name,
                status="offline",
                fleet_group=self._fleet_group,
            )
            self._client.publish(TOPIC_FLEET_ROSTER, hb.to_json(), qos=1, retain=True)
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to %s:%d", self._broker_host, self._broker_port)
            self.connection_changed.emit(True)

            # Subscribe to fleet topics
            client.subscribe("wake_focus/devices/+/telemetry", qos=0)
            client.subscribe("wake_focus/incidents/+", qos=1)
            client.subscribe(TOPIC_FLEET_ROSTER, qos=1)

            # Announce online
            hb = HeartbeatMessage(
                device_id=self._device_id,
                device_name=self._device_name,
                status="online",
                fleet_group=self._fleet_group,
            )
            client.publish(TOPIC_FLEET_ROSTER, hb.to_json(), qos=1, retain=True)
        else:
            logger.error("MQTT connection failed with code %d", rc)
            self.connection_changed.emit(False)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        logger.warning("MQTT disconnected (rc=%d)", rc)
        self.connection_changed.emit(False)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = msg.payload.decode("utf-8")
            topic = msg.topic

            if "telemetry" in topic:
                telemetry = TelemetryMessage.from_json(payload)
                if telemetry.device_id != self._device_id:
                    self.telemetry_received.emit(telemetry)

            elif "incidents" in topic:
                incident = IncidentMessage.from_json(payload)
                self.incident_received.emit(incident)

            elif topic == TOPIC_FLEET_ROSTER:
                hb = HeartbeatMessage.from_json(payload)
                if hb.device_id != self._device_id:
                    if hb.status == "online":
                        self.device_online.emit(hb.device_id, hb.device_name)
                    else:
                        self.device_offline.emit(hb.device_id)

        except Exception as e:
            logger.error("Error processing MQTT message: %s", e)

    def publish_telemetry(self, telemetry: TelemetryMessage) -> None:
        """Publish device telemetry."""
        if self._client and self._connected:
            topic = TOPIC_TELEMETRY.format(device_id=self._device_id)
            self._client.publish(topic, telemetry.to_json(), qos=0)

    def publish_incident(self, incident: IncidentMessage) -> None:
        """Publish an incident report."""
        if self._client and self._connected:
            topic = TOPIC_INCIDENTS.format(incident_id=incident.incident_id)
            self._client.publish(topic, incident.to_json(), qos=1, retain=True)

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def is_connected(self) -> bool:
        return self._connected
