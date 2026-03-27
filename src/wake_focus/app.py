"""
Wake Focus - Application Bootstrap

Main application class that wires all components together:
- Camera → Perception → Alert State Machine → UI
- GPS → Map → Fleet → Routing
- Settings / Profile dialogs
"""

import logging
import os
import sys

import numpy as np
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QApplication

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join("/tmp", "wake_focus", "matplotlib"),
)

if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    os.environ.setdefault("WAKE_FOCUS_DISABLE_WEBENGINE", "1")

from wake_focus.config import Config
from wake_focus.constants import AlertState
from wake_focus.core.alert_state_machine import AlertStateMachine
from wake_focus.core.async_perception import AsyncPerceptionProcessor
from wake_focus.core.audio_manager import AudioManager
from wake_focus.core.camera_manager import CameraManager
from wake_focus.core.perception_engine import PerceptionEngine, PerceptionResult
from wake_focus.fleet.fleet_monitor import FleetMonitor
from wake_focus.fleet.gps_manager import GPSManager
from wake_focus.fleet.incident_manager import IncidentManager
from wake_focus.fleet.mqtt_client import MQTTFleetClient
from wake_focus.fleet.route_manager import RouteManager
from wake_focus.fleet.telemetry_schema import TelemetryMessage
from wake_focus.ui.main_window import MainWindow
from wake_focus.ui.permission_dialog import PermissionDialog
from wake_focus.ui.profile_dialog import ProfileDialog
from wake_focus.ui.settings_dialog import SettingsDialog
from wake_focus.vehicle.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)


class WakeFocusApp:
    """Main application controller. Wires all components together."""

    def __init__(self, config: Config):
        self._config = config
        self._app: QApplication | None = None
        self._window: MainWindow | None = None

        # Core components (initialized later)
        self._camera_manager: CameraManager | None = None
        self._perception: AsyncPerceptionProcessor | None = None
        self._alert_fsm: AlertStateMachine | None = None
        self._audio: AudioManager | None = None

        # Fleet/Nav
        self._gps: GPSManager | None = None
        self._mqtt: MQTTFleetClient | None = None
        self._fleet: FleetMonitor | None = None
        self._incidents: IncidentManager | None = None
        self._router: RouteManager | None = None

        # Vehicle
        self._stats: StatsTracker | None = None

        # Timers
        self._map_timer: QTimer | None = None
        self._telemetry_timer: QTimer | None = None

        # Frame counter for scaling
        self._frame_count = 0
        self._last_processing_timestamp = 0.0
        self._latest_perception_result = PerceptionResult()

    def run(self) -> int:
        """Initialize and run the application. Returns exit code."""
        # Create Qt Application
        self._app = QApplication(sys.argv)

        # Setup logging
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        logger.info("Wake Focus v1.0.0 starting...")

        # Camera permission
        if not self._request_camera_permission():
            logger.info("Camera permission denied. Running without camera.")

        # Create main window
        self._window = MainWindow(config=self._config)

        # Initialize subsystems
        self._init_perception()
        self._init_alerts()
        self._init_fleet()
        self._init_vehicle_stats()
        self._wire_connections()

        # Show window
        self._window.show()

        logger.info("Wake Focus ready. Main window displayed.")

        return self._app.exec()

    def _request_camera_permission(self) -> bool:
        """Show camera permission dialog."""
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            logger.info("Offscreen mode detected, camera permission dialog skipped")
            return True

        dialog = PermissionDialog()
        result = dialog.exec()

        if result != PermissionDialog.DialogCode.Accepted:
            return False

        return True

    def _init_perception(self) -> None:
        """Initialize ML/CV pipeline."""
        # Camera manager
        self._camera_manager = CameraManager(
            device_index=self._config.camera_index,
            width=self._config.camera_width,
            height=self._config.camera_height,
            fps=self._config.camera_fps,
        )

        # Perception engine
        engine = PerceptionEngine(
            face_mesh_config={
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            drowsiness_config={
                "ear_threshold": self._config.ear_threshold,
                "drowsy_consec_frames": self._config.drowsy_consec_frames,
            },
            head_pose_config={
                "yaw_threshold": self._config.yaw_threshold,
                "pitch_threshold": self._config.pitch_threshold,
            },
            object_detection_config={
                "model_path": self._config.yolo_model_path,
                "confidence_threshold": self._config.yolo_confidence,
                "imgsz": self._config.yolo_imgsz,
            },
            frame_skip=self._config.frame_skip,
            object_detection_interval_frames=self._config.object_detection_interval_frames,
        )
        self._perception = AsyncPerceptionProcessor(engine=engine)
        self._perception.start()

        self._camera_manager.error_occurred.connect(self._on_camera_error)

    def _init_alerts(self) -> None:
        """Initialize alert system."""
        self._alert_fsm = AlertStateMachine(
            drowsy_recovery_seconds=self._config.drowsy_recovery_seconds,
            distraction_onset_seconds=self._config.distraction_onset_seconds,
            distraction_beep_interval_ms=self._config.distraction_beep_interval_ms,
            distraction_beeps_per_alert=self._config.distraction_beeps_per_alert,
            distraction_recovery_seconds=self._config.distraction_recovery_seconds,
            drowsy_beep_interval_ms=self._config.drowsy_beep_interval_ms,
            border_color_drowsy=self._config.border_color_drowsy,
            border_color_distraction=self._config.border_color_distraction,
        )

        self._audio = AudioManager(
            beep_frequency=self._config.beep_frequency,
            beep_duration_ms=self._config.beep_duration_ms,
            beep_volume=self._config.beep_volume,
            drowsy_interval_ms=self._config.drowsy_beep_interval_ms,
            distraction_interval_ms=self._config.distraction_beep_interval_ms,
            distraction_beep_count=self._config.distraction_beeps_per_alert,
        )

    def _init_fleet(self) -> None:
        """Initialize fleet/navigation components."""
        # GPS
        self._gps = GPSManager(
            source_type=self._config.gps_source,
            gpsd_host=self._config.get("gps.gpsd_host", "localhost"),
            serial_port=self._config.gps_serial_port,
            serial_baud=self._config.gps_serial_baud,
        )

        # Fleet monitor
        self._fleet = FleetMonitor()

        # MQTT client
        self._mqtt = MQTTFleetClient(
            device_id=self._config.device_id,
            device_name=self._config.device_name,
            broker_host=self._config.mqtt_host,
            broker_port=self._config.mqtt_port,
            username=self._config.mqtt_username,
            password=self._config.mqtt_password,
            fleet_group=self._config.fleet_group,
        )

        # Incident manager
        self._incidents = IncidentManager(fleet_monitor=self._fleet)

        # Route manager
        self._router = RouteManager(osrm_url=self._config.osrm_url)

        # Telemetry publish timer
        self._telemetry_timer = QTimer()
        self._telemetry_timer.setInterval(
            int(self._config.get("fleet.telemetry_interval", 1.0) * 1000)
        )
        self._telemetry_timer.timeout.connect(self._publish_telemetry)

        # Map refresh timer
        self._map_timer = QTimer()
        self._map_timer.setInterval(self._config.map_update_interval_ms)
        self._map_timer.timeout.connect(self._refresh_map)

    def _init_vehicle_stats(self) -> None:
        """Initialize vehicle stats tracking."""
        self._stats = StatsTracker(fuel_economy_l_per_100km=self._config.fuel_economy)

    def _wire_connections(self) -> None:
        """Wire all signal/slot connections."""
        if not self._window:
            return

        # ── Camera → Perception → UI ─────────────────────────────
        self._camera_manager.frame_ready.connect(self._on_frame)
        self._perception.result_ready.connect(self._on_perception_result)

        # ── Alert FSM → Audio ─────────────────────────────────────
        self._alert_fsm.start_drowsy_beep.connect(self._audio.start_drowsy_beep)
        self._alert_fsm.stop_drowsy_beep.connect(self._audio.stop_drowsy_beep)
        self._alert_fsm.play_distraction_beep.connect(self._audio.play_distraction_beep_once)
        self._alert_fsm.stop_distraction_beep.connect(self._audio.stop_distraction_beep)

        # ── GPS → Map + Stats ─────────────────────────────────────
        self._gps.position_updated.connect(self._on_gps_update)
        self._gps.fix_changed.connect(self._window.fleet_panel.update_gps_status)

        # ── Fleet ─────────────────────────────────────────────────
        self._mqtt.telemetry_received.connect(self._fleet.update_device)
        self._mqtt.incident_received.connect(self._incidents.handle_incident_message)
        self._mqtt.device_online.connect(self._fleet.set_device_online)
        self._mqtt.device_offline.connect(self._fleet.set_device_offline)
        self._fleet.fleet_updated.connect(self._window.fleet_panel.update_fleet_devices)

        # ── Incidents ─────────────────────────────────────────────
        self._incidents.incident_detected.connect(self._on_incident)
        self._incidents.reroute_suggested.connect(self._on_reroute_suggested)

        # ── Vehicle stats → UI ────────────────────────────────────
        self._stats.distance_updated.connect(self._window.stats_panel.update_distance)
        self._stats.fuel_updated.connect(self._window.stats_panel.update_fuel)
        self._stats.speed_updated.connect(self._window.stats_panel.update_speed)
        self._stats.trip_time_updated.connect(self._window.stats_panel.update_trip_time)

        # ── Button panel ──────────────────────────────────────────
        self._window.button_panel.settings_clicked.connect(self._show_settings)
        self._window.button_panel.profile_clicked.connect(self._show_profile)
        self._window.button_panel.exit_clicked.connect(self._shutdown)

        # ── Fleet panel monitoring toggle ─────────────────────────
        self._window.fleet_panel.monitoring_toggled.connect(self._on_monitoring_toggled)

    @Slot(np.ndarray, float)
    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Display camera frames immediately and feed perception asynchronously."""
        max_processing_fps = self._config.max_processing_fps
        self._window.camera_panel.update_frame(frame)

        should_submit = True
        if max_processing_fps > 0 and self._last_processing_timestamp > 0:
            min_interval = 1.0 / max_processing_fps
            should_submit = (timestamp - self._last_processing_timestamp) >= min_interval

        if should_submit:
            self._perception.submit_frame(frame, timestamp)
            self._last_processing_timestamp = timestamp

        self._frame_count += 1

    @Slot(object, object)
    def _on_perception_result(self, frame: np.ndarray, result: PerceptionResult) -> None:
        """Apply the newest async perception result without blocking the camera path."""
        alert_status = self._alert_fsm.update(result)
        self._latest_perception_result = result

        self._window.camera_panel.update_perception(result, schedule_repaint=False)
        self._window.camera_panel.update_alert(alert_status, schedule_repaint=False)
        self._window.camera_panel.update()

        if result.process_time_ms > 0:
            self._window.fleet_panel.update_fps(1000.0 / result.process_time_ms)

    @Slot(object)
    def _on_gps_update(self, position) -> None:
        """Handle GPS position update."""
        self._window.map_panel.update_position(position.lat, position.lon)
        self._stats.update(position)

    @Slot(object)
    def _on_incident(self, incident) -> None:
        """Handle new incident detection."""
        self._window.map_panel.add_incident(
            incident.incident_id, incident.lat, incident.lon, incident.description
        )
        self._window.fleet_panel.add_event(
            f"⚠️ {incident.incident_type}: {incident.description}"
        )

    @Slot(float, float, float)
    def _on_reroute_suggested(self, lat: float, lon: float, radius_km: float) -> None:
        """Handle reroute suggestion from incident manager."""
        self._window.fleet_panel.add_event(
            f"🔄 Reroute suggested: avoid ({lat:.3f}, {lon:.3f}) r={radius_km:.1f}km"
        )

        if self._gps.last_position and self._config.routing_enabled:
            # TODO: Compute and display alternative route
            logger.info("Reroute computation triggered")

    def _publish_telemetry(self) -> None:
        """Publish current telemetry via MQTT."""
        if not self._gps.last_position:
            return

        pos = self._gps.last_position
        alert_state = "normal"
        if self._alert_fsm.current_state == AlertState.ALERTING_DROWSY:
            alert_state = "drowsy"
        elif self._alert_fsm.current_state == AlertState.ALERTING_DISTRACTION:
            alert_state = "distracted"

        msg = TelemetryMessage.create(
            device_id=self._mqtt.device_id,
            device_name=self._config.device_name,
            lat=pos.lat,
            lon=pos.lon,
            speed=pos.speed_kmh,
            heading=pos.heading,
            accuracy=pos.accuracy_m,
            alert_state=alert_state,
            monitoring=self._window.fleet_panel.is_monitoring if self._window else False,
            fleet_group=self._config.fleet_group,
        )
        self._mqtt.publish_telemetry(msg)

    def _refresh_map(self) -> None:
        """Refresh the map panel with latest data."""
        if self._window and self._fleet:
            self._window.map_panel.update_fleet_devices(self._fleet.get_devices_map_dict())
            self._window.map_panel.refresh_map()

    @Slot(bool)
    def _on_monitoring_toggled(self, start: bool) -> None:
        """Handle Start/Stop monitoring."""
        if start:
            logger.info("Monitoring STARTED")
            self._window.fleet_panel.add_event("✅ Monitoring started")
            self._camera_manager.start()
            self._gps.start()
            self._mqtt.connect()
            self._telemetry_timer.start()
            self._map_timer.start()
            self._window.fleet_panel.update_model_status(self._perception.is_object_detection_available)
        else:
            logger.info("Monitoring STOPPED")
            self._window.fleet_panel.add_event("⏹ Monitoring stopped")
            self._camera_manager.stop()
            self._gps.stop()
            self._telemetry_timer.stop()
            self._map_timer.stop()
            self._alert_fsm.reset()
            self._audio.stop_all()

    def _show_settings(self) -> None:
        """Open settings dialog."""
        dialog = SettingsDialog(self._config, parent=self._window)
        dialog.exec()

    def _show_profile(self) -> None:
        """Open profile dialog."""
        dialog = ProfileDialog(self._config, parent=self._window)
        dialog.exec()

    @Slot(str)
    def _on_camera_error(self, error: str) -> None:
        """Handle camera errors."""
        self._window.fleet_panel.add_event(f"❌ Camera: {error}")

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down Wake Focus...")
        if self._camera_manager:
            self._camera_manager.stop()
        if self._gps:
            self._gps.stop()
        if self._mqtt:
            self._mqtt.disconnect()
        if self._perception:
            self._perception.close()
        if self._audio:
            self._audio.stop_all()
        if self._app:
            self._app.quit()
