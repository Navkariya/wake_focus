"""
Wake Focus - Configuration System

Loads YAML config with defaults, supports edge profile overlay,
and provides typed access to all configuration values.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from wake_focus.constants import (
    BEEP_DURATION_MS,
    BEEP_FREQUENCY_HZ,
    BEEP_VOLUME,
    BORDER_COLOR_DISTRACTION,
    BORDER_COLOR_DROWSY,
    BORDER_FALLBACK_PX,
    BORDER_THICKNESS_MM,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAMERA_WIDTH,
    DISTRACTION_BEEP_INTERVAL_MS,
    DISTRACTION_BEEPS_PER_ALERT,
    DISTRACTION_ONSET_SECONDS,
    DISTRACTION_RECOVERY_STABLE_SECONDS,
    DROWSY_BEEP_INTERVAL_MS,
    DROWSY_CONSEC_FRAMES,
    DROWSY_RECOVERY_STABLE_SECONDS,
    EAR_THRESHOLD,
    HEAD_PITCH_THRESHOLD,
    HEAD_YAW_THRESHOLD,
    MAP_DEFAULT_ZOOM,
    MAP_UPDATE_INTERVAL_MS,
    MQTT_DEFAULT_PORT,
    OSRM_DEFAULT_URL,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

logger = logging.getLogger(__name__)


def _find_config_file() -> Path | None:
    """Search for config file in standard locations."""
    search_paths = [
        Path.cwd() / "config" / "default_config.yaml",
        Path.cwd() / "default_config.yaml",
        Path(__file__).parent.parent.parent.parent / "config" / "default_config.yaml",
        Path.home() / ".config" / "wake_focus" / "config.yaml",
        Path("/etc/wake_focus/config.yaml"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Application configuration with typed accessors."""

    def __init__(self, config_path: str | Path | None = None, edge_mode: bool = False):
        self._data: dict[str, Any] = {}
        self._load(config_path, edge_mode)

    def _load(self, config_path: str | Path | None, edge_mode: bool) -> None:
        """Load configuration from file(s)."""
        # Start with empty dict
        self._data = {}

        # Try to find and load config file
        path = Path(config_path) if config_path else _find_config_file()
        if path and path.exists():
            logger.info("Loading config from %s", path)
            with open(path, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self._data = loaded
        else:
            logger.warning("No config file found, using defaults")

        # Apply edge overlay if requested
        if edge_mode:
            edge_path = path.parent / "edge_config.yaml" if path else None
            if edge_path and edge_path.exists():
                logger.info("Applying edge config overlay from %s", edge_path)
                with open(edge_path, "r") as f:
                    edge_data = yaml.safe_load(f)
                    if edge_data:
                        self._data = _deep_merge(self._data, edge_data)

        # Also check environment variable overrides
        env_config = os.environ.get("WAKE_FOCUS_CONFIG")
        if env_config:
            env_path = Path(env_config)
            if env_path.exists():
                logger.info("Applying env config override from %s", env_path)
                with open(env_path, "r") as f:
                    env_data = yaml.safe_load(f)
                    if env_data:
                        self._data = _deep_merge(self._data, env_data)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a config value by dot-separated path. E.g., 'camera.width'."""
        keys = key_path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    # ── Typed property accessors ────────────────────────────────────────────

    # Window
    @property
    def window_width(self) -> int:
        return self.get("app.window_width", WINDOW_WIDTH)

    @property
    def window_height(self) -> int:
        return self.get("app.window_height", WINDOW_HEIGHT)

    @property
    def log_level(self) -> str:
        return self.get("app.log_level", "INFO")

    # Camera
    @property
    def camera_index(self) -> int:
        return self.get("camera.device_index", DEFAULT_CAMERA_INDEX)

    @property
    def camera_width(self) -> int:
        return self.get("camera.width", DEFAULT_CAMERA_WIDTH)

    @property
    def camera_height(self) -> int:
        return self.get("camera.height", DEFAULT_CAMERA_HEIGHT)

    @property
    def camera_fps(self) -> int:
        return self.get("camera.fps", DEFAULT_CAMERA_FPS)

    @property
    def camera_simulation(self) -> bool:
        return self.get("camera.simulation", False)

    # Perception
    @property
    def frame_skip(self) -> int:
        return self.get("perception.frame_skip", 1)

    @property
    def max_processing_fps(self) -> float:
        return float(self.get("perception.max_processing_fps", 10.0))

    @property
    def object_detection_interval_frames(self) -> int:
        return int(self.get("perception.object_detection_interval_frames", 2))

    @property
    def yolo_model_path(self) -> str:
        return self.get("perception.object_detection.model_path", "models/yolo26n.onnx")

    @property
    def yolo_confidence(self) -> float:
        return self.get("perception.object_detection.confidence_threshold", 0.5)

    @property
    def yolo_imgsz(self) -> int:
        return self.get("perception.object_detection.imgsz", 640)

    # Drowsiness
    @property
    def ear_threshold(self) -> float:
        return self.get("alerts.drowsiness.ear_threshold", EAR_THRESHOLD)

    @property
    def drowsy_consec_frames(self) -> int:
        return self.get("alerts.drowsiness.drowsy_consec_frames", DROWSY_CONSEC_FRAMES)

    @property
    def drowsy_beep_interval_ms(self) -> int:
        return int(
            self.get("alerts.drowsiness.beep_interval_seconds", DROWSY_BEEP_INTERVAL_MS / 1000)
            * 1000
        )

    @property
    def drowsy_recovery_seconds(self) -> float:
        return self.get(
            "alerts.drowsiness.recovery_stable_seconds", DROWSY_RECOVERY_STABLE_SECONDS
        )

    # Distraction
    @property
    def distraction_onset_seconds(self) -> float:
        return self.get("alerts.distraction.onset_seconds", DISTRACTION_ONSET_SECONDS)

    @property
    def distraction_beep_interval_ms(self) -> int:
        return int(
            self.get(
                "alerts.distraction.beep_interval_seconds", DISTRACTION_BEEP_INTERVAL_MS / 1000
            )
            * 1000
        )

    @property
    def distraction_beeps_per_alert(self) -> int:
        return self.get("alerts.distraction.beeps_per_alert", DISTRACTION_BEEPS_PER_ALERT)

    @property
    def distraction_recovery_seconds(self) -> float:
        return self.get(
            "alerts.distraction.recovery_stable_seconds", DISTRACTION_RECOVERY_STABLE_SECONDS
        )

    @property
    def yaw_threshold(self) -> float:
        return self.get("alerts.distraction.yaw_threshold", HEAD_YAW_THRESHOLD)

    @property
    def pitch_threshold(self) -> float:
        return self.get("alerts.distraction.pitch_threshold", HEAD_PITCH_THRESHOLD)

    # Border
    @property
    def border_thickness_mm(self) -> float:
        return self.get("alerts.border.thickness_mm", BORDER_THICKNESS_MM)

    @property
    def border_fallback_px(self) -> int:
        return self.get("alerts.border.fallback_thickness_px", BORDER_FALLBACK_PX)

    @property
    def border_color_drowsy(self) -> tuple[int, int, int]:
        c = self.get("alerts.border.drowsy_color", list(BORDER_COLOR_DROWSY))
        return tuple(c) if isinstance(c, list) else BORDER_COLOR_DROWSY

    @property
    def border_color_distraction(self) -> tuple[int, int, int]:
        c = self.get("alerts.border.distraction_color", list(BORDER_COLOR_DISTRACTION))
        return tuple(c) if isinstance(c, list) else BORDER_COLOR_DISTRACTION

    # Audio
    @property
    def beep_frequency(self) -> int:
        return self.get("alerts.audio.beep_frequency_hz", BEEP_FREQUENCY_HZ)

    @property
    def beep_duration_ms(self) -> int:
        return self.get("alerts.audio.beep_duration_ms", BEEP_DURATION_MS)

    @property
    def beep_volume(self) -> float:
        return self.get("alerts.audio.beep_volume", BEEP_VOLUME)

    # GPS
    @property
    def gps_source(self) -> str:
        return self.get("gps.source", "simulation")

    @property
    def gps_serial_port(self) -> str:
        return self.get("gps.serial_port", "/dev/ttyUSB0")

    @property
    def gps_serial_baud(self) -> int:
        return self.get("gps.serial_baud", 9600)

    # Map
    @property
    def map_zoom(self) -> int:
        return self.get("map.default_zoom", MAP_DEFAULT_ZOOM)

    @property
    def map_center(self) -> list[float]:
        return self.get("map.default_center", [41.311, 69.279])

    @property
    def map_update_interval_ms(self) -> int:
        return self.get("map.update_interval_ms", MAP_UPDATE_INTERVAL_MS)

    @property
    def map_provider(self) -> str:
        return self.get("map.provider", "yandex")

    @property
    def yandex_maps_api_key(self) -> str:
        return self.get("map.yandex.api_key", os.environ.get("YANDEX_MAPS_API_KEY", ""))

    @property
    def yandex_maps_lang(self) -> str:
        return self.get("map.yandex.lang", "en_US")

    @property
    def map_traffic_enabled(self) -> bool:
        return self.get("map.yandex.traffic_enabled", True)

    @property
    def map_auto_follow(self) -> bool:
        return self.get("map.auto_follow", True)

    # Fleet MQTT
    @property
    def device_id(self) -> str:
        return self.get("fleet.device_id", "")

    @property
    def device_name(self) -> str:
        return self.get("fleet.device_name", "WakeFocus-Device")

    @property
    def mqtt_host(self) -> str:
        return self.get("fleet.mqtt.broker_host", "localhost")

    @property
    def mqtt_port(self) -> int:
        return self.get("fleet.mqtt.broker_port", MQTT_DEFAULT_PORT)

    @property
    def mqtt_username(self) -> str:
        return self.get("fleet.mqtt.username", "")

    @property
    def mqtt_password(self) -> str:
        return self.get("fleet.mqtt.password", "")

    # Routing
    @property
    def osrm_url(self) -> str:
        return self.get("routing.osrm_url", OSRM_DEFAULT_URL)

    @property
    def routing_enabled(self) -> bool:
        return self.get("routing.enabled", True)

    @property
    def auto_reroute(self) -> bool:
        return self.get("routing.auto_reroute", False)

    # Vehicle
    @property
    def fuel_economy(self) -> float:
        return self.get("vehicle.fuel_economy", 8.0)

    @property
    def obd_enabled(self) -> bool:
        return self.get("vehicle.obd.enabled", False)

    # Profile
    @property
    def driver_name(self) -> str:
        return self.get("profile.driver_name", "")

    @property
    def vehicle_id(self) -> str:
        return self.get("profile.vehicle_id", "")

    @property
    def fleet_group(self) -> str:
        return self.get("profile.fleet_group", "default")
