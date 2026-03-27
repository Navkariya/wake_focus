"""
Wake Focus - Constants and Layout Definitions

All magic numbers, layout sizes, thresholds, and timing constants.
These serve as compile-time defaults; runtime values come from config.yaml.
"""

from enum import Enum, auto


# ── Window & Panel Layout (pixels) ──────────────────────────────────────────
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

CAMERA_PANEL_W = 500
CAMERA_PANEL_H = 500

MAP_PANEL_W = 300
MAP_PANEL_H = 500

VEHICLE_STATS_W = 400
VEHICLE_STATS_H = 300

FLEET_STATUS_W = 300
FLEET_STATUS_H = 300

BUTTON_PANEL_W = 100
BUTTON_PANEL_H = 300

# ── Camera ──────────────────────────────────────────────────────────────────
DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

# ── MediaPipe Face Mesh Landmark Indices ────────────────────────────────────
# 6-point EAR landmarks per eye (standard for Eye Aspect Ratio calculation)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Full eye contour indices for green dot visualization
LEFT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
]
RIGHT_EYE_CONTOUR = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
]

# Iris landmarks (refined)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Key face landmarks for head pose estimation (solvePnP)
# [nose tip, chin, left eye left corner, right eye right corner,
#  left mouth corner, right mouth corner]
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# 3D model points for head pose (generic face model, mm)
HEAD_POSE_3D_POINTS = [
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
]

# ── Drowsiness Detection ───────────────────────────────────────────────────
EAR_THRESHOLD = 0.21
BLINK_CONSEC_FRAMES = 3
DROWSY_CONSEC_FRAMES = 20  # ~0.67s at 30fps
DROWSY_RECOVERY_STABLE_SECONDS = 3.0
DROWSY_BEEP_INTERVAL_MS = 1000  # 1 second

# ── Distraction Detection ──────────────────────────────────────────────────
DISTRACTION_ONSET_SECONDS = 30.0
DISTRACTION_BEEP_INTERVAL_MS = 5000  # 5 seconds
DISTRACTION_BEEPS_PER_ALERT = 2
DISTRACTION_RECOVERY_STABLE_SECONDS = 3.0
HEAD_YAW_THRESHOLD = 30.0   # degrees
HEAD_PITCH_THRESHOLD = 25.0  # degrees

# ── Alert Border ────────────────────────────────────────────────────────────
BORDER_THICKNESS_MM = 2.0
BORDER_FALLBACK_PX = 8
BORDER_COLOR_DROWSY = (255, 0, 0)      # Red (R, G, B)
BORDER_COLOR_DISTRACTION = (255, 165, 0)  # Orange (R, G, B)

# ── Audio ───────────────────────────────────────────────────────────────────
BEEP_FREQUENCY_HZ = 440
BEEP_DURATION_MS = 200
BEEP_SAMPLE_RATE = 44100
BEEP_VOLUME = 0.8

# ── Overlay Colors ──────────────────────────────────────────────────────────
EYE_LANDMARK_COLOR = (0, 255, 0)      # Green (R, G, B) for Qt
EYE_LANDMARK_RADIUS = 2
DETECTION_BOX_COLOR = (0, 255, 0)     # Green
DETECTION_BOX_THICKNESS = 2
DETECTION_LABEL_FONT_SIZE = 12

# ── YOLO Detection Classes ─────────────────────────────────────────────────
# YOLO26 / COCO handheld-distraction classes used by the app by default.
YOLO_HANDHELD_CLASSES = [
    "cell phone",
    "book",      # Used as a practical proxy for paper/document-like objects
    "laptop",
    "mouse",
    "remote",
    "keyboard",
]

YOLO_HANDHELD_TARGET_ALIASES = [
    "cell phone",
    "phone",
    "mobile phone",
    "smartphone",
    "paper",
    "document",
    "notebook",
    "book",
    "laptop",
    "mouse",
    "remote",
    "remote control",
    "keyboard",
    "electronic gadget",
]

# Custom training target classes
CUSTOM_DETECTION_CLASSES = [
    "cell_phone",
    "paper",
    "tablet",
    "food_drink",
    "cigarette",
    "book",
    "handheld_device",
    "makeup_tool",
    "wallet",
    "headphones",
]

# ── GPS ─────────────────────────────────────────────────────────────────────
GPS_UPDATE_RATE_HZ = 1.0
EARTH_RADIUS_KM = 6371.0

# ── Fleet / MQTT ────────────────────────────────────────────────────────────
MQTT_DEFAULT_PORT = 1883
MQTT_KEEPALIVE = 60
HEARTBEAT_INTERVAL_S = 5.0
TELEMETRY_INTERVAL_S = 1.0

# MQTT Topics
TOPIC_TELEMETRY = "wake_focus/devices/{device_id}/telemetry"
TOPIC_ALERTS = "wake_focus/devices/{device_id}/alerts"
TOPIC_INCIDENTS = "wake_focus/incidents/{incident_id}"
TOPIC_FLEET_ROSTER = "wake_focus/fleet/roster"

# ── Routing ─────────────────────────────────────────────────────────────────
OSRM_DEFAULT_URL = "http://router.project-osrm.org"
INCIDENT_AVOIDANCE_RADIUS_M = 500
CONGESTION_SPEED_THRESHOLD_KMH = 5.0
CONGESTION_DEVICE_COUNT = 2
CONGESTION_DURATION_S = 60.0

# ── Map ─────────────────────────────────────────────────────────────────────
OSM_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
MAP_DEFAULT_ZOOM = 15
MAP_UPDATE_INTERVAL_MS = 1000


# ── Enums ───────────────────────────────────────────────────────────────────
class AlertState(Enum):
    """Alert state machine states."""
    NORMAL = auto()
    DROWSINESS_DETECTED = auto()
    ALERTING_DROWSY = auto()
    DISTRACTION_TRACKING = auto()
    ALERTING_DISTRACTION = auto()
    RECOVERY_CHECK_DROWSY = auto()
    RECOVERY_CHECK_DISTRACTION = auto()


class AlertPriority(Enum):
    """Alert priority levels (higher number = higher priority)."""
    NONE = 0
    DISTRACTION = 1
    DROWSINESS = 2


class GPSSource(Enum):
    """GPS data source types."""
    GPSD = "gpsd"
    SERIAL = "serial"
    SIMULATION = "simulation"


class DeviceStatus(Enum):
    """Fleet device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ALERTING = "alerting"
    UNKNOWN = "unknown"
