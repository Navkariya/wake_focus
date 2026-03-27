# Wake Focus Research Brief

## Purpose

This brief converts the current Wake Focus specification into a research-backed implementation baseline for a coding agent. It is optimized for a greenfield build, but it is also reality-checked against the neighboring `../wake_focus` repository so the resulting prompt stays technically plausible.

Current date used for this brief: **2026-03-26**.

## Executive Takeaways

- The requested stack is feasible as a Python desktop application if we keep the architecture disciplined: `PySide6` for desktop UI, `OpenCV` for camera capture, `MediaPipe` for face/eye landmarks, `Ultralytics` for object detection, `Qt Multimedia` for beep playback, and `OSRM + MQTT` for routing and fleet telemetry.
- The most important implementation risks are not the UI layout; they are real-time inference latency, camera permission failure modes, Qt map/provider setup, and ARM64 deployment constraints on Orange Pi Zero 2W.
- The final prompt should force the coding agent to treat every unspecified area as a documented default instead of improvising silent behavior.
- The final prompt should instruct the coding agent to **write files into the repo and verify them**, not to print every full file inline. That is more reliable for an actual build task.

## Primary Technical Findings

### 1. Camera and real-time perception

- OpenCV `VideoCapture` is the right cross-platform baseline for desktop capture. The official docs describe opening a camera by index, checking `isOpened()`, and releasing the device when done: <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>.
- MediaPipe Face Landmarker in live-stream mode is a strong fit for real-time facial analysis. The official Python guide requires monotonically increasing timestamps and is explicit that live stream mode is asynchronous: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python>.
- MediaPipe Face Mesh remains a useful landmark reference for eye overlays. The official Face Mesh docs describe 468 3D face landmarks suitable for eye contour extraction and head-pose-adjacent geometry work: <https://chuoling.github.io/mediapipe/solutions/face_mesh.html>.

Recommended prompt default:
- Use a dedicated capture thread.
- Resize/crop frames into a fixed `500x500` display surface.
- Use monotonic timestamps for MediaPipe live-stream callbacks.
- Permit frame dropping under load rather than blocking the UI.

### 2. Eye landmarks, drowsiness, and alert logic

Your requirement names should stay literal:
- `<<ogohlantirish>>`: red border, repeating short beep until stable recovery
- `<<ogohlantirish2>>`: orange border, two beeps every 5 seconds after more than 30 seconds of sustained distraction

The prompt should force a deterministic alert state machine with:
- priority: drowsiness over distraction
- onset timers
- recovery stability windows
- hysteresis so the UI and audio do not flicker

The neighboring repo already uses this split in [`../wake_focus/src/wake_focus/core/alert_state_machine.py`](../../wake_focus/src/wake_focus/core/alert_state_machine.py) and validates it in [`../wake_focus/tests/test_alert_state_machine.py`](../../wake_focus/tests/test_alert_state_machine.py).

Recommended prompt default:
- `<<ogohlantirish>>` beep interval: 1 second
- `<<ogohlantirish2>>` alert cadence: 2 short beeps every 5 seconds
- stable recovery window for both: 3 seconds unless config overrides it

### 3. Border thickness and audio

- Qt `QSoundEffect` is the best cross-platform baseline for deterministic short beeps. The official docs describe it as suitable for low-latency feedback sounds and optimized for uncompressed formats such as WAV: <https://doc.qt.io/qt-6/qsoundeffect.html>.
- The `2 mm` border requirement should be treated as **DPI-aware best effort**, not absolute physical certainty. Qt exposes logical DPI, physical DPI, and device-pixel-ratio metrics through `QPaintDevice`: <https://doc.qt.io/qt-6/qpaintdevice.html>.

Recommended prompt default:
- Calculate `mm -> px` from screen DPI, prefer physical DPI when sensible, fall back to logical DPI, then fall back to a documented static thickness such as `8 px`.
- Ship tiny WAV assets or generate WAV beeps once at startup; reuse the loaded effect object instead of recreating it on every alert.

### 4. Map panel, GPS, and routing

- Qt `Map` requires a `plugin` to display actual map data. The official docs are explicit about that: <https://doc.qt.io/qt-6/qml-qtlocation-map.html>.
- Qt’s maps/navigation overview also reinforces that the client must create a `Plugin` object before using a `Map`: <https://doc.qt.io/qt-6/location-maps-qml.html>.
- The Qt Positioning NMEA plugin supports serial, socket, and file input, which maps well to Linux GPS receivers and simulation playback: <https://doc.qt.io/qt-6/position-plugin-nmea.html>.
- OSRM’s HTTP API is a practical open routing baseline with route alternatives and route geometry: <https://project-osrm.org/docs/v5.24.0/api/>.

Recommended prompt default:
- Use OSRM over HTTP and draw returned polylines in the map pane.
- Treat GPS input as pluggable with three sources:
  - `gpsd`
  - NMEA serial
  - simulation
- Default rerouting policy:
  - accident creates incident zone
  - congestion is inferred from multiple low-speed devices near the incident over a rolling window
  - alternative route is suggested or auto-applied based on config

### 5. Fleet telemetry

- MQTT remains a strong fit for lightweight device telemetry. The MQTT 3.1.1 specification is the official OASIS standard: <https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/mqtt-v3.1.1.html>.

Recommended prompt default:
- Publish three message types:
  - heartbeat
  - telemetry
  - incident
- Minimum telemetry fields:
  - device identity
  - UTC timestamp
  - position and speed
  - alert state
  - monitoring active flag
  - fleet group

This also matches the neighboring repo’s message split in [`../wake_focus/src/wake_focus/fleet/telemetry_schema.py`](../../wake_focus/src/wake_focus/fleet/telemetry_schema.py).

### 6. Packaging: Windows, Kali, and Orange Pi

- PyInstaller is appropriate for Windows desktop packaging, but its own documentation is clear that it is **not a cross-compiler**. Build Windows apps on Windows and Linux apps on Linux: <https://pyinstaller.org/en/stable/operating-mode.html>.
- Debian packaging should use `dpkg-buildpackage`, the standard Debian package build tool: <https://manpages.debian.org/unstable/dpkg-dev/dpkg-buildpackage.1.en.html>.
- Debian’s packaging guide is the right baseline for `.deb` structure and workflow: <https://www.debian.org/doc/manuals/debmake-doc/>.
- Kali tracks Debian testing closely enough that Debian-style packaging is the right default. See Kali’s relationship and branch documentation: <https://www.kali.org/docs/policy/kali-linux-relationship-with-debian/> and <https://www.kali.org/docs/general-use/kali-branches/>.

Recommended prompt default:
- Windows:
  - build on Windows with PyInstaller
  - ship a `dist/` `.exe` and document camera/privacy prerequisites
- Kali x86_64:
  - build native `.deb` with `dpkg-buildpackage`
- Orange Pi Zero 2W ARM64:
  - build on ARM64 hardware or an ARM64-native environment
  - prefer ARM64 `.deb`
  - allow editable install fallback if binary packaging is blocked by dependency constraints

### 7. Orange Pi Zero 2W feasibility

- The official Orange Pi Zero 2W page and wiki are the correct sources for board capabilities: <https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-2W.html> and <https://www.orangepi.org/orangepiwiki/index.php/Orange_Pi_Zero_2W>.
- The board class is resource-constrained compared with desktop targets, so the prompt should explicitly require an edge profile.

Recommended prompt default:
- `YOLO11n`
- smaller input size such as `320`
- frame skipping
- slower map refresh
- slower telemetry cadence
- optional ONNX export path
- explicit statement that FPS must be **measured on-device**, not promised from theory alone

The neighboring repo already captures these ideas in [`../wake_focus/config/edge_config.yaml`](../../wake_focus/config/edge_config.yaml).

### 8. Ultralytics / YOLO positioning

- Use Ultralytics documentation as the primary source: <https://docs.ultralytics.com/>.
- Keep the product target as **YOLO11**, because that is the user’s explicit requirement and the neighboring repo already follows it.
- Add a prompt note that Ultralytics documentation and model branding can change over time; the agent should keep the implementation pinned to YOLO11 unless the user explicitly approves a model-family change.
- Add a licensing caution. Ultralytics documentation presents both open-source and enterprise licensing paths, so a production prompt should tell the coding agent to document license implications instead of assuming unrestricted commercial use.

## Requirement Clarifications and Chosen Defaults

These defaults should be baked into the final prompt unless the user later overrides them.

| Area | Decision-complete default |
|---|---|
| Main window | Fixed `800x800`; no resize |
| Top row | `500x500` camera panel + `300x500` map panel |
| Bottom row | `400x300` vehicle stats + `300x300` fleet status/event log + `100x300` button panel |
| Unspecified `300x300` region | Implement as `Fleet Status & Event Log` |
| Camera permission UX | In-app dialog first, then camera-open attempt, then OS-specific help if blocked |
| Drowsiness border/audio | Red inner border, short beep every 1 second until stable recovery |
| Distraction border/audio | Orange inner border, 2 short beeps every 5 seconds after more than 30 seconds of continuous distracted orientation |
| Recovery stability | 3 seconds by default |
| GPS sources | `gpsd`, NMEA serial, simulation |
| Fuel data | OBD-II if available; otherwise estimate from distance plus configured fuel economy |
| Routing | OSRM HTTP baseline |
| Fleet networking | MQTT baseline |
| ARM64 deployment | Native ARM64 build preferred; editable-install fallback allowed if packaging dependencies fail |

## Repo Reality Check

The neighboring `../wake_focus` repo is not the target workspace, but it is a useful realism check.

Observed subsystem split:
- `src/wake_focus/core`
- `src/wake_focus/ml`
- `src/wake_focus/fleet`
- `src/wake_focus/ui`
- `src/wake_focus/vehicle`
- `config/`
- `training/`
- `packaging/`
- `tests/`

This is a sensible architecture pattern for the final prompt to encourage, without requiring those exact paths.

Useful existing defaults from the neighboring repo:
- alert thresholds and recovery windows in [`../wake_focus/config/default_config.yaml`](../../wake_focus/config/default_config.yaml)
- ARM64 edge tuning in [`../wake_focus/config/edge_config.yaml`](../../wake_focus/config/edge_config.yaml)
- YOLO dataset class list in [`../wake_focus/training/dataset.yaml`](../../wake_focus/training/dataset.yaml)
- OSRM route manager pattern in [`../wake_focus/src/wake_focus/fleet/route_manager.py`](../../wake_focus/src/wake_focus/fleet/route_manager.py)

## Risks That The Prompt Must Address Explicitly

### High-risk implementation gaps

1. **Camera permissions are platform-dependent.**
   The prompt must require graceful denial/blocked flows, not just `VideoCapture(0)`.

2. **MediaPipe live-stream semantics are easy to misuse.**
   The prompt must require monotonic timestamps and a frame-dropping strategy.

3. **Qt map setup is brittle if plugin/provider assumptions stay implicit.**
   The prompt must require explicit map plugin initialization and a documented fallback path.

4. **Orange Pi performance cannot be hand-waved.**
   The prompt must require an edge profile and measured verification on-device.

5. **Packaging instructions must respect build-host boundaries.**
   The prompt must explicitly say Windows builds happen on Windows and Linux/ARM builds happen on matching Linux targets.

6. **Ultralytics licensing should not be silently ignored.**
   The prompt must instruct the coding agent to document licensing considerations for intended deployment.

## Recommended Validation Order For The Final Prompt

The coding agent should validate in this order:

1. Scaffold repo and config
2. Launch fixed `800x800` UI shell
3. Camera permission flow
4. Camera feed in `500x500`
5. Face landmarks overlay
6. Object detection overlay
7. Alert state machine and audio
8. GPS simulation and map track
9. Fleet telemetry and incident flow
10. Rerouting behavior
11. Vehicle stats estimation / OBD integration
12. Windows packaging
13. Kali `.deb`
14. Orange Pi ARM64 profile and packaging/instructions

## Bottom Line

The build is feasible, but the prompt has to behave like a production spec, not a brainstorming request. The strongest final prompt will:

- keep YOLO11 as the explicit target
- lock every timing and ownership decision that matters
- avoid asking the implementer to invent telemetry schemas or recovery behavior
- require source-backed documentation for packaging limits
- tell the coding agent to create and verify files in-repo rather than dump massive inline code
