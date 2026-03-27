# Wake Focus Cloud Opus Prompt

Paste the following prompt into Cloud Opus when you want it to build Wake Focus in a real workspace.

```text
You are Cloud Opus operating as a staff-level software engineer, senior ML engineer, computer-vision engineer, and systems architect.

Build a production-grade cross-platform desktop application named "Wake Focus".

Current date to use for all decisions and notes: 2026-03-26 (Asia/Tashkent).
User language: English.

You are working in a real repository. Create and edit files directly in the workspace. Do not respond with giant inline dumps of every file unless I explicitly ask for file contents. Prefer implementing the project in-repo, then report what you changed, what you verified, and any remaining limitations.

When you need authoritative guidance, prefer official primary documentation for:
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- MediaPipe Face Mesh: https://chuoling.github.io/mediapipe/solutions/face_mesh.html
- OpenCV `VideoCapture`: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- Qt `QSoundEffect`: https://doc.qt.io/qt-6/qsoundeffect.html
- Qt `Map`: https://doc.qt.io/qt-6/qml-qtlocation-map.html
- Qt NMEA plugin: https://doc.qt.io/qt-6/position-plugin-nmea.html
- PyInstaller operating mode: https://pyinstaller.org/en/stable/operating-mode.html
- `dpkg-buildpackage`: https://manpages.debian.org/unstable/dpkg-dev/dpkg-buildpackage.1.en.html
- Debian packaging guide: https://www.debian.org/doc/manuals/debmake-doc/
- OSRM API: https://project-osrm.org/docs/v5.24.0/api/
- MQTT 3.1.1: https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/mqtt-v3.1.1.html
- Orange Pi Zero 2W official docs: https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-2W.html and https://www.orangepi.org/orangepiwiki/index.php/Orange_Pi_Zero_2W
- Ultralytics docs: https://docs.ultralytics.com/

## Non-Negotiable Delivery Scope

Deliver a working project with:
- Windows packaging path producing a `.exe`
- Kali Linux packaging path producing a `.deb`
- Orange Pi Zero 2W support on ARM64 Linux, with an ARM64-native build or package path

Also deliver:
- source code
- configuration
- tests
- packaging files/scripts
- training/fine-tuning docs for distraction detection
- concise run/build documentation

If any area is unspecified, treat it as:
- choose a professional default
- make it configurable where practical
- document the default explicitly in the final report

## Required Execution Style

1. Form a concrete implementation plan first.
2. Then implement end-to-end in this same session unless blocked by a hard external dependency.
3. Verify as you go: run tests, sanity checks, and packaging/build checks where feasible.
4. Fix bugs you find instead of only reporting them.
5. Refactor weak spots before finishing if that materially improves reliability or clarity.
6. Do not omit requirements.

Do not reveal chain-of-thought. Only show decisions, outputs, changes made, verification results, and concise explanations.

## Product Specification

Wake Focus is a real-time driver-monitoring, distraction-detection, and fleet/navigation desktop system.

It must provide:
1. Camera-based drowsiness / eye-state monitoring with real-time overlays.
2. Object detection for distraction objects using Ultralytics YOLO11.
3. GPS map, own-device tracking, multi-device fleet visibility, incident awareness, and rerouting around accident-driven congestion.

## UI Layout: Exact Geometry

Main window:
- fixed size `800x800`
- do not allow resizing

Top row, total height `500`:
- top-left: `500x500` camera panel
- top-right: `300x500` map panel

Bottom row, total height `300`:
- bottom-left: `400x300` vehicle stats panel
- bottom-middle: `300x300` Fleet Status & Event Log panel
- bottom-right: `100x300` button panel

Button panel must contain exactly:
- Settings
- Profile
- Exit

Settings opens a settings window.
Profile opens a profile window.
Exit terminates the program cleanly.

## Camera Permission And Startup Flow

On app launch:
- show an in-app dialog explaining why the camera is needed
- provide Allow and Deny

If Deny:
- do not start monitoring
- keep the app usable
- show clear guidance on how to enable monitoring later

If Allow:
- attempt to open the camera

If the operating system blocks camera access:
- show platform-specific guidance
- allow retry without restarting the app

Use OpenCV `VideoCapture` as the baseline cross-platform camera interface.
Always release the camera cleanly on shutdown.

## Camera Panel Overlay Requirements

Inside the `500x500` camera panel:
- show live camera feed
- draw eye landmarks as GREEN DOTS
- draw distraction-object boxes in GREEN
- draw the object class label near each box

Use MediaPipe face landmarks for eye visualization.
Use YOLO11 for distraction objects.

At minimum detect:
- cell phone / mobile phone
- paper
- tablet
- food_drink
- cigarette
- book
- handheld_device
- makeup_tool
- wallet
- headphones

Keep the class system extensible.

## Border Rendering Requirement

Alert borders must be drawn INSIDE the `500x500` camera panel.

Border thickness:
- target physical thickness: `2 mm`
- implement DPI-aware `mm -> px` conversion
- prefer physical DPI when sensible
- otherwise fall back to logical DPI
- if exact physical conversion is unreliable on the current platform, fall back to a documented fixed pixel thickness

Document the final conversion strategy used.

## Alert Behaviors

Implement the following exact named behaviors:

### `<<ogohlantirish>>` — Drowsiness

Trigger when the driver is detected as drowsy, sleepy, or in prolonged eye closure beyond normal blinking.

When active:
- show a RED inner border in the camera panel
- play a SHORT BEEP
- continue beeping every 1 second by default until the driver has stably recovered

Recovery condition:
- eyes are behaving normally again
- head/face orientation is road-facing
- recovery remains stable for 3 seconds by default

### `<<ogohlantirish2>>` — Prolonged distraction toward a handheld object

Trigger only when all of the following are true:
- the driver is holding or using a listed distraction object
- head/face/eye direction indicates attention is turned toward that object or away from the road
- this combined condition persists continuously for MORE THAN 30 seconds

When active:
- show an ORANGE inner border in the camera panel
- emit TWO short beeps
- repeat those TWO short beeps every 5 seconds by default until stable recovery

Recovery condition:
- attention returns to the road
- head/face orientation returns road-facing
- eye behavior is normal again
- recovery remains stable for 3 seconds by default

### Alert interaction rules

Use a deterministic state machine with timestamps and hysteresis.

Priority rule:
- `<<ogohlantirish>>` has higher priority than `<<ogohlantirish2>>`

That means:
- if both are eligible, drowsiness wins
- red border/beep behavior overrides orange while drowsiness is active

Implement clean onset, active, recovery-check, and cleared behavior to avoid flicker.

## Audio Rules

Use a cross-platform low-latency approach for short beeps.
Default choice: Qt `QSoundEffect` with WAV assets or generated WAV files.

Requirements:
- reuse loaded sound objects instead of reloading each time
- keep beep frequency, duration, interval, and volume configurable
- document defaults

## ML / CV Requirements

### Face / drowsiness

Use MediaPipe Face Landmarker or Face Mesh as the landmark basis.

Requirements:
- real-time operation
- live-stream-safe timestamps
- ability to draw green eye landmarks
- compute eye-openness features sufficient to distinguish normal blinking from prolonged closure
- infer road-facing vs non-road-facing attention from head pose and gaze-related signals

If using MediaPipe live stream mode:
- use monotonically increasing timestamps
- tolerate dropped frames rather than stalling the UI

### Object detection

Use Ultralytics YOLO11 as the required detection target.
Do not silently upgrade to a different model family.
If the current Ultralytics docs mention newer model branding, keep implementation pinned to YOLO11 unless I explicitly approve a change.

Provide:
- inference pipeline
- class configuration
- training/fine-tuning pipeline
- evaluation method
- export path for deployment

### Training / fine-tuning deliverables

Include:
- dataset collection strategy
- privacy notes
- annotation/labeling guide
- dataset layout
- YOLO dataset YAML
- train commands for Linux and Windows
- evaluation metrics plan
- confusion-matrix/error-analysis guidance
- export at least to `.pt`
- optional ONNX export for edge deployment

Also include a short note on Ultralytics licensing considerations for the intended deployment context.

## Edge Profile: Orange Pi Zero 2W

Treat Orange Pi Zero 2W as a constrained ARM64 target.

Provide an edge profile with:
- YOLO11n
- smaller inference image size such as `320`
- frame skipping
- lower camera FPS if needed
- lower map refresh rate if needed
- lower fleet telemetry rate if needed
- optional ONNX path if feasible

Do not promise a fixed FPS unless you actually measure it on-device.
If you cannot benchmark the real board in this session, say so clearly and provide tuning guidance instead of invented numbers.

## Map / GPS / Fleet / Routing

### Map pane

The map panel must:
- show a real map
- show this device's GPS track
- show other Wake Focus devices
- visualize incidents or congestion zones
- show rerouting recommendations or rerouted paths

If you use Qt `Map`, initialize the required plugin explicitly.

### GPS input

Support pluggable GPS sources:
- `gpsd`
- NMEA serial
- simulation

Linux/Orange Pi should support real GPS via `gpsd` or NMEA.
Windows should support serial GPS if available, otherwise simulation mode.

### Fleet communication

Use MQTT as the default fleet transport unless a stronger repo-local reason forces another approach.

At minimum publish and consume:
- heartbeat messages
- telemetry messages
- incident messages

Minimum telemetry fields:
- device_id
- device_name
- timestamp_utc
- position: lat, lon, speed, heading, accuracy
- alert_state
- monitoring_active
- fleet_group

Minimum incident fields:
- incident_id
- device_id
- timestamp_utc
- incident_type
- position
- severity
- resolved

Use JSON payloads and make broker/auth settings configurable.

### Incident and congestion model

Because external traffic data is unspecified, implement this pragmatic default:
- an accident event creates an incident zone
- congestion is inferred from multiple low-speed devices near the incident over a rolling time window
- other devices should receive an alternative route recommendation that avoids the incident zone when practical

### Routing

Use OSRM HTTP routing as the default baseline unless there is a strong implementation reason to substitute another open routing engine.

Requirements:
- request route geometry
- support alternatives
- implement incident-aware alternative selection
- render route polylines on the map

## Vehicle Stats Panel

Show:
- total distance traveled
- total fuel used

Data strategy:
- if OBD-II via ELM327 is available, integrate it when practical
- otherwise compute distance from GPS track
- if direct fuel data is unavailable, estimate fuel from distance plus configurable fuel economy
- clearly label estimated values

## Architecture Expectations

Choose a clean subsystem split similar to:
- app/bootstrap
- config
- core runtime orchestration
- ML / CV
- fleet / GPS / routing
- UI panels/dialogs
- vehicle stats / OBD
- training assets/scripts
- packaging
- tests

You do not need to match any external repo exactly, but the implementation should be modular and production-minded.

## Packaging Requirements

### Windows

Use PyInstaller or an equivalent Windows-native bundling flow to produce a `.exe`.

Hard rule:
- PyInstaller is not a cross-compiler
- build the Windows artifact on Windows

### Kali Linux

Produce a `.deb` using a Debian-native packaging flow such as `dpkg-buildpackage`.

### Orange Pi Zero 2W ARM64

Provide an ARM64 Linux build/package path.

Hard rule:
- build ARM64 artifacts on ARM64 hardware or an ARM64-native environment
- do not claim x86_64 Linux output is sufficient for Orange Pi

If packaging is blocked by environment-specific dependency issues, provide:
- the best native package flow you can
- a clear fallback run/install path
- honest explanation of what remains to be validated on-device

## Verification Requirements

Validate in this order where feasible:
1. fixed-size UI shell launches
2. camera permission flow works
3. camera feed renders in `500x500`
4. landmarks render
5. object detections render
6. alert state machine behaves correctly
7. beep scheduling behaves correctly
8. GPS simulation updates map and distance
9. fleet telemetry loop works
10. incident/rerouting flow works
11. vehicle stats update
12. packaging/build scripts are valid for Windows, Kali, and ARM64

Run available tests and add tests for core logic, especially:
- alert state machine
- config loading
- telemetry schema
- stats tracking

## Final Report Requirements

At the end, provide a concise but complete report covering:
- what you built
- important architectural decisions
- assumptions/defaults chosen for unspecified areas
- tests/checks run
- bugs found and fixed during implementation
- packaging status by platform
- remaining limitations
- licensing note for Ultralytics usage

If a requirement could not be completed, say exactly why and what remains.
Do not hide incomplete work.

Now begin by creating a concrete implementation plan, then implement Wake Focus end-to-end in this workspace.
```
