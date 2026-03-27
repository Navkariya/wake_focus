# Wake Focus Acceptance Checklist

Use this checklist to review the output of the final Cloud Opus prompt.

## Hard Failures

- [ ] The result is not a real repo/workspace implementation.
- [ ] The main window is not fixed at `800x800`.
- [ ] The `500x500` camera panel or `300x500` map panel dimensions are wrong.
- [ ] The alert names `<<ogohlantirish>>` and `<<ogohlantirish2>>` are missing or behaviorally altered.
- [ ] Drowsiness does not have priority over prolonged distraction.
- [ ] The build instructions imply cross-compiling Windows with PyInstaller from Linux.
- [ ] Orange Pi support is claimed without an ARM64-native build/run path.
- [ ] The implementation silently ignores camera-permission failure flows.

## UI And Layout

- [ ] Window is fixed to `800x800`.
- [ ] Top-left camera panel is `500x500`.
- [ ] Top-right map panel is `300x500`.
- [ ] Bottom-left vehicle stats panel is `400x300`.
- [ ] Bottom-right button panel is `100x300`.
- [ ] Bottom-middle `300x300` region is implemented and documented as `Fleet Status & Event Log`.
- [ ] Button panel contains exactly `Settings`, `Profile`, and `Exit`.

## Camera And Overlay Behavior

- [ ] App asks for camera access in-app on entry.
- [ ] Deny flow keeps monitoring disabled and shows guidance.
- [ ] Allow flow attempts to open the camera.
- [ ] Blocked-by-OS flow gives platform-specific help and retry.
- [ ] Eye landmarks are drawn as green dots.
- [ ] Distraction objects are shown with green boxes and labels.
- [ ] Camera border is drawn inside the `500x500` view.
- [ ] Border thickness uses DPI-aware `mm -> px` conversion plus documented fallback.

## Alert State Machine

- [ ] `<<ogohlantirish>>` triggers on drowsiness / prolonged eye closure.
- [ ] `<<ogohlantirish>>` shows a red border.
- [ ] `<<ogohlantirish>>` repeats short beeps until stable recovery.
- [ ] `<<ogohlantirish2>>` requires object use plus off-road orientation plus more than 30 seconds continuous persistence.
- [ ] `<<ogohlantirish2>>` shows an orange border.
- [ ] `<<ogohlantirish2>>` emits 2 short beeps every 5 seconds until stable recovery.
- [ ] Stable recovery window is documented and configurable.
- [ ] Priority and hysteresis are implemented deterministically.

## ML / CV

- [ ] Face landmarking is implemented with MediaPipe Face Landmarker or Face Mesh.
- [ ] Real-time mode uses monotonic timestamps and handles dropped frames correctly.
- [ ] Drowsiness distinguishes normal blinks from prolonged closure.
- [ ] Object detection target remains YOLO11 unless an explicit user-approved change is documented.
- [ ] Minimum object coverage includes phone, paper, and additional distraction classes.
- [ ] Training/fine-tuning docs include dataset strategy, labeling guide, splits, YAML, train commands, eval, and export.
- [ ] Edge profile exists for Orange Pi class hardware.

## Map / GPS / Fleet / Routing

- [ ] Map pane shows own location track.
- [ ] Map pane shows other Wake Focus devices.
- [ ] GPS sources include `gpsd`, NMEA serial, and simulation.
- [ ] Fleet communication includes heartbeat, telemetry, and incident events.
- [ ] Telemetry schema includes device identity, UTC timestamp, position, speed, alert state, and monitoring state.
- [ ] Accident/incident zones affect rerouting behavior.
- [ ] Routing is implemented with a documented engine such as OSRM.
- [ ] Rerouting away from congestion/incidents is demonstrated or test-covered.

## Vehicle Stats

- [ ] Vehicle stats pane shows distance traveled.
- [ ] Vehicle stats pane shows fuel used.
- [ ] OBD-II path is documented.
- [ ] Fuel estimation fallback is documented when OBD is unavailable.

## Packaging And Platform Support

- [ ] Windows packaging path produces a `.exe` via a Windows-native build.
- [ ] Kali packaging path produces a `.deb`.
- [ ] Orange Pi path supports ARM64 Linux with a native or ARM64-matched build flow.
- [ ] Packaging docs explicitly state that PyInstaller is not a cross-compiler.
- [ ] Debian packaging uses `dpkg-buildpackage` or an equivalent Debian-native workflow.
- [ ] ARM64 limitations and fallback behavior are documented honestly.

## Validation And Reporting

- [ ] Validation order covers UI shell, camera, inference, alerts, GPS simulation, fleet sync, and packaging.
- [ ] Tests exist for the alert state machine and at least minimal config/schema behavior.
- [ ] The final report documents assumptions chosen for every previously unspecified area.
- [ ] The final report includes a short licensing caution for Ultralytics usage.

## Source-Backed Review Anchors

Reviewer should confirm that the implementation and docs stay compatible with these sources:

- MediaPipe Face Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python>
- MediaPipe Face Mesh: <https://chuoling.github.io/mediapipe/solutions/face_mesh.html>
- OpenCV `VideoCapture`: <https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html>
- Qt `QSoundEffect`: <https://doc.qt.io/qt-6/qsoundeffect.html>
- Qt `Map`: <https://doc.qt.io/qt-6/qml-qtlocation-map.html>
- Qt NMEA plugin: <https://doc.qt.io/qt-6/position-plugin-nmea.html>
- PyInstaller operating mode: <https://pyinstaller.org/en/stable/operating-mode.html>
- `dpkg-buildpackage`: <https://manpages.debian.org/unstable/dpkg-dev/dpkg-buildpackage.1.en.html>
- Debian packaging guide: <https://www.debian.org/doc/manuals/debmake-doc/>
- OSRM API: <https://project-osrm.org/docs/v5.24.0/api/>
- MQTT 3.1.1: <https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/mqtt-v3.1.1.html>
- Orange Pi Zero 2W: <https://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-2W.html>
- Orange Pi Zero 2W wiki: <https://www.orangepi.org/orangepiwiki/index.php/Orange_Pi_Zero_2W>
- Ultralytics docs: <https://docs.ultralytics.com/>
