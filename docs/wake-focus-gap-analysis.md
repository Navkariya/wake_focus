# Wake Focus Prompt Gap Analysis

## Scope

This document compares the current draft prompt from the user against:

- the product requirements embedded in that draft
- official primary documentation
- the neighboring `../wake_focus` implementation shape

Goal: tighten the prompt until it is decision-complete and execution-friendly for a coding agent.

## What The Current Draft Already Gets Right

- It is unusually explicit about layout geometry.
- It names the two alert behaviors and their timing expectations.
- It already chooses a sensible senior-grade stack: PySide6, OpenCV, MediaPipe, Ultralytics, Qt Location/Positioning, QSoundEffect.
- It correctly calls out Windows `.exe`, Kali `.deb`, and Orange Pi ARM64 support.
- It already treats some undefined behavior as “NO SPECIFIC CONSTRAINT”, which is the right instinct.

## Main Gaps To Fix

| Area | Gap in current draft | Why it matters | Change in final prompt |
|---|---|---|---|
| Execution format | It asks for “full source code by file” printed inline | That is fragile and wasteful for a coding agent working in a real repo | Instruct Cloud Opus to create files in the workspace and summarize results; only print file contents when explicitly requested |
| Plan/execute sequencing | “Output the PLAN only” followed immediately by “Then implement” is internally awkward | Some agents may stop after the plan or conflate the two phases | Make the sequence explicit: plan first, then implement in the same session unless blocked |
| Packaging realism | The prompt mentions packaging, but build-host constraints need to be non-optional | PyInstaller is not a cross-compiler; ARM builds must match architecture | Add hard requirement that Windows builds happen on Windows and ARM64 builds happen on ARM64-native environments |
| Orange Pi performance | It asks for expected FPS without guarding against overpromising | Resource-constrained boards need measured validation, not theoretical claims | Require measured or estimated-on-device reporting and an edge profile with frame skipping and reduced input size |
| Ultralytics licensing | No explicit licensing caution | A production deliverable should not silently ignore licensing implications | Require a short licensing note in docs/build output |
| Map provider assumptions | The draft assumes “show a map” but not the provider/plugin initialization details | Qt `Map` requires a plugin and provider behavior varies | Force explicit map plugin setup and fallback behavior |
| Camera permissions | It asks for permission but not the platform-specific failure model | Desktop camera permission flows vary by OS | Require in-app consent, camera-open attempt, blocked-state guidance, and retry |
| Telemetry contract | It says to use MQTT or hub, but the schema is still under-specified | Fleet interoperability depends on stable message shape | Lock a minimum JSON schema for heartbeat, telemetry, and incidents |
| Alert recovery semantics | “Fully refocuses” is human-readable but not machine-decision-complete | The coding agent still has to invent thresholds and stable windows | Explicitly define recovery as stable road-facing + normal eye behavior for N seconds |
| Validation flow | Verification is requested, but the order is not pinned | Agents often skip the most failure-prone runtime checks | Add a required validation order from UI shell to camera to inference to fleet to packaging |

## Risky Or Contradictory Instructions In The Current Draft

### 1. “Print everything” is the wrong optimization

The draft asks for architecture, folder table, code by file, training pipeline, validation log, build steps, and more in a single giant printed response. For an agent that can actually write files, this increases the risk of truncation and inconsistency.

Recommended fix:
- Tell Cloud Opus to **create the repository contents directly**.
- Ask for a concise final report with key paths, tests run, limitations, and packaging notes.

### 2. The “plan only, then implement” instruction should be normalized

This is recoverable, but it is easy for an agent to interpret as “stop after the plan” in one turn.

Recommended fix:
- Use wording like: “Start by forming a concrete implementation plan. Then implement it end-to-end in this same session unless a hard blocker appears.”

### 3. “No specific constraint” needs stronger handling

The draft correctly uses the phrase, but it should also require:
- documented default
- config knob if practical
- explicit mention in the final report

Without that, the coding agent may still choose hidden defaults.

## Repo-Reality Alignment

The neighboring `../wake_focus` repo suggests that the product naturally decomposes into:

- `core`: orchestration, alerts, audio, camera
- `ml`: face mesh, drowsiness, gaze, object detection
- `fleet`: GPS, MQTT, incidents, routing
- `ui`: main window and panels
- `vehicle`: OBD and stats
- `config`, `training`, `packaging`, `tests`

This is useful evidence that the prompt should encourage a similar **subsystem split**, but it should not hardcode those exact filenames. A repo-agnostic prompt should describe the architecture pattern, not force a clone of the neighboring tree.

## Decisions Added In The Final Prompt

These are the biggest “implementer-choice” gaps that the final prompt should remove:

| Topic | Final prompt decision |
|---|---|
| Bottom-middle `300x300` panel | Fleet Status & Event Log |
| Drowsy beep interval | 1 second by default |
| Distraction beep cadence | 2 beeps every 5 seconds |
| Stable recovery window | 3 seconds by default |
| Routing baseline | OSRM HTTP |
| Fleet baseline | MQTT |
| GPS baseline | gpsd, NMEA serial, simulation |
| Fuel fallback | GPS distance + configured fuel economy estimate |
| Border thickness fallback | DPI-aware mm-to-px, static pixel fallback documented |
| Orange Pi fallback | edge profile and editable-install fallback if packaging is blocked |

## Things The Final Prompt Should Still Avoid Over-Specifying

- Exact neural thresholds beyond the defaults needed for deterministic behavior
- A guaranteed Orange Pi FPS number
- A promise that every dependency will package cleanly on every ARM image without adaptation
- Silent migration from YOLO11 to another model family

## Bottom Line

The user’s draft is already strong as a product brief. The final prompt improves it by making it more executable:

- fewer contradictory output instructions
- stronger build-host and packaging realism
- explicit telemetry and recovery defaults
- clearer validation sequence
- better fit for a coding agent that can modify a real workspace
