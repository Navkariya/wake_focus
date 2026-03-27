# Wake Focus

Wake Focus is a production-oriented desktop application for:

- driver drowsiness monitoring
- distraction-object detection
- GPS/fleet tracking and incident-aware rerouting

This workspace includes:

- the application under `src/wake_focus`
- configs in `config/`
- training assets in `training/`
- packaging scripts in `packaging/`
- tests in `tests/`
- research/prompt assets in `docs/` and `prompts/`

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-build-isolation -e .[dev]
QT_QPA_PLATFORM=offscreen pytest tests -q
python -m wake_focus
```

## Run The App From This Repo

Direct launcher:

```bash
./apps/linux/run_wake_focus.sh
```

Install a local desktop entry on this machine:

```bash
./apps/linux/install_local_desktop_entry.sh
```

This creates:

- `~/.local/bin/wake-focus-local`
- `~/.local/share/applications/wake-focus.desktop`

Build a local-machine launcher `.deb`:

```bash
./apps/linux/build_local_launcher_deb.sh
```

This produces a small launcher package in `artifacts/` that points to this
exact repo path on this machine.

If your environment already has the dependencies installed and you only need the
local package/CLI entry point, use:

```bash
pip install --no-build-isolation --no-deps -e .
```

## Packaging

Linux `.deb` packaging is provided under `packaging/`:

- `packaging/build_deb.sh` for Debian/Kali on the current architecture
- `packaging/build_arm64_deb.sh` for ARM64 boards such as Orange Pi Zero 2W

Both scripts require system packaging tools such as `dpkg-buildpackage` and
`debhelper` to be installed first.

## Optional OBD-II Support

OBD-II integration is optional because the `obd` package is not available on
all platforms and Python versions used by this project. Install it separately
when you need ELM327 support:

```bash
pip install "obd>=0.7.2"
```

## Optional YOLO Weights

Place local detector weights in `models/`, for example:

- `models/yolo26n.pt`
- `models/yolo26n.onnx`

If no local model is present, Wake Focus now starts with object detection
disabled instead of attempting an online download during startup.

## Integrated YOLO26 Handheld Classes

Current default YOLO26 distraction filter:

- `cell phone`
- `book` as a practical proxy for `paper` / `document`
- `laptop`
- `mouse`
- `remote`
- `keyboard`

Configured aliases also include:

- `phone`
- `mobile phone`
- `smartphone`
- `paper`
- `document`
- `notebook`
- `remote control`
- `electronic gadget`
