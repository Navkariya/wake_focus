# -*- mode: python ; coding: utf-8 -*-
# Wake Focus - PyInstaller Spec File
# Build with: pyinstaller packaging/windows/wake_focus.spec

import sys
from pathlib import Path

block_cipher = None

# Project root
ROOT = Path(SPECPATH).parent.parent

a = Analysis(
    [str(ROOT / 'src' / 'wake_focus' / '__main__.py')],
    pathex=[str(ROOT / 'src')],
    binaries=[],
    datas=[
        # Config files
        (str(ROOT / 'config'), 'config'),
        # Training class list (for reference)
        (str(ROOT / 'training' / 'class_list.txt'), 'training'),
        (str(ROOT / 'training' / 'dataset.yaml'), 'training'),
    ],
    hiddenimports=[
        'wake_focus',
        'wake_focus.app',
        'wake_focus.config',
        'wake_focus.constants',
        'wake_focus.core',
        'wake_focus.core.camera_manager',
        'wake_focus.core.perception_engine',
        'wake_focus.core.alert_state_machine',
        'wake_focus.core.audio_manager',
        'wake_focus.ml',
        'wake_focus.ml.face_mesh',
        'wake_focus.ml.drowsiness_detector',
        'wake_focus.ml.head_pose',
        'wake_focus.ml.gaze_analyzer',
        'wake_focus.ml.object_detector',
        'wake_focus.fleet',
        'wake_focus.fleet.gps_manager',
        'wake_focus.fleet.mqtt_client',
        'wake_focus.fleet.fleet_monitor',
        'wake_focus.fleet.incident_manager',
        'wake_focus.fleet.route_manager',
        'wake_focus.fleet.telemetry_schema',
        'wake_focus.vehicle',
        'wake_focus.vehicle.stats_tracker',
        'wake_focus.vehicle.obd_interface',
        'wake_focus.ui',
        'wake_focus.ui.main_window',
        'wake_focus.ui.camera_panel',
        'wake_focus.ui.map_panel',
        'wake_focus.ui.vehicle_stats_panel',
        'wake_focus.ui.fleet_status_panel',
        'wake_focus.ui.button_panel',
        'wake_focus.ui.permission_dialog',
        'wake_focus.ui.settings_dialog',
        'wake_focus.ui.profile_dialog',
        'wake_focus.ui.styles',
        # External dependencies that may be missed
        'mediapipe',
        'cv2',
        'ultralytics',
        'paho.mqtt',
        'paho.mqtt.client',
        'folium',
        'scipy.spatial',
        'yaml',
        'numpy',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtMultimedia',
        'PySide6.QtPositioning',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WakeFocus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WakeFocus',
)
