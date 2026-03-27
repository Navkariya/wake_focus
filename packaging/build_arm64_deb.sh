#!/bin/bash
# Wake Focus - ARM64 Build Script (Orange Pi Zero 2W)
#
# This script MUST be run on the ARM64 device itself or in an
# ARM64 chroot/container. PyInstaller and pip cannot cross-compile.
#
# Prerequisites on Orange Pi:
#   sudo apt update
#   sudo apt install python3 python3-pip python3-venv python3-dev
#   sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6
#   sudo apt install qt6-base-dev qt6-webengine-dev qt6-multimedia-dev
#   sudo apt install dpkg-dev debhelper

set -e

echo "========================================"
echo " Wake Focus - ARM64 Build (Orange Pi)"
echo " Device: $(uname -m)"
echo "========================================"

# Verify ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: This script is intended for ARM64 (aarch64)"
    echo "Current architecture: $ARCH"
    echo "Continue anyway? (y/N)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

if ! command -v dpkg-buildpackage >/dev/null 2>&1; then
    echo "ERROR: dpkg-buildpackage is not installed."
    echo "Install it first with: sudo apt install dpkg-dev debhelper"
    exit 1
fi

# Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv-arm64
source .venv-arm64/bin/activate

# Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-arm64.txt

# Verify critical imports
echo "[3/5] Verifying imports..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "
try:
    import mediapipe
    print(f'MediaPipe: {mediapipe.__version__}')
except ImportError:
    print('MediaPipe: NOT AVAILABLE (may need to build from source)')
"
python3 -c "
try:
    import onnxruntime
    print(f'ONNX Runtime: {onnxruntime.__version__}')
except ImportError:
    print('ONNX Runtime: NOT AVAILABLE')
"

# Build .deb
echo "[4/5] Building Debian package..."
cp -r packaging/debian .
dpkg-buildpackage -us -uc -b || {
    echo "dpkg-buildpackage failed. Trying manual install instead..."
    pip install --no-build-isolation --no-deps -e .
    echo "Installed in development mode."
    echo "Run with: python -m wake_focus --edge"
}

# Summary
echo ""
echo "[5/5] Build Summary"
echo "========================================"
echo "Architecture: $ARCH"
echo "Python: $(python3 --version)"

DEB=$(ls ../*.deb 2>/dev/null | head -1)
if [ -n "$DEB" ]; then
    echo "Package: $DEB"
    echo ""
    echo "Install: sudo dpkg -i $DEB && sudo apt install -f"
else
    echo "No .deb built. Run manually:"
    echo "  source .venv-arm64/bin/activate"
    echo "  python -m wake_focus --edge --config config/edge_config.yaml"
fi

echo ""
echo "Edge mode runs with:"
echo "  wake-focus --edge"
echo "  (Uses YOLO11n @ 320px, frame_skip=3, 15fps capture)"
