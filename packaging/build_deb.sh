#!/bin/bash
# Wake Focus - Debian Package Build Script
# Run on the TARGET architecture (x86_64 or ARM64)
#
# Prerequisites:
#   sudo apt install dpkg-dev debhelper python3 python3-venv python3-pip

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v dpkg-buildpackage >/dev/null 2>&1; then
    echo "ERROR: dpkg-buildpackage is not installed."
    echo "Install it first with: sudo apt install dpkg-dev debhelper"
    exit 1
fi

if ! command -v dh >/dev/null 2>&1; then
    echo "ERROR: debhelper (dh) is not installed."
    echo "Install it first with: sudo apt install debhelper"
    exit 1
fi

echo "========================================"
echo " Wake Focus - Debian Package Build"
echo " Architecture: $(dpkg --print-architecture)"
echo "========================================"

cd "$PROJECT_DIR"

# Copy debian directory to project root
cp -r packaging/debian .

# Build the package
dpkg-buildpackage -us -uc -b

echo ""
echo "Build complete!"
echo "Package: ../wake-focus_1.0.0-1_$(dpkg --print-architecture).deb"
echo ""
echo "Install with:"
echo "  sudo dpkg -i ../wake-focus_1.0.0-1_$(dpkg --print-architecture).deb"
echo "  sudo apt install -f  # Fix any missing dependencies"
