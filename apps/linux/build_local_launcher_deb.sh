#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACT_DIR="$PROJECT_DIR/artifacts"
BUILD_DIR="$PROJECT_DIR/build/local-launcher-deb"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$PROJECT_DIR/pyproject.toml" | head -n 1)"
VERSION="${VERSION:-1.0.0}"
ARCH="all"
PACKAGE_NAME="wake-focus-local-launcher"
OUTPUT_DEB="$ARTIFACT_DIR/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
PKG_ROOT="$BUILD_DIR/${PACKAGE_NAME}_${VERSION}_${ARCH}"

rm -rf "$PKG_ROOT"
mkdir -p \
    "$PKG_ROOT/DEBIAN" \
    "$PKG_ROOT/usr/bin" \
    "$PKG_ROOT/usr/share/applications"

cat > "$PKG_ROOT/DEBIAN/control" <<EOF
Package: ${PACKAGE_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: Wake Focus Team <wakefocus@example.com>
Depends: bash
Description: Local launcher for the Wake Focus repo on this machine
 This package installs a desktop launcher and command wrapper that point to:
 ${PROJECT_DIR}
 .
 It is intended for this local machine and this exact project path.
EOF

cat > "$PKG_ROOT/usr/bin/wake-focus-local" <<EOF
#!/usr/bin/env bash
exec "${PROJECT_DIR}/apps/linux/run_wake_focus.sh" "\$@"
EOF
chmod 0755 "$PKG_ROOT/usr/bin/wake-focus-local"

cat > "$PKG_ROOT/usr/share/applications/wake-focus-local.desktop" <<EOF
[Desktop Entry]
Name=Wake Focus (Local Repo)
Comment=Run Wake Focus from the local repo checkout
Exec=/usr/bin/wake-focus-local
Terminal=false
Type=Application
Categories=Utility;
Path=${PROJECT_DIR}
EOF

mkdir -p "$ARTIFACT_DIR"
dpkg-deb --build "$PKG_ROOT" "$OUTPUT_DEB"

echo "Built launcher package:"
echo "  $OUTPUT_DEB"
