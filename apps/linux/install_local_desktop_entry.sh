#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
BIN_DIR="${HOME}/.local/bin"

mkdir -p "$DESKTOP_DIR" "$BIN_DIR"

cat > "$BIN_DIR/wake-focus-local" <<EOF
#!/usr/bin/env bash
exec "${PROJECT_DIR}/apps/linux/run_wake_focus.sh" "\$@"
EOF
chmod +x "$BIN_DIR/wake-focus-local"

sed "s|Exec=.*|Exec=$BIN_DIR/wake-focus-local|; s|Path=.*|Path=$PROJECT_DIR|" \
    "$PROJECT_DIR/apps/linux/wake-focus.desktop" \
    > "$DESKTOP_DIR/wake-focus.desktop"

chmod +x "$DESKTOP_DIR/wake-focus.desktop"

echo "Installed desktop entry:"
echo "  $DESKTOP_DIR/wake-focus.desktop"
echo "Launcher:"
echo "  $BIN_DIR/wake-focus-local"
