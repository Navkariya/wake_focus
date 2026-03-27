#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -x "$PROJECT_DIR/.venv/bin/wake-focus" ]]; then
    echo "Wake Focus virtual environment is missing."
    echo "Expected launcher: $PROJECT_DIR/.venv/bin/wake-focus"
    echo "From the project root, run:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    echo "  .venv/bin/pip install -e . --no-build-isolation"
    exit 1
fi

export PYTHONPATH="$PROJECT_DIR/src"
export WAKE_FOCUS_CONFIG="${WAKE_FOCUS_CONFIG:-$PROJECT_DIR/config/default_config.yaml}"

exec "$PROJECT_DIR/.venv/bin/wake-focus" "$@"
