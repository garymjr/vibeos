#!/usr/bin/env bash
set -euo pipefail

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -is uv run python3 main.py "$@"
fi

exec uv run python3 main.py "$@"
