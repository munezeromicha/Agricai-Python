#!/usr/bin/env bash
# Production start (Linux). Run from anywhere:
#   ./scripts/start-production.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UVICORN="${ROOT}/.venv/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  echo "Missing ${UVICORN}. Create the venv and install deps:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec "$UVICORN" app.main:app --host "$HOST" --port "$PORT"
