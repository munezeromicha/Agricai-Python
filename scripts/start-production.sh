#!/usr/bin/env bash
# Production start (Linux). Run from anywhere:
#   ./scripts/start-production.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UVICORN=""
for dir in ${VENV_DIR:-} .venv venv; do
  [[ -z "$dir" ]] && continue
  candidate="${ROOT}/${dir}/bin/uvicorn"
  if [[ -x "$candidate" ]]; then
    UVICORN="$candidate"
    break
  fi
done

if [[ -z "$UVICORN" ]]; then
  echo "No venv uvicorn found (tried .venv and venv under ${ROOT})." >&2
  echo "  python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec "$UVICORN" app.main:app --host "$HOST" --port "$PORT"
