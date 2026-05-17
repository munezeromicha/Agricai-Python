#!/usr/bin/env bash
# Idempotent PM2 deploy: one Agricai-Python process on port 8000.
#   chmod +x scripts/pm2-deploy.sh && ./scripts/pm2-deploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_NAME="Agricai-Python"

# Remove every process with this name (avoids duplicate ids fighting for port 8000).
pm2 delete "$APP_NAME" 2>/dev/null || true

pm2 start ecosystem.config.cjs
pm2 save

echo ""
pm2 list
echo ""
echo "Smoke test:"
curl -sf "http://127.0.0.1:8000/health" && echo || {
  echo "Health check failed. Run: pm2 logs $APP_NAME --lines 50" >&2
  exit 1
}
