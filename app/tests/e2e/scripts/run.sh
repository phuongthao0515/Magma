#!/usr/bin/env bash
# Launch the agent or the frontend (Linux/macOS).
#   bash run.sh agent
#   bash run.sh frontend
# Paths derive from this script's location. Backend defaults to the deployed HF
# Space; override with DOME_SERVER_URL.
set -euo pipefail
TARGET="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # tests/e2e/scripts -> app
BACKEND="${DOME_SERVER_URL:-https://wafair-dome.hf.space}"

case "$TARGET" in
  agent)
    cd "$APP_DIR"
    echo "[run agent] app=$APP_DIR backend=$BACKEND"
    exec python agent/executor.py --server-url "$BACKEND"
    ;;
  frontend)
    # VITE_API_URL = URL the browser calls directly (overrides the
    # localhost:8000 default in public/config.js). VITE_BACKEND_URL = proxy target.
    export VITE_API_URL="$BACKEND"
    export VITE_BACKEND_URL="$BACKEND"
    cd "$APP_DIR/client"
    echo "[run frontend] client=$APP_DIR/client VITE_API_URL=$VITE_API_URL"
    exec npx vite --port 4000
    ;;
  *)
    echo "usage: run.sh <agent|frontend>" >&2
    exit 2
    ;;
esac
