#!/bin/bash
# Serve the pre-built BTC vol-arb backtest dashboard locally.
#
# Usage:
#   bash serve.sh           # serves on http://127.0.0.1:8765
#   bash serve.sh 8080      # serves on the chosen port
#
# Then open the URL in any modern browser.
#
# Requires: Python 3 (any 3.x) — uses only the standard library
# http.server. No Node.js, no npm install, no Python packages.

set -e
PORT="${1:-8765}"
HERE="$(cd "$(dirname "$0")" && pwd)"

cd "$HERE"
echo "=========================================================="
echo "  BTC vol-arb dashboard — static bundle"
echo "  Open: http://127.0.0.1:${PORT}"
echo "  Ctrl-C to stop"
echo "=========================================================="
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
