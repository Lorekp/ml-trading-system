#!/usr/bin/env bash
# Auto-restart wrapper for the Deribit paper-trading engine.
#
# Restarts the engine automatically on crashes with exponential
# backoff (5s → 10s → 20s → ... → 300s max).  Stops permanently on:
#   - Clean exit (rc=0): engine.stop() was called normally
#   - Shutdown exit (rc=42): operator requested permanent stop
#
# Usage:
#   bash deribit_paper/scripts/run_engine.sh [engine args...]
#
# Examples:
#   bash deribit_paper/scripts/run_engine.sh --duration-minutes 60
#   bash deribit_paper/scripts/run_engine.sh --paper --duration-minutes 60
#
# To stop without restart:  kill -TERM <engine_pid>  (clean stop, rc=0)
# To drain then stop:       touch deribit_paper/state/drain
# To pause new trades:      touch deribit_paper/state/pause
# To resume after pause:    rm deribit_paper/state/pause
set -uo pipefail

MAX_BACKOFF=300   # 5 minutes max wait between restarts
INITIAL_BACKOFF=5
EXIT_CODE_SHUTDOWN=42

backoff=$INITIAL_BACKOFF
restart_count=0

# Pick the Python interpreter — homeserver has python3, dev boxes
# may have either. Prefer python3, fall back to python.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY=python3
    elif command -v python >/dev/null 2>&1; then
        PY=python
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FATAL: no python interpreter found (tried python3, python)"
        exit 127
    fi
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Using interpreter: $PY ($($PY --version 2>&1))"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting engine (restart #${restart_count})..."
    "$PY" -m deribit_paper.cli --run-paper "$@"
    rc=$?

    if [ $rc -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Engine exited cleanly (rc=0). Stopping wrapper."
        break
    fi

    if [ $rc -eq $EXIT_CODE_SHUTDOWN ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Engine requested permanent shutdown (rc=42). Stopping wrapper."
        break
    fi

    restart_count=$((restart_count + 1))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Engine crashed (rc=$rc). Restarting in ${backoff}s... (restart #${restart_count})"
    sleep $backoff

    # Exponential backoff, capped
    backoff=$((backoff * 2))
    if [ $backoff -gt $MAX_BACKOFF ]; then
        backoff=$MAX_BACKOFF
    fi

    # Reset backoff if the engine ran for >5 minutes (was stable)
    # This is handled by the engine's own uptime tracking.
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wrapper exiting after ${restart_count} restart(s)."
