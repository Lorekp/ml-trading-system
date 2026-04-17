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
#   bash deribit_paper/scripts/run_engine.sh --venue testnet --dry-run
#   bash deribit_paper/scripts/run_engine.sh --venue prod --live
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

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting engine (restart #${restart_count})..."
    python -m deribit_paper.cli --run-paper "$@"
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
