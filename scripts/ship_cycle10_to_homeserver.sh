#!/bin/bash
# Cycle-10 ship: deploy + restart homeserver tmux + smoke-check.
#
# Pre-conditions:
# 1. cycle-10 LGBM walk-forward done (51/51 LGBM checkpoints in
#    expanding_pruned_svi_har_pd_v1/, 51/51 TransVol in expanding_hybrid/)
# 2. cycle-10 predictions parquet regenerated (post 2026-05-10 mtime on
#    BTCUSDT_pruned_svi_har_pd_predictions*.parquet)
# 3. cycle-10 aligner_betas_by_month.json regenerated (post 2026-05-10 mtime)
# 4. post_retrain_impact_cycle10.md showed acceptable PnL/Sharpe trade-off
#    (operator MUST review before running this script)
#
# Sequence:
# A. deploy_to_homeserver.sh — local pytest + scp + remote validation
# B. Capture pre-restart SHA256 of remote LGBM + TransVol bytes
# C. Restart homeserver tmux 'deribit' session
# D. Tail boot log until "LightGBM loaded" + "TransVol loaded" lines
#    appear; print SHA256 fingerprints for operator confirmation

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="monitor"
REMOTE_DIR="/hdd/ml-trading-system"

echo ""
echo "================================================================"
echo "CYCLE-10 SHIP TO HOMESERVER  $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ── A. Deploy ──────────────────────────────────────────────────────
echo ""
echo "[A/D] Running deploy_to_homeserver.sh (local tests + scp + remote validate)..."
bash scripts/deploy_to_homeserver.sh
echo "[A/D] DONE"

# ── B. Capture remote pre-restart SHA256 ─────────────────────────
echo ""
echo "[B/D] Capturing remote pre-restart SHA256 fingerprints..."
PRE_LGBM_SHA=$(ssh "$HOST" "sha256sum $REMOTE_DIR/vol_model/output/models/expanding_pruned_svi_har_pd_v1/2026-04.txt | head -c 12")
PRE_TV_SHA=$(ssh "$HOST" "sha256sum $REMOTE_DIR/vol_model/output/models/expanding_hybrid/2026-04_transvol.pt | head -c 12")
echo "  remote LGBM 2026-04 sha256:    $PRE_LGBM_SHA"
echo "  remote TransVol 2026-04 sha256: $PRE_TV_SHA"

# ── C. Restart homeserver tmux ───────────────────────────────────
echo ""
echo "[C/D] Restarting homeserver tmux 'deribit' session..."
LOG_NAME="paper_v2_$(date -u +%Y%m%d_%H%M).log"
ssh "$HOST" "tmux kill-session -t deribit 2>/dev/null || true; \
  tmux new-session -d -s deribit -c $REMOTE_DIR \
    'PYTHONPATH=. bash deribit_paper/scripts/run_engine.sh \
       --paper --config-preset v2 --duration-minutes 0 \
       --log-level INFO \
       2>&1 | tee deribit_paper/logs/$LOG_NAME'"
echo "  tmux 'deribit' session restarted; logging to deribit_paper/logs/$LOG_NAME"

# ── D. Wait for boot + verify SHA256 in boot log ─────────────────
echo ""
echo "[D/D] Waiting up to 90s for engine boot lines (LightGBM + TransVol load)..."
DEADLINE=$(($(date +%s) + 90))
LGBM_LINE=""
TV_LINE=""
while [ $(date +%s) -lt $DEADLINE ]; do
    LGBM_LINE=$(ssh "$HOST" "grep -m1 'LightGBM loaded:' $REMOTE_DIR/deribit_paper/logs/$LOG_NAME 2>/dev/null || true")
    TV_LINE=$(ssh "$HOST" "grep -m1 'TransVol loaded:' $REMOTE_DIR/deribit_paper/logs/$LOG_NAME 2>/dev/null || true")
    if [ -n "$LGBM_LINE" ] && [ -n "$TV_LINE" ]; then
        break
    fi
    sleep 5
done

if [ -z "$LGBM_LINE" ] || [ -z "$TV_LINE" ]; then
    echo "  WARN: didn't see both load lines within 90s. Engine may be slow to boot."
    echo "  Manually tail: ssh $HOST 'tail -f $REMOTE_DIR/deribit_paper/logs/$LOG_NAME'"
    exit 2
fi

echo ""
echo "  LGBM:     $LGBM_LINE"
echo "  TransVol: $TV_LINE"
echo ""

# Extract sha256 from logs and confirm match
LOG_LGBM_SHA=$(echo "$LGBM_LINE" | grep -oE "sha256=[a-f0-9]{12}" | sed "s/sha256=//")
LOG_TV_SHA=$(echo "$TV_LINE" | grep -oE "sha256=[a-f0-9]{12}" | sed "s/sha256=//")

echo "  Verification:"
if [ "$LOG_LGBM_SHA" = "$PRE_LGBM_SHA" ]; then
    echo "    LGBM sha256 matches deployed bytes: $LOG_LGBM_SHA OK"
else
    echo "    LGBM sha256 MISMATCH: log=$LOG_LGBM_SHA  deployed=$PRE_LGBM_SHA"
fi
if [ "$LOG_TV_SHA" = "$PRE_TV_SHA" ]; then
    echo "    TransVol sha256 matches deployed bytes: $LOG_TV_SHA OK"
else
    echo "    TransVol sha256 MISMATCH: log=$LOG_TV_SHA  deployed=$PRE_TV_SHA"
fi

echo ""
echo "================================================================"
echo "SHIP COMPLETE  $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""
echo "Monitor live engine: ssh $HOST 'tail -f $REMOTE_DIR/deribit_paper/logs/$LOG_NAME'"
echo "Telegram heartbeats fire every ~hour. First heartbeat in <60min confirms full boot OK."
