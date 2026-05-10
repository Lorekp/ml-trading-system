#!/bin/bash
# Cycle-10 post-TransVol-retrain orchestration (2026-05-09).
#
# Runs the canonical post-retrain pipeline as a single fail-fast chain:
#   1. LightGBM walk-forward retrain (--force-retrain, 200/50 prod config)
#   2. Re-extract predictions parquet
#   3. Run walk-forward aligner → aligned-bias parquet
#   4. Extract per-month β* JSON for live engine
#   5. V2 backtest comparison vs pre-retrain baseline (uses existing
#      Kelly K1 sidecar — refresh is an optional follow-up, see notes)
#
# Pre-condition: TransVol retrain has finished (51 checkpoints in
# vol_model/output/models/expanding_hybrid/{YYYY-MM}_transvol.pt and the
# summary CSV exists at vol_model/output/transvol_optuna_retrain_summary.csv).
#
# Logs to vol_model/output/cycle10_post_retrain.log; fail-fast on any step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_FILE="vol_model/output/cycle10_post_retrain.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "================================================================"
echo "CYCLE-10 POST-RETRAIN ORCHESTRATION  $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ── Pre-check: TransVol retrain finished ──────────────────────────────
TV_DIR="vol_model/output/models/expanding_hybrid"
TV_COUNT=$(ls "$TV_DIR"/*_transvol.pt 2>/dev/null | wc -l)
TV_SUMMARY="vol_model/output/transvol_optuna_retrain_summary.csv"
if [ "$TV_COUNT" -lt 51 ]; then
    echo "ERROR: Only $TV_COUNT TransVol checkpoints in $TV_DIR (expected 51)."
    echo "       The retrain hasn't finished yet. Aborting."
    exit 1
fi
if [ ! -f "$TV_SUMMARY" ]; then
    echo "ERROR: TransVol retrain summary not found at $TV_SUMMARY."
    echo "       Either retrain hasn't finished or the run errored. Aborting."
    exit 1
fi
echo "Pre-check PASS: $TV_COUNT TransVol checkpoints + summary CSV present."
echo ""

# ── Step 1: LightGBM walk-forward retrain ─────────────────────────────
# Cycle-9 reference runtime (51 months, 200/50 prod config): 11h 09min.
# Cycle-10 PAUSE-RESUME (2026-05-10): --force-retrain dropped so already-
# completed cycle-10 months are skipped. The script's default behaviour
# at run_walkforward_pruned_svi_har_pd.py:441 skips a month whose .txt
# already exists. Resume picks up at the first missing month and trains
# forward. To force a full re-do, add --force-retrain back.
echo "[STEP 1/5] LightGBM walk-forward retrain (200/50, resume — skip done)..."
PYTHONPATH=. python vol_model/run_walkforward_pruned_svi_har_pd.py
echo "[STEP 1/5] DONE"
echo ""

# ── Step 2: Re-extract predictions ────────────────────────────────────
echo "[STEP 2/5] Re-extracting predictions parquet..."
PYTHONPATH=. python vol_model/extract_pruned_svi_har_pd_predictions.py
echo "[STEP 2/5] DONE"
echo ""

# ── Step 3: Walk-forward aligner ──────────────────────────────────────
echo "[STEP 3/5] Walk-forward aligner (writes aligned-bias parquet)..."
PYTHONPATH=. python -m vol_model.run_walkforward_aligner
echo "[STEP 3/5] DONE"
echo ""

# ── Step 4: Extract per-month β* JSON for live engine ─────────────────
echo "[STEP 4/5] Extracting per-month β* JSON for live engine..."
PYTHONPATH=. python scripts/extract_aligner_betas.py
echo "[STEP 4/5] DONE"
echo ""

# ── Step 5: V2 backtest comparison ────────────────────────────────────
echo "[STEP 5/5] V2 backtest comparison vs pre-retrain baseline..."
echo "  NB: uses existing Kelly K1 sidecar (BTC_v2_kelly_mult_K1.parquet)"
echo "  from 2026-05-08. Optional 2nd-pass refresh: rerun"
echo "  scripts/generate_kelly_sidecar.py against the new trade log,"
echo "  then re-run this comparator. Cycle-8 used 1-pass (no Kelly refresh)."
PYTHONPATH=. python scripts/compare_post_transvol_optuna_retrain.py
echo "[STEP 5/5] DONE"
echo ""

echo "================================================================"
echo "ORCHESTRATION COMPLETE  $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""
echo "Next operator action (live deployment):"
echo "  1. Restart the live engine — load_model_bundle is called once at boot."
echo "  2. Verify boot log shows new SHA256: 'LightGBM loaded: ... sha256=<new>'"
echo "                                       'TransVol loaded: ... sha256=<new>'"
echo "  3. Smoke test: pytest deribit_paper/tests/test_transvol_live_offline_parity.py"
echo ""
echo "Impact report: vol_arb/output/post_retrain_impact_cycle10.md"
echo "Log:           $LOG_FILE"
