#!/usr/bin/env bash
# Orchestrator for partial-window V2 backtests.
#
# Chains the eight pipeline steps in order:
#   1. Download latest exchange data (Binance bookTicker/aggTrades,
#      Deribit options/perp/DVOL).
#   2. TSRV incremental: extend the canonical TSRV parquet through
#      the latest available bar.
#   3. vol_arb merged features: rebuild with --end-date cutoff.
#   4. HAR-PD predictions: extract for the partial month, falling back
#      to the latest available walk-forward model for in-progress bars.
#   5. Aligner: apply per-month β* (cal slice [M-3, M) is fully
#      complete for the partial month, so no special handling needed).
#   6. Kelly sidecar: two-pass bootstrap — draft V2 backtest →
#      regenerate K1 (and K1+slope) → final V2 backtest.
#   7. (VPIN deferred: ffill from canonical parquet — acceptable
#       approximation for partial-month bars.)
#   8. Final V2 backtest sliced to [start, end_dt] → suffixed output.
#
# Usage:
#   bash scripts/run_partial_backtest.sh                       # → now-30min UTC
#   bash scripts/run_partial_backtest.sh 2026-05-11T12:15      # → explicit cutoff
#
# Output:
#   vol_arb/output/BTC_v2_trade_log_partial_<YYYYMMDDTHHMM>.parquet
#
# Determinism contract: for any bar T, the partial run's row is
# byte-identical to a full-sample run's row at the same T (test:
# tests/test_partial_window_parity.py).

set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

# ── Resolve cutoff timestamp ──
DEFAULT_END="$(python -c "
import pandas as pd
print((pd.Timestamp.utcnow().floor('15min') - pd.Timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%S'))
")"
END_DT="${1:-$DEFAULT_END}"
END_DATE="$(python -c "import pandas as pd; print(pd.Timestamp('$END_DT').strftime('%Y-%m-%d'))")"
SUFFIX="$(python -c "import pandas as pd; print(pd.Timestamp('$END_DT').strftime('%Y%m%dT%H%M'))")"

# Resolve paths
DRAFT_TL="vol_arb/output/BTC_v2_trade_log_draft_${SUFFIX}.parquet"
PARTIAL_K1="vol_arb/output/BTC_v2_kelly_mult_K1_partial.parquet"
PARTIAL_K1_SLOPE="vol_arb/output/BTC_v2_kelly_mult_K1_slope_partial.parquet"
ALIGNED_PREDS="vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
FINAL_TL="vol_arb/output/BTC_v2_trade_log_partial_${SUFFIX}.parquet"

echo "================================================================"
echo "  PARTIAL-WINDOW V2 BACKTEST ORCHESTRATOR"
echo "  End timestamp: $END_DT (UTC)"
echo "  Date for downloads: $END_DATE"
echo "  Output: $FINAL_TL"
echo "================================================================"

# ── Step 1: Data download ──
echo
echo "[Step 1/8] Downloading latest exchange data ..."
bash scripts/download_latest_daily.sh "$END_DATE"

# ── Step 2: TSRV incremental ──
echo
echo "[Step 2/8] TSRV incremental update ..."
# Cached calibration threshold for BTCUSDT 15m. Pinned to preserve
# byte-identity with the canonical batch; do NOT recalibrate.
TSRV_THRESHOLD="${TSRV_THRESHOLD:-250000}"
PYTHONPATH=. python TSRV/run_tsrv_incremental.py \
    --asset BTCUSDT --tf 900 \
    --threshold "$TSRV_THRESHOLD" \
    --end-date "$END_DT"

# ── Step 3: vol_arb merged features ──
echo
echo "[Step 3/8] vol_arb merged features (--end-date $END_DATE) ..."
PYTHONPATH=. python vol_arb/run_pipeline.py \
    --currency BTC --svi \
    --end-date "$END_DT"

# ── Step 4: HAR-PD predictions ──
echo
echo "[Step 4/8] HAR-PD partial-month predictions ..."
PYTHONPATH=. python vol_model/extract_pruned_svi_har_pd_predictions.py \
    --end-date "$END_DT"

# ── Step 5: Aligner ──
echo
echo "[Step 5/8] Walk-forward aligner ..."
PYTHONPATH=. python -m vol_model.run_walkforward_aligner

# ── Step 6: Kelly sidecar (two-pass) ──
echo
echo "[Step 6/8] Kelly sidecar two-pass bootstrap ..."
echo "  6a) draft V2 backtest (kelly off) ..."
PYTHONPATH=. python scripts/run_draft_backtest.py \
    --end-dt "$END_DT" \
    --output "$DRAFT_TL" \
    --predictions-path "$ALIGNED_PREDS"

echo "  6b) regenerate K1 from draft trade log ..."
PYTHONPATH=. python scripts/generate_kelly_sidecar.py \
    --trade-log "$DRAFT_TL" \
    --preds-path "$ALIGNED_PREDS" \
    --output-suffix "_partial"

echo "  6c) regenerate K1+slope from partial K1 ..."
PYTHONPATH=. python scripts/generate_kelly_slope_amped_sidecar.py \
    --k1-path "$PARTIAL_K1" \
    --trade-log "$DRAFT_TL" \
    --output "$PARTIAL_K1_SLOPE"

# ── Step 7: VPIN — deferred (ffill from canonical) ──
echo
echo "[Step 7/8] VPIN sidecar — partial-month uses canonical ffill (no regen)."

# ── Step 8: Final partial backtest with the new Kelly sidecar ──
echo
echo "[Step 8/8] Final partial backtest ..."
PYTHONPATH=. python scripts/run_partial_backtest.py \
    --end-dt "$END_DT" \
    --kelly-mult-path "$PARTIAL_K1_SLOPE" \
    --predictions-path "$ALIGNED_PREDS" \
    --output "$FINAL_TL"

echo
echo "================================================================"
echo "  DONE"
echo "  Partial trade log: $FINAL_TL"
echo "  Draft trade log:   $DRAFT_TL (intermediate, can be deleted)"
echo "  Partial Kelly K1:  $PARTIAL_K1"
echo "  Partial Kelly K1S: $PARTIAL_K1_SLOPE"
echo "================================================================"
