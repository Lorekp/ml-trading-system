#!/usr/bin/env bash
# Daily-mode wrapper around the existing exchange downloaders.
#
# Fetches the latest available data for every source the partial-window
# pipeline consumes, so the backtester can run through "today's bar".
# All underlying downloaders skip files that already exist on disk —
# this script just invokes them with the right date range.
#
# Sources covered:
#   * Binance BTCUSDT spot bookTicker (daily) → data/bookTicker_parquet/
#   * Binance BTCUSDT perp aggTrades (daily)  → data/binance-aggTrades-parquet/
#   * Deribit BTC option trades (daily)       → data/deribit/trades/BTC_option/
#   * Deribit BTC perp trades (daily)         → data/deribit/trades/BTC_future/
#   * Deribit DVOL (covered by deribit_paper or its own downloader)
#
# Usage:
#   bash scripts/download_latest_daily.sh                 # → from last-fetched to today
#   bash scripts/download_latest_daily.sh 2026-05-11     # → explicit cutoff date
#
# Designed to be idempotent and cheap on the no-op path. First call
# after a long gap may take 10-30 min; routine catch-up is minutes.

set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

END_DATE="${1:-$(date -u +%Y-%m-%d)}"
SYMBOLS="${SYMBOLS:-BTCUSDT}"

echo "================================================================"
echo "  DAILY DATA INGESTION WRAPPER"
echo "  End date: $END_DATE"
echo "  Symbols:  $SYMBOLS"
echo "================================================================"

# Detect the most recent existing daily file for a directory pattern.
# Returns YYYY-MM-DD on stdout, or empty if no files match.
latest_daily() {
    local dir="$1"
    local pattern="${2:-*.parquet}"
    if [ -d "$dir" ]; then
        ls -1 "$dir"/$pattern 2>/dev/null \
            | sed 's|.*/||; s|\.parquet||' \
            | sort | tail -1
    fi
}

# ───────────────────────── 1) Binance spot bookTicker ─────────────────────────
echo
echo "[1/4] Binance spot bookTicker (per-day parquets) ..."
BT_DIR="data/bookTicker_parquet/BTCUSDT"
LAST_BT="$(latest_daily "$BT_DIR")"
START_BT="${LAST_BT:-2022-01-01}"
echo "      last cached: ${LAST_BT:-<none>}"
echo "      requesting:  $START_BT  →  $END_DATE"
python scripts/download_binance_bbo.py --symbols "$SYMBOLS" \
    --start "$START_BT" --end "$END_DATE" \
    || echo "      bookTicker downloader exited non-zero (will continue)"

# ───────────────────────── 2) Binance perp aggTrades ─────────────────────────
# The Binance-Vision downloader uses --all or --symbols flags and
# monthly bundles. Newer days within the current month are typically
# fetched in a second pass via a daily endpoint — invoke it once for
# now; user can add a daily-AggTrades downloader in a follow-up cycle.
echo
echo "[2/4] Binance perp aggTrades (Vision daily / monthly) ..."
AT_DIR="data/binance-aggTrades-parquet/BTCUSDT"
LAST_AT="$(ls -1 "$AT_DIR" 2>/dev/null | sed 's|\.parquet||' | sort | tail -1)"
echo "      last cached: ${LAST_AT:-<none>}"
if [ -f "scripts/download_binance_vision.py" ]; then
    python scripts/download_binance_vision.py --symbols "$SYMBOLS" \
        || echo "      Binance Vision exited non-zero (continuing)"
else
    echo "      scripts/download_binance_vision.py not found — skip"
fi

# ───────────────────────── 3) Deribit option + perp trades ─────────────────────────
echo
echo "[3/4] Deribit BTC option + perp trades (daily JSON) ..."
OPT_DIR="data/deribit/trades/BTC_option"
LAST_OPT="$(latest_daily "$OPT_DIR")"
START_DR="${LAST_OPT:-2022-01-01}"
echo "      last cached option: ${LAST_OPT:-<none>}"
echo "      requesting:  $START_DR  →  $END_DATE"
python scripts/download_deribit_robust.py \
    --data-dir "data/deribit/trades" \
    --start "$START_DR" --end "$END_DATE" \
    --currencies BTC --kinds option,future \
    || echo "      Deribit downloader exited non-zero (continuing)"

# ───────────────────────── 4) Deribit DVOL ─────────────────────────
echo
echo "[4/4] Deribit DVOL (60m parquet) ..."
DVOL_PATH="data/deribit/dvol/BTC_dvol_60m.parquet"
if [ -f "$DVOL_PATH" ]; then
    python -c "
import pandas as pd
df = pd.read_parquet('$DVOL_PATH')
ts_col = 'timestamp' if 'timestamp' in df.columns else None
last = df[ts_col].max() if ts_col else df.index.max()
print(f'      last DVOL bar: {last}')
"
fi
# DVOL doesn't have a dedicated CLI downloader here; the deribit_paper
# engine appends to it during normal operation, or a manual fetch from
# Deribit's REST endpoint can extend it. For now, log status only.
echo "      (DVOL refresh path TBD — engine auto-updates during live run)"

echo
echo "================================================================"
echo "  DAILY INGESTION DONE"
echo "================================================================"
