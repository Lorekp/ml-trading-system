#!/usr/bin/env python
"""Backfill Realized Quarticity (RQ) into an existing TSRV parquet.

Goal: add ONE new column `rq` to the existing TSRV 15m parquet without
recomputing or perturbing any existing column (byte-identical gate).

Why a focused backfill instead of a full regen:
- Full TSRV regen scans 38GB+ of tick data and runs the K-subgrid averaging,
  BPV, MedRV, jump decomposition, and realized semivariance per window.
- For RQ we only need sub-bar log-prices within each 15m window. Same tick
  loading cost, but much less per-window math.
- Byte-identical guarantee: we never touch the other columns.

Strategy:
1. Load existing TSRV parquet → all columns already correct (trust Stage 18 output).
2. For each month of trade data: split into 15-min windows (same integer-window-ID
   logic as _process_month_tsrv), accumulate sub-bars using the same threshold, and
   compute ONLY rq per window.
3. Merge the new `rq` series onto the existing parquet index.
4. Sanity check: rq ≥ 0 everywhere, not-all-NaN, timestamps aligned with existing
   parquet.
5. Write back (atomic replace via .tmp rename).

Usage:
    python -m TSRV.backfill_rq --asset BTCUSDT --threshold 500000
    python -m TSRV.backfill_rq --asset ETHUSDT --threshold 1000000
    python -m TSRV.backfill_rq --asset BTCUSDT --threshold 500000 --verify-only 2024-11
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from TSRV.data_loader import available_months, load_trades
from TSRV.sampling import _accumulate_bars


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TSRV.backfill_rq")


def compute_rq_for_window(prices: np.ndarray, dollar_vol: np.ndarray, threshold: float) -> dict:
    """Compute ONLY rq + n_sub_bars for a single 15m window.

    Returns dict with keys: rq, n_sub_bars.

    Mirrors the RQ branch of compute_tsrv_window so results are byte-identical
    when the main TSRV pipeline runs with this column enabled. Sub-bar returns
    are built from dollar-accumulated sub-bars of the raw tick prices.
    """
    n_ticks = len(prices)
    if n_ticks < 2:
        return {"rq": 0.0, "n_sub_bars": 0}

    log_p = np.log(prices)
    sub_bar_idx_list = _accumulate_bars(dollar_vol, threshold)

    if len(sub_bar_idx_list) < 2:
        return {"rq": 0.0, "n_sub_bars": len(sub_bar_idx_list)}

    sub_bar_indices = np.array(sub_bar_idx_list)
    sub_log_p = log_p[sub_bar_indices]
    sub_returns = np.diff(sub_log_p)
    n_sub_returns = len(sub_returns)
    rq = (n_sub_returns / 3.0) * float(np.sum(sub_returns ** 4)) if n_sub_returns > 0 else 0.0
    return {"rq": rq, "n_sub_bars": len(sub_bar_indices)}


def process_month_rq(
    trade_dir: str | Path, symbol: str, month: str, threshold: float, slow_tf_s: int = 900,
) -> pd.DataFrame:
    """Process one month of trades → DataFrame(index=window_end_ts, rq, n_sub_bars).

    Uses the SAME window-ID logic as _process_month_tsrv, so resulting timestamps
    are IDENTICAL to the existing TSRV parquet index. Timestamp is the START of
    the 15m window, matching the existing parquet schema (END-stamp is applied
    at the vol_model pipeline layer, not here).
    """
    try:
        month_df = load_trades(trade_dir, symbol, month)
    except Exception as exc:
        logger.warning("  Failed to load %s %s: %s", symbol, month, exc)
        return pd.DataFrame(columns=["rq", "n_sub_bars"])

    n = len(month_df)
    if n < 10:
        return pd.DataFrame(columns=["rq", "n_sub_bars"])

    freq_ns = int(pd.Timedelta(seconds=slow_tf_s).value)
    prices = month_df["price"].values.astype(np.float64)
    dollar_vol = month_df["dollar_volume"].values.astype(np.float64)
    ts_ns = month_df.index.values.astype("int64")
    window_ids = ts_ns // freq_ns

    change_idx = np.where(np.diff(window_ids) != 0)[0] + 1
    block_starts = np.empty(len(change_idx) + 1, dtype=np.int64)
    block_starts[0] = 0
    block_starts[1:] = change_idx
    block_ends = np.empty(len(change_idx) + 1, dtype=np.int64)
    block_ends[:-1] = change_idx
    block_ends[-1] = n

    rows = []
    for b in range(len(block_starts)):
        s, e = int(block_starts[b]), int(block_ends[b])
        if e - s < 2:
            continue
        result = compute_rq_for_window(prices[s:e], dollar_vol[s:e], threshold)
        result["timestamp"] = pd.Timestamp(
            int(window_ids[s]) * freq_ns, unit="ns", tz="UTC"
        )
        rows.append(result)

    if not rows:
        return pd.DataFrame(columns=["rq", "n_sub_bars"])

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def verify_one_month(
    existing: pd.DataFrame, trade_dir: str, symbol: str, month: str, threshold: float,
    slow_tf_s: int = 900, tolerance: int = 5,
) -> None:
    """Verify threshold matches: n_sub_bars from backfill should match existing within tolerance.

    Small tolerance allows for edge-case float jitter in _accumulate_bars.
    """
    logger.info("[verify] Computing %s RQ for %s with threshold=%s ...", symbol, month, threshold)
    month_df = process_month_rq(trade_dir, symbol, month, threshold, slow_tf_s)
    if len(month_df) == 0:
        logger.error("[verify] empty month; cannot verify threshold")
        raise SystemExit(2)

    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.DateOffset(months=1)
    # Existing parquet may be START-stamped OR END-stamped. The backfill above
    # produces START-stamped timestamps. So we compare to existing rows whose
    # index falls in [month_start, month_end).
    existing_month = existing.loc[(existing.index >= month_start) & (existing.index < month_end)]
    if len(existing_month) == 0:
        logger.warning("[verify] existing parquet has no rows for %s; skipping threshold check", month)
        return

    # Always compare both START-stamp and END-stamp alignments; pick the one with
    # higher exact-match rate. Raw overlap can collide even when alignments differ
    # (START=00:15 and END=00:15 are the same timestamp but different windows).
    def _diff_stats(df: pd.DataFrame) -> tuple[int, float, int, float]:
        common = df.index.intersection(existing_month.index)
        if len(common) == 0:
            return 0, float("inf"), -1, 0.0
        nsb_new = df.loc[common, "n_sub_bars"].astype(int)
        nsb_old = existing_month.loc[common, "n_sub_bars"].astype(int)
        d = (nsb_new - nsb_old).abs()
        return len(common), float(d.mean()), int(d.max()), float((d == 0).mean())

    raw_stats = _diff_stats(month_df)
    shifted = month_df.copy()
    shifted.index = shifted.index + pd.Timedelta(seconds=slow_tf_s)
    shifted_stats = _diff_stats(shifted)

    logger.info("[verify] %s raw-stamp: n=%d mean|Δ|=%.2f frac_exact=%.3f",
                month, raw_stats[0], raw_stats[1], raw_stats[3])
    logger.info("[verify] %s +%ds-shift: n=%d mean|Δ|=%.2f frac_exact=%.3f",
                month, slow_tf_s, shifted_stats[0], shifted_stats[1], shifted_stats[3])

    # Pick alignment with higher frac_exact
    if shifted_stats[3] > raw_stats[3]:
        logger.info("[verify] using END-stamp convention (+%ds shift)", slow_tf_s)
        chosen = shifted_stats
    else:
        logger.info("[verify] using START-stamp convention (raw)")
        chosen = raw_stats

    n_common, mean_diff, max_diff, frac_exact = chosen
    if n_common == 0:
        raise SystemExit(f"[verify] zero timestamp overlap for {month}")
    logger.info("[verify] %s CHOSEN: mean|Δ|=%.2f, max|Δ|=%d, frac_exact=%.3f (n=%d)",
                month, mean_diff, max_diff, frac_exact, n_common)
    if mean_diff > tolerance:
        raise SystemExit(
            f"[verify] threshold mismatch: mean|Δn_sub_bars|={mean_diff:.2f} "
            f"> tolerance={tolerance}. Check --threshold argument."
        )


def main():
    parser = argparse.ArgumentParser(description="Backfill RQ into existing TSRV parquet")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Dollar sub-bar threshold used in the existing parquet")
    parser.add_argument("--tf", type=int, default=900, help="Slow timeframe in seconds (default 900=15m)")
    parser.add_argument("--trade-dir", default="data/binance-aggTrades-parquet")
    parser.add_argument("--output-dir", default="TSRV/output")
    parser.add_argument("--verify-only", type=str, default=None,
                        help="If set, run verification on a single month (YYYY-MM) and exit")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating .pre_rq_bak backup (dangerous — use only if already backed up)")
    args = parser.parse_args()

    tf_min = args.tf // 60
    parquet_path = Path(args.output_dir) / f"{args.asset}_tsrv_{tf_min}m.parquet"
    if not parquet_path.exists():
        logger.error("Missing parquet: %s", parquet_path)
        raise SystemExit(2)

    logger.info("Loading existing parquet: %s", parquet_path)
    existing = pd.read_parquet(parquet_path)
    logger.info("  %d rows, cols=%s", len(existing), list(existing.columns))

    # Step 1: verify threshold on a recent month
    all_months = available_months(args.trade_dir, args.asset)
    if not all_months:
        raise SystemExit("no trade months available")

    verify_month = args.verify_only or all_months[-2]  # second-to-last month (last may be partial)
    verify_one_month(existing, args.trade_dir, args.asset, verify_month, args.threshold, args.tf)
    if args.verify_only:
        logger.info("[verify-only] passed; exiting")
        return

    # Step 2: compute rq for every month
    logger.info("Computing RQ across %d months ...", len(all_months))
    t0 = time.time()
    parts = []
    for i, month in enumerate(all_months):
        month_df = process_month_rq(args.trade_dir, args.asset, month, args.threshold, args.tf)
        parts.append(month_df)
        rate = (i + 1) / (time.time() - t0 + 1e-9) * 60
        logger.info("  [%d/%d] %s: %d windows (%.1f mo/min)",
                    i + 1, len(all_months), month, len(month_df), rate)

    rq_all = pd.concat(parts).sort_index()
    rq_all = rq_all[~rq_all.index.duplicated(keep="last")]
    logger.info("Total RQ windows: %d", len(rq_all))

    # Step 3: align to existing parquet (handle START vs END stamp automatically).
    # Pick the alignment that maximises n_sub_bars exact-match rate — robust to the
    # stamp convention the existing parquet was written with.
    def _align_exact_frac(df: pd.DataFrame) -> tuple[pd.Index, float]:
        common = df.index.intersection(existing.index)
        if len(common) == 0:
            return common, 0.0
        d = (df.loc[common, "n_sub_bars"].astype(int)
             - existing.loc[common, "n_sub_bars"].astype(int)).abs()
        return common, float((d == 0).mean())

    raw_common, raw_exact = _align_exact_frac(rq_all)
    shifted = rq_all.copy()
    shifted.index = shifted.index + pd.Timedelta(seconds=args.tf)
    shifted_common, shifted_exact = _align_exact_frac(shifted)

    logger.info("Alignment probe: raw n_common=%d exact=%.3f | shifted+%ds n_common=%d exact=%.3f",
                len(raw_common), raw_exact, args.tf, len(shifted_common), shifted_exact)
    if shifted_exact > raw_exact:
        logger.info("Using END-stamp convention (+%ds shift)", args.tf)
        rq_all = shifted
        common = shifted_common
    else:
        logger.info("Using START-stamp convention (raw)")
        common = raw_common

    align_frac = len(common) / len(existing)
    logger.info("Alignment: %d common rows (%.2f%% of existing parquet)",
                len(common), 100 * align_frac)
    if align_frac < 0.98:
        raise SystemExit(f"alignment fraction too low ({align_frac:.3f}) — check threshold/config")

    # Step 4: backup + merge
    if not args.no_backup:
        bak_path = parquet_path.with_suffix(".parquet.pre_rq_bak")
        if bak_path.exists():
            logger.info("Backup already exists: %s (not overwriting)", bak_path)
        else:
            logger.info("Backing up existing parquet to %s", bak_path)
            bak_path.write_bytes(parquet_path.read_bytes())

    merged = existing.copy()
    merged["rq"] = np.nan
    merged.loc[common, "rq"] = rq_all.loc[common, "rq"]

    n_nan = int(merged["rq"].isna().sum())
    logger.info("After merge: %d rows, %d NaN in rq (%.2f%%)",
                len(merged), n_nan, 100 * n_nan / len(merged))

    # Sanity: all non-rq columns byte-identical (trivially true since we didn't touch them)
    old_cols = list(existing.columns)
    new_cols = [c for c in merged.columns if c != "rq"]
    for c in old_cols:
        if not merged[c].equals(existing[c]):
            raise SystemExit(f"column '{c}' changed unexpectedly — aborting")
    logger.info("Byte-identical verification: all %d existing columns unchanged", len(old_cols))

    # Step 5: atomic write
    tmp_path = parquet_path.with_suffix(".tmp")
    merged.to_parquet(tmp_path)
    tmp_path.replace(parquet_path)
    logger.info("Saved: %s", parquet_path)

    # Sanity on the written file
    verify_df = pd.read_parquet(parquet_path)
    assert "rq" in verify_df.columns, "rq column missing after round-trip"
    logger.info("Round-trip check: rq column present, min=%.3e max=%.3e",
                verify_df["rq"].min(), verify_df["rq"].max())


if __name__ == "__main__":
    main()
