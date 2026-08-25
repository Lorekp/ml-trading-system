"""Compute TSRV time series — one noise-corrected RV per slow-timeframe window.

Fast scale: all ticks (tick-level returns).
Slow scale: dollar sub-bars within each window.

Works with any asset and any timeframe.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from TSRV.config import TSRVConfig
from TSRV.data_loader import iter_trade_days
from TSRV.jump_robust import (
    bipower_variation, median_rv, realized_semivariance,
    decompose_jump_continuous, directional_jump_fractions, subgrid_rv_slow,
)
from TSRV.sampling import _accumulate_bars
from TSRV.signature_plot import find_knee, plot_signature
from TSRV.windowing import timestamp_window_ids

logger = logging.getLogger(__name__)


def compute_tsrv_window(
    prices: np.ndarray,
    dollar_vol: np.ndarray,
    threshold: float,
) -> dict:
    """Compute TSRV for a single window of trades.

    Uses K-subgrid averaging (Zhang et al. 2005) for lower-variance estimation
    and includes BPV/MedRV jump decomposition (BNS 2004, ADS 2012).

    Args:
        prices: price array for all ticks in window
        dollar_vol: dollar_volume array for all ticks
        threshold: dollar sub-bar threshold

    Returns:
        dict with tsrv, rv_fast, rv_slow, bpv, medrv, continuous_var,
        jump_var, jump_fraction, rs_down, rs_up, down_jump_fraction,
        up_jump_fraction, rq, n_ticks, n_sub_bars, n_dash
    """
    _zero = {"tsrv": 0.0, "rv_fast": 0.0, "rv_slow": 0.0,
             "bpv": 0.0, "medrv": 0.0, "continuous_var": 0.0,
             "jump_var": 0.0, "jump_fraction": 0.0,
             "rs_down": 0.0, "rs_up": 0.0,
             "down_jump_fraction": 0.0, "up_jump_fraction": 0.0,
             "rq": 0.0,
             "n_ticks": len(prices), "n_sub_bars": 0, "n_dash": 0.0}

    n_ticks = len(prices)
    if n_ticks < 2:
        return _zero

    # RV(fast) -- all-tick returns
    log_p = np.log(prices)
    rv_fast = float(np.sum(np.diff(log_p) ** 2))

    # Dollar sub-bars within window
    sub_bar_idx_list = _accumulate_bars(dollar_vol, threshold)

    if len(sub_bar_idx_list) < 2:
        _zero["rv_fast"] = rv_fast
        _zero["tsrv"] = rv_fast
        _zero["rv_slow"] = rv_fast
        _zero["n_sub_bars"] = len(sub_bar_idx_list)
        return _zero

    sub_bar_indices = np.array(sub_bar_idx_list)
    n_sub_bars = len(sub_bar_indices)
    n = n_ticks - 1

    # K-subgrid-averaged RV(slow) (Zhang et al. 2005)
    K = max(1, int(np.ceil(n_sub_bars ** (2 / 3))))
    rv_slow_avg, n_bar_avg = subgrid_rv_slow(log_p, sub_bar_indices, K)

    # Also compute single-grid RV(slow) for BPV comparison
    sub_log_p = log_p[sub_bar_indices]
    rv_slow_single = float(np.sum(np.diff(sub_log_p) ** 2))

    # TSRV with subgrid averaging + bias correction
    if n > 0 and n_bar_avg > 0:
        tsrv_raw = rv_slow_avg - (n_bar_avg / n) * rv_fast
        denom = 1.0 - n_bar_avg / n
        tsrv_val = tsrv_raw / denom if denom > 0.01 else tsrv_raw
    else:
        tsrv_val = rv_slow_avg

    # BPV and MedRV on sub-bar returns (noise-attenuated)
    bpv = bipower_variation(sub_log_p)
    mrv = median_rv(sub_log_p)

    # Jump/continuous decomposition (using single-grid RV for consistency with BPV)
    continuous, jump = decompose_jump_continuous(rv_slow_single, bpv)

    # Signed realized semivariance on sub-bar returns
    rs_down, rs_up = realized_semivariance(sub_log_p)

    # Directional jump fractions (BNS-Kinnebrock-Shephard 2010)
    # Down-jumps = liquidation cascades (sustained vol)
    # Up-jumps = sell-wall absorption (quick mean-revert)
    down_jf, up_jf = directional_jump_fractions(rv_slow_single, bpv, rs_down, rs_up)

    # Realized Quarticity (Barndorff-Nielsen/Shephard 2002 → HARQ, BPQ 2016)
    # RQ = (n/3) * Σ r_i^4 on sub-bar returns — vol-of-vol / quarticity proxy
    sub_returns = np.diff(sub_log_p)
    n_sub_returns = len(sub_returns)
    rq = (n_sub_returns / 3.0) * float(np.sum(sub_returns ** 4)) if n_sub_returns > 0 else 0.0

    n_dash = n / n_sub_bars if n_sub_bars > 0 else 0.0

    return {
        "tsrv": max(tsrv_val, 0.0),
        "rv_fast": rv_fast,
        "rv_slow": rv_slow_single,
        "bpv": bpv,
        "medrv": mrv,
        "continuous_var": continuous,
        "jump_var": jump,
        "jump_fraction": jump / max(rv_slow_single, 1e-20),
        "rs_down": rs_down,
        "rs_up": rs_up,
        "down_jump_fraction": down_jf,
        "up_jump_fraction": up_jf,
        "rq": rq,
        "n_ticks": n_ticks,
        "n_sub_bars": n_sub_bars,
        "n_dash": n_dash,
    }


def calibrate_sub_bar_threshold(
    trade_dir: str | Path,
    symbol: str,
    thresholds: list[float],
    slow_tf_s: int,
    config: TSRVConfig,
    max_windows: int = 500,
) -> dict:
    """Find optimal sub-bar threshold via signature plot within slow_tf windows.

    Samples windows across the dataset, computes RV(slow) at each threshold,
    averages across windows, and finds the knee.

    Returns: {threshold, knee_result, signature_df}
    """
    results = {t: [] for t in thresholds}
    window_count = 0
    freq = pd.Timedelta(seconds=slow_tf_s)

    for date_str, day_df in iter_trade_days(trade_dir, symbol):
        if len(day_df) < 100:
            continue

        # Split day into windows
        prices = day_df["price"].values.astype(np.float64)
        dollar_vol = day_df["dollar_volume"].values.astype(np.float64)
        timestamps = day_df.index

        # Window boundaries
        day_start = timestamps[0].floor(freq)
        day_end = timestamps[-1]
        window_start = day_start

        while window_start < day_end:
            window_end = window_start + freq
            mask = (timestamps >= window_start) & (timestamps < window_end)
            w_idx = np.where(mask)[0]

            if len(w_idx) < 10:
                window_start = window_end
                continue

            w_prices = prices[w_idx]
            w_dvol = dollar_vol[w_idx]

            for t in thresholds:
                sub_idx = _accumulate_bars(w_dvol, t)
                if len(sub_idx) < 2:
                    continue
                sub_p = w_prices[sub_idx]
                rv_slow = float(np.sum(np.diff(np.log(sub_p)) ** 2))
                results[t].append(rv_slow)

            window_count += 1
            window_start = window_end

            if max_windows and window_count >= max_windows:
                break

        if max_windows and window_count >= max_windows:
            break

    logger.info("Calibration: %d windows sampled", window_count)

    # Build signature
    rows = []
    for t in thresholds:
        vals = np.array(results[t])
        if len(vals) < 5:
            continue
        rows.append({
            "threshold": t,
            "rv_mean": float(vals.mean()),
            "rv_std": float(vals.std()),
            "rv_median": float(np.median(vals)),
            "n_days": len(vals),
        })

    sig_df = pd.DataFrame(rows)
    if len(sig_df) < 3:
        logger.warning("Not enough data points for knee detection, using smallest threshold")
        return {"threshold": thresholds[0], "knee_result": {}, "signature_df": sig_df}

    knee_result = find_knee(sig_df["threshold"].values, sig_df["rv_mean"].values, config)

    optimal = knee_result.get("knee_x", thresholds[len(thresholds) // 2])
    if optimal is None:
        optimal = thresholds[len(thresholds) // 2]

    return {
        "threshold": float(optimal),
        "knee_result": knee_result,
        "signature_df": sig_df,
    }


def _process_month_tsrv(
    trade_dir: str | Path,
    symbol: str,
    month: str,
    threshold: float,
    freq: pd.Timedelta,
    periods_per_year: float,
) -> list[dict]:
    """Process one month of trades into TSRV windows.

    Uses O(n) contiguous-block splitting instead of O(n*k) boolean masking.
    Data is sorted by time, so windows are contiguous — find boundaries via
    integer division on nanosecond timestamps, then np.diff to detect changes.

    Returns list of result dicts (one per 15m window).
    """
    from TSRV.data_loader import load_trades

    try:
        month_df = load_trades(trade_dir, symbol, month)
    except Exception:
        return []

    n = len(month_df)
    if n < 10:
        return []

    prices = month_df["price"].values.astype(np.float64)
    dollar_vol = month_df["dollar_volume"].values.astype(np.float64)

    # Integer window IDs via floor division on normalized nanosecond timestamps.
    window_ids, freq_ns = timestamp_window_ids(month_df.index, freq)

    # Find contiguous block boundaries — O(n)
    change_idx = np.where(np.diff(window_ids) != 0)[0] + 1
    # Split tick indices into per-window groups — O(n) total
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

        result = compute_tsrv_window(prices[s:e], dollar_vol[s:e], threshold)
        # Reconstruct window timestamp from integer ID
        result["timestamp"] = pd.Timestamp(int(window_ids[s]) * freq_ns, unit="ns", tz="UTC")
        result["rv_ann"] = result["tsrv"] * periods_per_year
        result["vol_ann"] = float(np.sqrt(max(result["rv_ann"], 0.0)))
        rows.append(result)

    return rows


def compute_tsrv_series(
    trade_dir: str | Path,
    symbol: str,
    threshold: float,
    slow_tf_s: int,
    config: TSRVConfig,
    output_path: str | Path | None = None,
    save_every: int = 6,
) -> pd.DataFrame:
    """Compute TSRV time series — one value per slow_tf window.

    Processes one month at a time (vectorized window assignment).
    Saves intermediate results every `save_every` months for crash recovery.
    Resumes from partial output if it exists.

    Returns DataFrame indexed by window start time.
    """
    from TSRV.data_loader import available_months

    freq = pd.Timedelta(seconds=slow_tf_s)
    periods_per_year = 525960.0 / (slow_tf_s / 60.0)

    months = available_months(trade_dir, symbol)
    logger.info("  TSRV series: %d months to process", len(months))

    # Resume from partial output
    existing_rows = []
    last_month_done = None
    if output_path and Path(output_path).exists():
        existing_df = pd.read_parquet(output_path)
        if len(existing_df) > 0:
            last_ts = existing_df.index.max()
            # Find which months are already covered
            for m in months:
                # Month string like "2023-05", compare to last timestamp
                month_end = pd.Timestamp(f"{m}-01", tz="UTC") + pd.DateOffset(months=1)
                if month_end <= last_ts:
                    last_month_done = m
                else:
                    break
            if last_month_done:
                existing_rows = existing_df.reset_index().to_dict("records")
                logger.info("  Resuming: %d existing windows through %s",
                            len(existing_rows), last_month_done)

    # Filter to remaining months
    if last_month_done:
        start_idx = months.index(last_month_done) + 1
        months = months[start_idx:]

    all_rows = existing_rows
    t0 = __import__("time").time()

    for i, month in enumerate(months):
        month_rows = _process_month_tsrv(
            trade_dir, symbol, month, threshold, freq, periods_per_year,
        )
        all_rows.extend(month_rows)

        elapsed = __import__("time").time() - t0
        rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        remaining = len(months) - (i + 1)
        eta = remaining / rate if rate > 0 else 0

        logger.info("  [%d/%d] %s: +%d windows (total=%d, %.1f mo/min, ETA %.0fmin)",
                     i + 1, len(months), month, len(month_rows), len(all_rows),
                     rate, eta)

        # Periodic save
        if output_path and (i + 1) % save_every == 0:
            _save_partial(all_rows, output_path)
            logger.info("  Saved checkpoint: %d windows", len(all_rows))

    logger.info("  TSRV series: DONE — %d months, %d windows", len(months), len(all_rows))

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df = df.set_index("timestamp").sort_index()
        # Deduplicate (in case of resume overlap)
        df = df[~df.index.duplicated(keep="last")]
    return df


def _save_partial(rows: list[dict], output_path: str | Path) -> None:
    """Save intermediate TSRV results to parquet."""
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        tmp = Path(output_path).with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(output_path)
