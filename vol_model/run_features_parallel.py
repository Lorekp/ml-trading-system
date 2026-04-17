#!/usr/bin/env python
"""Compute liquidity features in parallel — vectorized OFI, one worker per month.

Key optimizations:
1. OFI computed ONCE per day via numpy vectorization (no Python loop)
2. Dollar bar boundaries found via searchsorted (O(n_bars * log(n_ticks)))
3. Per-bar aggregation via cumsum slicing (no per-bar loop for OFI)
4. One month parquet loaded once per worker

Usage:
    PYTHONPATH=. python vol_model/run_features_parallel.py --workers 8
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def _vectorized_ofi(bp, bq, ap, aq):
    """Compute true OFI (Kolm 2023) fully vectorized — no Python loop.

    Returns OFI array (length n-1, first tick has no delta).
    """
    bp_diff = np.diff(bp)
    ap_diff = np.diff(ap)

    # Bid side: UP -> new_qty, SAME -> delta, DOWN -> -old_qty
    bid_of = np.where(bp_diff > 0, bq[1:],
             np.where(bp_diff == 0, bq[1:] - bq[:-1],
                      -bq[:-1]))

    # Ask side: DOWN -> -new_qty, SAME -> -delta, UP -> old_qty
    ask_of = np.where(ap_diff < 0, -aq[1:],
             np.where(ap_diff == 0, -(aq[1:] - aq[:-1]),
                      aq[:-1]))

    return bid_of - ask_of


def _process_day(tick_df, trade_df, freq_s, dollar_threshold, tick_size):
    """Process one day: vectorized OFI + dollar bar aggregation."""
    from TSRV.sampling import _accumulate_bars

    if len(tick_df) < 100 or len(trade_df) < 10:
        return []

    # Compute OFI for entire day at once (vectorized)
    bp = tick_df["bid"].values.astype(np.float64)
    bq = tick_df["bid_qty"].values.astype(np.float64)
    ap = tick_df["ask"].values.astype(np.float64)
    aq = tick_df["ask_qty"].values.astype(np.float64)

    day_ofi = _vectorized_ofi(bp, bq, ap, aq)
    # Prepend 0 for first tick (no delta)
    day_ofi_full = np.concatenate([[0.0], day_ofi])

    # Pre-compute spread in ticks for entire day
    day_spread = (ap - bp) / tick_size

    # Pre-compute order book imbalance for entire day
    # Static snapshot — noisy alone but useful alongside OFI
    total_qty = bq + aq
    safe_total = np.where(total_qty > 0, total_qty, 1.0)
    day_imbalance = (bq - aq) / safe_total  # [-1, 1]

    tick_ts_int = tick_df.index.view(np.int64)
    trade_ts = trade_df.index
    trade_dvol = trade_df["dollar_volume"].values.astype(np.float64)

    freq_td = pd.Timedelta(seconds=freq_s)
    day_start = trade_ts[0].floor(freq_td)
    day_end = trade_ts[-1]
    rows = []
    ws = day_start

    while ws < day_end:
        we = ws + freq_td
        t_mask = (trade_ts >= ws) & (trade_ts < we)
        t_idx = np.where(t_mask)[0]
        if len(t_idx) < 10:
            ws = we
            continue

        bar_indices = _accumulate_bars(trade_dvol[t_idx], dollar_threshold)
        if len(bar_indices) < 2:
            ws = we
            continue

        bar_times = trade_ts[t_idx[bar_indices]]
        bar_starts = np.searchsorted(tick_ts_int, bar_times.view(np.int64), side="left")

        # Per-bar aggregation using pre-computed day-level arrays
        ofi_sums = []
        ofi_means = []
        spreads = []
        spread_maxs = []
        imbalances = []
        n_quotes = []

        for b in range(len(bar_starts) - 1):
            qs, qe = bar_starts[b], bar_starts[b + 1]
            if qe - qs < 3:
                continue

            bar_ofi = day_ofi_full[qs:qe]
            bar_spread = day_spread[qs:qe]
            bar_imb = day_imbalance[qs:qe]

            ofi_sums.append(bar_ofi.sum())
            ofi_means.append(bar_ofi.mean())
            spreads.append(bar_spread.mean())
            spread_maxs.append(bar_spread.max())
            imbalances.append(bar_imb[-1])  # end-of-bar snapshot
            n_quotes.append(qe - qs)

        if len(ofi_sums) < 2:
            ws = we
            continue

        om = np.array(ofi_means)
        sp = np.array(spreads)
        imb = np.array(imbalances)

        rows.append({
            "timestamp": ws,
            # True OFI (Kolm 2023) — conditioned on price changes
            "ofi_sum": float(np.sum(ofi_sums)),
            "ofi_mean": float(np.mean(om)),
            "ofi_std": float(np.std(om)),
            "ofi_abs_mean": float(np.mean(np.abs(om))),
            # Order book imbalance — end-of-bar snapshot per dollar bar
            "imbalance_mean": float(np.mean(imb)),
            "imbalance_std": float(np.std(imb)),
            "imbalance_end": float(imb[-1]),
            # Spread in tick multiples
            "spread_ticks": float(np.mean(sp)),
            "spread_max_ticks": float(np.max(spread_maxs)),
            "spread_std_ticks": float(np.std(sp)),
            # Activity
            "quote_per_bar": float(np.mean(n_quotes)),
        })
        ws = we

    return rows


def process_one_month(args):
    """Process all days in one month."""
    month_str, trade_dir, tick_dir, asset, freq_s, threshold, tick_size, bt_dates_list = args
    bt_dates = set(bt_dates_list)

    from TSRV.data_loader import load_day, load_trades

    try:
        month_df = load_trades(trade_dir, asset, month_str)
    except Exception:
        return []

    all_rows = []
    for day_date, trade_df in month_df.groupby(month_df.index.date):
        date_str = str(day_date)
        if date_str not in bt_dates:
            continue
        try:
            tick_df = load_day(tick_dir, asset, date_str)
        except Exception:
            continue

        rows = _process_day(tick_df, trade_df, freq_s, threshold, tick_size)
        all_rows.extend(rows)

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--asset", default="BTCUSDT")
    args = parser.parse_args()

    from vol_model.config import VolModelConfig
    from TSRV.data_loader import available_dates, available_months

    config = VolModelConfig(asset=args.asset)
    bt_dates = sorted(available_dates(config.tick_dir, config.asset))
    months = available_months(config.trade_dir, config.asset)

    print(f"{len(months)} months, {args.workers} workers, {len(bt_dates)} bookTicker days", flush=True)
    t0 = time.time()

    task_args = [
        (m, config.trade_dir, config.tick_dir, config.asset,
         config.freq_s, config.dollar_bar_threshold, config.tick_size, bt_dates)
        for m in months
    ]

    all_rows = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_month, a): a[0] for a in task_args}
        for future in as_completed(futures):
            month = futures[future]
            try:
                rows = future.result()
            except Exception as e:
                print(f"  {month}: FAILED ({e})", flush=True)
                continue
            all_rows.extend(rows)
            done += 1
            elapsed = time.time() - t0
            print(f"  {month}: {len(rows)} windows ({done}/{len(months)}, {elapsed:.0f}s)", flush=True)

    if not all_rows:
        print("No data!", flush=True)
        return

    df = pd.DataFrame(all_rows).set_index("timestamp").sort_index()
    out_path = Path(config.output_dir) / f"{config.asset}_features_liquidity.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    elapsed = time.time() - t0
    print(f"\nDone: {len(df)} windows -> {out_path} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
