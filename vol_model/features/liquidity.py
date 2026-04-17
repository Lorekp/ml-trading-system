"""Liquidity features — true OFI (Cont/Kolm) per dollar bar, tick-scaled.

Uses the existing Kolm (2023) OFI implementation from src/signal_engine/l2_features.py:
- Conditions volume on price changes (not static snapshot VOI)
- Filters HFT flickering by tracking deltas, not levels

Flow per 15m window:
1. Form dollar bars from aggTrades
2. For each dollar bar, compute true OFI + spread from bookTicker ticks
3. Aggregate per-bar features to window level

All features tick-scaled for clean LightGBM splits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from TSRV.data_loader import load_day, available_dates, iter_trade_days
from TSRV.sampling import _accumulate_bars
from vol_model.config import VolModelConfig

logger = logging.getLogger(__name__)


def _compute_ofi_per_bar(
    bid_price: np.ndarray,
    bid_qty: np.ndarray,
    ask_price: np.ndarray,
    ask_qty: np.ndarray,
    tick_size: float,
) -> dict:
    """Compute true OFI (Kolm 2023) + spread for one dollar bar's ticks.

    True OFI conditions volume on price changes:
        bid_price UP:    bid_of = +new_qty  (new level)
        bid_price SAME:  bid_of = qty_delta (change at same level)
        bid_price DOWN:  bid_of = -old_qty  (level removed)
        (opposite for ask side)
        OFI = bid_of - ask_of
    """
    n = len(bid_price)
    if n < 3:
        return None

    ofi = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        # Bid side
        if bid_price[i] > bid_price[i - 1]:
            bid_of = bid_qty[i]
        elif bid_price[i] == bid_price[i - 1]:
            bid_of = bid_qty[i] - bid_qty[i - 1]
        else:
            bid_of = -bid_qty[i - 1]

        # Ask side
        if ask_price[i] < ask_price[i - 1]:
            ask_of = -ask_qty[i]
        elif ask_price[i] == ask_price[i - 1]:
            ask_of = -(ask_qty[i] - ask_qty[i - 1])
        else:
            ask_of = ask_qty[i - 1]

        ofi[i] = bid_of - ask_of

    # Skip first tick (no delta)
    ofi_valid = ofi[1:]

    spread_ticks = (ask_price - bid_price) / tick_size
    mid = (bid_price + ask_price) * 0.5

    return {
        "ofi_sum": float(np.sum(ofi_valid)),
        "ofi_mean": float(np.mean(ofi_valid)),
        "ofi_std": float(np.std(ofi_valid)),
        "spread_ticks": float(np.mean(spread_ticks)),
        "spread_max_ticks": float(np.max(spread_ticks)),
        "spread_std_ticks": float(np.std(spread_ticks)),
        "n_quotes": n,
    }


def compute_liquidity_features(config: VolModelConfig) -> pd.DataFrame:
    """Compute dollar-bar liquidity features using true OFI (Kolm 2023).

    Memory-safe: loads one day of bookTicker at a time.
    """
    freq_s = config.freq_s
    freq_td = pd.Timedelta(seconds=freq_s)
    threshold = config.dollar_bar_threshold
    tick_size = config.tick_size

    bt_dates = set(available_dates(config.tick_dir, config.asset))
    rows = []
    day_count = 0

    for date_str, trade_df in iter_trade_days(config.trade_dir, config.asset):
        if len(trade_df) < 10 or date_str not in bt_dates:
            continue

        try:
            tick_df = load_day(config.tick_dir, config.asset, date_str)
        except Exception:
            continue
        if len(tick_df) < 100:
            continue

        # Raw bookTicker arrays (load_day renames to bid/ask/bid_qty/ask_qty)
        tick_ts_int = tick_df.index.view(np.int64)
        tick_bid = tick_df["bid"].values.astype(np.float64)
        tick_ask = tick_df["ask"].values.astype(np.float64)
        tick_bid_qty = tick_df["bid_qty"].values.astype(np.float64)
        tick_ask_qty = tick_df["ask_qty"].values.astype(np.float64)

        trade_timestamps = trade_df.index
        trade_dvol = trade_df["dollar_volume"].values.astype(np.float64)

        day_start = trade_timestamps[0].floor(freq_td)
        day_end = trade_timestamps[-1]
        window_start = day_start

        while window_start < day_end:
            window_end = window_start + freq_td

            t_mask = (trade_timestamps >= window_start) & (trade_timestamps < window_end)
            t_idx = np.where(t_mask)[0]
            if len(t_idx) < 10:
                window_start = window_end
                continue

            # Dollar bars within window
            w_dvol = trade_dvol[t_idx]
            bar_indices = _accumulate_bars(w_dvol, threshold)
            if len(bar_indices) < 2:
                window_start = window_end
                continue

            bar_times = trade_timestamps[t_idx[bar_indices]]
            bar_ts_int = bar_times.view(np.int64)
            bar_tick_starts = np.searchsorted(tick_ts_int, bar_ts_int, side="left")

            # Per dollar bar: compute true OFI + spread
            bar_results = []
            for b in range(len(bar_tick_starts) - 1):
                qs = bar_tick_starts[b]
                qe = bar_tick_starts[b + 1]
                if qe - qs < 3:
                    continue

                result = _compute_ofi_per_bar(
                    tick_bid[qs:qe], tick_bid_qty[qs:qe],
                    tick_ask[qs:qe], tick_ask_qty[qs:qe],
                    tick_size,
                )
                if result:
                    bar_results.append(result)

            if len(bar_results) < 2:
                window_start = window_end
                continue

            # Aggregate per-bar features to window level
            ofi_sums = np.array([r["ofi_sum"] for r in bar_results])
            ofi_means = np.array([r["ofi_mean"] for r in bar_results])
            ofi_stds = np.array([r["ofi_std"] for r in bar_results])
            spreads = np.array([r["spread_ticks"] for r in bar_results])
            spread_maxs = np.array([r["spread_max_ticks"] for r in bar_results])
            quotes = np.array([r["n_quotes"] for r in bar_results], dtype=float)

            rows.append({
                "timestamp": window_start,
                # True OFI (Kolm 2023) — conditioned on price changes
                "ofi_sum": float(np.sum(ofi_sums)),        # net flow across all bars
                "ofi_mean": float(np.mean(ofi_means)),     # avg flow per bar
                "ofi_std": float(np.std(ofi_means)),       # flow variability across bars
                "ofi_abs_mean": float(np.mean(np.abs(ofi_means))),  # unsigned flow intensity
                # Spread in tick multiples
                "spread_ticks": float(np.mean(spreads)),
                "spread_max_ticks": float(np.max(spread_maxs)),
                "spread_std_ticks": float(np.std(spreads)),
                # Activity per dollar bar
                "quote_per_bar": float(np.mean(quotes)),
            })

            window_start = window_end

        day_count += 1
        if day_count % 30 == 0:
            logger.info("  Liquidity features: %d days, %d windows", day_count, len(rows))

    logger.info("  Liquidity features: DONE — %d days, %d windows", day_count, len(rows))

    if rows:
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df
    return pd.DataFrame()
