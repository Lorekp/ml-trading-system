"""Toxicity features from aggTrades — bar velocity, VPIN, tick imbalance.

These capture informed trading activity and market microstructure:
- Bar velocity: how fast dollar bars form (fast = high activity)
- VPIN: probability of informed trading (high = toxic flow)
- Tick imbalance: directional flow pressure

All features are computed per completed 15m window using only trades
within [window_start, window_end). No future data used.

Binance is_buyer_maker interpretation:
    True  = buyer was maker -> taker was SELLER -> sell aggression
    False = seller was maker -> taker was BUYER -> buy aggression
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from TSRV.data_loader import iter_trade_days
from TSRV.sampling import _accumulate_bars
from vol_model.config import VolModelConfig

logger = logging.getLogger(__name__)


def _compute_vpin_window(
    dollar_vol: np.ndarray,
    is_buyer_maker: np.ndarray,
    bucket_size: float,
    n_buckets: int,
) -> float:
    """Compute VPIN for a single window.

    Volume-Synchronized Probability of Informed Trading:
    1. Bucket trades by cumulative dollar volume
    2. Within each bucket: |V_buy - V_sell| / V_total
    3. VPIN = mean over last n_buckets

    Returns VPIN in [0, 1], or NaN if insufficient buckets.
    """
    if len(dollar_vol) < 2:
        return np.nan

    # Accumulate into volume buckets
    cumvol = np.cumsum(dollar_vol)
    total_vol = cumvol[-1]

    if total_vol < bucket_size:
        # Not enough volume for even one bucket
        # Compute single-bucket VPIN from all trades
        buy_vol = dollar_vol[~is_buyer_maker].sum()
        sell_vol = dollar_vol[is_buyer_maker].sum()
        if total_vol > 0:
            return abs(buy_vol - sell_vol) / total_vol
        return np.nan

    # Find bucket boundaries
    bucket_boundaries = np.arange(bucket_size, total_vol + bucket_size, bucket_size)
    bucket_idx = np.searchsorted(cumvol, bucket_boundaries)
    bucket_idx = np.clip(bucket_idx, 0, len(dollar_vol) - 1)
    bucket_idx = np.unique(bucket_idx)

    if len(bucket_idx) < 1:
        return np.nan

    # Compute imbalance per bucket
    imbalances = []
    prev = 0
    for idx in bucket_idx:
        if idx <= prev:
            continue
        bucket_mask_buy = ~is_buyer_maker[prev:idx]
        bucket_dvol = dollar_vol[prev:idx]
        buy_v = bucket_dvol[bucket_mask_buy].sum()
        sell_v = bucket_dvol[~bucket_mask_buy].sum()
        total = buy_v + sell_v
        if total > 0:
            imbalances.append(abs(buy_v - sell_v) / total)
        prev = idx

    if not imbalances:
        return np.nan

    # Use last n_buckets (or all if fewer)
    tail = imbalances[-n_buckets:]
    return float(np.mean(tail))


def compute_toxicity_features(config: VolModelConfig) -> pd.DataFrame:
    """Compute toxicity features from aggTrades.

    For each 15m window [t, t+15m), compute:
        bar_velocity_mean: mean time (seconds) for dollar bars to form
        bar_velocity_std: std of bar formation times
        vpin: volume-synchronized probability of informed trading
        tick_imbalance: (buy_vol - sell_vol) / total_vol
        n_trades: trade count in window
        trade_intensity: trades per second

    Returns DataFrame indexed by window start time (UTC).
    """
    freq = pd.Timedelta(seconds=config.freq_s)
    rows = []
    day_count = 0

    for date_str, day_df in iter_trade_days(config.trade_dir, config.asset):
        if len(day_df) < 10:
            continue

        prices = day_df["price"].values.astype(np.float64)
        dollar_vol = day_df["dollar_volume"].values.astype(np.float64)
        is_buyer_maker = day_df["is_buyer_maker"].values.astype(bool)
        timestamps = day_df.index

        # Window boundaries for this day
        day_start = timestamps[0].floor(freq)
        day_end = timestamps[-1]
        window_start = day_start

        while window_start < day_end:
            window_end = window_start + freq
            # Strict [start, end) — no future data
            mask = (timestamps >= window_start) & (timestamps < window_end)
            w_idx = np.where(mask)[0]

            if len(w_idx) < 2:
                window_start = window_end
                continue

            w_prices = prices[w_idx]
            w_dvol = dollar_vol[w_idx]
            w_buyer = is_buyer_maker[w_idx]
            w_times = timestamps[w_idx]

            row = {"timestamp": window_start}

            # 1. Dollar bar velocity
            sub_bar_indices = _accumulate_bars(w_dvol, config.dollar_bar_threshold)
            if len(sub_bar_indices) >= 2:
                bar_times = w_times[sub_bar_indices]
                deltas = np.diff(bar_times.values).astype("timedelta64[s]").astype(float)
                row["bar_velocity_mean"] = float(np.mean(deltas))
                row["bar_velocity_std"] = float(np.std(deltas)) if len(deltas) > 1 else 0.0
            else:
                # Window too quiet for dollar bars — use window duration
                total_secs = (w_times[-1] - w_times[0]).total_seconds()
                row["bar_velocity_mean"] = total_secs
                row["bar_velocity_std"] = 0.0

            # 2. VPIN
            row["vpin"] = _compute_vpin_window(
                w_dvol, w_buyer, config.vpin_bucket_size, config.vpin_n_buckets)

            # 3. Per-dollar-bar imbalance (not window-level sum)
            bar_imbalances = []
            bar_trade_counts = []
            prev_idx = 0
            for bi in range(1, len(sub_bar_indices)):
                bar_start = sub_bar_indices[bi - 1] if bi > 1 else 0
                bar_end = sub_bar_indices[bi]
                bar_dvol = w_dvol[bar_start:bar_end]
                bar_buyer = w_buyer[bar_start:bar_end]
                bar_buy = bar_dvol[~bar_buyer].sum()
                bar_sell = bar_dvol[bar_buyer].sum()
                bar_total = bar_buy + bar_sell
                if bar_total > 0:
                    bar_imbalances.append((bar_buy - bar_sell) / bar_total)
                bar_trade_counts.append(bar_end - bar_start)

            if bar_imbalances:
                bar_imb = np.array(bar_imbalances)
                row["imbalance_per_bar_mean"] = float(np.mean(bar_imb))
                row["imbalance_per_bar_std"] = float(np.std(bar_imb))
            else:
                row["imbalance_per_bar_mean"] = 0.0
                row["imbalance_per_bar_std"] = 0.0

            # 4. Trades per dollar bar (fragmentation measure)
            if bar_trade_counts:
                row["trades_per_bar"] = float(np.mean(bar_trade_counts))
            else:
                row["trades_per_bar"] = float(len(w_idx))

            rows.append(row)
            window_start = window_end

        day_count += 1
        if day_count % 30 == 0:
            logger.info("  Toxicity features: %d days, %d windows", day_count, len(rows))

    logger.info("  Toxicity features: DONE — %d days, %d windows", day_count, len(rows))

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.set_index("timestamp").sort_index()
    return df
