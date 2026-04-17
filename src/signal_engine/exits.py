"""Path-dependent exit computation.

Given signals and OHLCV data, walks the actual price path to determine
exit points based on configurable rules: stop-loss, take-profit,
trailing stops, and adaptive hold periods.

All exit decisions are causal — only information available at or before
the decision point is used. ATR is computed with a backward-looking EWM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine.config import ExitConfig


def compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using EWM (causal — backward-looking only)."""
    high = ohlcv["high"]
    low = ohlcv["low"]
    prev_close = ohlcv["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean()


def compute_path_exits(
    signals_df: pd.DataFrame,
    ohlcv_dict: dict[str, pd.DataFrame],
    exit_config: ExitConfig,
) -> pd.DataFrame:
    """Compute path-dependent exit returns for each signal.

    Walks the future OHLCV bars for each signal to check exit conditions.
    Returns a copy of signals_df with ``fwd_return`` replaced by the actual
    exit return and a new ``bars_held`` column.

    Barrier priority when multiple hit in the same bar (conservative):
    SL > TP > trail > time expiry.

    Args:
        signals_df: Must contain columns: timestamp, direction, asset,
            fwd_return, signal_strength.
        ohlcv_dict: Mapping of asset name -> 15m OHLCV DataFrame with
            columns: open, high, low, close.
        exit_config: Exit rule configuration.

    Returns:
        Modified signals_df with path-computed fwd_return and bars_held.
        Signals where exits couldn't be computed are dropped.
    """
    n = len(signals_df)
    exit_returns = np.full(n, np.nan, dtype=np.float64)
    bars_held_arr = np.full(n, 0, dtype=np.int32)
    exit_timestamps = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    # Pre-compute ATR per asset (causal)
    atr_cache: dict[str, pd.Series] = {}
    for asset in signals_df["asset"].unique():
        if asset in ohlcv_dict:
            atr_cache[asset] = compute_atr(ohlcv_dict[asset])

    assets = signals_df["asset"].values
    timestamps = signals_df["timestamp"].values
    directions = signals_df["direction"].values.astype(np.float64)
    sig_strengths = (
        signals_df["signal_strength"].values.astype(np.float64)
        if "signal_strength" in signals_df.columns
        else np.zeros(n)
    )

    has_sl_pct = exit_config.stop_loss_pct > 0
    has_tp_pct = exit_config.take_profit_pct > 0
    has_sl_atr = exit_config.stop_loss_atr_mult > 0
    has_tp_atr = exit_config.take_profit_atr_mult > 0
    has_trail = exit_config.trail_atr_mult > 0

    # Pre-build index lookups per asset (tz-normalized)
    _ohlcv_lookup: dict[str, dict] = {}
    for asset_name in signals_df["asset"].unique():
        if asset_name not in ohlcv_dict:
            continue
        ohlcv = ohlcv_dict[asset_name]
        # Normalize to tz-naive for matching
        ohlcv_idx = ohlcv.index
        if ohlcv_idx.tz is not None:
            ohlcv_idx = ohlcv_idx.tz_localize(None)
        _ohlcv_lookup[asset_name] = {
            "ohlcv": ohlcv,
            "idx_naive": ohlcv_idx,
            "ts_map": {t: i for i, t in enumerate(ohlcv_idx)},
        }

    for i in range(n):
        asset = assets[i]
        if asset not in _ohlcv_lookup:
            continue

        lkp = _ohlcv_lookup[asset]
        ohlcv = lkp["ohlcv"]
        ts = pd.Timestamp(timestamps[i])
        direction = directions[i]

        # Normalize timestamp to tz-naive for matching
        ts_naive = ts.tz_localize(None) if ts.tzinfo is not None else ts

        # Find bar index using pre-built map (O(1) lookup)
        idx = lkp["ts_map"].get(ts_naive)
        if idx is None:
            # Fallback: searchsorted on tz-naive index
            idx = lkp["idx_naive"].searchsorted(ts_naive)
            if idx >= len(ohlcv):
                continue
            # Verify match
            if abs((lkp["idx_naive"][idx] - ts_naive).total_seconds()) > 60:
                continue

        entry_price = float(ohlcv["close"].iat[idx])
        if entry_price <= 0:
            continue

        # ATR at entry (for barrier scaling)
        atr_series = atr_cache.get(asset)
        entry_atr = float(atr_series.iat[idx]) if (atr_series is not None and idx < len(atr_series)) else np.nan

        # --- Barrier placement ---
        sl_price = np.nan
        tp_price = np.nan

        # Percentage-based stop-loss (e.g., 2% for live-executor consistency)
        if has_sl_pct:
            sl_dist = exit_config.stop_loss_pct * entry_price
            sl_price = entry_price - sl_dist if direction > 0 else entry_price + sl_dist

        # ATR-based stop-loss (overrides pct if both set)
        if has_sl_atr and np.isfinite(entry_atr) and entry_atr > 0:
            sl_dist = exit_config.stop_loss_atr_mult * entry_atr
            sl_price = entry_price - sl_dist if direction > 0 else entry_price + sl_dist

        # Percentage-based take-profit
        if has_tp_pct:
            tp_dist = exit_config.take_profit_pct * entry_price
            tp_price = entry_price + tp_dist if direction > 0 else entry_price - tp_dist

        # ATR-based take-profit (overrides pct if both set)
        if has_tp_atr and np.isfinite(entry_atr) and entry_atr > 0:
            tp_dist = exit_config.take_profit_atr_mult * entry_atr
            tp_price = entry_price + tp_dist if direction > 0 else entry_price - tp_dist

        # Adaptive hold period (stronger signals held longer)
        max_hold = exit_config.max_hold_bars
        if exit_config.adaptive_hold and exit_config.adaptive_hold_extra > 0:
            ss = sig_strengths[i]
            if np.isfinite(ss):
                max_hold += round(ss * exit_config.adaptive_hold_extra)

        # --- Walk forward bar by bar ---
        exited = False
        peak_favorable = entry_price  # tracks best price for trailing stop
        last_valid_j = 0

        for j in range(1, max_hold + 1):
            bar_idx = idx + j
            if bar_idx >= len(ohlcv):
                break

            bar_high = float(ohlcv["high"].iat[bar_idx])
            bar_low = float(ohlcv["low"].iat[bar_idx])
            last_valid_j = j

            # Update peak for trailing stop
            if direction > 0:
                if bar_high > peak_favorable:
                    peak_favorable = bar_high
            else:
                if bar_low < peak_favorable:
                    peak_favorable = bar_low

            # Check stop-loss (HIGHEST priority — assume hit first on ambiguous bars)
            if np.isfinite(sl_price):
                sl_hit = False
                if direction > 0 and bar_low <= sl_price:
                    sl_hit = True
                elif direction < 0 and bar_high >= sl_price:
                    sl_hit = True

                if sl_hit:
                    exit_returns[i] = (sl_price - entry_price) / entry_price
                    bars_held_arr[i] = j
                    exit_timestamps[i] = np.datetime64(ohlcv.index[bar_idx].to_datetime64())
                    exited = True
                    break

            # Check take-profit
            if np.isfinite(tp_price):
                tp_hit = False
                if direction > 0 and bar_high >= tp_price:
                    tp_hit = True
                elif direction < 0 and bar_low <= tp_price:
                    tp_hit = True

                if tp_hit:
                    exit_returns[i] = (tp_price - entry_price) / entry_price
                    bars_held_arr[i] = j
                    exit_timestamps[i] = np.datetime64(ohlcv.index[bar_idx].to_datetime64())
                    exited = True
                    break

            # Check trailing stop (only after minimum profit reached)
            if has_trail and np.isfinite(entry_atr) and entry_atr > 0:
                trail_dist = exit_config.trail_atr_mult * entry_atr
                bar_close = float(ohlcv["close"].iat[bar_idx])

                if direction > 0:
                    unrealized_pct = (bar_close - entry_price) / entry_price
                else:
                    unrealized_pct = (entry_price - bar_close) / entry_price

                if unrealized_pct >= exit_config.trail_min_profit_pct:
                    if direction > 0:
                        trail_stop = peak_favorable - trail_dist
                        if bar_low <= trail_stop:
                            exit_returns[i] = (trail_stop - entry_price) / entry_price
                            bars_held_arr[i] = j
                            exit_timestamps[i] = np.datetime64(ohlcv.index[bar_idx].to_datetime64())
                            exited = True
                            break
                    else:
                        trail_stop = peak_favorable + trail_dist
                        if bar_high >= trail_stop:
                            exit_returns[i] = (trail_stop - entry_price) / entry_price
                            bars_held_arr[i] = j
                            exit_timestamps[i] = np.datetime64(ohlcv.index[bar_idx].to_datetime64())
                            exited = True
                            break

        # Time expiry — exit at close of last bar
        if not exited:
            end_idx = idx + last_valid_j
            if last_valid_j > 0 and end_idx < len(ohlcv):
                exit_price = float(ohlcv["close"].iat[end_idx])
                exit_returns[i] = (exit_price - entry_price) / entry_price
                bars_held_arr[i] = last_valid_j
                exit_timestamps[i] = np.datetime64(ohlcv.index[end_idx].to_datetime64())

    out = signals_df.copy()
    out["fwd_return"] = exit_returns
    out["bars_held"] = bars_held_arr
    out["exit_timestamp"] = pd.to_datetime(exit_timestamps, utc=True)

    # Drop signals where exit couldn't be computed
    valid = np.isfinite(out["fwd_return"].values) & (out["bars_held"].values > 0)
    return out[valid].reset_index(drop=True)
