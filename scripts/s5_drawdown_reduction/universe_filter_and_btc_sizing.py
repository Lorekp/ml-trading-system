"""Historical S5 C2/C4 drawdown-reduction candidate.

S5 remains uncertain: independent PnL realism was not evaluated and the full
universe needs a clean rerun. The historical within-model comparison favored
this overlay, but that did not validate or authorize it.

This module provides two pure-function post-processors that
take (pnl, sizes, skipped, timestamps, dpnl_adj, assets) and return modified
versions. The wrapped live engine would call these BEFORE persisting the
trade log (the engine's existing simulate path is unchanged; the
overlay only affects how the position is sized and whether it enters).

Look-ahead safety:
  - C2 uses shift(1) on the rolling 30d Sharpe: only T-1 data is in the
    decision for trade entered at T.
  - C4 uses shift(1) on the rolling 30d BTC correlation: only T-1 data.
  - Both are evaluated per-asset per-day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def c2_universe_filter(
    pnl: np.ndarray, sizes: np.ndarray, skipped: np.ndarray,
    timestamps: np.ndarray, dpnl_adj: pd.Series, assets: pd.Series,
    window: int = 30, min_periods: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Skip trades whose asset has rolling-30d Sharpe < 0 (T-1 lookback).

    Args:
        pnl, sizes, skipped: per-trade arrays from simulate_with_breaker.
        timestamps: per-trade timestamp array (datetime64).
        dpnl_adj: daily PnL series (informational; not used in this filter).
        assets: per-trade asset name.

    Returns:
        (new_pnl, sizes, new_skipped, new_dpnl) — modified versions.
    """
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "asset": assets})
    ts_norm = pd.DatetimeIndex(df["ts"]).normalize()
    asset_daily = df.groupby([ts_norm, "asset"])["pnl"].sum().unstack(fill_value=0.0)
    # shift(1) — T-1 rolling Sharpe only (NO LOOK-AHEAD)
    asset_roll = asset_daily.shift(1).rolling(window, min_periods=min_periods).sum()
    asset_std = asset_daily.shift(1).rolling(window, min_periods=min_periods).std().fillna(0.0) + 1e-9
    asset_sharpe_proxy = (asset_roll / asset_std).fillna(0.0)
    skip_mask = pd.Series(False, index=range(len(pnl)))
    trade_dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    for i in range(len(pnl)):
        if skipped[i] or pd.isna(assets.iloc[i]):
            continue
        a = assets.iloc[i]
        if a not in asset_sharpe_proxy.columns:
            continue
        d = trade_dates[i]
        if d not in asset_sharpe_proxy.index:
            continue
        sh = asset_sharpe_proxy.at[d, a]
        if pd.notna(sh) and sh < 0:
            skip_mask[i] = True
    new_pnl = np.where(skip_mask.values | skipped, 0.0, pnl)
    new_skipped = skipped | skip_mask.values
    df2 = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df2[~new_skipped].groupby(df2[~new_skipped]["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, sizes, new_skipped, new_dpnl


def c4_btc_corr_sizing(
    pnl: np.ndarray, sizes: np.ndarray, skipped: np.ndarray,
    timestamps: np.ndarray, dpnl_adj: pd.Series, assets: pd.Series,
    window: int = 30, min_periods: int = 15,
    high_corr_thr: float = 0.5, low_corr_thr: float = 0.3,
    high_mult: float = 0.5, low_mult: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Reduce trade size for assets with high rolling-30d BTC correlation.

    Uses T-1 rolling correlation only (shift(1), NO LOOK-AHEAD).
    """
    df = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": pnl, "asset": assets})
    asset_daily = df.groupby([df["ts"].dt.normalize(), "asset"])["pnl"].sum().unstack(fill_value=0.0)
    if "BTC" not in asset_daily.columns:
        return pnl, sizes, skipped, dpnl_adj
    btc = asset_daily["BTC"]
    rolling_corr = {}
    for a in asset_daily.columns:
        if a == "BTC":
            rolling_corr[a] = pd.Series(0.0, index=asset_daily.index)
            continue
        rolling_corr[a] = asset_daily[a].shift(1).rolling(window, min_periods=min_periods).corr(btc.shift(1)).fillna(0.0)
    rolling_corr_df = pd.DataFrame(rolling_corr)
    new_pnl = pnl.copy()
    new_sizes = sizes.copy()
    new_skipped = skipped.copy()
    trade_dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    for i in range(len(pnl)):
        if skipped[i] or pd.isna(assets.iloc[i]):
            continue
        a = assets.iloc[i]
        if a not in rolling_corr_df.columns:
            continue
        d = trade_dates[i]
        if d not in rolling_corr_df.index:
            continue
        c = rolling_corr_df.at[d, a]
        if pd.notna(c) and c > high_corr_thr:
            mult = high_mult
        elif pd.notna(c) and c > low_corr_thr:
            mult = low_mult
        else:
            mult = 1.0
        new_pnl[i] = pnl[i] * mult
        new_sizes[i] = sizes[i] * mult
    df2 = pd.DataFrame({"ts": pd.to_datetime(timestamps), "pnl": new_pnl})
    daily = df2.groupby(df2["ts"].dt.normalize())["pnl"].sum()
    full = pd.date_range(pd.to_datetime(timestamps).min().normalize(),
                         pd.to_datetime(timestamps).max().normalize(), freq="D")
    new_dpnl = daily.reindex(full, fill_value=0.0)
    return new_pnl, new_sizes, new_skipped, new_dpnl


def apply_stack(
    pnl: np.ndarray, sizes: np.ndarray, skipped: np.ndarray,
    timestamps: np.ndarray, dpnl_adj: pd.Series, assets: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Apply C2 then C4 — the ACKed stack."""
    new_pnl, new_sizes, new_skipped, new_dpnl = c2_universe_filter(
        pnl, sizes, skipped, timestamps, dpnl_adj, assets)
    return c4_btc_corr_sizing(
        new_pnl, new_sizes, new_skipped, timestamps, new_dpnl, assets)
