"""Realistic execution cost modeling — MkrTkr60 with BBO data.

Implements per-trade execution cost computation using actual BBO (best bid/offer)
data to determine maker fill rates, or a statistical fallback when BBO is unavailable.

Target exchange: Bybit (VIP tier fee schedule).
BBO data source: Hyperliquid S3 archive or Binance bookTicker parquets.

Usage:
    from signal_engine.execution import adjust_fwd_returns, BYBIT_VIP_TIERS

    # Adjust fwd_returns for realistic execution costs before portfolio sim
    signals = adjust_fwd_returns(signals_df, bbo_data, tick_sizes, equity=250_000)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# Fee schedules
# ============================================================

# Bybit VIP tiers: (name, min_equity, min_30d_volume, maker_bps, taker_bps)
BYBIT_VIP_TIERS = [
    ("VIP0",    0,           0,             2.00, 5.50),
    ("VIP1",    100_000,     10_000_000,    1.80, 4.00),
    ("VIP2",    250_000,     25_000_000,    1.60, 3.75),
    ("VIP3",    500_000,     50_000_000,    1.40, 3.50),
    ("VIP4",    1_000_000,   100_000_000,   1.20, 3.20),
    ("VIP5",    2_000_000,   250_000_000,   1.00, 3.20),
    ("Supreme", 5_000_000,   500_000_000,   0.00, 3.00),
]
SPREAD_BPS = 0.5


def get_vip_tier(equity: float, volume_30d: float = 0) -> tuple[str, float, float]:
    """Get Bybit VIP tier fees based on equity and 30d volume.

    Returns (tier_name, maker_bps, taker_bps).
    """
    result = BYBIT_VIP_TIERS[0]
    for tier in BYBIT_VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ============================================================
# BBO data loading
# ============================================================

BBO_DIRS = [Path("data/hl-bbo"), Path("data/bookTicker_parquet")]


def load_bbo_for_dates(
    symbol: str,
    dates: list[str],
    bbo_dirs: list[Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load BBO data for a symbol across specified dates.

    Checks hl-bbo/ first (Hyperliquid), then bookTicker_parquet/ (Binance).

    Returns (timestamps_ns, best_bid, best_ask) as numpy arrays,
    or None if no data found.
    """
    if bbo_dirs is None:
        bbo_dirs = BBO_DIRS

    all_ts, all_bid, all_ask = [], [], []

    for date_str in dates:
        found = False
        for bdir in bbo_dirs:
            pq_path = bdir / symbol / f"{date_str}.parquet"
            if pq_path.exists():
                try:
                    df = pd.read_parquet(pq_path)
                    ts = df["transaction_time"].values.astype(np.int64) * 1_000_000  # ms → ns
                    all_ts.append(ts)
                    all_bid.append(df["best_bid_price"].values.astype(np.float64))
                    all_ask.append(df["best_ask_price"].values.astype(np.float64))
                    found = True
                    break
                except Exception:
                    continue

    if not all_ts:
        return None

    return (np.concatenate(all_ts), np.concatenate(all_bid), np.concatenate(all_ask))


def load_bbo_dict(
    symbols: list[str],
    date_range: tuple[str, str],
    bbo_dirs: list[Path] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load BBO data for multiple symbols across a date range.

    Returns {symbol: (timestamps_ns, best_bid, best_ask)}.
    Only includes symbols with available data.
    """
    if bbo_dirs is None:
        bbo_dirs = BBO_DIRS

    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    dates = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()

    result = {}
    for sym in symbols:
        data = load_bbo_for_dates(sym, dates, bbo_dirs)
        if data is not None and len(data[0]) > 0:
            result[sym] = data

    return result


# ============================================================
# MkrTkr60 fill checking — ported from execution_strategy_ranking.py:250
# ============================================================

def check_maker_fill(
    bts: np.ndarray,
    bbid: np.ndarray,
    bask: np.ndarray,
    times_ns: np.ndarray,
    directions: np.ndarray,
    timeout_ns: int = 60_000_000_000,
    tick: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Check if maker orders fill within timeout window.

    For LONG: place limit buy at best bid, check if ask touches bid within timeout.
    For SHORT: place limit sell at best ask, check if bid touches ask within timeout.

    Args:
        bts: BBO timestamps in nanoseconds (sorted)
        bbid: Best bid prices
        bask: Best ask prices
        times_ns: Signal timestamps in nanoseconds
        directions: +1 for long, -1 for short
        timeout_ns: Maximum wait time in nanoseconds (default 60s)
        tick: Tick size for the asset

    Returns:
        (filled: bool array, limit_prices: float array)
    """
    n = len(directions)
    filled = np.zeros(n, dtype=bool)
    limit_px = np.full(n, np.nan)

    for i in range(n):
        s = np.searchsorted(bts, times_ns[i], side="right") - 1
        if s < 0 or s >= len(bbid):
            continue
        e = np.searchsorted(bts, times_ns[i] + timeout_ns, side="right")
        if directions[i] > 0:
            lp = bbid[s]
            if np.any(bask[s:e] <= lp + tick * 0.01):
                filled[i] = True
                limit_px[i] = lp
        else:
            lp = bask[s]
            if np.any(bbid[s:e] >= lp - tick * 0.01):
                filled[i] = True
                limit_px[i] = lp

    return filled, limit_px


# ============================================================
# Statistical fallback model
# ============================================================

# Default calibration: volume-tier → maker fill rate + avg spread
# Calibrated from Binance BBO data (37 symbols, May 2023 - Mar 2024)
DEFAULT_CALIBRATION = {
    "tier1": {"volume_min": 500_000_000, "fill_rate": 0.97, "spread_bps": 0.3,
              "desc": "BTC, ETH — mega-cap"},
    "tier2": {"volume_min": 50_000_000, "fill_rate": 0.95, "spread_bps": 0.5,
              "desc": "Top 10 by volume"},
    "tier3": {"volume_min": 10_000_000, "fill_rate": 0.90, "spread_bps": 1.0,
              "desc": "Top 30-50 by volume"},
    "tier4": {"volume_min": 0, "fill_rate": 0.80, "spread_bps": 2.0,
              "desc": "Everything else"},
}


def load_calibration(path: Path | None = None) -> dict:
    """Load calibrated execution model parameters.

    Falls back to DEFAULT_CALIBRATION if file not found.
    """
    if path is None:
        path = Path("data/execution_calibration.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return DEFAULT_CALIBRATION


def get_statistical_cost(
    volume: float,
    maker_bps: float,
    taker_bps: float,
    calibration: dict | None = None,
) -> float:
    """Estimate round-trip execution cost for an asset without BBO data.

    Uses volume-tier-based maker fill rate and spread estimate.

    Returns cost in bps (round-trip: entry + exit).
    """
    if calibration is None:
        calibration = DEFAULT_CALIBRATION

    # Find tier
    for tier in ["tier1", "tier2", "tier3", "tier4"]:
        if volume >= calibration[tier]["volume_min"]:
            fill_rate = calibration[tier]["fill_rate"]
            spread = calibration[tier]["spread_bps"]
            break
    else:
        fill_rate = 0.80
        spread = 2.0

    # RT cost: entry + exit, each with fill_rate chance of maker
    rt = fill_rate * 2 * maker_bps + (1 - fill_rate) * 2 * taker_bps + spread
    return rt


# ============================================================
# Per-trade execution cost computation
# ============================================================

# Default tick sizes for common symbols
DEFAULT_TICK_SIZES = {
    "BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.001, "XRPUSDT": 0.0001,
    "BNBUSDT": 0.01, "ADAUSDT": 0.0001, "AVAXUSDT": 0.001, "DOTUSDT": 0.001,
    "LINKUSDT": 0.001, "NEARUSDT": 0.001, "UNIUSDT": 0.001, "AAVEUSDT": 0.01,
    "LTCUSDT": 0.01, "BCHUSDT": 0.01, "FILUSDT": 0.001, "FTMUSDT": 0.0001,
    "HBARUSDT": 0.00001, "TRXUSDT": 0.0001, "SUIUSDT": 0.0001, "APTUSDT": 0.001,
    "ENJUSDT": 0.0001, "SXPUSDT": 0.0001, "ZECUSDT": 0.01, "XLMUSDT": 0.00001,
    "FETUSDT": 0.0001, "WLDUSDT": 0.001, "BNXUSDT": 0.001, "ANKRUSDT": 0.00001,
    "TAOUSDT": 0.01, "ENAUSDT": 0.0001, "DOGEUSDT": 0.00001, "ARBUSDT": 0.0001,
    "1000PEPEUSDT": 0.0000001, "1000SHIBUSDT": 0.000001,
}


def compute_execution_costs(
    signals_df: pd.DataFrame,
    bbo_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    tick_sizes: dict[str, float] | None = None,
    equity: float = 250_000,
    volume_30d: float = 0,
    calibration: dict | None = None,
    asset_volumes: dict[str, float] | None = None,
    timeout_ns: int = 60_000_000_000,
) -> tuple[np.ndarray, dict]:
    """Compute per-trade round-trip execution cost in bps.

    For signals with BBO data: uses MkrTkr60 fill checking on ENTRY leg.
    For signals without BBO: uses statistical fallback.
    Exit leg uses same fill rate assumption (symmetric).

    Args:
        signals_df: DataFrame with columns [timestamp, asset, direction, strategy, hold_bars]
        bbo_data: {SYMBOL: (timestamps_ns, best_bid, best_ask)}
        tick_sizes: {SYMBOL: tick_size}
        equity: Current equity for VIP tier
        volume_30d: 30-day trading volume for VIP tier
        calibration: Statistical fallback calibration dict
        asset_volumes: {ASSET_NAME: approx daily dollar volume} for statistical fallback
        timeout_ns: Maker fill timeout in nanoseconds

    Returns:
        (cost_bps_array, stats_dict)
    """
    if tick_sizes is None:
        tick_sizes = DEFAULT_TICK_SIZES
    if calibration is None:
        calibration = load_calibration()
    if asset_volumes is None:
        asset_volumes = {}

    n = len(signals_df)
    costs = np.full(n, np.nan)

    _, maker_bps, taker_bps = get_vip_tier(equity, volume_30d)

    # Parse timestamps
    ts = pd.to_datetime(signals_df["timestamp"].values)
    if hasattr(ts, "tz") and ts.tz is not None:
        ts = ts.tz_localize(None)
    ts_ns = ts.astype(np.int64)

    # Parse directions
    if "direction" in signals_df.columns:
        dirs = signals_df["direction"].values.astype(np.float64)
    else:
        dirs = np.ones(n)

    assets = signals_df["asset"].values
    n_bbo = 0
    n_stat = 0

    # Group by asset for efficiency
    for asset_name in np.unique(assets):
        mask = assets == asset_name
        idxs = np.where(mask)[0]
        sym = f"{asset_name}USDT"
        tick = tick_sizes.get(sym, 0.01)

        if sym in bbo_data:
            # BBO-based: check actual maker fills
            bts, bbid, bask = bbo_data[sym]
            sig_ts_ns = ts_ns[idxs]
            sig_dirs = dirs[idxs]

            # Entry leg
            e_filled, e_px = check_maker_fill(bts, bbid, bask, sig_ts_ns, sig_dirs, timeout_ns, tick)
            # Exit leg (opposite direction)
            # Compute exit times based on hold_bars and strategy
            hold_bars = signals_df.iloc[idxs]["hold_bars"].values.astype(int)
            strategies = signals_df.iloc[idxs]["strategy"].values
            bar_ns = np.zeros(len(idxs), dtype=np.int64)
            for j in range(len(idxs)):
                st = strategies[j]
                if st == "15m":
                    bar_ns[j] = 15 * 60 * 1_000_000_000
                elif st == "1h":
                    bar_ns[j] = 60 * 60 * 1_000_000_000
                elif st == "3h":
                    bar_ns[j] = 3 * 60 * 60 * 1_000_000_000
                elif st == "4h":
                    bar_ns[j] = 4 * 60 * 60 * 1_000_000_000
                elif st == "24h":
                    bar_ns[j] = 24 * 60 * 60 * 1_000_000_000
                else:
                    bar_ns[j] = 60 * 60 * 1_000_000_000  # default 1h

            exit_ts_ns = sig_ts_ns + hold_bars * bar_ns
            x_filled, x_px = check_maker_fill(
                bts, bbid, bask, exit_ts_ns, -sig_dirs, timeout_ns, tick)

            # Compute per-trade cost
            for j in range(len(idxs)):
                e_fee = maker_bps if e_filled[j] else taker_bps
                x_fee = maker_bps if x_filled[j] else taker_bps
                # Spread cost: maker gets favorable price (0 spread cost),
                # taker pays half-spread on each leg
                e_spread = 0.0 if e_filled[j] else SPREAD_BPS / 2
                x_spread = 0.0 if x_filled[j] else SPREAD_BPS / 2
                costs[idxs[j]] = e_fee + x_fee + e_spread + x_spread

            n_bbo += len(idxs)
        else:
            # Statistical fallback
            vol = asset_volumes.get(asset_name, 10_000_000)
            rt_cost = get_statistical_cost(vol, maker_bps, taker_bps, calibration)
            costs[idxs] = rt_cost
            n_stat += len(idxs)

    # Fill any remaining NaNs with taker worst-case
    nan_mask = np.isnan(costs)
    if nan_mask.any():
        costs[nan_mask] = 2 * taker_bps + SPREAD_BPS

    stats = {
        "n_total": n,
        "n_bbo": n_bbo,
        "n_statistical": n_stat,
        "pct_bbo": n_bbo / n * 100 if n > 0 else 0,
        "avg_cost_bps": float(np.mean(costs)),
        "maker_bps": maker_bps,
        "taker_bps": taker_bps,
    }

    return costs, stats


def adjust_fwd_returns(
    signals_df: pd.DataFrame,
    bbo_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    tick_sizes: dict[str, float] | None = None,
    equity: float = 250_000,
    volume_30d: float = 0,
    calibration: dict | None = None,
    asset_volumes: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Adjust fwd_return to include realistic execution costs.

    After calling this, set PortfolioConfig.slippage = 0 since costs
    are already baked into fwd_return.

    Returns (modified_signals_df, stats_dict).
    """
    costs, stats = compute_execution_costs(
        signals_df, bbo_data, tick_sizes, equity, volume_30d,
        calibration, asset_volumes)

    out = signals_df.copy()
    out["fwd_return"] = out["fwd_return"].astype(float) - costs / 10000  # bps → decimal

    logger.info(
        "Execution costs: %.1f%% BBO-based, %.1f%% statistical, avg=%.1f bps RT",
        stats["pct_bbo"], 100 - stats["pct_bbo"], stats["avg_cost_bps"])

    return out, stats
