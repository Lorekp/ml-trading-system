"""Volume-aware slippage model for realistic backtesting.

Three-component model (all one-way, returned as decimal fraction):
  1. Half-spread (from BBO calibration or volume-tier default)
  2. Square-root market impact (Almgren-Chriss style)
  3. Fee cost (VIP-tier maker/taker blend weighted by fill rate)

Usage:
    from signal_engine.slippage import compute_slippage, lookup_volume

    vol = lookup_volume(asset_volumes, "BTC", some_timestamp)
    slip = compute_slippage("BTC", 40_000, vol, calibration, ("VIP0", 2.0, 5.5))
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Volume tier thresholds (daily USD volume)
TIER_THRESHOLDS = [
    ("tier1", 500_000_000),
    ("tier2", 50_000_000),
    ("tier3", 10_000_000),
    ("tier4", 0),
]

# Square-root impact coefficient (bps) per tier
# Conservative estimates for crypto perps
IMPACT_COEFF = {
    "tier1": 1.0,
    "tier2": 3.0,
    "tier3": 5.0,
    "tier4": 10.0,
}

# Default spread/fill when no calibration available
DEFAULT_TIER_PARAMS = {
    "tier1": {"spread_bps": 0.3, "fill_rate": 0.97},
    "tier2": {"spread_bps": 0.5, "fill_rate": 0.95},
    "tier3": {"spread_bps": 1.0, "fill_rate": 0.90},
    "tier4": {"spread_bps": 2.0, "fill_rate": 0.80},
}


def get_volume_tier(daily_volume: float) -> str:
    for tier, threshold in TIER_THRESHOLDS:
        if daily_volume >= threshold:
            return tier
    return "tier4"


def load_per_symbol_calibration(path: Path | None = None) -> dict[str, dict]:
    """Load per-symbol spread/fill calibration. Returns {ASSET: {spread, fill_rate}}."""
    if path is None:
        path = Path("data/execution_calibration.json")
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    result = {}
    for entry in data.get("per_symbol", []):
        sym = entry.get("symbol", "")
        # Strip USDT suffix to get asset name
        asset = sym.replace("USDT", "") if sym.endswith("USDT") else sym
        result[asset] = {
            "spread_bps": entry.get("spread_median_bps", 1.0),
            "fill_rate": entry.get("fill_rate", 0.85),
        }
    return result


def lookup_volume(
    asset_volumes: dict[str, pd.Series],
    asset: str,
    timestamp,
) -> float:
    """Look up daily dollar volume for an asset at a given timestamp.

    Uses searchsorted for fast lookup on the time-series index.
    Returns 0.0 if no data available.
    """
    vol_series = asset_volumes.get(asset)
    if vol_series is None or len(vol_series) == 0:
        return 0.0

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    idx = vol_series.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        idx = idx.tz_localize(None)

    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return 0.0
    val = vol_series.iloc[pos]
    if not np.isfinite(val):
        return 0.0
    return float(val)


def compute_slippage(
    asset: str,
    position_notional: float,
    daily_volume: float,
    per_symbol_cal: dict[str, dict] | None = None,
    vip_tier: tuple[str, float, float] = ("VIP0", 2.0, 5.5),
) -> float:
    """Compute one-way slippage as a decimal fraction.

    Args:
        asset: Asset name (e.g. "BTC")
        position_notional: Trade size in USD
        daily_volume: Asset's daily dollar volume at trade time
        per_symbol_cal: Per-symbol calibration {asset: {spread_bps, fill_rate}}
        vip_tier: (name, maker_bps, taker_bps)

    Returns:
        One-way slippage as decimal (e.g. 0.0003 for 3bps)
    """
    if daily_volume <= 0 or position_notional <= 0:
        # No volume data: use conservative flat estimate
        return 5.0 / 10_000  # 5bps

    tier = get_volume_tier(daily_volume)
    _, maker_bps, taker_bps = vip_tier

    # 1. Half-spread
    if per_symbol_cal and asset in per_symbol_cal:
        spread_bps = per_symbol_cal[asset]["spread_bps"]
        fill_rate = per_symbol_cal[asset]["fill_rate"]
    else:
        spread_bps = DEFAULT_TIER_PARAMS[tier]["spread_bps"]
        fill_rate = DEFAULT_TIER_PARAMS[tier]["fill_rate"]
    half_spread = spread_bps / 2.0

    # 2. Square-root market impact
    participation = position_notional / daily_volume
    impact_bps = IMPACT_COEFF[tier] * math.sqrt(participation)

    # 3. Fee cost (blended maker/taker)
    fee_bps = fill_rate * maker_bps + (1.0 - fill_rate) * taker_bps

    total_bps = half_spread + impact_bps + fee_bps
    return total_bps / 10_000


def max_profitable_size(
    asset: str,
    expected_edge: float,
    daily_volume: float,
    per_symbol_cal: dict[str, dict] | None = None,
    vip_tier: tuple[str, float, float] = ("VIP0", 2.0, 5.5),
    guard_pct: float = 0.50,
) -> float:
    """Compute the position size cap where marginal slippage = marginal edge.

    Derivation: PnL(S) = S * (edge - fixed - c*sqrt(S/V)/10000)
    dPnL/dS = 0  =>  S* = V * ((edge - fixed)*10000 / (1.5*c))^2
    Returns guard_pct * S* (default 50% safety margin).

    All inputs are lookahead-free: expected_edge comes from model probability,
    daily_volume from historical rolling median, calibration from past BBO data.

    Args:
        asset: Asset name
        expected_edge: Model-predicted edge as decimal (e.g. 2*|prob-0.5|)
        daily_volume: Asset's daily dollar volume
        per_symbol_cal: Per-symbol calibration
        vip_tier: (name, maker_bps, taker_bps)
        guard_pct: Fraction of S* to use (0.5 = 50% safety margin)

    Returns:
        Max position size in USD, or float('inf') if no binding constraint.
    """
    if daily_volume <= 0 or expected_edge <= 0:
        return 0.0

    tier = get_volume_tier(daily_volume)
    _, maker_bps, taker_bps = vip_tier

    # Fixed costs (one-way, as decimal)
    if per_symbol_cal and asset in per_symbol_cal:
        spread_bps = per_symbol_cal[asset]["spread_bps"]
        fill_rate = per_symbol_cal[asset]["fill_rate"]
    else:
        spread_bps = DEFAULT_TIER_PARAMS[tier]["spread_bps"]
        fill_rate = DEFAULT_TIER_PARAMS[tier]["fill_rate"]

    half_spread = spread_bps / 2.0
    fee_bps = fill_rate * maker_bps + (1.0 - fill_rate) * taker_bps
    fixed_cost = (half_spread + fee_bps) / 10_000

    # Edge must exceed fixed costs to be profitable at any size
    net_edge = expected_edge - fixed_cost
    if net_edge <= 0:
        return 0.0

    coeff = IMPACT_COEFF[tier]
    # S* = V * ((net_edge * 10000) / (1.5 * coeff))^2
    ratio = (net_edge * 10_000) / (1.5 * coeff)
    s_star = daily_volume * ratio * ratio

    return guard_pct * s_star
