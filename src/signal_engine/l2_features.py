"""L2 feature computation — aggregates tick/depth data to 15-minute bars.

Implements:
- Order Flow Imbalance (Kolm, Turiel & Westray 2023)
- Spread dynamics (Harris, Trading & Exchanges)
- Depth imbalance from aggregate bookDepth

All features are backward-looking (use data up to the bar's close, never future).
Output: DataFrame indexed by 15m bar timestamps, same alignment as OHLCV features.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

logger = logging.getLogger(__name__)

# Feature names (all prefixed l2_ to avoid collision with OHLCV features)
OFI_FEATURES = ["l2_ofi_sum", "l2_ofi_mean", "l2_ofi_std", "l2_ofi_skew"]
SPREAD_FEATURES = ["l2_spread_mean", "l2_spread_std", "l2_spread_max"]
IMBALANCE_FEATURES = ["l2_imbalance_mean", "l2_imbalance_end"]
ACTIVITY_FEATURES = ["l2_tick_count", "l2_mid_return"]
DEPTH_FEATURES = [
    "l2_depth_imbalance_1pct", "l2_depth_imbalance_2pct",
    "l2_total_bid_depth_2pct", "l2_depth_change_1pct",
]

ALL_BOOK_TICKER_FEATURES = OFI_FEATURES + SPREAD_FEATURES + IMBALANCE_FEATURES + ACTIVITY_FEATURES
ALL_BOOK_DEPTH_FEATURES = DEPTH_FEATURES
ALL_L2_FEATURES = ALL_BOOK_TICKER_FEATURES + ALL_BOOK_DEPTH_FEATURES


# ============================================================
# Tick-level computation (bookTicker)
# ============================================================

def compute_tick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-tick features from raw bookTicker data.

    Implements Kolm (2023) OFI at the best level:
        bOF = +qty if bid_price rises, (qty - prev_qty) if same, -prev_qty if drops
        aOF = -qty if ask_price drops, -(qty - prev_qty) if same, +prev_qty if rises
        OFI = bOF - aOF

    Args:
        df: Raw bookTicker with columns: best_bid_price, best_bid_qty,
            best_ask_price, best_ask_qty, transaction_time, event_time

    Returns:
        Same DataFrame with added columns: ofi, spread_pct, bid_ask_imbalance, mid_price
    """
    out = df.copy()

    bbp = out["best_bid_price"].values
    bbq = out["best_bid_qty"].values
    bap = out["best_ask_price"].values
    baq = out["best_ask_qty"].values

    n = len(out)
    ofi = np.zeros(n, dtype=np.float64)

    # Vectorized OFI computation (Kolm 2023, Eq. 1-3)
    for i in range(1, n):
        # Bid-side order flow
        if bbp[i] > bbp[i - 1]:
            bid_of = bbq[i]  # new higher bid, full volume
        elif bbp[i] == bbp[i - 1]:
            bid_of = bbq[i] - bbq[i - 1]  # same price, volume change
        else:
            bid_of = -bbq[i - 1]  # bid dropped, previous volume removed

        # Ask-side order flow
        if bap[i] < bap[i - 1]:
            ask_of = -baq[i]  # ask dropped, full volume
        elif bap[i] == bap[i - 1]:
            ask_of = -(baq[i] - baq[i - 1])  # same price, volume change
        else:
            ask_of = baq[i - 1]  # ask rose, previous volume removed

        ofi[i] = bid_of - ask_of

    out["ofi"] = ofi
    mid = (bbp + bap) / 2.0
    out["mid_price"] = mid
    spread = bap - bbp
    out["spread_pct"] = np.where(mid > 0, spread / mid, 0.0)
    out["bid_ask_imbalance"] = (bbq - baq) / (bbq + baq + 1e-10)

    return out


def aggregate_book_ticker_to_15m(tick_df: pd.DataFrame) -> pd.DataFrame:
    """Resample tick-level features to 15-minute bars.

    Args:
        tick_df: Output of compute_tick_features() with ofi, spread_pct, etc.

    Returns:
        DataFrame indexed by 15m bar timestamps with L2 features.
    """
    ts = pd.to_datetime(tick_df["transaction_time"], unit="ms", utc=True)
    tick_df = tick_df.set_index(ts)

    grouped = tick_df.resample("15min")

    def safe_skew(x):
        if len(x) < 3:
            return 0.0
        return float(scipy_skew(x, nan_policy="omit"))

    bars = pd.DataFrame({
        # OFI features (Kolm 2023)
        "l2_ofi_sum": grouped["ofi"].sum(),
        "l2_ofi_mean": grouped["ofi"].mean(),
        "l2_ofi_std": grouped["ofi"].std(),
        "l2_ofi_skew": grouped["ofi"].apply(safe_skew),

        # Spread features (Harris)
        "l2_spread_mean": grouped["spread_pct"].mean(),
        "l2_spread_std": grouped["spread_pct"].std(),
        "l2_spread_max": grouped["spread_pct"].max(),

        # Imbalance features
        "l2_imbalance_mean": grouped["bid_ask_imbalance"].mean(),
        "l2_imbalance_end": grouped["bid_ask_imbalance"].last(),

        # Activity
        "l2_tick_count": grouped["ofi"].count(),
    })

    # Mid-price return (bar-to-bar)
    mid_last = grouped["mid_price"].last()
    bars["l2_mid_return"] = mid_last.pct_change()

    # Drop bars with too few ticks (early/late partial bars)
    bars = bars[bars["l2_tick_count"] > 100]

    # Downcast
    for col in bars.columns:
        bars[col] = bars[col].astype(np.float32)

    return bars


# ============================================================
# Depth-level computation (bookDepth)
# ============================================================

def aggregate_book_depth_to_15m(depth_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bookDepth data to 15-minute bars.

    bookDepth has columns: timestamp, percentage, depth, notional
    where percentage is distance from mid (-5 to -1 = bid side, +1 to +5 = ask side).

    Args:
        depth_df: Raw bookDepth DataFrame

    Returns:
        DataFrame indexed by 15m bar timestamps with depth features.
    """
    df = depth_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Pivot: one row per timestamp, columns for each percentage level
    # Separate bid and ask sides
    bid_mask = df["percentage"] < 0
    ask_mask = df["percentage"] > 0

    # For each timestamp, compute depth imbalance
    def compute_depth_features(group):
        bids = group[group["percentage"] < 0]
        asks = group[group["percentage"] > 0]

        bid_1 = bids[bids["percentage"] == -1]["notional"].sum()
        ask_1 = asks[asks["percentage"] == 1]["notional"].sum()
        bid_2 = bids[bids["percentage"].isin([-1, -2])]["notional"].sum()
        ask_2 = asks[asks["percentage"].isin([1, 2])]["notional"].sum()

        total_1 = bid_1 + ask_1
        total_2 = bid_2 + ask_2

        return pd.Series({
            "depth_imbalance_1pct": (bid_1 - ask_1) / (total_1 + 1e-10),
            "depth_imbalance_2pct": (bid_2 - ask_2) / (total_2 + 1e-10),
            "total_bid_depth_2pct": bid_2,
        })

    # Group by timestamp snapshot
    per_snap = df.groupby("timestamp").apply(compute_depth_features, include_groups=False)

    # Resample to 15m
    per_snap.index = pd.to_datetime(per_snap.index)
    grouped = per_snap.resample("15min")

    bars = pd.DataFrame({
        "l2_depth_imbalance_1pct": grouped["depth_imbalance_1pct"].mean(),
        "l2_depth_imbalance_2pct": grouped["depth_imbalance_2pct"].mean(),
        "l2_total_bid_depth_2pct": grouped["total_bid_depth_2pct"].mean(),
    })

    # Depth change (shift of imbalance)
    bars["l2_depth_change_1pct"] = bars["l2_depth_imbalance_1pct"].diff()

    # Drop sparse bars
    bars = bars.dropna()

    for col in bars.columns:
        bars[col] = bars[col].astype(np.float32)

    return bars


# ============================================================
# Caching and orchestration
# ============================================================

def compute_and_cache_l2_features(
    symbol: str,
    data_type: str = "bookDepth",
    remote_dir: str = "/hdd/binance-l2",
    cache_dir: str = "/hdd/binance-l2/features_15m",
) -> pd.DataFrame:
    """Compute 15m L2 features for a symbol, cache as parquet on homeserver.

    Processes all available dates, concatenates, caches.

    Args:
        symbol: e.g. "BTCUSDT"
        data_type: "bookTicker" or "bookDepth"
        remote_dir: base L2 data directory on homeserver
        cache_dir: where to cache aggregated features

    Returns:
        DataFrame with 15m L2 features
    """
    from signal_engine.l2_data import list_available_dates, load_book_ticker, load_book_depth

    cache_path = Path(cache_dir) / f"{symbol}_{data_type}_15m.parquet"

    # Check if already cached (via SSH)
    import subprocess
    check = subprocess.run(
        ["ssh", "monitor", f"test -f {cache_path} && echo exists"],
        capture_output=True, text=True, timeout=10,
    )
    if "exists" in check.stdout:
        # Load from cache
        import io
        data = subprocess.run(
            ["ssh", "monitor", f"cat {cache_path}"],
            capture_output=True, timeout=120,
        )
        return pd.read_parquet(io.BytesIO(data.stdout))

    # Get available dates
    dates = list_available_dates(symbol, data_type)
    if not dates:
        logger.warning("No %s data for %s", data_type, symbol)
        return pd.DataFrame()

    logger.info("Computing %s features for %s (%d dates)", data_type, symbol, len(dates))
    t0 = time.time()

    all_bars = []
    for i, date in enumerate(dates):
        try:
            if data_type == "bookTicker":
                raw = load_book_ticker(symbol, date)
                ticked = compute_tick_features(raw)
                bars = aggregate_book_ticker_to_15m(ticked)
            elif data_type == "bookDepth":
                raw = load_book_depth(symbol, date)
                bars = aggregate_book_depth_to_15m(raw)
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            all_bars.append(bars)

            if (i + 1) % 30 == 0:
                logger.info("  %s %s: %d/%d dates processed", symbol, data_type, i + 1, len(dates))
        except Exception as e:
            logger.warning("  %s %s %s: failed (%s)", symbol, data_type, date, e)

    if not all_bars:
        return pd.DataFrame()

    combined = pd.concat(all_bars).sort_index()
    # Remove duplicates (overlapping bar boundaries)
    combined = combined[~combined.index.duplicated(keep="first")]

    elapsed = time.time() - t0
    logger.info("Computed %s features for %s: %d bars in %.0fs",
                data_type, symbol, len(combined), elapsed)

    # Cache to homeserver
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            combined.to_parquet(tmp.name)
            subprocess.run(
                ["ssh", "monitor", f"mkdir -p {cache_dir}"],
                timeout=10,
            )
            subprocess.run(
                ["scp", tmp.name, f"monitor:{cache_path}"],
                timeout=120,
            )
            Path(tmp.name).unlink()
        logger.info("Cached to %s", cache_path)
    except Exception as e:
        logger.warning("Failed to cache: %s", e)

    return combined


def load_cached_l2_features(
    symbol: str,
    data_type: str = "bookDepth",
    cache_dir: str = "/hdd/binance-l2/features_15m",
) -> pd.DataFrame:
    """Load pre-computed L2 features from homeserver cache."""
    cache_path = f"{cache_dir}/{symbol}_{data_type}_15m.parquet"
    import subprocess, io
    data = subprocess.run(
        ["ssh", "monitor", f"cat {cache_path}"],
        capture_output=True, timeout=120,
    )
    if data.returncode != 0:
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(data.stdout))


def merge_l2_with_ohlcv(
    ohlcv_features: pd.DataFrame,
    l2_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge L2 features into OHLCV features on aligned 15m index.

    L2 features are joined left — missing dates get NaN (filled with 0 for model).
    """
    if l2_features.empty:
        return ohlcv_features

    # Align timezones
    if ohlcv_features.index.tz is None and l2_features.index.tz is not None:
        l2_features = l2_features.copy()
        l2_features.index = l2_features.index.tz_localize(None)
    elif ohlcv_features.index.tz is not None and l2_features.index.tz is None:
        l2_features = l2_features.copy()
        l2_features.index = l2_features.index.tz_localize(ohlcv_features.index.tz)

    merged = ohlcv_features.join(l2_features, how="left")

    # Fill NaN L2 features with 0 (no L2 data = no signal, not missing data)
    l2_cols = [c for c in merged.columns if c.startswith("l2_")]
    merged[l2_cols] = merged[l2_cols].fillna(0)

    return merged
