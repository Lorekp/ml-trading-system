"""
Universal Signal Test v2 — Multi-Asset, Multi-Timeframe Direction Prediction.

Optimized version with:
  - Per-asset parallelism (load data once, run all timeframes)
  - Command-line args (--reverse, --workers, --output)
  - Configurable data paths via environment variables
  - Memory-efficient processing (float32, gc.collect)
  - Better progress reporting with ETA
  - Resume support (skip already-completed combos)

Tests the ML direction prediction signal across ALL available asset classes,
timeframes, and configurations. Answers:
  1. Does the signal generalize to unseen crypto assets?
  2. Does it work on stocks, commodities, forex, indices?
  3. How does it behave across timeframes (1m, 5m, 15m, 30m, 1H, 4H)?
  4. Split-sample stability for long-history assets.

Anti-overfitting discipline (AFML Ch 11, 14):
  - Pre-registered analysis: all combos defined upfront, ALL results reported.
  - Fixed model: LogReg L1 C=0.1 — NO tuning per asset.
  - Fixed filters: vol 50%, confidence 70% — NO tuning.
  - DSR computed with K = total combos tested.
  - Split-sample for assets with >4 years of data.
"""

import argparse
import gc
import os
import sys
import json
import time
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION — ALL PRE-REGISTERED, NO TUNING
# =============================================================================

# Data paths — configurable via environment variables
BINANCE_DIR = Path(os.environ.get("BINANCE_1M_DIR", "/home/user/ml-data/binance-1m"))
HISTDATA_DIR = Path(os.environ.get("HISTDATA_1M_DIR", "/hdd/ml-data/histdata-1m"))
YAHOO_DIR = Path(os.environ.get("NON_CRYPTO_DIR", "/hdd/ml-data/non-crypto"))
MEXC_DIR = Path(os.environ.get("MEXC_1M_DIR", "/hdd/ml-data/mexc-futures-1m"))
OUTPUT_FILE = Path(os.environ.get("OUTPUT_FILE", "/home/user/ml-data/universal_test_results.json"))
LOG_FILE = Path(os.environ.get("LOG_FILE", "/home/user/ml-data/universal_test_log.txt"))

ASSETS = {
    "crypto_core": {
        "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
        "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT",
    },
    "crypto_unseen": {
        "DOGE": "DOGEUSDT", "SOL": "SOLUSDT", "AVAX": "AVAXUSDT",
        "ADA": "ADAUSDT", "DOT": "DOTUSDT", "NEAR": "NEARUSDT",
        "UNI": "UNIUSDT", "AAVE": "AAVEUSDT", "ARB": "ARBUSDT",
        "OP": "OPUSDT",
    },
    "stocks": {
        "AAPL": "AAPLSTOCK_USDT", "TSLA": "TSLASTOCK_USDT",
        "NVDA": "NVDASTOCK_USDT", "AMZN": "AMZNSTOCK_USDT",
        "GOOGL": "GOOGLSTOCK_USDT",
    },
    "commodities": {"XAUUSD": "XAUUSD", "BCOUSD": "BCOUSD"},
    "forex": {"EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY"},
    "indices": {"SPXUSD": "SPXUSD", "NSXUSD": "NSXUSD"},
}

TIMEFRAMES = [
    {"bar_tf": "1min",  "fwd_bars": 5,  "label": "1m_5bar_5min"},
    {"bar_tf": "5min",  "fwd_bars": 6,  "label": "5m_6bar_30min"},
    {"bar_tf": "15min", "fwd_bars": 4,  "label": "15m_4bar_1h"},
    {"bar_tf": "30min", "fwd_bars": 4,  "label": "30m_4bar_2h"},
    {"bar_tf": "1h",    "fwd_bars": 2,  "label": "1h_2bar_2h"},
    {"bar_tf": "4h",    "fwd_bars": 1,  "label": "4h_1bar_4h"},
]

FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]

# Fixed model — no tuning
MODEL_C = 0.1
MODEL_PENALTY = "l1"
MODEL_SOLVER = "saga"
MODEL_MAX_ITER = 1000
MODEL_SEED = 42

# Portfolio parameters
STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001  # 1 bps
MAX_EXPOSURE = 0.60
MAX_PER_ASSET = 0.20
BASE_SIZING_PCT = 0.02
MAX_SIGNAL_MULT = 2.0

# Filters
VOL_FILTER_QUANTILE = 0.50
CONFIDENCE_FILTER_PERCENTILE = 70

# Winsorization
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Signal discretization
SIGNAL_STEP_SIZE = 0.1

# Split-sample threshold
SPLIT_SAMPLE_MIN_YEARS = 4.0


# =============================================================================
# INLINE TECHNICAL INDICATORS (no dependency on src/afml/ on homeserver)
# =============================================================================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def stochastic_rsi(close: pd.Series, rsi_period: int = 14,
                   stoch_period: int = 14) -> pd.Series:
    """Stochastic RSI."""
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min
    stoch = (rsi_val - rsi_min) / rsi_range.replace(0, np.nan) * 100
    return stoch.rename("stoch_rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD histogram, normalized by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    result = pd.DataFrame(index=close.index)
    result["macd_hist"] = histogram / close * 100
    return result


def bollinger_pct_b(close: pd.Series, window: int = 20,
                    n_std: float = 2.0) -> pd.Series:
    """Bollinger %B — position within Bollinger Bands."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    return pct_b.rename("bb_pct_b")


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range, normalized by close."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return (tr.ewm(span=period, adjust=False).mean() / close).rename(f"atr_{period}")


def swing_levels(high: pd.Series, low: pd.Series,
                 lookback: int = 20) -> pd.DataFrame:
    """Swing high/low detection — backward-looking only."""
    result = pd.DataFrame(index=high.index)
    rolling_high = high.rolling(lookback).max()
    rolling_low = low.rolling(lookback).min()
    is_swing_high = (high == rolling_high)
    is_swing_low = (low == rolling_low)
    swing_highs = high.where(is_swing_high).ffill()
    swing_lows = low.where(is_swing_low).ffill()
    mid = (high + low) / 2
    result["dist_swing_high"] = (swing_highs - mid) / mid
    result["dist_swing_low"] = (mid - swing_lows) / mid
    result["dist_nearest_level"] = pd.concat(
        [result["dist_swing_high"].abs(), result["dist_swing_low"].abs()],
        axis=1,
    ).min(axis=1)
    return result


# =============================================================================
# INLINE SIGNAL / SIZING HELPERS
# =============================================================================

def get_signal(probability, num_classes=2):
    """Convert probability to signal strength using z-score of divergence from uniform."""
    p = np.asarray(probability, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
    signal = 2.0 * norm.cdf(z) - 1.0
    return float(signal) if np.ndim(probability) == 0 else signal


def discretize_signal(signal, step_size=0.1):
    """Discretize signal to reduce noise."""
    return np.round(np.asarray(signal) / step_size) * step_size


# =============================================================================
# SAMPLE WEIGHTS — AFML Ch 4: Average Uniqueness
# =============================================================================

def compute_sample_weights(timestamps: np.ndarray, fwd_bars: int) -> np.ndarray:
    """AFML Ch 4: Average uniqueness for overlapping labels.

    OPTIMIZED: For regularly-spaced bars, concurrency is constant = fwd_bars+1
    in the interior, with lower concurrency at edges. O(n) instead of O(n²).
    """
    n = len(timestamps)
    if n == 0:
        return np.array([])
    # Interior concurrency = min(fwd_bars + 1, n)
    c = min(fwd_bars + 1, n)
    weights = np.full(n, 1.0 / c, dtype=np.float64)
    # Edge correction: first/last fwd_bars samples have lower concurrency
    for i in range(min(fwd_bars, n)):
        weights[i] = 1.0 / (i + 1)
        if n - 1 - i != i:  # avoid double-counting center
            weights[n - 1 - i] = 1.0 / (i + 1)
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights


# =============================================================================
# FULL STATISTICS COMPUTATION
# =============================================================================

def compute_full_stats(daily_pnl_series: pd.Series, starting_capital: float,
                       n_trades: int, trades_per_day: float) -> Dict[str, Any]:
    """Compute all statistics from daily PnL series.

    Args:
        daily_pnl_series: Daily PnL in dollar terms (sum of trade PnLs per day).
        starting_capital: Starting equity.
        n_trades: Total trade count.
        trades_per_day: Average trades per calendar day.

    Returns:
        Dictionary with all standard quant metrics.
    """
    daily_ret = daily_pnl_series / starting_capital
    n_days = len(daily_ret)

    if n_days < 5:
        return {"error": "insufficient_days", "n_days": n_days}

    # Core
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    sharpe = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0.0

    downside = daily_ret[daily_ret < 0].std()
    sortino = mean_r / downside * np.sqrt(365.25) if (downside is not None and downside > 0) else 0.0

    equity = (1 + daily_ret).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    calmar = (mean_r * 365.25) / max_dd if max_dd > 0 else 0.0

    total_ret = equity.iloc[-1] - 1 if len(equity) > 0 else 0.0
    years = n_days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0.0

    # Win/Loss (daily)
    win_days = (daily_ret > 0).sum()
    loss_days = (daily_ret < 0).sum()
    win_rate_days = win_days / n_days if n_days > 0 else 0.0

    winning = daily_ret[daily_ret > 0]
    losing = daily_ret[daily_ret < 0]
    avg_win = winning.mean() * 10000 if len(winning) > 0 else 0.0   # bps
    avg_loss = losing.mean() * 10000 if len(losing) > 0 else 0.0    # bps
    profit_factor = winning.sum() / abs(losing.sum()) if losing.sum() != 0 else float("inf")
    expectancy = mean_r * 10000  # bps per day

    # Risk
    ann_std = std_r * np.sqrt(365.25)
    ann_var = ann_std ** 2

    # Drawdown recovery (days from max DD to new high)
    dd_recovery = -1
    if len(drawdown) > 0:
        dd_idx = drawdown.idxmin()
        post_dd = equity.loc[dd_idx:]
        recovery_idx = post_dd[post_dd >= running_max.loc[dd_idx]].index
        if len(recovery_idx) > 0:
            try:
                dd_recovery = (recovery_idx[0] - dd_idx).days
            except (AttributeError, TypeError):
                dd_recovery = -1

    # Statistical
    skew = float(daily_ret.skew()) if len(daily_ret) > 3 else 0.0
    kurt = float(daily_ret.kurtosis() + 3) if len(daily_ret) > 4 else 3.0  # regular kurtosis

    # PSR (Probabilistic Sharpe Ratio)
    sr_unann = sharpe / np.sqrt(365.25)
    denom_inner = (1 - skew * sr_unann + ((kurt - 1) / 4) * sr_unann ** 2) / max(n_days - 1, 1)
    denom = np.sqrt(max(denom_inner, 0))
    psr = norm.cdf(sr_unann / denom) * 100 if denom > 0 else 50.0

    final_equity = (1 + total_ret) * starting_capital

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "cagr_pct": round(cagr * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_days": round(win_rate_days, 4),
        "loss_rate_days": round(1 - win_rate_days, 4),
        "profit_factor": round(min(profit_factor, 99.0), 3),
        "avg_win_bps": round(avg_win, 1),
        "avg_loss_bps": round(avg_loss, 1),
        "expectancy_bps": round(expectancy, 2),
        "ann_std": round(ann_std, 4),
        "ann_var": round(ann_var, 6),
        "dd_recovery_days": dd_recovery,
        "psr_pct": round(psr, 1),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 1),
        "n_days": n_days,
        "n_trades": n_trades,
        "trades_per_day": round(trades_per_day, 1),
        "final_equity": round(final_equity, 2),
    }


# =============================================================================
# DATA LOADING — 4 different formats
# =============================================================================

def load_binance_1m(symbol: str) -> Optional[pd.DataFrame]:
    """Load Binance 1m CSVs for a symbol.

    Path: /home/user/ml-data/binance-1m/{SYMBOL}/{SYMBOL}-1m-{YYYY-MM}.csv
    Format: CSV, no header (or sometimes with header),
            columns: open_time,open,high,low,close,volume,close_time,...
    """
    sym_dir = BINANCE_DIR / symbol
    if not sym_dir.exists():
        return None

    csvs = sorted(sym_dir.glob("*.csv"))
    if not csvs:
        return None

    dfs = []
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "count", "tbv", "tqv", "ignore"]

    for f in csvs:
        try:
            # Peek at first line to detect header
            with open(f) as fh:
                first_line = fh.readline().strip()
            first_val = first_line.split(",")[0]

            if first_val.replace(".", "").replace("-", "").replace("e", "").replace("+", "").isdigit():
                df = pd.read_csv(f, header=None, names=cols)
            else:
                df = pd.read_csv(f)
                if len(df.columns) >= 12:
                    df.columns = cols
                else:
                    # Fewer columns — try to map
                    df.columns = cols[:len(df.columns)]

            # Filter rows where open_time is not numeric (header rows in data)
            df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
            df = df.dropna(subset=["open_time"])

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.dropna(subset=["open", "close"])
    return combined


def load_histdata_1m(pair: str) -> Optional[pd.DataFrame]:
    """Load HistData 1m CSVs for forex/commodities/indices.

    Path: /hdd/ml-data/histdata-1m/{PAIR}/*.csv
    Format: Semicolon-delimited, no header
            columns: DateTime;Open;High;Low;Close;Volume
            DateTime format: 20190101 170000
    """
    pair_dir = HISTDATA_DIR / pair
    if not pair_dir.exists():
        return None

    csvs = sorted(pair_dir.glob("*.csv"))
    if not csvs:
        return None

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(
                f, sep=";", header=None,
                names=["datetime", "open", "high", "low", "close", "volume"],
                dtype={"datetime": str},
            )
            # Parse datetime: "20190101 170000" format
            df["timestamp"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S", utc=True)
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.dropna(subset=["open", "close"])
    return combined


def load_yahoo(asset_class: str, symbol: str) -> Optional[pd.DataFrame]:
    """Load Yahoo CSV data for stocks/commodities/indices.

    Path: /hdd/ml-data/non-crypto/{class}/{SYMBOL}/hourly.csv or 5min_60d.csv
    Format: Yahoo Finance CSVs often have a 2-row multi-level header:
        Row 0: Price,Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
        Row 1: Datetime,NaN,NaN,NaN,NaN,NaN,NaN
        Row 2+: actual data
    Tries hourly first (longest history), then 5min, then 1min.
    """
    base = YAHOO_DIR / asset_class / symbol

    # Try files in order of preference (longer history first)
    for fname in ["hourly.csv", "5min_60d.csv", "1min_recent.csv"]:
        fpath = base / fname
        if not fpath.exists():
            continue
        try:
            # First, try loading with multi-level header (Yahoo Finance format)
            # Peek at the first line to detect the format
            with open(fpath, "r") as fh:
                first_line = fh.readline().strip()

            first_fields = first_line.split(",")
            is_yahoo_multilevel = (
                len(first_fields) >= 2
                and first_fields[0].strip().lower() in ("price", "")
                and any(f.strip().lower() in ("ticker", "") for f in first_fields[1:3])
            )

            if is_yahoo_multilevel:
                # Yahoo multi-level header: skip first 2 rows, assign columns manually
                df = pd.read_csv(fpath, skiprows=[0, 1])
                # After skipping 2 header rows, first column is datetime, rest are OHLCV
                if len(df.columns) >= 6:
                    df.columns = ["datetime", "close", "high", "low", "open", "volume"][:len(df.columns)]
                elif len(df.columns) >= 5:
                    df.columns = ["datetime", "close", "high", "low", "open"][:len(df.columns)]
                else:
                    continue

                df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                df = df.dropna(subset=["datetime"])
                for col in ["close", "high", "low", "open", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.set_index("datetime")
                if "volume" not in df.columns:
                    df["volume"] = 0.0
            else:
                # Standard CSV format
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)

                # Handle multi-level headers: flatten if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

                # Normalize column names
                col_map = {}
                for c in df.columns:
                    cl = str(c).lower().strip()
                    if cl in ("open", "high", "low", "close", "volume", "adj close"):
                        col_map[c] = cl
                df = df.rename(columns=col_map)

            needed = ["open", "high", "low", "close"]
            if not all(c in df.columns for c in needed):
                continue

            df = df[needed + (["volume"] if "volume" in df.columns else [])]
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Ensure timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            if "volume" not in df.columns:
                df["volume"] = 0.0

            df = df.dropna(subset=["open", "close"])
            if len(df) > 100:
                return df
        except Exception:
            continue

    return None


def load_mexc_1m(symbol: str) -> Optional[pd.DataFrame]:
    """Load MEXC futures 1m data.

    Path: /hdd/ml-data/mexc-futures-1m/{SYMBOL}/all_1m.csv
    Format: CSV with header: timestamp,open,high,low,close,volume
    """
    fpath = MEXC_DIR / symbol / "all_1m.csv"
    if not fpath.exists():
        return None

    try:
        df = pd.read_csv(fpath)
        if "timestamp" in df.columns:
            # Could be ms epoch or ISO string
            ts = df["timestamp"]
            if pd.to_numeric(ts, errors="coerce").notna().all():
                df["timestamp"] = pd.to_datetime(pd.to_numeric(ts), unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(ts, utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.dropna(subset=["open", "close"])
        return df if len(df) > 100 else None
    except Exception:
        return None


def load_data_for_asset(asset_name: str, asset_class: str,
                        symbol: str) -> Optional[pd.DataFrame]:
    """Unified data loading dispatcher. Returns 1m OHLCV or native resolution.

    Returns the finest resolution available for the asset.
    """
    if asset_class in ("crypto_core", "crypto_unseen"):
        # Try Binance first (most data)
        data = load_binance_1m(symbol)
        if data is not None:
            return data
        # Fallback to MEXC
        data = load_mexc_1m(symbol)
        return data

    elif asset_class == "forex":
        return load_histdata_1m(symbol)

    elif asset_class == "commodities":
        # Try HistData first (1m, long history)
        data = load_histdata_1m(symbol)
        if data is not None:
            return data
        # Fallback to Yahoo
        return load_yahoo("commodities", asset_name)

    elif asset_class == "indices":
        # Try HistData first
        data = load_histdata_1m(symbol)
        if data is not None:
            return data
        return load_yahoo("indices", asset_name)

    elif asset_class == "stocks":
        return load_yahoo("stocks", asset_name)

    return None


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample OHLCV data to the target timeframe.

    If the data's native resolution is coarser than the target, return None
    (we don't upsample). Otherwise resample to the target.
    """
    if df is None or len(df) < 50:
        return None

    # Infer native resolution from median gap between consecutive bars
    gaps = df.index.to_series().diff().dropna()
    if len(gaps) == 0:
        return None
    median_gap = gaps.median()

    # Parse target timeframe to timedelta for comparison
    tf_map = {
        "1min": pd.Timedelta(minutes=1),
        "5min": pd.Timedelta(minutes=5),
        "15min": pd.Timedelta(minutes=15),
        "30min": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
    }
    target_td = tf_map.get(tf)
    if target_td is None:
        return None

    # Don't upsample: if native resolution is coarser than target, skip
    if median_gap > target_td * 1.5:
        return None

    # If native resolution matches target (within tolerance), return as-is
    if abs(median_gap - target_td) < target_td * 0.3:
        return df

    # Resample
    try:
        resampled = df.resample(tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open", "close"])
        return resampled if len(resampled) > 50 else None
    except Exception:
        return None


# =============================================================================
# FEATURE COMPUTATION — 14 universal OHLCV features
# =============================================================================

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all 14 universal features from OHLCV data.

    No asset-specific calibration. No HL derivatives. Pure OHLCV.
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    f = pd.DataFrame(index=ohlcv.index)

    # Returns
    for lb in [1, 3, 7, 14]:
        f[f"ret_{lb}"] = close.pct_change(lb)

    # Bar range
    f["bar_range"] = (high - low) / close.replace(0, np.nan)

    # Volume z-score (handle zero/constant volume — common in forex HistData)
    vol_mean = volume.rolling(30, min_periods=10).mean()
    vol_std = volume.rolling(30, min_periods=10).std()
    if vol_std.max() > 0:
        f["vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    else:
        f["vol_zscore"] = 0.0  # constant volume, no information

    # Technical indicators
    f["rsi_14"] = rsi(close, 14)
    f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    f["macd_hist"] = macd(close)["macd_hist"]
    f["bb_pct_b"] = bollinger_pct_b(close, 20)
    f["atr_14"] = atr(high, low, close, 14)

    # Swing levels
    sw = swing_levels(high, low, lookback=20)
    f["dist_swing_high"] = sw["dist_swing_high"]
    f["dist_swing_low"] = sw["dist_swing_low"]
    f["dist_nearest_level"] = sw["dist_nearest_level"]

    # Downcast to float32 where precision doesn't matter
    for col in f.columns:
        f[col] = f[col].astype(np.float32)

    return f


# =============================================================================
# WALK-FORWARD SIGNAL GENERATION
# =============================================================================

def generate_walk_forward_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
) -> pd.DataFrame:
    """Walk-forward monthly retrain with expanding window.

    - Monthly retrain (expanding window)
    - Embargo = fwd_bars gap
    - Sample weights (AFML Ch4 uniqueness)
    - Vol filter top 50%, confidence filter top 30%
    - LogReg L1 C=0.1 with StandardScaler

    Returns DataFrame with columns: timestamp, direction, fwd_return,
    confidence, probability, signal_strength.
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for vol filter
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Determine month range from the actual data (not hardcoded)
    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    # Need at least 6 months of data to start walk-forward
    if (end_date - start_date).days < 180:
        return pd.DataFrame()

    # Generate monthly periods from actual data range
    # Start testing from month 7 onward (6 months minimum training)
    # Use .to_period('M') to ensure proper Period objects for pd.period_range
    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame()

    # OPTIMIZATION: For fine timeframes, retrain quarterly (3x fewer model fits)
    # Still TEST on all months — just reuse the model trained at the quarter boundary
    retrain_every = 1  # default: every month
    if len(ohlcv) > 500_000:  # ~1m data with >1 year
        retrain_every = 3
    elif len(ohlcv) > 200_000:  # ~5m data with >1 year
        retrain_every = 2

    # Compute embargo timedelta based on bar timeframe and fwd_bars
    # We approximate: the embargo is fwd_bars * bar_duration
    if len(ohlcv) > 1:
        median_gap = ohlcv.index.to_series().diff().dropna().median()
        embargo_td = median_gap * fwd_bars
    else:
        embargo_td = pd.Timedelta(hours=2)

    rows = []
    cached_model = None
    cached_scaler = None
    last_retrain_idx = -999

    for month_idx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        # Ensure timezone compatibility
        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)
            embargo_cutoff = ms - embargo_td
        else:
            embargo_cutoff = ms - embargo_td

        # Train cutoff with embargo
        train_cutoff = embargo_cutoff

        # Select features available up to train_cutoff
        feat_cols = [c for c in FEATURES if c in features.columns]
        if len(feat_cols) < 8:
            continue  # Need at least 8 features

        feat = features[feat_cols].copy()

        # Per-window winsorization of returns using training-period quantiles
        train_mask = feat.index < train_cutoff
        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                train_vals = feat.loc[train_mask, col]
                if len(train_vals.dropna()) > 10:
                    lo = train_vals.quantile(WINSOR_LO)
                    hi = train_vals.quantile(WINSOR_HI)
                    feat[col] = feat[col].clip(lo, hi)

        # Clean: drop rows with any NaN feature
        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        # Train/test split
        tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 200 or len(te) < 10:
            continue

        # Vol filter: top 50% volatility bars in test set
        if vol is not None and len(vol) > 0:
            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 20:
                continue
            vt = tv.quantile(VOL_FILTER_QUANTILE)
            vol_aligned = vol.reindex(te)
            hv = (vol_aligned > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
        else:
            continue

        if len(te) < 5:
            continue

        X_train = fc.loc[tr]
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        # OPTIMIZATION: Only retrain every retrain_every months
        need_retrain = (month_idx - last_retrain_idx) >= retrain_every or cached_model is None

        if need_retrain:
            # Sample weights (AFML Ch4)
            train_timestamps = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars)

        # Fit LogReg L1 (or reuse cached model for intermediate months)
        try:
            if need_retrain:
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_train)
                model = LogisticRegression(
                    C=MODEL_C, penalty=MODEL_PENALTY, solver=MODEL_SOLVER,
                    max_iter=MODEL_MAX_ITER, random_state=MODEL_SEED,
                )
                model.fit(X_tr_scaled, y_train, sample_weight=sample_w)
                cached_model = model
                cached_scaler = scaler
                last_retrain_idx = month_idx
            else:
                scaler = cached_scaler
                model = cached_model

            X_te_scaled = scaler.transform(X_test)
            yp = model.predict_proba(X_te_scaled)[:, 1]
        except Exception:
            continue

        # Confidence filter (top 30%)
        conf = np.abs(yp - 0.5)
        if len(conf) > 5:
            conf_threshold = np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
            conf_mask = conf >= conf_threshold
        else:
            conf_mask = np.ones(len(conf), dtype=bool)

        yp_masked = yp[conf_mask]
        conf_masked = conf[conf_mask]
        te_masked = te[conf_mask]
        fwd_ret_masked = fwd_ret.reindex(te_masked)

        if len(yp_masked) == 0:
            continue

        directions = np.where(yp_masked > 0.5, 1.0, -1.0)
        signals = get_signal(yp_masked, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        # VECTORIZED signal construction (replaces per-signal Python loop)
        fr_vals = fwd_ret_masked.values if hasattr(fwd_ret_masked, "values") else np.array(fwd_ret_masked)
        finite_mask = np.isfinite(fr_vals)
        if finite_mask.any():
            sig_str = signals_disc if np.ndim(signals_disc) > 0 else np.full(len(yp_masked), float(signals_disc))
            batch = pd.DataFrame({
                "timestamp": te_masked[finite_mask],
                "direction": directions[finite_mask],
                "fwd_return": fr_vals[finite_mask],
                "confidence": conf_masked[finite_mask],
                "probability": yp_masked[finite_mask],
                "signal_strength": np.abs(sig_str[finite_mask]),
            })
            rows.append(batch)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name
    return df


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame) -> Optional[Dict]:
    """Portfolio simulation with shared capital pool.

    Same logic as validation_honest.py: probability-based sizing, exposure limits,
    dynamic deleveraging, DD halt.
    """
    if signals_df is None or len(signals_df) < 10:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values if "asset" in sub.columns else np.full(n, "SINGLE")
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = (
        sub["signal_strength"].values.astype(np.float64)
        if "signal_strength" in sub.columns else None
    )

    equity_arr = np.empty(n)
    pnl_arr = np.zeros(n)
    dd_arr = np.empty(n)
    size_arr = np.zeros(n)
    skipped = np.ones(n, dtype=bool)

    equity = STARTING_CAPITAL
    peak_equity = equity
    halted = False

    current_exposure = 0.0
    asset_exposure: dict = {}

    use_prob = signal_strengths is not None

    for i in range(n):
        if halted:
            equity_arr[i] = equity
            dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        if dd >= MAX_DD:
            halted = True
            equity_arr[i] = equity
            dd_arr[i] = dd
            continue

        # Dynamic deleveraging
        if dd > DELEVER_START:
            dd_ratio = (dd - DELEVER_START) / (MAX_DD - DELEVER_START)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        # Position sizing
        if use_prob:
            sig_abs = abs(signal_strengths[i])
            raw_size = equity * BASE_SIZING_PCT * sig_abs * size_mult
            raw_size = min(raw_size, equity * BASE_SIZING_PCT * MAX_SIGNAL_MULT * size_mult)
        else:
            raw_size = equity * BASE_SIZING_PCT * size_mult

        # Exposure limits
        asset = assets_arr[i]
        asset_exp = asset_exposure.get(asset, 0.0)

        if asset_exp + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - asset_exp, 0.0)

        if current_exposure + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - current_exposure, 0.0)

        if raw_size < 1.0:
            equity_arr[i] = equity
            dd_arr[i] = dd
            continue

        # Execute trade
        trade_return = directions[i] * fwd_returns[i] - SLIPPAGE
        pnl = raw_size * trade_return

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        current_exposure = min(
            current_exposure + raw_size * 0.1, equity * MAX_EXPOSURE
        )
        if i > 0 and timestamps[i] != timestamps[i - 1]:
            current_exposure *= 0.9
        asset_exposure[asset] = asset_exposure.get(asset, 0.0) * 0.9

        equity_arr[i] = equity
        pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        size_arr[i] = raw_size
        skipped[i] = False

    return {
        "equity": equity_arr,
        "pnl": pnl_arr,
        "dd": dd_arr,
        "sizes": size_arr,
        "skipped": skipped,
        "timestamps": timestamps,
        "halted": halted,
        "final_equity": equity,
        "n_trades": int((~skipped).sum()),
    }


def compute_daily_pnl_series(sim: Dict) -> Optional[pd.Series]:
    """Extract daily PnL series from simulation output."""
    if sim is None or sim["n_trades"] < 10:
        return None

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
    timestamps = sim["timestamps"][active]

    ts_series = pd.Series(pnls, index=pd.to_datetime(timestamps))
    daily = ts_series.resample("1D").sum()

    # Filter to only days with activity
    t0 = pd.Timestamp(timestamps[0])
    daily = daily[daily.index >= t0]
    daily_nonzero = daily[daily != 0]

    return daily_nonzero if len(daily_nonzero) >= 5 else None


# =============================================================================
# PIPELINE FOR A SINGLE TIMEFRAME (used within run_asset_group)
# =============================================================================

def _run_single_tf_pipeline(
    ohlcv_resampled: pd.DataFrame,
    bar_tf: str,
    fwd_bars: int,
    label: str,
    asset_name: str,
    asset_class: str,
    symbol: str,
    combo_key: str,
) -> Dict:
    """Run the full ML pipeline for one (asset, timeframe) combo.

    Expects already-resampled OHLCV data. Returns a result dict.
    This is the core logic extracted from run_single_combo, minus data loading.
    """
    t_start = time.time()

    try:
        ohlcv = ohlcv_resampled

        # OPTIMIZATION: For 1m timeframe with very long history, limit to last 18 months
        # to avoid extremely slow walk-forward (3M+ rows). 18 months still provides
        # 12 months OOS with 6 months initial training.
        if bar_tf == "1min" and len(ohlcv) > 800_000:
            cutoff = ohlcv.index.max() - pd.DateOffset(months=18)
            ohlcv = ohlcv.loc[ohlcv.index >= cutoff]

        if ohlcv is None or len(ohlcv) < 200:
            return {"combo": combo_key, "status": "insufficient_data_after_resample",
                    "asset": asset_name, "asset_class": asset_class,
                    "timeframe": label, "raw_bars": 0}

        # Compute features
        features = compute_features(ohlcv)

        # Walk-forward signal generation
        signals = generate_walk_forward_signals(ohlcv, features, fwd_bars, asset_name)
        if signals is None or len(signals) < 20:
            elapsed = time.time() - t_start
            return {"combo": combo_key, "status": "insufficient_signals",
                    "asset": asset_name, "asset_class": asset_class,
                    "timeframe": label, "n_bars": len(ohlcv),
                    "n_signals": len(signals) if signals is not None else 0,
                    "elapsed_s": round(elapsed, 1)}

        # Portfolio simulation
        sim = simulate_portfolio(signals)
        if sim is None or sim["n_trades"] < 20:
            elapsed = time.time() - t_start
            return {"combo": combo_key, "status": "insufficient_trades",
                    "asset": asset_name, "asset_class": asset_class,
                    "timeframe": label, "n_signals": len(signals),
                    "n_trades": sim["n_trades"] if sim else 0,
                    "elapsed_s": round(elapsed, 1)}

        # Compute daily PnL and full stats
        daily_pnl = compute_daily_pnl_series(sim)
        if daily_pnl is None:
            elapsed = time.time() - t_start
            return {"combo": combo_key, "status": "insufficient_daily_data",
                    "asset": asset_name, "asset_class": asset_class,
                    "timeframe": label, "n_trades": sim["n_trades"],
                    "elapsed_s": round(elapsed, 1)}

        active = ~sim["skipped"]
        t0 = pd.Timestamp(sim["timestamps"][active][0])
        t1 = pd.Timestamp(sim["timestamps"][active][-1])
        days = max((t1 - t0).total_seconds() / 86400, 1)
        trades_per_day = sim["n_trades"] / days

        stats = compute_full_stats(daily_pnl, STARTING_CAPITAL,
                                   sim["n_trades"], trades_per_day)

        # Split-sample analysis (if >4 years of data)
        data_years = (ohlcv.index[-1] - ohlcv.index[0]).total_seconds() / (365.25 * 86400)
        split_sample = None

        if data_years >= SPLIT_SAMPLE_MIN_YEARS and len(signals) > 50:
            midpoint = signals["timestamp"].median()

            first_half = signals[signals["timestamp"] < midpoint]
            second_half = signals[signals["timestamp"] >= midpoint]

            ss_results = {}
            for half_name, half_sigs in [("first_half", first_half),
                                          ("second_half", second_half)]:
                if len(half_sigs) >= 20:
                    h_sim = simulate_portfolio(half_sigs)
                    if h_sim and h_sim["n_trades"] >= 10:
                        h_daily = compute_daily_pnl_series(h_sim)
                        if h_daily is not None:
                            h_active = ~h_sim["skipped"]
                            h_t0 = pd.Timestamp(h_sim["timestamps"][h_active][0])
                            h_t1 = pd.Timestamp(h_sim["timestamps"][h_active][-1])
                            h_days = max((h_t1 - h_t0).total_seconds() / 86400, 1)
                            h_tpd = h_sim["n_trades"] / h_days
                            h_stats = compute_full_stats(
                                h_daily, STARTING_CAPITAL,
                                h_sim["n_trades"], h_tpd,
                            )
                            ss_results[half_name] = {
                                "sharpe": h_stats.get("sharpe", 0),
                                "total_return_pct": h_stats.get("total_return_pct", 0),
                                "n_trades": h_sim["n_trades"],
                                "n_days": h_stats.get("n_days", 0),
                            }

            if "first_half" in ss_results and "second_half" in ss_results:
                sh1 = ss_results["first_half"]["sharpe"]
                sh2 = ss_results["second_half"]["sharpe"]
                decay = sh2 / sh1 if sh1 != 0 else 0.0
                ss_results["decay_ratio"] = round(decay, 3)
                split_sample = ss_results

        elapsed = time.time() - t_start

        result = {
            "combo": combo_key,
            "status": "success",
            "asset": asset_name,
            "asset_class": asset_class,
            "symbol": symbol,
            "timeframe": label,
            "bar_tf": bar_tf,
            "fwd_bars": fwd_bars,
            "n_bars": len(ohlcv),
            "data_years": round(data_years, 2),
            "n_signals": len(signals),
            "elapsed_s": round(elapsed, 1),
            "stats": stats,
        }

        if split_sample is not None:
            result["split_sample"] = split_sample

        # Free memory from this timeframe
        del features, signals, sim, daily_pnl
        # ohlcv is freed by caller

        return result

    except Exception as e:
        elapsed = time.time() - t_start
        return {
            "combo": combo_key,
            "status": "error",
            "asset": asset_name,
            "asset_class": asset_class,
            "timeframe": label,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(elapsed, 1),
        }


# =============================================================================
# PER-ASSET GROUP RUNNER — loads data ONCE, runs all timeframes
# =============================================================================

def run_asset_group(
    asset_name: str,
    asset_class: str,
    symbol: str,
    tf_list: List[Dict],
    skip_combos: Optional[set] = None,
) -> List[Dict]:
    """Load data once for an asset, run all timeframes sequentially.

    Args:
        asset_name: Human-readable name (e.g., "BTC").
        asset_class: Category (e.g., "crypto_core").
        symbol: Trading symbol (e.g., "BTCUSDT").
        tf_list: List of timeframe config dicts.
        skip_combos: Set of combo keys to skip (already completed).

    Returns:
        List of result dicts (one per timeframe).
    """
    results = []
    t0 = time.time()

    if skip_combos is None:
        skip_combos = set()

    # Check if ALL combos for this asset are already done
    all_skipped = all(
        f"{asset_name}_{tf['label']}" in skip_combos for tf in tf_list
    )
    if all_skipped:
        for tf in tf_list:
            combo_key = f"{asset_name}_{tf['label']}"
            results.append({
                "combo": combo_key, "status": "resumed_skip",
                "asset": asset_name, "asset_class": asset_class,
                "timeframe": tf["label"],
            })
        return results

    # Load raw data ONCE
    raw = load_data_for_asset(asset_name, asset_class, symbol)
    if raw is None or len(raw) < 200:
        for tf in tf_list:
            combo_key = f"{asset_name}_{tf['label']}"
            if combo_key in skip_combos:
                results.append({
                    "combo": combo_key, "status": "resumed_skip",
                    "asset": asset_name, "asset_class": asset_class,
                    "timeframe": tf["label"],
                })
            else:
                results.append({
                    "combo": combo_key, "status": "no_data",
                    "asset": asset_name, "asset_class": asset_class,
                    "timeframe": tf["label"],
                })
        return results

    load_time = time.time() - t0
    raw_rows = len(raw)

    # Sort timeframes from coarsest to finest (4h first, 1m last)
    # Coarser timeframes are faster, so we get early results
    tf_order = {"4h": 0, "1h": 1, "30min": 2, "15min": 3, "5min": 4, "1min": 5}
    tf_sorted = sorted(tf_list, key=lambda t: tf_order.get(t["bar_tf"], 99))

    # Run each timeframe
    for tf in tf_sorted:
        bar_tf = tf["bar_tf"]
        fwd_bars = tf["fwd_bars"]
        label = tf["label"]
        combo_key = f"{asset_name}_{label}"

        # Skip if already completed
        if combo_key in skip_combos:
            results.append({
                "combo": combo_key, "status": "resumed_skip",
                "asset": asset_name, "asset_class": asset_class,
                "timeframe": label,
            })
            continue

        t1 = time.time()

        # Resample to target timeframe
        ohlcv = resample_ohlcv(raw, bar_tf)
        if ohlcv is None or len(ohlcv) < 200:
            elapsed_tf = time.time() - t1
            results.append({
                "combo": combo_key, "status": "insufficient_data_after_resample",
                "asset": asset_name, "asset_class": asset_class,
                "timeframe": label, "raw_bars": raw_rows,
                "elapsed_s": round(elapsed_tf, 1),
            })
            continue

        # Run the full pipeline on resampled data
        result = _run_single_tf_pipeline(
            ohlcv, bar_tf, fwd_bars, label,
            asset_name, asset_class, symbol, combo_key,
        )

        # Inject load_time into first result for this asset
        if len(results) == 0 or all(r.get("status") == "resumed_skip" for r in results):
            result["load_time_s"] = round(load_time, 1)
            result["raw_rows"] = raw_rows

        results.append(result)

        # Free resampled data
        del ohlcv

    # Free raw data and collect garbage
    del raw
    gc.collect()

    return results


# =============================================================================
# DSR COMPUTATION (overall, corrected for multiple testing)
# =============================================================================

def compute_dsr(sharpe_ratios: List[float], n_days_list: List[int]) -> Dict:
    """Compute Deflated Sharpe Ratio for the best combo given K trials.

    Args:
        sharpe_ratios: Annualized Sharpe for each combo.
        n_days_list: Number of daily observations for each combo.

    Returns:
        Dict with DSR, best combo info, and expected max Sharpe under null.
    """
    if not sharpe_ratios:
        return {"dsr_pct": 0, "k_trials": 0}

    K = len(sharpe_ratios)
    best_idx = int(np.argmax(sharpe_ratios))
    best_sharpe = sharpe_ratios[best_idx]
    best_n = n_days_list[best_idx]

    # Un-annualize
    sr_unann = best_sharpe / np.sqrt(365.25)

    # Variance of Sharpe across trials
    sr_var = np.var([s / np.sqrt(365.25) for s in sharpe_ratios])
    if sr_var == 0:
        sr_var = 0.01

    # Expected max Sharpe under null (Bailey & Lopez de Prado)
    gamma_c = 0.5772  # Euler-Mascheroni
    if K > 1:
        ppf_k = norm.ppf(1 - 1.0 / K) if K > 1 else 0
        ppf_ke = norm.ppf(1 - 1.0 / (K * np.e)) if K * np.e > 1 else 0
        sr_0 = np.sqrt(sr_var) * ((1 - gamma_c) * ppf_k + gamma_c * ppf_ke)
    else:
        sr_0 = 0

    # PSR denominator for the best combo
    # Use estimated skew/kurt from all daily returns if possible
    sigma_sr = np.sqrt(1.0 / max(best_n - 1, 1))

    dsr = norm.cdf((sr_unann - sr_0) / sigma_sr) * 100 if sigma_sr > 0 else 50.0

    return {
        "dsr_pct": round(dsr, 1),
        "k_trials": K,
        "best_sharpe": round(best_sharpe, 3),
        "best_idx": best_idx,
        "sr_0_unann": round(sr_0, 4),
        "sr_var": round(sr_var, 6),
    }


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(results: List[Dict], total_time: float):
    """Print comprehensive summary tables."""
    successful = [r for r in results if r.get("status") == "success" and "stats" in r]
    failed = [r for r in results if r.get("status") not in ("success", "resumed_skip")]
    skipped = [r for r in results if r.get("status") == "resumed_skip"]

    print("\n" + "=" * 120, flush=True)
    print("UNIVERSAL SIGNAL TEST v2 — RESULTS SUMMARY", flush=True)
    print("=" * 120, flush=True)
    print(f"Total combos attempted: {len(results)}", flush=True)
    print(f"Successful:             {len(successful)}", flush=True)
    print(f"Failed/skipped:         {len(failed)}", flush=True)
    print(f"Resumed (pre-existing): {len(skipped)}", flush=True)
    print(f"Total runtime:          {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)

    if not successful:
        print("\nNo successful combos to report.", flush=True)
        return

    # Extract Sharpes for distribution analysis
    sharpes = [r["stats"]["sharpe"] for r in successful]
    n_days_all = [r["stats"]["n_days"] for r in successful]

    print(f"\n--- SHARPE DISTRIBUTION ---", flush=True)
    print(f"Mean Sharpe:    {np.mean(sharpes):.3f}", flush=True)
    print(f"Median Sharpe:  {np.median(sharpes):.3f}", flush=True)
    print(f"Std Sharpe:     {np.std(sharpes):.3f}", flush=True)
    print(f"Min Sharpe:     {np.min(sharpes):.3f}", flush=True)
    print(f"Max Sharpe:     {np.max(sharpes):.3f}", flush=True)
    print(f"% Sharpe > 0:   {sum(1 for s in sharpes if s > 0) / len(sharpes) * 100:.1f}%", flush=True)
    print(f"% Sharpe > 0.5: {sum(1 for s in sharpes if s > 0.5) / len(sharpes) * 100:.1f}%", flush=True)
    print(f"% Sharpe > 1.0: {sum(1 for s in sharpes if s > 1.0) / len(sharpes) * 100:.1f}%", flush=True)
    print(f"% Sharpe > 1.5: {sum(1 for s in sharpes if s > 1.5) / len(sharpes) * 100:.1f}%", flush=True)
    print(f"% Sharpe > 2.0: {sum(1 for s in sharpes if s > 2.0) / len(sharpes) * 100:.1f}%", flush=True)

    # DSR
    dsr_result = compute_dsr(sharpes, n_days_all)
    print(f"\n--- DEFLATED SHARPE RATIO ---", flush=True)
    print(f"K trials:           {dsr_result['k_trials']}", flush=True)
    print(f"Best Sharpe:        {dsr_result['best_sharpe']:.3f}", flush=True)
    print(f"Expected max (null):{dsr_result['sr_0_unann']:.4f} (unannualized)", flush=True)
    print(f"DSR:                {dsr_result['dsr_pct']:.1f}%", flush=True)

    # Ranked table — top 30
    ranked = sorted(successful, key=lambda r: r["stats"]["sharpe"], reverse=True)

    print(f"\n--- TOP 30 COMBOS (ranked by Sharpe) ---", flush=True)
    header = (
        f"{'Rank':>4s}  {'Asset':<8s}  {'Class':<14s}  {'Timeframe':<16s}  "
        f"{'Sharpe':>7s}  {'Sortino':>7s}  {'Return%':>8s}  {'MaxDD%':>6s}  "
        f"{'WinRate':>7s}  {'PF':>5s}  {'Trades':>7s}  {'Days':>5s}  "
        f"{'PSR%':>5s}  {'E[bps]':>7s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for i, r in enumerate(ranked[:30]):
        s = r["stats"]
        print(
            f"{i+1:>4d}  {r['asset']:<8s}  {r['asset_class']:<14s}  {r['timeframe']:<16s}  "
            f"{s['sharpe']:>7.3f}  {s['sortino']:>7.3f}  {s['total_return_pct']:>+8.2f}  "
            f"{s['max_dd_pct']:>6.2f}  {s['win_rate_days']:>7.4f}  "
            f"{s['profit_factor']:>5.2f}  {s['n_trades']:>7d}  {s['n_days']:>5d}  "
            f"{s['psr_pct']:>5.1f}  {s['expectancy_bps']:>7.2f}",
            flush=True,
        )

    # Bottom 10
    if len(ranked) > 10:
        print(f"\n--- BOTTOM 10 COMBOS ---", flush=True)
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for i, r in enumerate(ranked[-10:]):
            rank = len(ranked) - 10 + i + 1
            s = r["stats"]
            print(
                f"{rank:>4d}  {r['asset']:<8s}  {r['asset_class']:<14s}  {r['timeframe']:<16s}  "
                f"{s['sharpe']:>7.3f}  {s['sortino']:>7.3f}  {s['total_return_pct']:>+8.2f}  "
                f"{s['max_dd_pct']:>6.2f}  {s['win_rate_days']:>7.4f}  "
                f"{s['profit_factor']:>5.2f}  {s['n_trades']:>7d}  {s['n_days']:>5d}  "
                f"{s['psr_pct']:>5.1f}  {s['expectancy_bps']:>7.2f}",
                flush=True,
            )

    # Heatmap: asset class x timeframe -> mean Sharpe
    print(f"\n--- HEATMAP: Asset Class x Timeframe (Mean Sharpe) ---", flush=True)
    heatmap = {}
    for r in successful:
        ac = r["asset_class"]
        tf = r["timeframe"]
        heatmap.setdefault(ac, {}).setdefault(tf, []).append(r["stats"]["sharpe"])

    # Collect all timeframes seen
    all_tfs = sorted(set(r["timeframe"] for r in successful),
                     key=lambda t: next((i for i, tc in enumerate(TIMEFRAMES)
                                         if tc["label"] == t), 99))

    tf_header = f"{'Class':<16s}" + "".join(f"  {tf:>16s}" for tf in all_tfs)
    print(tf_header, flush=True)
    print("-" * len(tf_header), flush=True)

    for ac in sorted(heatmap.keys()):
        row = f"{ac:<16s}"
        for tf in all_tfs:
            vals = heatmap[ac].get(tf, [])
            if vals:
                mean_s = np.mean(vals)
                row += f"  {mean_s:>16.3f}"
            else:
                row += f"  {'---':>16s}"
        print(row, flush=True)

    # Per asset-class summary
    print(f"\n--- PER ASSET CLASS SUMMARY ---", flush=True)
    class_summary = {}
    for r in successful:
        ac = r["asset_class"]
        class_summary.setdefault(ac, []).append(r["stats"]["sharpe"])

    print(f"{'Class':<16s}  {'N':>4s}  {'Mean':>7s}  {'Med':>7s}  {'Std':>7s}  {'Best':>7s}  {'>0':>5s}  {'>1':>5s}", flush=True)
    print("-" * 72, flush=True)
    for ac in sorted(class_summary.keys()):
        vals = class_summary[ac]
        n = len(vals)
        print(
            f"{ac:<16s}  {n:>4d}  {np.mean(vals):>7.3f}  {np.median(vals):>7.3f}  "
            f"{np.std(vals):>7.3f}  {np.max(vals):>7.3f}  "
            f"{sum(1 for v in vals if v > 0)/n*100:>4.0f}%  "
            f"{sum(1 for v in vals if v > 1.0)/n*100:>4.0f}%",
            flush=True,
        )

    # Per timeframe summary
    print(f"\n--- PER TIMEFRAME SUMMARY ---", flush=True)
    tf_summary = {}
    for r in successful:
        tf = r["timeframe"]
        tf_summary.setdefault(tf, []).append(r["stats"]["sharpe"])

    print(f"{'Timeframe':<16s}  {'N':>4s}  {'Mean':>7s}  {'Med':>7s}  {'Std':>7s}  {'Best':>7s}  {'>0':>5s}  {'>1':>5s}", flush=True)
    print("-" * 72, flush=True)
    for tf in all_tfs:
        vals = tf_summary.get(tf, [])
        if vals:
            n = len(vals)
            print(
                f"{tf:<16s}  {n:>4d}  {np.mean(vals):>7.3f}  {np.median(vals):>7.3f}  "
                f"{np.std(vals):>7.3f}  {np.max(vals):>7.3f}  "
                f"{sum(1 for v in vals if v > 0)/n*100:>4.0f}%  "
                f"{sum(1 for v in vals if v > 1.0)/n*100:>4.0f}%",
                flush=True,
            )

    # Split-sample report
    split_results = [r for r in successful if "split_sample" in r]
    if split_results:
        print(f"\n--- SPLIT-SAMPLE ANALYSIS ({len(split_results)} combos with >4yr data) ---", flush=True)
        print(
            f"{'Asset':<8s}  {'Timeframe':<16s}  {'1st Half SR':>12s}  {'2nd Half SR':>12s}  "
            f"{'Decay Ratio':>12s}  {'Status':<10s}",
            flush=True,
        )
        print("-" * 80, flush=True)
        for r in sorted(split_results, key=lambda x: x["split_sample"].get("decay_ratio", 0), reverse=True):
            ss = r["split_sample"]
            sh1 = ss.get("first_half", {}).get("sharpe", 0)
            sh2 = ss.get("second_half", {}).get("sharpe", 0)
            decay = ss.get("decay_ratio", 0)
            status = "STABLE" if 0.5 <= decay <= 1.5 else ("IMPROVED" if decay > 1.5 else "DECAYED")
            print(
                f"{r['asset']:<8s}  {r['timeframe']:<16s}  {sh1:>12.3f}  {sh2:>12.3f}  "
                f"{decay:>12.3f}  {status:<10s}",
                flush=True,
            )

    # Failed combos summary
    if failed:
        print(f"\n--- FAILED/SKIPPED COMBOS ({len(failed)}) ---", flush=True)
        status_counts = {}
        for r in failed:
            s = r.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"  {s:<35s}: {c}", flush=True)

    print("\n" + "=" * 120, flush=True)
    print("END OF REPORT", flush=True)
    print("=" * 120, flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Universal Signal Test v2 — Multi-Asset, Multi-Timeframe Direction Prediction"
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="Process assets in reverse order (for parallel with another machine)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (0 = auto, uses all CPUs)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (overrides OUTPUT_FILE / env var)",
    )
    args = parser.parse_args()

    # Resolve output path
    output_file = Path(args.output) if args.output else OUTPUT_FILE

    # Resolve worker count
    n_workers = args.workers if args.workers > 0 else os.cpu_count()

    overall_start = time.time()

    print("=" * 120, flush=True)
    print("UNIVERSAL SIGNAL TEST v2 — Multi-Asset, Multi-Timeframe Direction Prediction", flush=True)
    print("=" * 120, flush=True)
    print(f"Started:          {datetime.now().isoformat()}", flush=True)
    print(f"Model:            LogReg L1 C={MODEL_C} (FIXED, no tuning)", flush=True)
    print(f"Features:         {len(FEATURES)} universal OHLCV features", flush=True)
    print(f"Capital:          ${STARTING_CAPITAL:,.0f}", flush=True)
    print(f"Sizing:           {BASE_SIZING_PCT*100:.0f}% base, prob-scaled", flush=True)
    print(f"Max DD halt:      {MAX_DD*100:.0f}%", flush=True)
    print(f"Slippage:         {SLIPPAGE*10000:.0f} bps", flush=True)
    print(f"Vol filter:       top {(1-VOL_FILTER_QUANTILE)*100:.0f}%", flush=True)
    print(f"Confidence filter:top {(100-CONFIDENCE_FILTER_PERCENTILE):.0f}%", flush=True)
    print(f"Workers:          {n_workers}", flush=True)
    print(f"Reverse order:    {args.reverse}", flush=True)
    print(f"Split-sample:     assets with >{SPLIT_SAMPLE_MIN_YEARS:.0f} years", flush=True)
    print(f"Output file:      {output_file}", flush=True)
    print(f"Data paths:", flush=True)
    print(f"  BINANCE_DIR:    {BINANCE_DIR}", flush=True)
    print(f"  HISTDATA_DIR:   {HISTDATA_DIR}", flush=True)
    print(f"  YAHOO_DIR:      {YAHOO_DIR}", flush=True)
    print(f"  MEXC_DIR:       {MEXC_DIR}", flush=True)
    print("=" * 120, flush=True)

    # -------------------------------------------------------------------------
    # Build asset group list (one entry per unique asset)
    # -------------------------------------------------------------------------
    asset_groups = []
    for asset_class, assets_dict in ASSETS.items():
        for asset_name, symbol in assets_dict.items():
            asset_groups.append({
                "asset_name": asset_name,
                "asset_class": asset_class,
                "symbol": symbol,
                "tf_list": list(TIMEFRAMES),  # all timeframes for each asset
            })

    # Reverse order if requested
    if args.reverse:
        asset_groups = list(reversed(asset_groups))

    total_assets = len(asset_groups)
    total_combos = total_assets * len(TIMEFRAMES)

    print(f"\nTotal assets: {total_assets}", flush=True)
    print(f"Total combos to test: {total_combos}", flush=True)
    print(f"Timeframes per asset: {len(TIMEFRAMES)}", flush=True)
    print(f"Asset classes: {list(ASSETS.keys())}", flush=True)
    print(f"Timeframes: {[t['label'] for t in TIMEFRAMES]}", flush=True)

    # Count assets per class
    for ac, ad in ASSETS.items():
        print(f"  {ac}: {list(ad.keys())}", flush=True)

    # -------------------------------------------------------------------------
    # Resume support: load existing results if output file exists
    # -------------------------------------------------------------------------
    skip_combos = set()
    resumed_results = []

    if output_file.exists():
        try:
            with open(output_file) as f:
                existing = json.load(f)
            existing_results = existing.get("results", [])
            for r in existing_results:
                combo_key = r.get("combo", "")
                status = r.get("status", "")
                if status == "success" and "stats" in r:
                    skip_combos.add(combo_key)
                    resumed_results.append(r)
            if skip_combos:
                print(f"\nResuming: found {len(skip_combos)} completed combos in {output_file}", flush=True)
                print(f"  Will skip: {sorted(skip_combos)[:10]}{'...' if len(skip_combos) > 10 else ''}", flush=True)
        except Exception as e:
            print(f"\nWarning: could not load existing results from {output_file}: {e}", flush=True)
            skip_combos = set()
            resumed_results = []

    # -------------------------------------------------------------------------
    # Run per-asset groups in parallel
    # -------------------------------------------------------------------------
    print(f"\nStarting per-asset parallel execution with {n_workers} workers...\n", flush=True)

    all_results = list(resumed_results)  # start with resumed successes
    completed_assets = 0
    completed_combos = len(skip_combos)
    asset_times = []  # for ETA tracking

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for ag in asset_groups:
            fut = pool.submit(
                run_asset_group,
                ag["asset_name"],
                ag["asset_class"],
                ag["symbol"],
                ag["tf_list"],
                skip_combos,
            )
            futures[fut] = ag

        for fut in as_completed(futures):
            ag = futures[fut]
            completed_assets += 1
            asset_start = time.time()

            try:
                group_results = fut.result()
            except Exception as e:
                # Entire asset group failed
                group_results = []
                for tf in ag["tf_list"]:
                    combo_key = f"{ag['asset_name']}_{tf['label']}"
                    group_results.append({
                        "combo": combo_key,
                        "status": "executor_error",
                        "asset": ag["asset_name"],
                        "asset_class": ag["asset_class"],
                        "timeframe": tf["label"],
                        "error": str(e),
                    })

            # Process results for this asset group
            new_results = [r for r in group_results if r.get("status") != "resumed_skip"]
            n_new = len(new_results)
            completed_combos += n_new

            # Print per-asset progress
            # Find load info from first non-skip result
            load_info = ""
            for r in group_results:
                if "raw_rows" in r:
                    load_info = f" -- loaded {r['raw_rows']:,} rows in {r.get('load_time_s', 0):.1f}s"
                    break

            print(
                f"[{completed_assets}/{total_assets} assets] "
                f"{ag['asset_name']} ({ag['asset_class']})"
                f"{load_info}",
                flush=True,
            )

            # Print per-timeframe results within this asset
            for r in group_results:
                if r.get("status") == "resumed_skip":
                    continue  # don't print skipped combos

                status = r.get("status", "unknown")
                sharpe_str = ""
                if r.get("status") == "success" and "stats" in r:
                    sharpe_str = f"Sharpe={r['stats']['sharpe']:.3f}"

                elapsed_str = f"({r.get('elapsed_s', 0):.0f}s)" if "elapsed_s" in r else ""

                print(
                    f"  {r.get('timeframe', '?'):<16s}: {sharpe_str or status}  {elapsed_str}",
                    flush=True,
                )

            # Add new results (not resumed ones which are already in all_results)
            all_results.extend(new_results)

            # ETA calculation
            asset_elapsed = time.time() - overall_start
            asset_times.append(asset_elapsed)
            remaining_assets = total_assets - completed_assets
            if completed_assets > 0:
                avg_time_per_asset = asset_elapsed / completed_assets
                eta_s = avg_time_per_asset * remaining_assets
                eta_str = f"{eta_s/60:.1f}m" if eta_s < 3600 else f"{eta_s/3600:.1f}h"
            else:
                eta_str = "?"

            print(
                f"  Total: {n_new} combos. "
                f"ETA for remaining {remaining_assets} assets: ~{eta_str}",
                flush=True,
            )
            print("", flush=True)

            # Incremental save after each asset group completes
            _save_results(all_results, output_file, overall_start, total_combos, incremental=True)

    total_time = time.time() - overall_start

    # -------------------------------------------------------------------------
    # Final summary and save
    # -------------------------------------------------------------------------
    print_summary(all_results, total_time)
    _save_results(all_results, output_file, overall_start, total_combos, incremental=False)

    print(f"\nResults saved to: {output_file}", flush=True)
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)


def _save_results(
    results: List[Dict],
    output_file: Path,
    overall_start: float,
    total_combos: int,
    incremental: bool = False,
):
    """Save results to JSON. Called incrementally and at the end."""
    # Convert numpy types for JSON serialization
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if obj is np.nan or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj

    # Clean results for serialization
    serializable_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, dict):
                clean[k] = {kk: json_safe(vv) for kk, vv in v.items()}
            else:
                clean[k] = json_safe(v)
        # Remove large arrays from output
        if "traceback" in clean and len(str(clean["traceback"])) > 500:
            clean["traceback"] = str(clean["traceback"])[:500] + "..."
        serializable_results.append(clean)

    # Compute overall DSR for the output
    successful = [r for r in serializable_results if r.get("status") == "success" and "stats" in r]
    sharpes = [r["stats"]["sharpe"] for r in successful]
    n_days_list = [r["stats"]["n_days"] for r in successful]
    dsr_result = compute_dsr(sharpes, n_days_list) if sharpes else {}

    total_time = time.time() - overall_start

    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "universal_signal_test_v2",
        "incremental": incremental,
        "config": {
            "model": f"LogReg_L1_C{MODEL_C}",
            "features": FEATURES,
            "starting_capital": STARTING_CAPITAL,
            "max_dd": MAX_DD,
            "slippage": SLIPPAGE,
            "max_exposure": MAX_EXPOSURE,
            "max_per_asset": MAX_PER_ASSET,
            "base_sizing_pct": BASE_SIZING_PCT,
            "vol_filter_quantile": VOL_FILTER_QUANTILE,
            "confidence_filter_pct": CONFIDENCE_FILTER_PERCENTILE,
            "split_sample_min_years": SPLIT_SAMPLE_MIN_YEARS,
        },
        "summary": {
            "total_combos": total_combos,
            "successful": len(successful),
            "failed": total_combos - len(successful),
            "mean_sharpe": round(np.mean(sharpes), 3) if sharpes else 0,
            "median_sharpe": round(np.median(sharpes), 3) if sharpes else 0,
            "pct_positive_sharpe": round(
                sum(1 for s in sharpes if s > 0) / len(sharpes) * 100, 1
            ) if sharpes else 0,
            "pct_above_1": round(
                sum(1 for s in sharpes if s > 1.0) / len(sharpes) * 100, 1
            ) if sharpes else 0,
            "pct_above_2": round(
                sum(1 for s in sharpes if s > 2.0) / len(sharpes) * 100, 1
            ) if sharpes else 0,
            "runtime_s": round(total_time, 1),
        },
        "dsr": dsr_result,
        "results": serializable_results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
