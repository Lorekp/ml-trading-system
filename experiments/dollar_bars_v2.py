"""
Dollar Bars v2 — Rigorous Dollar vs Time Bars Comparison.

Tests information-driven (dollar volume) sampling against fixed time bars at
3 matched frequency tiers: ~15min, ~30min, ~1hour. Key improvements over v1:

  1. ADAPTIVE threshold: rolling 30-day median daily dollar volume, shifted(1)
     to prevent lookahead. Yesterday's rolling median sets today's threshold.
  2. BAR-COUNT forward returns for both bar types (same fwd_bars). Since dollar
     bars are calibrated to the same frequency, bar-count returns cover
     approximately the same wall-clock horizon.
  3. THREE frequency tiers for robust comparison, not just one.
  4. Bar quality diagnostics: autocorrelation, kurtosis, inter-bar time stats.
  5. Same features (bar-count based RSI/MACD/etc.) for both — no time-window
     features needed since frequencies are matched.

All 6 bar types share:
  - Same walk-forward methodology (monthly expanding, embargo, sample weights)
  - Same portfolio simulation (shared capital, exposure limits, DD halt)
  - Same slippage model (1 bps)
  - Same 14 features, same LogReg L1 model
  - Percentile 70 confidence filter
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION — ALL PRE-REGISTERED, NO TUNING
# =============================================================================

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "BNB": "BNBUSDT",
    "SUI": "SUIUSDT",
    "LINK": "LINKUSDT",
}

DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")

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

# Winsorization
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Signal discretization
SIGNAL_STEP_SIZE = 0.1

# Rolling threshold window
ROLLING_THRESHOLD_DAYS = 30

# Bar configurations: (name, resample_arg, fwd_bars)
#   For time bars: resample_arg is a pandas frequency string.
#   For dollar bars: resample_arg is target_bars_per_day (int).
BAR_CONFIGS = [
    ("time_15m",  "15min",  4),   # baseline ~15m
    ("time_30m",  "30min",  4),   # baseline ~30m
    ("time_1h",   "1h",     2),   # baseline ~1h
    ("dollar_96", 96,       4),   # match 15m frequency
    ("dollar_48", 48,       4),   # match 30m frequency
    ("dollar_24", 24,       2),   # match 1h frequency
]

# Feature list (14 core features only)
FEATURES_14 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]


# =============================================================================
# INLINE TECHNICAL INDICATORS (no dependency on src/afml/)
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
    in the interior, with lower concurrency at edges. O(n) instead of O(n^2).
    """
    n = len(timestamps)
    if n == 0:
        return np.array([])
    c = min(fwd_bars + 1, n)
    weights = np.full(n, 1.0 / c, dtype=np.float64)
    for i in range(min(fwd_bars, n)):
        weights[i] = 1.0 / (i + 1)
        if n - 1 - i != i:
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
    """Compute all statistics from daily PnL series."""
    daily_ret = daily_pnl_series / starting_capital
    n_days = len(daily_ret)

    if n_days < 5:
        return {"error": "insufficient_days", "n_days": n_days}

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

    win_days = (daily_ret > 0).sum()
    win_rate_days = win_days / n_days if n_days > 0 else 0.0

    winning = daily_ret[daily_ret > 0]
    losing = daily_ret[daily_ret < 0]
    avg_win = winning.mean() * 10000 if len(winning) > 0 else 0.0
    avg_loss = losing.mean() * 10000 if len(losing) > 0 else 0.0
    profit_factor = winning.sum() / abs(losing.sum()) if losing.sum() != 0 else float("inf")
    expectancy = mean_r * 10000

    ann_std = std_r * np.sqrt(365.25)
    ann_var = ann_std ** 2

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

    skew = float(daily_ret.skew()) if len(daily_ret) > 3 else 0.0
    kurt = float(daily_ret.kurtosis() + 3) if len(daily_ret) > 4 else 3.0

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
# FEATURE COMPUTATION
# =============================================================================

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all 14 universal features from OHLCV data."""
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    f = pd.DataFrame(index=ohlcv.index)

    for lb in [1, 3, 7, 14]:
        f[f"ret_{lb}"] = close.pct_change(lb)

    f["bar_range"] = (high - low) / close.replace(0, np.nan)

    vol_mean = volume.rolling(30, min_periods=10).mean()
    vol_std = volume.rolling(30, min_periods=10).std()
    if vol_std.max() > 0:
        f["vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    else:
        f["vol_zscore"] = 0.0

    f["rsi_14"] = rsi(close, 14)
    f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    f["macd_hist"] = macd(close)["macd_hist"]
    f["bb_pct_b"] = bollinger_pct_b(close, 20)
    f["atr_14"] = atr(high, low, close, 14)

    sw = swing_levels(high, low, lookback=20)
    f["dist_swing_high"] = sw["dist_swing_high"]
    f["dist_swing_low"] = sw["dist_swing_low"]
    f["dist_nearest_level"] = sw["dist_nearest_level"]

    for col in f.columns:
        f[col] = f[col].astype(np.float32)

    return f


# =============================================================================
# DATA LOADING — Binance 1m CSVs
# =============================================================================

def load_binance_1m(symbol: str, data_dir: Path) -> pd.DataFrame:
    """Load all monthly CSV files for a symbol from Binance 1m data.

    Handles both header and no-header formats. Returns OHLCV DataFrame
    with UTC DatetimeIndex.
    """
    symbol_dir = data_dir / symbol
    pattern = f"{symbol}-1m-*.csv"
    files = sorted(symbol_dir.glob(pattern))
    if not files:
        files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {symbol_dir} or {data_dir}")

    col_names = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_vol", "taker_buy_quote_vol", "ignore",
    ]

    frames = []
    for fp in files:
        try:
            first_line = pd.read_csv(fp, nrows=1, header=None)
            first_val = first_line.iloc[0, 0]
            has_header = False
            try:
                pd.to_numeric(first_val)
            except (ValueError, TypeError):
                has_header = True

            if has_header:
                df = pd.read_csv(fp, header=0)
                df.columns = col_names[:len(df.columns)]
            else:
                df = pd.read_csv(fp, header=None, names=col_names[:12])

            frames.append(df)
        except Exception as e:
            print(f"  Warning: failed to load {fp.name}: {e}")
            continue

    if not frames:
        raise FileNotFoundError(f"No valid CSV files for {symbol}")

    combined = pd.concat(frames, ignore_index=True)

    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    combined = combined.dropna(subset=["open_time"])
    combined["timestamp"] = pd.to_datetime(combined["open_time"], unit="ms", utc=True)
    combined = combined.set_index("timestamp").sort_index()

    ohlcv = combined[["open", "high", "low", "close", "volume"]].copy()
    for col in ohlcv.columns:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
    ohlcv = ohlcv.dropna()

    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]

    print(f"  Loaded {symbol}: {len(ohlcv):,} bars, "
          f"{ohlcv.index.min().date()} to {ohlcv.index.max().date()}")
    return ohlcv


# =============================================================================
# RESAMPLING — Time Bars
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Standard OHLCV resampling to fixed time bars."""
    resampled = df.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


# =============================================================================
# DOLLAR BARS — ADAPTIVE THRESHOLD (v2)
# =============================================================================

def compute_rolling_dollar_threshold(
    ohlcv_1m: pd.DataFrame,
    target_bars_per_day: int,
    rolling_days: int = ROLLING_THRESHOLD_DAYS,
) -> pd.Series:
    """Compute daily-adaptive dollar bar threshold using rolling median.

    For each calendar day, the threshold is:
        rolling_median(daily_dollar_volume, window=rolling_days) / target_bars_per_day

    Uses shift(1) to prevent lookahead: yesterday's rolling median sets
    today's threshold.

    Args:
        ohlcv_1m: 1-minute OHLCV data.
        target_bars_per_day: Desired number of bars per day.
        rolling_days: Rolling window in calendar days for median computation.

    Returns:
        pd.Series indexed by date with the dollar threshold for each day.
    """
    # Compute dollar volume per 1m bar, then aggregate to daily
    dollar_vol_1m = ohlcv_1m["close"] * ohlcv_1m["volume"]
    daily_dollar_vol = dollar_vol_1m.resample("1D").sum()

    # Drop zero-volume days (shouldn't happen for crypto but be safe)
    daily_dollar_vol = daily_dollar_vol.replace(0, np.nan)

    # Rolling median of daily dollar volume
    rolling_median = daily_dollar_vol.rolling(
        window=rolling_days, min_periods=1
    ).median()

    # Shift(1): use yesterday's rolling median for today's threshold
    rolling_median_shifted = rolling_median.shift(1)

    # First day(s) may be NaN after shift — fill with first non-NaN value
    first_valid = rolling_median_shifted.dropna()
    if len(first_valid) > 0:
        fallback = first_valid.iloc[0]
        rolling_median_shifted = rolling_median_shifted.fillna(fallback)
    else:
        # Extreme edge case: no valid data at all — use raw median
        fallback = daily_dollar_vol.median()
        rolling_median_shifted = rolling_median_shifted.fillna(fallback)

    # Threshold = daily dollar volume / target bars per day
    threshold_per_day = rolling_median_shifted / target_bars_per_day

    # Safety: ensure no zero or NaN thresholds
    threshold_per_day = threshold_per_day.replace(0, np.nan)
    threshold_per_day = threshold_per_day.ffill().bfill()

    # Convert index to date (not datetime) for lookup
    threshold_per_day.index = threshold_per_day.index.date

    return threshold_per_day


def resample_dollar_bars_adaptive(
    ohlcv_1m: pd.DataFrame,
    threshold_per_day: pd.Series,
) -> pd.DataFrame:
    """Resample 1m OHLCV to dollar bars with daily-adaptive threshold.

    For each 1m bar, looks up the threshold for that bar's date. Accumulates
    dollar volume (close * volume) and emits a new OHLCV bar when cumulative
    exceeds the threshold.

    Args:
        ohlcv_1m: 1-minute OHLCV data with DatetimeIndex.
        threshold_per_day: Series indexed by date with dollar threshold per day.

    Returns:
        OHLCV DataFrame with irregular DatetimeIndex.
    """
    closes = ohlcv_1m["close"].values
    highs = ohlcv_1m["high"].values
    lows = ohlcv_1m["low"].values
    opens = ohlcv_1m["open"].values
    volumes = ohlcv_1m["volume"].values
    timestamps = ohlcv_1m.index

    # Pre-compute date for each 1m bar for threshold lookup
    dates = timestamps.date

    # Build a fast lookup dict from threshold_per_day
    thresh_dict = threshold_per_day.to_dict()

    # Fallback threshold: median of all thresholds
    all_thresh_vals = threshold_per_day.dropna().values
    fallback_threshold = float(np.median(all_thresh_vals)) if len(all_thresh_vals) > 0 else 1e12

    bars = []
    cum_dollar = 0.0
    bar_open = opens[0]
    bar_high = highs[0]
    bar_low = lows[0]
    bar_volume = 0.0
    bar_start_idx = 0

    for i in range(len(closes)):
        # Dollar volume for this 1m bar
        dv = closes[i] * volumes[i]
        cum_dollar += dv
        bar_high = max(bar_high, highs[i])
        bar_low = min(bar_low, lows[i])
        bar_volume += volumes[i]

        # Look up threshold for this bar's date
        bar_date = dates[i]
        threshold = thresh_dict.get(bar_date, fallback_threshold)

        if cum_dollar >= threshold:
            bars.append({
                "timestamp": timestamps[bar_start_idx],
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": closes[i],
                "volume": bar_volume,
            })
            # Reset for next bar
            cum_dollar = 0.0
            if i + 1 < len(closes):
                bar_open = opens[i + 1]
                bar_high = highs[i + 1]
                bar_low = lows[i + 1]
                bar_volume = 0.0
                bar_start_idx = i + 1

    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    result = pd.DataFrame(bars)
    result = result.set_index("timestamp")
    return result


# =============================================================================
# BAR QUALITY DIAGNOSTICS
# =============================================================================

def compute_bar_diagnostics(bars: pd.DataFrame, bar_name: str) -> Dict[str, Any]:
    """Compute quality diagnostics for a bar type.

    Reports: bar count, frequency stability, inter-bar time distribution,
    return distribution properties (skewness, kurtosis, autocorrelation).
    """
    if len(bars) < 10:
        return {"bar_name": bar_name, "n_bars": len(bars), "error": "too_few_bars"}

    returns = bars["close"].pct_change().dropna()
    inter_bar = bars.index.to_series().diff().dropna()

    # Total days spanned
    total_days = (bars.index[-1] - bars.index[0]).total_seconds() / 86400.0
    total_days = max(total_days, 1.0)

    # Bars per day: compute daily then take mean/std
    bars_daily = bars.resample("1D").size()
    bars_daily = bars_daily[bars_daily > 0]  # only days with data

    inter_bar_seconds = inter_bar.dt.total_seconds()

    return {
        "bar_name": bar_name,
        "n_bars": len(bars),
        "bars_per_day_mean": round(float(bars_daily.mean()), 1) if len(bars_daily) > 0 else 0.0,
        "bars_per_day_std": round(float(bars_daily.std()), 1) if len(bars_daily) > 1 else 0.0,
        "inter_bar_mean_min": round(float(inter_bar_seconds.mean()) / 60, 1),
        "inter_bar_std_min": round(float(inter_bar_seconds.std()) / 60, 1),
        "inter_bar_median_min": round(float(inter_bar_seconds.median()) / 60, 1),
        "return_skewness": round(float(returns.skew()), 4) if len(returns) > 3 else 0.0,
        "return_kurtosis": round(float(returns.kurtosis() + 3), 2) if len(returns) > 4 else 3.0,
        "return_autocorr_lag1": round(float(returns.autocorr(lag=1)), 4) if len(returns) > 10 else 0.0,
        "return_autocorr_lag2": round(float(returns.autocorr(lag=2)), 4) if len(returns) > 10 else 0.0,
    }


# =============================================================================
# RAW SIGNAL GENERATION — Walk-Forward (no confidence filter)
# =============================================================================

def generate_raw_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Walk-forward monthly retrain with expanding window.

    Uses bar-count forward returns for both time and dollar bars.
    Same pipeline from multi_approach_comparison.py.

    Returns DataFrame with raw signals (no confidence filter applied).
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for vol filter
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Determine date range
    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    if (end_date - start_date).days < 180:
        return pd.DataFrame()

    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame()

    # OPTIMIZATION: For fine timeframes, retrain quarterly
    retrain_every = 1
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    # Compute embargo timedelta based on bar spacing and fwd_bars
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
        train_cutoff = embargo_cutoff

        feat_cols = [c for c in feature_cols if c in features.columns]
        if len(feat_cols) < 8:
            continue

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

        need_retrain = (month_idx - last_retrain_idx) >= retrain_every or cached_model is None

        if need_retrain:
            train_timestamps = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars)

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

        conf = np.abs(yp - 0.5)
        fwd_ret_vals = fwd_ret.reindex(te)

        directions = np.where(yp > 0.5, 1.0, -1.0)
        signals = get_signal(yp, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        fr_vals = fwd_ret_vals.values if hasattr(fwd_ret_vals, "values") else np.array(fwd_ret_vals)
        finite_mask = np.isfinite(fr_vals)
        if finite_mask.any():
            sig_str = signals_disc if np.ndim(signals_disc) > 0 else np.full(len(yp), float(signals_disc))
            batch = pd.DataFrame({
                "timestamp": te[finite_mask],
                "direction": directions[finite_mask],
                "fwd_return": fr_vals[finite_mask],
                "confidence": conf[finite_mask],
                "probability": yp[finite_mask],
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
# FILTER FUNCTION
# =============================================================================

def apply_confidence_filter(
    signals_df: pd.DataFrame,
    mode: str,
    threshold: float,
) -> pd.DataFrame:
    """Apply confidence filter to raw signals."""
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    df = signals_df.copy()

    if mode == "percentile":
        conf = df["confidence"].values
        if len(conf) > 5:
            conf_threshold = np.percentile(conf, threshold)
            mask = conf >= conf_threshold
        else:
            mask = np.ones(len(conf), dtype=bool)
        return df[mask].reset_index(drop=True)

    elif mode == "absolute":
        prob = df["probability"].values
        mask = (prob > threshold) | (prob < (1.0 - threshold))
        return df[mask].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown filter mode: {mode}")


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame) -> Optional[Dict]:
    """Portfolio simulation with shared capital pool."""
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

    t0 = pd.Timestamp(timestamps[0])
    daily = daily[daily.index >= t0]
    daily_nonzero = daily[daily != 0]

    return daily_nonzero if len(daily_nonzero) >= 5 else None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("DOLLAR BARS v2 — RIGOROUS DOLLAR vs TIME BARS COMPARISON")
    print("3 frequency tiers x 6 assets, adaptive threshold, bar-count fwd returns")
    print("=" * 90)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()

    # Storage: signals per bar type per asset
    # Key: bar_name, Value: list of filtered signal DataFrames
    all_signals: Dict[str, List[pd.DataFrame]] = {
        cfg[0]: [] for cfg in BAR_CONFIGS
    }
    all_diagnostics: Dict[str, Dict[str, Any]] = {}
    per_asset_signals: Dict[str, Dict[str, int]] = {}  # asset -> bar_name -> signal count

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'=' * 70}")
        print(f"Processing {asset_name} ({symbol})...")
        asset_start = time.time()

        # Load 1m data
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        per_asset_signals[asset_name] = {}

        # ==============================================================
        # TIME BARS — resample at 3 frequencies
        # ==============================================================
        print(f"  --- Time Bars ---")
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        ohlcv_30m = resample_ohlcv(ohlcv_1m, "30min")
        ohlcv_1h = resample_ohlcv(ohlcv_1m, "1h")
        print(f"  15m: {len(ohlcv_15m):,} | 30m: {len(ohlcv_30m):,} | 1h: {len(ohlcv_1h):,}")

        # ==============================================================
        # DOLLAR BARS — adaptive threshold at 3 frequencies
        # ==============================================================
        print(f"  --- Dollar Bars (adaptive threshold) ---")

        thresh_96 = compute_rolling_dollar_threshold(ohlcv_1m, 96)
        thresh_48 = compute_rolling_dollar_threshold(ohlcv_1m, 48)
        thresh_24 = compute_rolling_dollar_threshold(ohlcv_1m, 24)

        t0 = time.time()
        dollar_96 = resample_dollar_bars_adaptive(ohlcv_1m, thresh_96)
        dollar_48 = resample_dollar_bars_adaptive(ohlcv_1m, thresh_48)
        dollar_24 = resample_dollar_bars_adaptive(ohlcv_1m, thresh_24)
        print(f"  Dollar 96: {len(dollar_96):,} | 48: {len(dollar_48):,} | 24: {len(dollar_24):,}  "
              f"({time.time()-t0:.1f}s)")

        # ==============================================================
        # Build config list for this asset
        # ==============================================================
        configs = [
            ("time_15m", ohlcv_15m, 4),
            ("time_30m", ohlcv_30m, 4),
            ("time_1h",  ohlcv_1h,  2),
            ("dollar_96", dollar_96, 4),
            ("dollar_48", dollar_48, 4),
            ("dollar_24", dollar_24, 2),
        ]

        for bar_name, bars, fwd_bars in configs:
            if len(bars) < 500:
                print(f"  {bar_name}: SKIPPED (too few bars: {len(bars)})")
                per_asset_signals[asset_name][bar_name] = 0
                continue

            # Diagnostics
            diag = compute_bar_diagnostics(bars, bar_name)
            all_diagnostics[f"{asset_name}_{bar_name}"] = diag

            # Features
            features = compute_features(bars)

            # Walk-forward signal generation
            t0 = time.time()
            raw = generate_raw_signals(
                bars, features, fwd_bars, asset_name, FEATURES_14,
            )

            if raw is None or len(raw) == 0:
                print(f"  {bar_name}: no raw signals generated")
                per_asset_signals[asset_name][bar_name] = 0
                continue

            # Apply percentile-70 confidence filter
            filtered = apply_confidence_filter(raw, "percentile", 70)
            elapsed = time.time() - t0

            n_raw = len(raw)
            n_filt = len(filtered)
            print(f"  {bar_name}: {n_raw:,} raw -> {n_filt:,} filtered ({elapsed:.1f}s)")

            per_asset_signals[asset_name][bar_name] = n_filt

            if n_filt > 0:
                all_signals[bar_name].append(filtered)

        elapsed = time.time() - asset_start
        print(f"  {asset_name} complete in {elapsed:.1f}s")

    # =========================================================================
    # Portfolio simulation per bar type (combine all assets)
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("PORTFOLIO SIMULATIONS (per bar type, all assets combined)")
    print("=" * 90)

    results: Dict[str, Dict[str, Any]] = {}

    for bar_name in [cfg[0] for cfg in BAR_CONFIGS]:
        if not all_signals[bar_name]:
            print(f"  {bar_name}: no signals collected, skipping")
            continue

        combined = pd.concat(all_signals[bar_name]).sort_values("timestamp")
        print(f"  {bar_name}: {len(combined):,} total signals across all assets")

        sim = simulate_portfolio(combined)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                trades_per_day = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(
                    daily, STARTING_CAPITAL, sim["n_trades"], trades_per_day,
                )
                results[bar_name] = stats
                results[bar_name]["n_trades"] = sim["n_trades"]
                results[bar_name]["halted"] = sim["halted"]
                print(f"    -> Sharpe={stats.get('sharpe', 0):.3f}, "
                      f"Return={stats.get('total_return_pct', 0):.2f}%, "
                      f"MaxDD={stats.get('max_dd_pct', 0):.2f}%, "
                      f"Trades={sim['n_trades']}")
            else:
                print(f"    -> insufficient daily PnL data")
        else:
            n_trades = sim["n_trades"] if sim else 0
            print(f"    -> insufficient trades ({n_trades})")

    # =========================================================================
    # TABLE 1: Bar Quality Diagnostics
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("TABLE 1 — BAR QUALITY DIAGNOSTICS")
    print("=" * 90)
    print(f"{'Bar Type':<12} {'Asset':<6} {'Bars/day(mean+/-std)':<22} "
          f"{'InterBar(mean+/-std)':<22} {'Autocorr(lag1)':<16} {'Kurtosis':<10}")
    print("-" * 90)

    for asset_name in ASSETS.keys():
        for bar_name in [cfg[0] for cfg in BAR_CONFIGS]:
            key = f"{asset_name}_{bar_name}"
            if key not in all_diagnostics:
                continue
            d = all_diagnostics[key]
            if "error" in d:
                print(f"{bar_name:<12} {asset_name:<6} {'SKIPPED':<22}")
                continue
            bpd = f"{d['bars_per_day_mean']:.1f}+/-{d['bars_per_day_std']:.1f}"
            ibt = f"{d['inter_bar_mean_min']:.1f}+/-{d['inter_bar_std_min']:.1f}min"
            ac1 = f"{d['return_autocorr_lag1']:.4f}"
            kurt_val = f"{d['return_kurtosis']:.2f}"
            print(f"{bar_name:<12} {asset_name:<6} {bpd:<22} {ibt:<22} {ac1:<16} {kurt_val:<10}")

    # =========================================================================
    # TABLE 2: Performance (portfolio level)
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("TABLE 2 — PERFORMANCE (portfolio level, all assets combined)")
    print("=" * 90)
    print(f"{'Bar Type':<12} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
          f"{'Trades':>8} {'WinRate':>8} {'PSR%':>6}")
    print("-" * 60)

    for bar_name in [cfg[0] for cfg in BAR_CONFIGS]:
        if bar_name in results:
            r = results[bar_name]
            print(f"{bar_name:<12} {r.get('sharpe', 0):>8.3f} "
                  f"{r.get('total_return_pct', 0):>8.2f}% "
                  f"{r.get('max_dd_pct', 0):>7.2f}% "
                  f"{r.get('n_trades', 0):>8} "
                  f"{r.get('win_rate_days', 0) * 100:>7.1f}% "
                  f"{r.get('psr_pct', 0):>5.1f}%")
        else:
            print(f"{bar_name:<12} {'N/A':>8}")

    # =========================================================================
    # TABLE 3: Paired Comparison (dollar vs time at matched frequency)
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("TABLE 3 — PAIRED COMPARISON (dollar vs time at matched frequency)")
    print("=" * 90)

    pairs = [
        ("~15min", "time_15m", "dollar_96"),
        ("~30min", "time_30m", "dollar_48"),
        ("~1hour", "time_1h",  "dollar_24"),
    ]

    print(f"{'Frequency':<10} {'Time Sharpe':>12} {'Dollar Sharpe':>14} "
          f"{'Delta':>8} {'Winner':>10} {'Consistent?(4+/6)':>18}")
    print("-" * 75)

    for freq_label, time_key, dollar_key in pairs:
        time_sharpe = results.get(time_key, {}).get("sharpe", None)
        dollar_sharpe = results.get(dollar_key, {}).get("sharpe", None)

        if time_sharpe is not None and dollar_sharpe is not None:
            delta = dollar_sharpe - time_sharpe
            winner = "Dollar" if delta > 0 else ("Time" if delta < 0 else "Tie")

            # Per-asset consistency check: how many assets does the winner win?
            asset_wins_dollar = 0
            asset_wins_time = 0
            asset_count = 0
            for asset_name in ASSETS.keys():
                t_sigs = per_asset_signals.get(asset_name, {}).get(time_key, 0)
                d_sigs = per_asset_signals.get(asset_name, {}).get(dollar_key, 0)
                if t_sigs > 0 and d_sigs > 0:
                    asset_count += 1
                    # We need per-asset Sharpe to check consistency, but we only
                    # have portfolio-level. Use signal count as proxy for participation.
                    # For true consistency, we'd need per-asset simulations.
                    # Mark as "N/A" since we don't run per-asset portfolios.

            consistent = "N/A"  # Would need per-asset portfolio sims

            print(f"{freq_label:<10} {time_sharpe:>12.3f} {dollar_sharpe:>14.3f} "
                  f"{delta:>+8.3f} {winner:>10} {consistent:>18}")
        else:
            ts = f"{time_sharpe:.3f}" if time_sharpe is not None else "N/A"
            ds = f"{dollar_sharpe:.3f}" if dollar_sharpe is not None else "N/A"
            print(f"{freq_label:<10} {ts:>12} {ds:>14} {'N/A':>8} {'N/A':>10} {'N/A':>18}")

    # =========================================================================
    # Per-asset signal counts summary
    # =========================================================================
    print(f"\n{'=' * 90}")
    print("SIGNAL COUNTS PER ASSET x BAR TYPE")
    print("=" * 90)
    bar_names = [cfg[0] for cfg in BAR_CONFIGS]
    header = f"{'Asset':<8}" + "".join(f"{bn:>12}" for bn in bar_names)
    print(header)
    print("-" * (8 + 12 * len(bar_names)))
    for asset_name in ASSETS.keys():
        row = f"{asset_name:<8}"
        for bn in bar_names:
            cnt = per_asset_signals.get(asset_name, {}).get(bn, 0)
            row += f"{cnt:>12,}"
        print(row)

    # =========================================================================
    # Save to JSON
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    output = {
        "meta": {
            "experiment": "dollar_bars_v2",
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(total_elapsed, 1),
            "assets": list(ASSETS.keys()),
            "n_assets": len(ASSETS),
            "description": (
                "Rigorous dollar vs time bars comparison with adaptive threshold, "
                "bar-count fwd returns, 3 frequency tiers x 6 assets"
            ),
        },
        "config": {
            "model_c": MODEL_C,
            "model_penalty": MODEL_PENALTY,
            "model_solver": MODEL_SOLVER,
            "model_max_iter": MODEL_MAX_ITER,
            "model_seed": MODEL_SEED,
            "starting_capital": STARTING_CAPITAL,
            "max_dd": MAX_DD,
            "delever_start": DELEVER_START,
            "slippage": SLIPPAGE,
            "max_exposure": MAX_EXPOSURE,
            "max_per_asset": MAX_PER_ASSET,
            "base_sizing_pct": BASE_SIZING_PCT,
            "max_signal_mult": MAX_SIGNAL_MULT,
            "vol_filter_quantile": VOL_FILTER_QUANTILE,
            "winsor_lo": WINSOR_LO,
            "winsor_hi": WINSOR_HI,
            "signal_step_size": SIGNAL_STEP_SIZE,
            "rolling_threshold_days": ROLLING_THRESHOLD_DAYS,
            "features": FEATURES_14,
            "confidence_filter": "percentile_70",
            "bar_configs": [
                {"name": name, "arg": str(arg), "fwd_bars": fwd}
                for name, arg, fwd in BAR_CONFIGS
            ],
        },
        "results": results,
        "diagnostics": all_diagnostics,
        "per_asset_signal_counts": per_asset_signals,
        "paired_comparisons": {},
    }

    # Add paired comparison data
    for freq_label, time_key, dollar_key in pairs:
        time_sharpe = results.get(time_key, {}).get("sharpe", None)
        dollar_sharpe = results.get(dollar_key, {}).get("sharpe", None)
        if time_sharpe is not None and dollar_sharpe is not None:
            output["paired_comparisons"][freq_label] = {
                "time_bar": time_key,
                "dollar_bar": dollar_key,
                "time_sharpe": time_sharpe,
                "dollar_sharpe": dollar_sharpe,
                "delta": round(dollar_sharpe - time_sharpe, 3),
                "winner": "dollar" if dollar_sharpe > time_sharpe else "time",
            }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/dollar_bars_v2_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 90}")
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
