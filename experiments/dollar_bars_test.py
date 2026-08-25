"""
Dollar Bars vs 15m Time Bars — Comparative Experiment.

Tests whether information-driven (dollar volume) sampling improves signal
quality compared to fixed 15-minute time bars. Dollar bars create a new bar
whenever cumulative dollar volume exceeds a threshold, producing more bars
during high-activity periods and fewer during quiet periods.

Both bar types use:
  - Same walk-forward methodology (monthly expanding, embargo, sample weights)
  - Same portfolio simulation (shared capital, exposure limits, DD halt)
  - Same slippage model (1 bps)
  - Same 14 features, same LogReg L1 model
  - Approach A baseline (percentile 70 confidence filter)

Key difference: Dollar bars use a TIME-BASED forward horizon (1 hour) since
bars are irregularly spaced, while time bars use bar-count (4 bars = 1 hour).
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

# Dollar bar calibration
TARGET_BARS_PER_DAY = 96  # same as 15m time bars (24h / 15min)
CALIBRATION_DAYS = 30     # first 30 days used to compute threshold

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
    """Compute all statistics from daily PnL series."""
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
# FEATURE COMPUTATION
# =============================================================================

def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all 14 universal features from OHLCV data."""
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

    # Volume z-score (handle zero/constant volume)
    vol_mean = volume.rolling(30, min_periods=10).mean()
    vol_std = volume.rolling(30, min_periods=10).std()
    if vol_std.max() > 0:
        f["vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    else:
        f["vol_zscore"] = 0.0

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

    # Downcast to float32
    for col in f.columns:
        f[col] = f[col].astype(np.float32)

    return f


# =============================================================================
# DATA LOADING — Binance 1m CSVs
# =============================================================================

def load_binance_1m(symbol: str, data_dir: Path) -> pd.DataFrame:
    """Load all monthly CSV files for a symbol from Binance 1m data.

    Handles both header and no-header formats. Some 2025+ files include
    a header row. Detection: try pd.to_numeric on the first value of
    the first column — if it fails, the first row is a header.

    Returns OHLCV DataFrame with UTC DatetimeIndex.
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
# RESAMPLING — Time Bars & Dollar Bars
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


def resample_dollar_bars(ohlcv_1m: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """Resample 1m OHLCV to dollar bars.

    A new bar is created whenever cumulative dollar volume (close * volume)
    exceeds dollar_threshold since the last bar.

    Returns OHLCV DataFrame with irregular DatetimeIndex.
    """
    closes = ohlcv_1m["close"].values
    highs = ohlcv_1m["high"].values
    lows = ohlcv_1m["low"].values
    opens = ohlcv_1m["open"].values
    volumes = ohlcv_1m["volume"].values
    timestamps = ohlcv_1m.index

    dollar_vol = closes * volumes  # dollar volume per 1m bar

    bars = []
    cum_dollar = 0.0
    bar_open = opens[0]
    bar_high = highs[0]
    bar_low = lows[0]
    bar_volume = 0.0
    bar_start_idx = 0

    for i in range(len(closes)):
        cum_dollar += dollar_vol[i]
        bar_high = max(bar_high, highs[i])
        bar_low = min(bar_low, lows[i])
        bar_volume += volumes[i]

        if cum_dollar >= dollar_threshold:
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
                bar_high = highs[i + 1] if i + 1 < len(highs) else highs[i]
                bar_low = lows[i + 1] if i + 1 < len(lows) else lows[i]
                bar_volume = 0.0
                bar_start_idx = i + 1

    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    result = pd.DataFrame(bars)
    result = result.set_index("timestamp")
    return result


def calibrate_dollar_threshold(ohlcv_1m: pd.DataFrame,
                                target_bars_per_day: int = TARGET_BARS_PER_DAY,
                                calibration_days: int = CALIBRATION_DAYS) -> float:
    """Auto-calibrate dollar bar threshold for target bar frequency.

    Uses median daily dollar volume from first calibration_days days.
    Threshold = median_daily_dollar_volume / target_bars_per_day.
    """
    # Use first calibration_days of data
    start = ohlcv_1m.index.min()
    end = start + pd.Timedelta(days=calibration_days)
    calib_data = ohlcv_1m.loc[start:end]

    # Compute dollar volume per 1m bar
    dollar_vol = calib_data["close"] * calib_data["volume"]

    # Daily dollar volume
    daily_dollar_vol = dollar_vol.resample("1D").sum()
    daily_dollar_vol = daily_dollar_vol[daily_dollar_vol > 0]

    if len(daily_dollar_vol) < 5:
        # Fallback: use all available data
        daily_dollar_vol = (ohlcv_1m["close"] * ohlcv_1m["volume"]).resample("1D").sum()
        daily_dollar_vol = daily_dollar_vol[daily_dollar_vol > 0]

    median_daily = daily_dollar_vol.median()
    threshold = median_daily / target_bars_per_day

    return threshold


# =============================================================================
# FORWARD RETURN COMPUTATION FOR DOLLAR BARS (TIME-BASED)
# =============================================================================

def compute_dollar_bar_fwd_returns(
    dollar_bars: pd.DataFrame,
    ohlcv_1m: pd.DataFrame,
    horizon_hours: float = 1.0,
) -> pd.Series:
    """Compute forward returns for dollar bars using TIME-BASED horizon.

    Since dollar bars are irregularly spaced, we cannot use bar-count for the
    forward horizon. Instead, for each dollar bar at time t, we find the close
    price ~horizon_hours later in the original 1m data.

    Args:
        dollar_bars: OHLCV DataFrame with irregular DatetimeIndex.
        ohlcv_1m: Original 1-minute OHLCV data.
        horizon_hours: Forward horizon in hours (default 1.0).

    Returns:
        pd.Series of forward returns indexed by dollar bar timestamps.
    """
    horizon_td = pd.Timedelta(hours=horizon_hours)
    bar_times = dollar_bars.index
    bar_closes = dollar_bars["close"].values

    ohlcv_1m_index = ohlcv_1m.index
    ohlcv_1m_close = ohlcv_1m["close"].values

    fwd_returns = np.full(len(bar_times), np.nan)

    for i in range(len(bar_times)):
        target_time = bar_times[i] + horizon_td
        # Find the closest 1m bar at or after target_time
        fwd_idx = ohlcv_1m_index.searchsorted(target_time)
        if fwd_idx < len(ohlcv_1m_close):
            fwd_price = ohlcv_1m_close[fwd_idx]
            fwd_returns[i] = fwd_price / bar_closes[i] - 1.0

    return pd.Series(fwd_returns, index=bar_times, name="fwd_return")


# =============================================================================
# RAW SIGNAL GENERATION — Walk-Forward (shared for both bar types)
# =============================================================================

def generate_raw_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_ret: pd.Series,
    asset_name: str,
    feature_cols: List[str],
    fwd_bars_for_weights: int = 4,
) -> pd.DataFrame:
    """Walk-forward monthly retrain with expanding window.

    Accepts a pre-computed fwd_ret series, which allows both time bars
    (bar-count based) and dollar bars (time-based) to use the same pipeline.

    Args:
        ohlcv: OHLCV data (used for vol filter).
        features: Feature DataFrame aligned to ohlcv index.
        fwd_ret: Forward returns series (pre-computed, aligned to ohlcv index).
        asset_name: Asset identifier.
        feature_cols: Which features to use.
        fwd_bars_for_weights: Number of forward bars for sample weight computation.

    Returns:
        DataFrame with raw signals.
    """
    close = ohlcv["close"]

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

    # Compute embargo timedelta based on bar timeframe
    if len(ohlcv) > 1:
        median_gap = ohlcv.index.to_series().diff().dropna().median()
        embargo_td = median_gap * fwd_bars_for_weights
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
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars_for_weights)

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
    print("=" * 80)
    print("DOLLAR BARS vs 15m TIME BARS — COMPARATIVE EXPERIMENT")
    print("LogReg L1, 14 features, Approach A (percentile 70 filter)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()

    # Storage for results
    time_bar_signals: List[pd.DataFrame] = []
    dollar_bar_signals: List[pd.DataFrame] = []

    per_asset_details: Dict[str, Dict[str, Any]] = {}

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {asset_name} ({symbol})...")
        asset_start = time.time()

        # Load 1m data
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        # ==============================================================
        # TIME BARS (15m) — baseline
        # ==============================================================
        print(f"  --- Time Bars (15m) ---")
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        feat_15m = compute_features(ohlcv_15m)
        print(f"  15m bars: {len(ohlcv_15m):,}")

        # Forward return: 4 bars ahead (= 1 hour)
        fwd_ret_time = ohlcv_15m["close"].pct_change(4).shift(-4)

        raw_time = generate_raw_signals(
            ohlcv_15m, feat_15m, fwd_ret_time,
            asset_name=asset_name, feature_cols=FEATURES_14,
            fwd_bars_for_weights=4,
        )
        sig_time = apply_confidence_filter(raw_time, "percentile", 70)
        print(f"  Time bar signals: {len(raw_time):,} raw -> {len(sig_time):,} filtered")

        if len(sig_time) > 0:
            time_bar_signals.append(sig_time)

        # ==============================================================
        # DOLLAR BARS — auto-calibrated
        # ==============================================================
        print(f"  --- Dollar Bars ---")

        # Calibrate threshold
        threshold = calibrate_dollar_threshold(ohlcv_1m)
        print(f"  Dollar threshold: ${threshold:,.0f}")

        # Resample to dollar bars
        ohlcv_dollar = resample_dollar_bars(ohlcv_1m, threshold)
        print(f"  Dollar bars: {len(ohlcv_dollar):,}")

        if len(ohlcv_dollar) < 500:
            print(f"  SKIPPING dollar bars for {asset_name}: too few bars ({len(ohlcv_dollar)})")
            per_asset_details[asset_name] = {
                "time_bars_count": len(ohlcv_15m),
                "dollar_bars_count": len(ohlcv_dollar),
                "dollar_threshold": float(threshold),
                "dollar_bars_skipped": True,
            }
            continue

        # Bar frequency statistics
        inter_bar_times = ohlcv_dollar.index.to_series().diff().dropna()
        inter_bar_minutes = inter_bar_times.dt.total_seconds() / 60.0
        bars_per_day = len(ohlcv_dollar) / max(
            (ohlcv_dollar.index.max() - ohlcv_dollar.index.min()).days, 1
        )

        print(f"  Inter-bar time: mean={inter_bar_minutes.mean():.1f}min, "
              f"std={inter_bar_minutes.std():.1f}min, "
              f"median={inter_bar_minutes.median():.1f}min")
        print(f"  Bars/day: {bars_per_day:.1f} (target: {TARGET_BARS_PER_DAY})")

        # Compute features on dollar bars
        feat_dollar = compute_features(ohlcv_dollar)

        # Forward return: TIME-BASED (1 hour) from 1m data
        fwd_ret_dollar = compute_dollar_bar_fwd_returns(
            ohlcv_dollar, ohlcv_1m, horizon_hours=1.0,
        )

        # Estimate equivalent fwd_bars for sample weights
        # median inter-bar time in minutes -> how many bars in 1 hour
        median_bar_mins = inter_bar_minutes.median()
        equiv_fwd_bars = max(int(round(60.0 / median_bar_mins)), 1) if median_bar_mins > 0 else 4

        raw_dollar = generate_raw_signals(
            ohlcv_dollar, feat_dollar, fwd_ret_dollar,
            asset_name=asset_name, feature_cols=FEATURES_14,
            fwd_bars_for_weights=equiv_fwd_bars,
        )
        sig_dollar = apply_confidence_filter(raw_dollar, "percentile", 70)
        print(f"  Dollar bar signals: {len(raw_dollar):,} raw -> {len(sig_dollar):,} filtered")

        if len(sig_dollar) > 0:
            dollar_bar_signals.append(sig_dollar)

        # Store per-asset details
        per_asset_details[asset_name] = {
            "time_bars_count": len(ohlcv_15m),
            "dollar_bars_count": len(ohlcv_dollar),
            "dollar_threshold": float(threshold),
            "dollar_bars_per_day": round(bars_per_day, 1),
            "inter_bar_mean_min": round(float(inter_bar_minutes.mean()), 1),
            "inter_bar_std_min": round(float(inter_bar_minutes.std()), 1),
            "inter_bar_median_min": round(float(inter_bar_minutes.median()), 1),
            "equiv_fwd_bars": equiv_fwd_bars,
            "time_bar_raw_signals": len(raw_time),
            "time_bar_filtered_signals": len(sig_time),
            "dollar_bar_raw_signals": len(raw_dollar),
            "dollar_bar_filtered_signals": len(sig_dollar),
        }

        elapsed = time.time() - asset_start
        print(f"  {asset_name} complete in {elapsed:.1f}s")

    # =========================================================================
    # Portfolio simulations
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PORTFOLIO SIMULATIONS")
    print("=" * 80)

    results = {}

    for bar_type, signal_list in [("time_15m", time_bar_signals), ("dollar", dollar_bar_signals)]:
        if not signal_list:
            print(f"  {bar_type}: no signals collected, skipping")
            continue

        combined = pd.concat(signal_list).sort_values("timestamp")
        print(f"  {bar_type}: {len(combined):,} total signals across all assets")

        sim = simulate_portfolio(combined)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                trades_per_day = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(
                    daily, STARTING_CAPITAL, sim["n_trades"], trades_per_day,
                )
                results[bar_type] = stats
                results[bar_type]["n_trades"] = sim["n_trades"]
                results[bar_type]["halted"] = sim["halted"]
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
    # Summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY — DOLLAR BARS vs 15m TIME BARS")
    print("=" * 80)

    # Per-asset bar counts
    print(f"\n{'Asset':<8} {'Time Bars':>12} {'Dollar Bars':>13} {'$/day':>10} {'InterBar Mean':>15} {'InterBar Std':>14}")
    print("-" * 75)
    for asset_name, details in per_asset_details.items():
        tb = details.get("time_bars_count", 0)
        db = details.get("dollar_bars_count", 0)
        bpd = details.get("dollar_bars_per_day", 0)
        ibm = details.get("inter_bar_mean_min", 0)
        ibs = details.get("inter_bar_std_min", 0)
        print(f"{asset_name:<8} {tb:>12,} {db:>13,} {bpd:>10.1f} {ibm:>13.1f}min {ibs:>12.1f}min")

    # Sharpe comparison
    print(f"\n{'Bar Type':<20} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'Trades':>8} {'WinRate':>8} {'PSR%':>6}")
    print("-" * 70)
    for bar_type in ["time_15m", "dollar"]:
        if bar_type in results:
            r = results[bar_type]
            label = "15m Time Bars" if bar_type == "time_15m" else "Dollar Bars"
            print(f"{label:<20} {r.get('sharpe', 0):>8.3f} "
                  f"{r.get('total_return_pct', 0):>8.2f}% "
                  f"{r.get('max_dd_pct', 0):>7.2f}% "
                  f"{r.get('n_trades', 0):>8} "
                  f"{r.get('win_rate_days', 0) * 100:>7.1f}% "
                  f"{r.get('psr_pct', 0):>5.1f}%")

    # Delta
    if "time_15m" in results and "dollar" in results:
        sharpe_delta = results["dollar"]["sharpe"] - results["time_15m"]["sharpe"]
        ret_delta = results["dollar"]["total_return_pct"] - results["time_15m"]["total_return_pct"]
        print(f"\n  Sharpe delta (dollar - time): {sharpe_delta:+.3f}")
        print(f"  Return delta (dollar - time): {ret_delta:+.2f}%")
        if sharpe_delta > 0:
            print("  -> Dollar bars OUTPERFORM time bars")
        elif sharpe_delta < 0:
            print("  -> Time bars OUTPERFORM dollar bars")
        else:
            print("  -> No difference")

    # =========================================================================
    # Save to JSON
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    output = {
        "meta": {
            "experiment": "dollar_bars_vs_time_bars",
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(total_elapsed, 1),
            "assets": list(ASSETS.keys()),
            "n_assets": len(ASSETS),
            "description": "Dollar bars (information-driven sampling) vs 15m time bars",
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
            "target_bars_per_day": TARGET_BARS_PER_DAY,
            "calibration_days": CALIBRATION_DAYS,
            "dollar_bar_fwd_horizon_hours": 1.0,
            "time_bar_fwd_bars": 4,
            "features": FEATURES_14,
            "confidence_filter": "percentile_70",
        },
        "results": results,
        "per_asset": per_asset_details,
    }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/dollar_bars_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
