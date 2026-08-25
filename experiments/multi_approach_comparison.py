"""
Multi-Approach Comparison — 8 approaches on 6 crypto assets at 15m.

Tests different signal generation and filtering strategies on the SAME data
to find the best approach for live trading. All approaches share:
  - Same walk-forward methodology (monthly expanding, embargo, sample weights)
  - Same portfolio simulation (shared capital, exposure limits, DD halt)
  - Same slippage model (1 bps)

Approaches:
  A: 15m 14-feature baseline (percentile 70 confidence filter)
  B: 15m 22-feature multi-TF (percentile 70 confidence filter)
  C: A + 4H regime filter (suppress counter-trend signals)
  D: Multi-timeframe blend (60% 15m + 40% 4H signals)
  E1: A with absolute confidence > 0.55
  E2: A with absolute confidence > 0.60
  E3: A with absolute confidence > 0.65
  F: B with absolute confidence > 0.60

Key optimization: Train 3 unique models per asset, not 8.
  Config 1: 15m, 14 features -> A, C, E1, E2, E3
  Config 2: 15m, 22 features -> B, F
  Config 3: 4H, 14 features  -> D
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

# Feature lists
FEATURES_14 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]

MTF_FEATURES = [
    "rsi_14_1h", "bb_pct_b_1h", "atr_14_1h", "macd_hist_1h",
    "rsi_14_4h", "bb_pct_b_4h", "ret_1_4h", "atr_14_4h",
]

FEATURES_22 = FEATURES_14 + MTF_FEATURES

APPROACHES = ["A", "B", "C", "D", "E1", "E2", "E3", "F"]

APPROACH_DESCRIPTIONS = {
    "A":  "15m 14-feat, percentile-70 filter",
    "B":  "15m 22-feat (multi-TF), percentile-70 filter",
    "C":  "A + 4H regime filter",
    "D":  "Multi-TF blend (60% 15m + 40% 4H)",
    "E1": "15m 14-feat, absolute conf > 0.55",
    "E2": "15m 14-feat, absolute conf > 0.60",
    "E3": "15m 14-feat, absolute conf > 0.65",
    "F":  "15m 22-feat, absolute conf > 0.60",
}


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
# FEATURE COMPUTATION
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
# DATA LOADING — Binance 1m CSVs
# =============================================================================

def load_binance_1m(symbol: str, data_dir: Path) -> pd.DataFrame:
    """Load all monthly CSV files for a symbol from Binance 1m data.

    Handles both header and no-header formats. Some 2025+ files include
    a header row. Detection: try pd.to_numeric on the first value of
    the first column — if it fails, the first row is a header.

    Returns OHLCV DataFrame with UTC DatetimeIndex.
    """
    # Files are in data_dir/SYMBOL/SYMBOL-1m-*.csv
    symbol_dir = data_dir / symbol
    pattern = f"{symbol}-1m-*.csv"
    files = sorted(symbol_dir.glob(pattern))
    if not files:
        # Try flat directory as fallback
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
        # Read first line to detect header
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
                # Rename to standard columns
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

    # Convert open_time to datetime index
    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    combined = combined.dropna(subset=["open_time"])
    combined["timestamp"] = pd.to_datetime(combined["open_time"], unit="ms", utc=True)
    combined = combined.set_index("timestamp").sort_index()

    # Keep OHLCV columns, cast to float
    ohlcv = combined[["open", "high", "low", "close", "volume"]].copy()
    for col in ohlcv.columns:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
    ohlcv = ohlcv.dropna()

    # Drop duplicates
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]

    print(f"  Loaded {symbol}: {len(ohlcv):,} bars, "
          f"{ohlcv.index.min().date()} to {ohlcv.index.max().date()}")
    return ohlcv


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Standard OHLCV resampling."""
    resampled = df.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled


# =============================================================================
# MULTI-TIMEFRAME FEATURE COMPUTATION
# =============================================================================

def compute_multi_tf_features(
    ohlcv_1m: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resample 1m to 15m, 1H, 4H. Compute features at each timeframe.

    Returns (ohlcv_15m, ohlcv_4h, feat_15m, feat_22).

    CRITICAL no-lookahead for multi-TF features:
    - 1H features: compute on 1H bars, then SHIFT(1) before reindexing to 15m
      (so at 09:45 you use the 08:00-09:00 bar, not the one still forming)
    - 4H features: same shift(1) logic
    - Then reindex to 15m timestamps with method='ffill'
    """
    # Resample
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    ohlcv_1h = resample_ohlcv(ohlcv_1m, "1h")
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")

    # Compute features at each timeframe
    feat_15m = compute_features(ohlcv_15m)
    feat_1h_all = compute_features(ohlcv_1h)
    feat_4h_all = compute_features(ohlcv_4h)

    # Select and align 1H features (no lookahead: shift(1) before reindex)
    feat_1h_sel = feat_1h_all[["rsi_14", "bb_pct_b", "atr_14", "macd_hist"]]
    feat_1h_shifted = feat_1h_sel.shift(1)  # Use PREVIOUS completed bar
    feat_1h_aligned = feat_1h_shifted.reindex(ohlcv_15m.index, method="ffill")
    feat_1h_aligned.columns = [c + "_1h" for c in feat_1h_aligned.columns]

    # Select and align 4H features (no lookahead: shift(1) before reindex)
    feat_4h_sel = feat_4h_all[["rsi_14", "bb_pct_b", "atr_14", "ret_1"]]
    feat_4h_shifted = feat_4h_sel.shift(1)  # Use PREVIOUS completed bar
    feat_4h_aligned = feat_4h_shifted.reindex(ohlcv_15m.index, method="ffill")
    feat_4h_aligned.columns = [c + "_4h" for c in feat_4h_aligned.columns]

    # Combine into 22-feature set
    feat_22 = pd.concat([feat_15m, feat_1h_aligned, feat_4h_aligned], axis=1)

    return ohlcv_15m, ohlcv_4h, feat_15m, feat_22


# =============================================================================
# RAW SIGNAL GENERATION — Walk-Forward (no confidence filter)
# =============================================================================

def generate_raw_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
    feature_cols: List[str],
    collect_coefficients: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Walk-forward monthly retrain with expanding window.

    Same logic as universal_signal_test_v2.py generate_walk_forward_signals,
    but:
      1. Accepts custom feature_cols list (14 or 22)
      2. Applies vol filter (top 50%) but NO confidence filter
      3. Returns ALL signals with their probabilities
      4. Optionally collects L1 coefficients from each retrain

    Returns (signals_df, coefficients_dict_or_None).
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for vol filter
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Determine month range from the actual data
    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    # Need at least 6 months of data to start walk-forward
    if (end_date - start_date).days < 180:
        return pd.DataFrame(), None

    # Generate monthly periods from actual data range
    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame(), None

    # OPTIMIZATION: For fine timeframes, retrain quarterly (3x fewer model fits)
    retrain_every = 1  # default: every month
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    # Compute embargo timedelta based on bar timeframe and fwd_bars
    if len(ohlcv) > 1:
        median_gap = ohlcv.index.to_series().diff().dropna().median()
        embargo_td = median_gap * fwd_bars
    else:
        embargo_td = pd.Timedelta(hours=2)

    rows = []
    cached_model = None
    cached_scaler = None
    last_retrain_idx = -999
    coefficients = {} if collect_coefficients else None

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
        feat_cols = [c for c in feature_cols if c in features.columns]
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

                # Collect coefficients if requested
                if collect_coefficients:
                    coef_dict = {}
                    for j, col_name in enumerate(feat_cols):
                        coef_dict[col_name] = float(model.coef_[0][j])
                    coefficients[str(month)] = coef_dict

            else:
                scaler = cached_scaler
                model = cached_model

            X_te_scaled = scaler.transform(X_test)
            yp = model.predict_proba(X_te_scaled)[:, 1]
        except Exception:
            continue

        # NO confidence filter — return ALL signals
        conf = np.abs(yp - 0.5)
        fwd_ret_vals = fwd_ret.reindex(te)

        directions = np.where(yp > 0.5, 1.0, -1.0)
        signals = get_signal(yp, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        # VECTORIZED signal construction
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
        return pd.DataFrame(), coefficients

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name
    return df, coefficients


# =============================================================================
# FILTER FUNCTIONS
# =============================================================================

def apply_confidence_filter(
    signals_df: pd.DataFrame,
    mode: str,
    threshold: float,
) -> pd.DataFrame:
    """Apply confidence filter to raw signals.

    Args:
        signals_df: Raw signals with 'probability' and 'confidence' columns.
        mode: "percentile" or "absolute".
        threshold: For percentile mode, the percentile (e.g. 70).
                   For absolute mode, the probability threshold (e.g. 0.55).

    Returns:
        Filtered signals DataFrame.
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    df = signals_df.copy()

    if mode == "percentile":
        # Keep where |prob - 0.5| >= Nth percentile (original behavior)
        conf = df["confidence"].values
        if len(conf) > 5:
            conf_threshold = np.percentile(conf, threshold)
            mask = conf >= conf_threshold
        else:
            mask = np.ones(len(conf), dtype=bool)
        return df[mask].reset_index(drop=True)

    elif mode == "absolute":
        # Keep where prob > threshold OR prob < (1 - threshold)
        prob = df["probability"].values
        mask = (prob > threshold) | (prob < (1.0 - threshold))
        return df[mask].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown filter mode: {mode}")


def apply_regime_filter(
    signals_df: pd.DataFrame,
    ohlcv_4h: pd.DataFrame,
) -> pd.DataFrame:
    """Apply 4H regime filter: suppress counter-trend signals.

    Compute 4H EMA(20). For each signal, check if direction agrees with 4H trend.
    - 4H close > EMA(20) -> bullish -> allow long, suppress short
    - 4H close < EMA(20) -> bearish -> allow short, suppress long

    Forward-fill 4H regime to signal timestamps.
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    df = signals_df.copy()

    # Compute 4H EMA(20) regime
    ema_20 = ohlcv_4h["close"].ewm(span=20, adjust=False).mean()
    regime_4h = pd.Series(
        np.where(ohlcv_4h["close"] > ema_20, 1.0, -1.0),
        index=ohlcv_4h.index,
        name="regime",
    )

    # Align to signal timestamps via forward-fill
    signal_times = pd.to_datetime(df["timestamp"])

    # Handle timezone compatibility
    if regime_4h.index.tz is not None and signal_times.dt.tz is None:
        signal_times = signal_times.dt.tz_localize("UTC")
    elif regime_4h.index.tz is None and signal_times.dt.tz is not None:
        signal_times = signal_times.dt.tz_localize(None)

    # Create a combined index for reindexing
    regime_aligned = regime_4h.reindex(
        regime_4h.index.union(signal_times.unique())
    ).ffill()
    signal_regimes = regime_aligned.reindex(signal_times).values

    # Filter: direction must agree with regime
    # regime=1 (bullish) -> allow long (direction=1), suppress short (direction=-1)
    # regime=-1 (bearish) -> allow short (direction=-1), suppress long (direction=1)
    directions = df["direction"].values
    mask = (directions * signal_regimes) > 0  # same sign = agree

    return df[mask].reset_index(drop=True)


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
# SIGNAL CORRELATION ANALYSIS
# =============================================================================

def compute_signal_correlations(
    all_signals: Dict[str, List[pd.DataFrame]],
) -> Dict[str, float]:
    """Compute pairwise Spearman rank correlation between approaches' signed signals.

    Matches on timestamp to compare overlapping signals.
    """
    # Combine all signals per approach into a single DataFrame
    approach_series = {}
    for key, dfs in all_signals.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        # Create signed signal: direction * signal_strength
        combined["signed_signal"] = combined["direction"] * combined["signal_strength"]
        # Aggregate by timestamp (mean if multiple assets at same time)
        ts_signal = combined.groupby("timestamp")["signed_signal"].mean()
        approach_series[key] = ts_signal

    correlations = {}
    keys = sorted(approach_series.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            s1, s2 = approach_series[k1], approach_series[k2]
            # Find overlapping timestamps
            common_idx = s1.index.intersection(s2.index)
            if len(common_idx) < 30:
                correlations[f"{k1}_vs_{k2}"] = None
                continue
            v1 = s1.reindex(common_idx).values
            v2 = s2.reindex(common_idx).values
            # Remove NaN pairs
            valid = np.isfinite(v1) & np.isfinite(v2)
            if valid.sum() < 30:
                correlations[f"{k1}_vs_{k2}"] = None
                continue
            corr, _ = spearmanr(v1[valid], v2[valid])
            correlations[f"{k1}_vs_{k2}"] = round(float(corr), 4)

    return correlations


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MULTI-APPROACH COMPARISON")
    print("8 approaches x 6 assets @ 15m (+ 4H for D)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()

    all_approach_signals: Dict[str, List[pd.DataFrame]] = {a: [] for a in APPROACHES}
    per_asset_results: Dict[str, Dict[str, Any]] = {}
    all_coefficients: Dict[str, Dict] = {}  # asset -> coefficients from Config 2

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {asset_name} ({symbol})...")
        asset_start = time.time()

        # Load and resample
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        ohlcv_15m, ohlcv_4h, feat_15m, feat_22 = compute_multi_tf_features(ohlcv_1m)
        feat_4h = compute_features(ohlcv_4h)
        print(f"  15m bars: {len(ohlcv_15m):,}, 4H bars: {len(ohlcv_4h):,}")

        # MODEL CONFIG 1: 15m, 14 features (for A, C, E1, E2, E3)
        print(f"  Config 1: 15m base model (14 features)...")
        t0 = time.time()
        raw_15m_14, _ = generate_raw_signals(
            ohlcv_15m, feat_15m, fwd_bars=4,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        print(f"    -> {len(raw_15m_14):,} raw signals in {time.time()-t0:.1f}s")

        # MODEL CONFIG 2: 15m, 22 features (for B, F)
        print(f"  Config 2: 15m multi-TF model (22 features)...")
        t0 = time.time()
        raw_15m_22, coefs_22 = generate_raw_signals(
            ohlcv_15m, feat_22, fwd_bars=4,
            asset_name=asset_name, feature_cols=FEATURES_22,
            collect_coefficients=True,
        )
        print(f"    -> {len(raw_15m_22):,} raw signals in {time.time()-t0:.1f}s")
        if coefs_22:
            all_coefficients[asset_name] = coefs_22

        # MODEL CONFIG 3: 4H, 14 features (for D)
        print(f"  Config 3: 4H model (14 features)...")
        t0 = time.time()
        raw_4h_14, _ = generate_raw_signals(
            ohlcv_4h, feat_4h, fwd_bars=1,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        print(f"    -> {len(raw_4h_14):,} raw signals in {time.time()-t0:.1f}s")

        # Apply filters to derive each approach
        sig_A = apply_confidence_filter(raw_15m_14, "percentile", 70)
        sig_B = apply_confidence_filter(raw_15m_22, "percentile", 70)
        sig_C = apply_regime_filter(sig_A, ohlcv_4h)
        sig_E1 = apply_confidence_filter(raw_15m_14, "absolute", 0.55)
        sig_E2 = apply_confidence_filter(raw_15m_14, "absolute", 0.60)
        sig_E3 = apply_confidence_filter(raw_15m_14, "absolute", 0.65)
        sig_F = apply_confidence_filter(raw_15m_22, "absolute", 0.60)

        # D: multi-timeframe blend (60% 15m + 40% 4H)
        sig_D_15m = sig_A.copy()
        if len(sig_D_15m) > 0:
            sig_D_15m["signal_strength"] *= 0.6
        sig_D_4h = apply_confidence_filter(raw_4h_14, "percentile", 70)
        if len(sig_D_4h) > 0:
            sig_D_4h["signal_strength"] *= 0.4
        sig_D = pd.concat([sig_D_15m, sig_D_4h]).sort_values("timestamp")

        # Report signal counts
        sigs = {"A": sig_A, "B": sig_B, "C": sig_C, "D": sig_D,
                "E1": sig_E1, "E2": sig_E2, "E3": sig_E3, "F": sig_F}
        print(f"  Signal counts: " + ", ".join(
            f"{k}={len(v)}" for k, v in sigs.items()
        ))

        # Collect
        per_asset_results[asset_name] = {}
        for key, sig in sigs.items():
            if sig is not None and len(sig) > 0:
                all_approach_signals[key].append(sig)
                per_asset_results[asset_name][key] = {
                    "n_signals": len(sig),
                    "long_pct": round(float((sig["direction"] > 0).mean()) * 100, 1)
                        if len(sig) > 0 else 0,
                    "mean_confidence": round(float(sig["confidence"].mean()), 4)
                        if len(sig) > 0 else 0,
                }

        elapsed = time.time() - asset_start
        print(f"  {asset_name} complete in {elapsed:.1f}s")

    # =========================================================================
    # Run portfolio simulation for each approach
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PORTFOLIO SIMULATIONS")
    print("=" * 80)

    results: Dict[str, Dict[str, Any]] = {}
    for key in APPROACHES:
        if not all_approach_signals[key]:
            print(f"  {key}: no signals collected, skipping")
            continue

        combined = pd.concat(all_approach_signals[key]).sort_values("timestamp")
        print(f"  {key}: {len(combined):,} total signals across all assets")

        sim = simulate_portfolio(combined)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                trades_per_day = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(
                    daily, STARTING_CAPITAL, sim["n_trades"], trades_per_day,
                )
                results[key] = stats
                results[key]["n_trades"] = sim["n_trades"]
                results[key]["halted"] = sim["halted"]
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
    # Signal correlations
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("SIGNAL CORRELATIONS")
    print("=" * 80)

    correlations = compute_signal_correlations(all_approach_signals)
    for pair, corr in sorted(correlations.items()):
        if corr is not None:
            print(f"  {pair}: {corr:.4f}")
        else:
            print(f"  {pair}: insufficient overlap")

    # =========================================================================
    # L1 coefficients from approach B (which multi-TF features matter?)
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("L1 COEFFICIENTS — APPROACH B (22-feature model)")
    print("=" * 80)

    # Average the last retrain's coefficients across assets
    if all_coefficients:
        avg_coefs = {}
        for asset_name, month_coefs in all_coefficients.items():
            # Get the last retrain month
            last_month = sorted(month_coefs.keys())[-1]
            last_coefs = month_coefs[last_month]
            print(f"\n  {asset_name} (last retrain: {last_month}):")
            for feat_name, coef_val in sorted(last_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
                marker = " ***" if feat_name in MTF_FEATURES else ""
                print(f"    {feat_name:>25s}: {coef_val:+.6f}{marker}")
                avg_coefs.setdefault(feat_name, []).append(coef_val)

        print(f"\n  AVERAGE across assets (last retrain):")
        avg_final = {}
        for feat_name in sorted(avg_coefs.keys(), key=lambda x: abs(np.mean(avg_coefs[x])), reverse=True):
            mean_val = np.mean(avg_coefs[feat_name])
            marker = " ***" if feat_name in MTF_FEATURES else ""
            print(f"    {feat_name:>25s}: {mean_val:+.6f}{marker}")
            avg_final[feat_name] = round(float(mean_val), 6)
    else:
        avg_final = {}

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Approach':<35} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} {'Trades':>8} {'WinRate':>8} {'PSR%':>6}")
    print("-" * 85)
    for key in sorted(results.keys()):
        r = results[key]
        desc = APPROACH_DESCRIPTIONS.get(key, key)
        label = f"{key}: {desc}"
        print(f"{label:<35} {r.get('sharpe', 0):>8.3f} "
              f"{r.get('total_return_pct', 0):>8.2f}% "
              f"{r.get('max_dd_pct', 0):>7.2f}% "
              f"{r.get('n_trades', 0):>8} "
              f"{r.get('win_rate_days', 0) * 100:>7.1f}% "
              f"{r.get('psr_pct', 0):>5.1f}%")

    # Signal correlations vs A
    print(f"\n{'=' * 80}")
    print("CORRELATIONS VS APPROACH A (baseline)")
    print("=" * 80)
    for key in sorted(results.keys()):
        if key == "A":
            continue
        pair_key = f"A_vs_{key}"
        corr = correlations.get(pair_key)
        if corr is not None:
            print(f"  A vs {key}: {corr:.4f}")
        else:
            print(f"  A vs {key}: insufficient overlap")

    # =========================================================================
    # Save to JSON
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    output = {
        "meta": {
            "experiment": "multi_approach_comparison",
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(total_elapsed, 1),
            "assets": list(ASSETS.keys()),
            "n_assets": len(ASSETS),
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
            "features_14": FEATURES_14,
            "features_22": FEATURES_22,
        },
        "approach_descriptions": APPROACH_DESCRIPTIONS,
        "approaches": results,
        "per_asset": per_asset_results,
        "signal_correlations": correlations,
        "l1_coefficients_avg": avg_final,
        "l1_coefficients_per_asset": {
            asset: {
                month: {k: round(v, 6) for k, v in coefs.items()}
                for month, coefs in month_coefs.items()
            }
            for asset, month_coefs in all_coefficients.items()
        },
    }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/multi_approach_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Also save a compact summary
    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
