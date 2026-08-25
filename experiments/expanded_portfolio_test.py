"""
Expanded Portfolio Test — 16 crypto assets with Approach A and C at 15m.

Tests whether diversification across 16 assets improves risk-adjusted returns
compared to the original 6-asset portfolio. Compares:
  1. 6-asset portfolio with Approach C (4H regime filter) — baseline from multi_approach
  2. 16-asset portfolio with Approach C (4H regime filter) — expanded
  3. 16-asset portfolio with Approach A (no regime filter) — expanded baseline

Assets:
  Core 6:   BTC, ETH, XRP, BNB, SUI, LINK
  Unseen 10: SOL, ADA, AVAX, DOT, NEAR, DOGE, ARB, OP, UNI, AAVE

Portfolio params adjusted for 16 assets: MAX_PER_ASSET = 0.06, MAX_EXPOSURE = 0.80.

Walk-forward handles shorter histories gracefully (fewer test months for newer tokens).
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
# CONFIGURATION
# =============================================================================

CORE_ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "BNB": "BNBUSDT",
    "SUI": "SUIUSDT",
    "LINK": "LINKUSDT",
}

EXPANDED_ASSETS = {
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT",
    "DOT": "DOTUSDT",
    "NEAR": "NEARUSDT",
    "DOGE": "DOGEUSDT",
    "ARB": "ARBUSDT",
    "OP": "OPUSDT",
    "UNI": "UNIUSDT",
    "AAVE": "AAVEUSDT",
}

ALL_ASSETS = {**CORE_ASSETS, **EXPANDED_ASSETS}

DATA_DIRS = [
    Path("C:/Claude/ml-trading-system/data/binance-1m"),
    Path("C:/Claude/ml-trading-system/data/homeserver-mirror/binance-1m"),
]

# Fixed model — no tuning
MODEL_C = 0.1
MODEL_PENALTY = "l1"
MODEL_SOLVER = "saga"
MODEL_MAX_ITER = 1000
MODEL_SEED = 42

# Portfolio parameters — ORIGINAL (6-asset)
STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001  # 1 bps
MAX_EXPOSURE_6 = 0.60
MAX_PER_ASSET_6 = 0.20
BASE_SIZING_PCT = 0.02
MAX_SIGNAL_MULT = 2.0

# Portfolio parameters — EXPANDED (16-asset)
MAX_EXPOSURE_16 = 0.80
MAX_PER_ASSET_16 = 0.06

# Filters
VOL_FILTER_QUANTILE = 0.50

# Winsorization
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Signal discretization
SIGNAL_STEP_SIZE = 0.1

# Feature list
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
# DATA LOADING — Binance 1m CSVs (checks multiple directories)
# =============================================================================

def load_binance_1m(symbol: str, data_dirs: List[Path]) -> pd.DataFrame:
    """Load all monthly CSV files for a symbol from Binance 1m data.

    Checks multiple data directories and uses whichever has data.
    Handles both header and no-header formats.

    Returns OHLCV DataFrame with UTC DatetimeIndex.
    """
    col_names = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_vol", "taker_buy_quote_vol", "ignore",
    ]

    files = []
    used_dir = None
    for data_dir in data_dirs:
        symbol_dir = data_dir / symbol
        pattern = f"{symbol}-1m-*.csv"
        found = sorted(symbol_dir.glob(pattern))
        if not found:
            found = sorted(data_dir.glob(pattern))
        if found:
            files = found
            used_dir = data_dir
            break

    if not files:
        raise FileNotFoundError(
            f"No files matching {symbol}-1m-*.csv in any of: "
            + ", ".join(str(d / symbol) for d in data_dirs)
        )

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

    dir_label = "primary" if used_dir == data_dirs[0] else "mirror"
    print(f"  Loaded {symbol} ({dir_label}): {len(ohlcv):,} bars, "
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

    Returns all signals with their probabilities (no confidence filter applied).
    Handles shorter histories gracefully (fewer test months).
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
        print(f"    {asset_name}: insufficient history ({(end_date - start_date).days} days < 180)")
        return pd.DataFrame()

    # Generate monthly periods from actual data range
    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame()

    # OPTIMIZATION: For fine timeframes, retrain quarterly (3x fewer model fits)
    retrain_every = 1
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
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name
    return df


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
    directions = df["direction"].values
    mask = (directions * signal_regimes) > 0  # same sign = agree

    return df[mask].reset_index(drop=True)


# =============================================================================
# PORTFOLIO SIMULATION (parameterized exposure limits)
# =============================================================================

def simulate_portfolio(
    signals_df: pd.DataFrame,
    max_exposure: float,
    max_per_asset: float,
) -> Optional[Dict]:
    """Portfolio simulation with shared capital pool.

    Same logic as multi_approach_comparison: probability-based sizing, exposure limits,
    dynamic deleveraging, DD halt. Accepts parameterized exposure limits.
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

        # Exposure limits (parameterized)
        asset = assets_arr[i]
        asset_exp = asset_exposure.get(asset, 0.0)

        if asset_exp + raw_size > equity * max_per_asset:
            raw_size = max(equity * max_per_asset - asset_exp, 0.0)

        if current_exposure + raw_size > equity * max_exposure:
            raw_size = max(equity * max_exposure - current_exposure, 0.0)

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
            current_exposure + raw_size * 0.1, equity * max_exposure
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
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EXPANDED PORTFOLIO TEST")
    print("16 assets: Approach A (baseline) vs Approach C (4H regime filter)")
    print("Comparison: 6-asset C vs 16-asset C vs 16-asset A")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()

    # Storage for signals: per-asset raw signals + 4H OHLCV for regime filter
    asset_raw_signals: Dict[str, pd.DataFrame] = {}
    asset_ohlcv_4h: Dict[str, pd.DataFrame] = {}
    per_asset_info: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # Phase 1: Load data and generate raw signals for ALL 16 assets
    # =========================================================================
    print("PHASE 1: Signal generation for 16 assets")
    print("=" * 80)

    for asset_name, symbol in ALL_ASSETS.items():
        print(f"\n{'-' * 60}")
        print(f"Processing {asset_name} ({symbol})...")
        asset_start = time.time()

        # Load 1m data (check both directories)
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        # Resample to 15m and 4H
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
        print(f"  15m bars: {len(ohlcv_15m):,}, 4H bars: {len(ohlcv_4h):,}")

        # Compute 15m features
        feat_15m = compute_features(ohlcv_15m)

        # Generate raw signals (15m, 14 features, fwd_bars=4)
        print(f"  Generating raw signals (15m, 14 features)...")
        t0 = time.time()
        raw_signals = generate_raw_signals(
            ohlcv_15m, feat_15m, fwd_bars=4,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        elapsed = time.time() - t0

        if raw_signals is not None and len(raw_signals) > 0:
            asset_raw_signals[asset_name] = raw_signals
            asset_ohlcv_4h[asset_name] = ohlcv_4h
            print(f"    -> {len(raw_signals):,} raw signals in {elapsed:.1f}s")
        else:
            print(f"    -> No signals generated (insufficient history?) in {elapsed:.1f}s")
            continue

        # Store per-asset info
        per_asset_info[asset_name] = {
            "symbol": symbol,
            "n_1m_bars": len(ohlcv_1m),
            "n_15m_bars": len(ohlcv_15m),
            "date_range": f"{ohlcv_1m.index.min().date()} to {ohlcv_1m.index.max().date()}",
            "n_raw_signals": len(raw_signals),
            "is_core": asset_name in CORE_ASSETS,
        }

        total_asset_time = time.time() - asset_start
        print(f"  {asset_name} complete in {total_asset_time:.1f}s")

    # =========================================================================
    # Phase 2: Apply filters and build portfolio signals
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PHASE 2: Apply filters and build portfolios")
    print("=" * 80)

    # Build 3 sets of signals:
    # 1. 6-asset C (core only, regime filter)
    # 2. 16-asset C (all, regime filter)
    # 3. 16-asset A (all, percentile-70 confidence filter, no regime)

    signals_6_C = []    # 6-asset with approach C
    signals_16_C = []   # 16-asset with approach C
    signals_16_A = []   # 16-asset with approach A

    per_asset_approach_results: Dict[str, Dict[str, Any]] = {}

    for asset_name in sorted(asset_raw_signals.keys()):
        raw = asset_raw_signals[asset_name]
        ohlcv_4h = asset_ohlcv_4h[asset_name]
        is_core = asset_name in CORE_ASSETS

        # Approach A: percentile-70 confidence filter
        sig_A = apply_confidence_filter(raw, "percentile", 70)

        # Approach C: A + 4H regime filter
        sig_C = apply_regime_filter(sig_A, ohlcv_4h)

        # Per-asset stats
        asset_results = {
            "n_raw": len(raw),
            "n_A": len(sig_A),
            "n_C": len(sig_C),
            "is_core": is_core,
        }

        # Compute per-asset Sharpe/return/trades for approach A
        if len(sig_A) >= 10:
            sim_A = simulate_portfolio(sig_A, MAX_PER_ASSET_6, MAX_EXPOSURE_6)
            if sim_A and sim_A["n_trades"] >= 10:
                daily_A = compute_daily_pnl_series(sim_A)
                if daily_A is not None:
                    n_days_A = len(daily_A)
                    tpd_A = sim_A["n_trades"] / max(n_days_A, 1)
                    stats_A = compute_full_stats(daily_A, STARTING_CAPITAL, sim_A["n_trades"], tpd_A)
                    asset_results["A_sharpe"] = stats_A.get("sharpe", 0)
                    asset_results["A_return_pct"] = stats_A.get("total_return_pct", 0)
                    asset_results["A_trades"] = sim_A["n_trades"]

        # Compute per-asset Sharpe/return/trades for approach C
        if len(sig_C) >= 10:
            sim_C = simulate_portfolio(sig_C, MAX_PER_ASSET_6, MAX_EXPOSURE_6)
            if sim_C and sim_C["n_trades"] >= 10:
                daily_C = compute_daily_pnl_series(sim_C)
                if daily_C is not None:
                    n_days_C = len(daily_C)
                    tpd_C = sim_C["n_trades"] / max(n_days_C, 1)
                    stats_C = compute_full_stats(daily_C, STARTING_CAPITAL, sim_C["n_trades"], tpd_C)
                    asset_results["C_sharpe"] = stats_C.get("sharpe", 0)
                    asset_results["C_return_pct"] = stats_C.get("total_return_pct", 0)
                    asset_results["C_trades"] = sim_C["n_trades"]

        per_asset_approach_results[asset_name] = asset_results

        # Collect for portfolio simulations
        if len(sig_A) > 0:
            signals_16_A.append(sig_A)
        if len(sig_C) > 0:
            signals_16_C.append(sig_C)
            if is_core:
                signals_6_C.append(sig_C)

        tag = "CORE" if is_core else "NEW "
        print(f"  [{tag}] {asset_name}: raw={len(raw):,}, A={len(sig_A):,}, C={len(sig_C):,}")

    # =========================================================================
    # Phase 3: Portfolio simulations
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PHASE 3: Portfolio simulations")
    print("=" * 80)

    portfolio_results: Dict[str, Dict[str, Any]] = {}

    # --- Portfolio 1: 6-asset C (original params) ---
    label = "6-asset C (regime filter)"
    if signals_6_C:
        combined_6C = pd.concat(signals_6_C).sort_values("timestamp")
        print(f"\n  {label}: {len(combined_6C):,} total signals")
        sim = simulate_portfolio(combined_6C, MAX_EXPOSURE_6, MAX_PER_ASSET_6)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                tpd = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(daily, STARTING_CAPITAL, sim["n_trades"], tpd)
                stats["n_trades"] = sim["n_trades"]
                stats["halted"] = sim["halted"]
                stats["n_assets"] = 6
                stats["max_exposure"] = MAX_EXPOSURE_6
                stats["max_per_asset"] = MAX_PER_ASSET_6
                portfolio_results["6_asset_C"] = stats
                print(f"    -> Sharpe={stats['sharpe']:.3f}, Return={stats['total_return_pct']:.2f}%, "
                      f"MaxDD={stats['max_dd_pct']:.2f}%, Trades={sim['n_trades']}")
            else:
                print(f"    -> insufficient daily PnL data")
        else:
            print(f"    -> insufficient trades")
    else:
        print(f"\n  {label}: no signals")

    # --- Portfolio 2: 16-asset C (expanded params) ---
    label = "16-asset C (regime filter)"
    if signals_16_C:
        combined_16C = pd.concat(signals_16_C).sort_values("timestamp")
        print(f"\n  {label}: {len(combined_16C):,} total signals")
        sim = simulate_portfolio(combined_16C, MAX_EXPOSURE_16, MAX_PER_ASSET_16)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                tpd = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(daily, STARTING_CAPITAL, sim["n_trades"], tpd)
                stats["n_trades"] = sim["n_trades"]
                stats["halted"] = sim["halted"]
                stats["n_assets"] = len(signals_16_C)
                stats["max_exposure"] = MAX_EXPOSURE_16
                stats["max_per_asset"] = MAX_PER_ASSET_16
                portfolio_results["16_asset_C"] = stats
                print(f"    -> Sharpe={stats['sharpe']:.3f}, Return={stats['total_return_pct']:.2f}%, "
                      f"MaxDD={stats['max_dd_pct']:.2f}%, Trades={sim['n_trades']}")
            else:
                print(f"    -> insufficient daily PnL data")
        else:
            print(f"    -> insufficient trades")
    else:
        print(f"\n  {label}: no signals")

    # --- Portfolio 3: 16-asset A (expanded params, no regime filter) ---
    label = "16-asset A (no regime filter)"
    if signals_16_A:
        combined_16A = pd.concat(signals_16_A).sort_values("timestamp")
        print(f"\n  {label}: {len(combined_16A):,} total signals")
        sim = simulate_portfolio(combined_16A, MAX_EXPOSURE_16, MAX_PER_ASSET_16)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                tpd = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(daily, STARTING_CAPITAL, sim["n_trades"], tpd)
                stats["n_trades"] = sim["n_trades"]
                stats["halted"] = sim["halted"]
                stats["n_assets"] = len(signals_16_A)
                stats["max_exposure"] = MAX_EXPOSURE_16
                stats["max_per_asset"] = MAX_PER_ASSET_16
                portfolio_results["16_asset_A"] = stats
                print(f"    -> Sharpe={stats['sharpe']:.3f}, Return={stats['total_return_pct']:.2f}%, "
                      f"MaxDD={stats['max_dd_pct']:.2f}%, Trades={sim['n_trades']}")
            else:
                print(f"    -> insufficient daily PnL data")
        else:
            print(f"    -> insufficient trades")
    else:
        print(f"\n  {label}: no signals")

    # =========================================================================
    # Phase 4: Summary tables
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PER-ASSET RESULTS")
    print("=" * 80)
    print(f"{'Asset':<8} {'Type':<6} {'Signals':>8} "
          f"{'A Sharpe':>9} {'A Ret%':>8} {'A Trades':>9} "
          f"{'C Sharpe':>9} {'C Ret%':>8} {'C Trades':>9}")
    print("-" * 95)

    for asset_name in sorted(per_asset_approach_results.keys()):
        r = per_asset_approach_results[asset_name]
        atype = "CORE" if r.get("is_core") else "NEW"
        a_sharpe = r.get("A_sharpe", "-")
        a_ret = r.get("A_return_pct", "-")
        a_trades = r.get("A_trades", "-")
        c_sharpe = r.get("C_sharpe", "-")
        c_ret = r.get("C_return_pct", "-")
        c_trades = r.get("C_trades", "-")

        a_sharpe_s = f"{a_sharpe:>9.3f}" if isinstance(a_sharpe, (int, float)) else f"{a_sharpe:>9}"
        a_ret_s = f"{a_ret:>7.2f}%" if isinstance(a_ret, (int, float)) else f"{a_ret:>8}"
        a_trades_s = f"{a_trades:>9}" if isinstance(a_trades, int) else f"{a_trades:>9}"
        c_sharpe_s = f"{c_sharpe:>9.3f}" if isinstance(c_sharpe, (int, float)) else f"{c_sharpe:>9}"
        c_ret_s = f"{c_ret:>7.2f}%" if isinstance(c_ret, (int, float)) else f"{c_ret:>8}"
        c_trades_s = f"{c_trades:>9}" if isinstance(c_trades, int) else f"{c_trades:>9}"

        print(f"{asset_name:<8} {atype:<6} {r['n_raw']:>8} "
              f"{a_sharpe_s} {a_ret_s} {a_trades_s} "
              f"{c_sharpe_s} {c_ret_s} {c_trades_s}")

    print(f"\n{'=' * 80}")
    print("PORTFOLIO-LEVEL COMPARISON")
    print("=" * 80)
    print(f"{'Portfolio':<30} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
          f"{'Trades':>8} {'WinRate':>8} {'PSR%':>6} {'Sortino':>8}")
    print("-" * 95)

    for key in ["6_asset_C", "16_asset_C", "16_asset_A"]:
        if key not in portfolio_results:
            label_map = {
                "6_asset_C": "6-asset C (regime)",
                "16_asset_C": "16-asset C (regime)",
                "16_asset_A": "16-asset A (baseline)",
            }
            print(f"{label_map[key]:<30} {'N/A':>8} {'N/A':>9} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>8}")
            continue

        r = portfolio_results[key]
        label_map = {
            "6_asset_C": f"6-asset C (regime)",
            "16_asset_C": f"16-asset C (regime, {r.get('n_assets', '?')}a)",
            "16_asset_A": f"16-asset A (baseline, {r.get('n_assets', '?')}a)",
        }
        label = label_map.get(key, key)
        print(f"{label:<30} {r.get('sharpe', 0):>8.3f} "
              f"{r.get('total_return_pct', 0):>8.2f}% "
              f"{r.get('max_dd_pct', 0):>7.2f}% "
              f"{r.get('n_trades', 0):>8} "
              f"{r.get('win_rate_days', 0) * 100:>7.1f}% "
              f"{r.get('psr_pct', 0):>5.1f}% "
              f"{r.get('sortino', 0):>8.3f}")

    # Diversification benefit analysis
    if "6_asset_C" in portfolio_results and "16_asset_C" in portfolio_results:
        r6 = portfolio_results["6_asset_C"]
        r16 = portfolio_results["16_asset_C"]
        print(f"\n  Diversification impact (16-asset C vs 6-asset C):")
        print(f"    Sharpe delta:  {r16['sharpe'] - r6['sharpe']:+.3f}")
        print(f"    Return delta:  {r16['total_return_pct'] - r6['total_return_pct']:+.2f}%")
        print(f"    MaxDD delta:   {r16['max_dd_pct'] - r6['max_dd_pct']:+.2f}%")
        print(f"    Trade delta:   {r16['n_trades'] - r6['n_trades']:+d}")

    if "16_asset_C" in portfolio_results and "16_asset_A" in portfolio_results:
        rC = portfolio_results["16_asset_C"]
        rA = portfolio_results["16_asset_A"]
        print(f"\n  Regime filter impact (16-asset C vs 16-asset A):")
        print(f"    Sharpe delta:  {rC['sharpe'] - rA['sharpe']:+.3f}")
        print(f"    Return delta:  {rC['total_return_pct'] - rA['total_return_pct']:+.2f}%")
        print(f"    MaxDD delta:   {rC['max_dd_pct'] - rA['max_dd_pct']:+.2f}%")
        print(f"    Trade delta:   {rC['n_trades'] - rA['n_trades']:+d}")

    # =========================================================================
    # Save to JSON
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    output = {
        "meta": {
            "experiment": "expanded_portfolio_test",
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(total_elapsed, 1),
            "core_assets": list(CORE_ASSETS.keys()),
            "expanded_assets": list(EXPANDED_ASSETS.keys()),
            "all_assets": list(ALL_ASSETS.keys()),
            "n_core": len(CORE_ASSETS),
            "n_expanded": len(EXPANDED_ASSETS),
            "n_total": len(ALL_ASSETS),
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
            "6_asset_max_exposure": MAX_EXPOSURE_6,
            "6_asset_max_per_asset": MAX_PER_ASSET_6,
            "16_asset_max_exposure": MAX_EXPOSURE_16,
            "16_asset_max_per_asset": MAX_PER_ASSET_16,
            "base_sizing_pct": BASE_SIZING_PCT,
            "max_signal_mult": MAX_SIGNAL_MULT,
            "vol_filter_quantile": VOL_FILTER_QUANTILE,
            "winsor_lo": WINSOR_LO,
            "winsor_hi": WINSOR_HI,
            "signal_step_size": SIGNAL_STEP_SIZE,
            "features_14": FEATURES_14,
        },
        "portfolios": portfolio_results,
        "per_asset": per_asset_approach_results,
        "per_asset_info": per_asset_info,
    }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/expanded_portfolio_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
