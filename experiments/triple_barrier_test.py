"""
Triple-Barrier Labeling Test — Compare 3 labeling approaches on 6 crypto assets at 15m.

Tests whether AFML triple-barrier labeling produces better training signals
than simple fixed-horizon labeling. The key insight: triple-barrier changes
ONLY the training labels. Portfolio simulation still uses actual forward returns.

Approaches:
  1. Fixed-Horizon: y = (fwd_ret > 0).astype(int)  [current baseline]
  2. Triple-Barrier [1,1]: symmetric profit-take / stop-loss at 1x vol
  3. Triple-Barrier [2,1]: asymmetric — let winners run (2x vol PT, 1x vol SL)

All approaches share:
  - Same walk-forward methodology (monthly expanding, embargo, sample weights)
  - Same portfolio simulation (shared capital, exposure limits, DD halt)
  - Same slippage model (1 bps)
  - Same 14-feature set at 15m
  - Same LogReg L1 model (C=0.1, saga solver)
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

# Feature list (14 universal features)
FEATURES_14 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]

# Triple-barrier configs
FWD_BARS = 4  # 4 bars ahead at 15m = 1 hour
VOL_SPAN = 50  # EWM span for volatility estimate
VOL_MIN_PERIODS = 20

LABELING_APPROACHES = {
    "Fixed-horizon": {"type": "fixed"},
    "Triple-barrier [1,1]": {"type": "triple_barrier", "pt_sl": [1.0, 1.0]},
    "Triple-barrier [2,1]": {"type": "triple_barrier", "pt_sl": [2.0, 1.0]},
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
    """Bollinger %B -- position within Bollinger Bands."""
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
    """Swing high/low detection -- backward-looking only."""
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
# SAMPLE WEIGHTS -- AFML Ch 4: Average Uniqueness
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
    loss_days = (daily_ret < 0).sum()
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
# DATA LOADING -- Binance 1m CSVs
# =============================================================================

def load_binance_1m(symbol: str, data_dir: Path) -> pd.DataFrame:
    """Load all monthly CSV files for a symbol from Binance 1m data."""
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
# TRIPLE-BARRIER LABELING (from src/afml/labeling.py, inlined for self-containment)
# =============================================================================

def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int,
) -> pd.Series:
    """Add vertical (time) barriers to events.

    For each event timestamp, find the bar index `num_bars` ahead.
    If the series ends before num_bars, use the last available bar.
    """
    t1 = close.index.searchsorted(t_events) + num_bars
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[: len(t1)])
    return t1


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: list,
) -> pd.DataFrame:
    """Find the first barrier touch (profit-take, stop-loss, or vertical).

    Vectorized where possible, with a loop fallback for correctness.
    """
    out = events[["t1"]].copy()
    nat_series = pd.Series(pd.NaT, index=events.index, dtype=events["t1"].dtype)
    out["pt"] = nat_series.copy()
    out["sl"] = nat_series.copy()

    if "side" not in events.columns:
        events = events.copy()
        events["side"] = 1.0

    for loc, row in events.iterrows():
        t1 = row["t1"]
        if pd.isna(t1):
            t1 = close.index[-1]

        path = close.loc[loc:t1]
        if len(path) < 2:
            continue

        path_returns = (path / close.loc[loc] - 1) * row["side"]

        if pt_sl[0] > 0:
            pt_threshold = pt_sl[0] * row["trgt"]
            pt_hits = path_returns[path_returns > pt_threshold]
            if len(pt_hits) > 0:
                out.at[loc, "pt"] = pt_hits.index[0]

        if pt_sl[1] > 0:
            sl_threshold = -pt_sl[1] * row["trgt"]
            sl_hits = path_returns[path_returns < sl_threshold]
            if len(sl_hits) > 0:
                out.at[loc, "sl"] = sl_hits.index[0]

    return out


def get_triple_barrier_labels(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list,
    target: pd.Series,
    num_bars_vertical: int,
) -> pd.Series:
    """Compute triple-barrier binary labels for given events.

    Orchestrates the full triple-barrier pipeline:
    1. Build events DataFrame with volatility targets
    2. Add vertical barriers
    3. Find first barrier touches (PT, SL, vertical)
    4. Convert to binary labels: 1 if PT first, 0 if SL first,
       sign(return) mapped to {0,1} if vertical hit

    Returns:
        Series of {0, 1} labels indexed by event timestamps.
        Events that cannot be labeled (e.g., near end of data) are dropped.
    """
    # Filter events to those in the target index
    target_aligned = target.reindex(t_events).dropna()
    t_events_clean = target_aligned.index

    if len(t_events_clean) == 0:
        return pd.Series(dtype=float)

    # Build events DataFrame
    events = pd.DataFrame(index=t_events_clean)
    events["trgt"] = target_aligned
    events["side"] = 1.0  # Standard labeling (not meta-labeling)

    # Add vertical barrier
    vertical = add_vertical_barrier(t_events_clean, close, num_bars_vertical)
    events["t1"] = vertical.reindex(t_events_clean)

    # Drop events where vertical barrier could not be set (near end of data)
    events = events.dropna(subset=["t1"])

    if len(events) == 0:
        return pd.Series(dtype=float)

    # Apply PT/SL barriers
    barriers = apply_pt_sl_on_t1(close, events, pt_sl)

    # Determine which barrier was hit first and assign labels
    # Vectorized approach: compare timestamps
    pt_times = barriers["pt"]
    sl_times = barriers["sl"]
    t1_times = barriers["t1"]  # vertical barrier from events

    labels = pd.Series(index=barriers.index, dtype=float)

    for idx in barriers.index:
        pt_t = pt_times.loc[idx]
        sl_t = sl_times.loc[idx]
        vb_t = t1_times.loc[idx]

        # Collect non-NaT barrier times
        candidates = {}
        if pd.notna(pt_t):
            candidates["pt"] = pt_t
        if pd.notna(sl_t):
            candidates["sl"] = sl_t
        if pd.notna(vb_t):
            candidates["vb"] = vb_t
        else:
            # Fallback: use last available close
            candidates["vb"] = close.index[-1]

        if not candidates:
            labels[idx] = np.nan
            continue

        # Which barrier was touched first?
        first_barrier = min(candidates, key=candidates.get)

        if first_barrier == "pt":
            labels[idx] = 1.0
        elif first_barrier == "sl":
            labels[idx] = 0.0
        else:
            # Vertical barrier hit: label by sign of return
            exit_time = candidates["vb"]
            if exit_time in close.index:
                ret = close.loc[exit_time] / close.loc[idx] - 1
            else:
                # Find nearest available close
                nearest_idx = close.index.searchsorted(exit_time)
                nearest_idx = min(nearest_idx, len(close) - 1)
                ret = close.iloc[nearest_idx] / close.loc[idx] - 1
            labels[idx] = 1.0 if ret > 0 else 0.0

    return labels.dropna()


# =============================================================================
# WALK-FORWARD SIGNAL GENERATION -- Fixed-Horizon (baseline)
# =============================================================================

def generate_fixed_horizon_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Walk-forward with fixed-horizon labeling (current baseline).

    Returns (signals_df, diagnostics_dict).
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for vol filter
    vol = close.pct_change().dropna().ewm(span=VOL_SPAN, min_periods=VOL_MIN_PERIODS).std()

    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    if (end_date - start_date).days < 180:
        return pd.DataFrame(), {"label_balance": None, "train_sizes": [], "filtered_events": 0}

    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame(), {"label_balance": None, "train_sizes": [], "filtered_events": 0}

    # Retrain optimization for large datasets
    retrain_every = 1
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    if len(ohlcv) > 1:
        median_gap = ohlcv.index.to_series().diff().dropna().median()
        embargo_td = median_gap * fwd_bars
    else:
        embargo_td = pd.Timedelta(hours=2)

    rows = []
    cached_model = None
    cached_scaler = None
    last_retrain_idx = -999
    all_train_labels = []
    train_sizes = []

    for month_idx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        embargo_cutoff = ms - embargo_td
        train_cutoff = embargo_cutoff

        feat_cols = [c for c in feature_cols if c in features.columns]
        if len(feat_cols) < 8:
            continue

        feat = features[feat_cols].copy()

        # Per-window winsorization
        train_mask = feat.index < train_cutoff
        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                train_vals = feat.loc[train_mask, col]
                if len(train_vals.dropna()) > 10:
                    lo = train_vals.quantile(WINSOR_LO)
                    hi = train_vals.quantile(WINSOR_HI)
                    feat[col] = feat[col].clip(lo, hi)

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 200 or len(te) < 10:
            continue

        # Vol filter
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

        all_train_labels.extend(y_train.values.tolist())
        train_sizes.append(len(tr))

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
        return pd.DataFrame(), {"label_balance": None, "train_sizes": train_sizes, "filtered_events": 0}

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name

    # Diagnostics
    label_arr = np.array(all_train_labels)
    label_balance = float(label_arr.mean()) if len(label_arr) > 0 else None

    diagnostics = {
        "label_balance": label_balance,
        "train_sizes": train_sizes,
        "filtered_events": 0,
        "total_train_labels": len(label_arr),
    }

    return df, diagnostics


# =============================================================================
# WALK-FORWARD SIGNAL GENERATION -- Triple-Barrier Labeling
# =============================================================================

def generate_triple_barrier_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
    feature_cols: List[str],
    pt_sl: List[float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Walk-forward with triple-barrier labeling instead of fixed-horizon.

    The triple-barrier changes ONLY the TRAINING LABELS.
    The portfolio simulation still uses actual forward returns to compute PnL.
    We are testing whether training on triple-barrier labels produces better
    signals, evaluated on the same PnL metric.

    Args:
        ohlcv: OHLCV DataFrame.
        features: Feature DataFrame.
        fwd_bars: Forward bars for vertical barrier + fixed-horizon return eval.
        asset_name: Asset identifier.
        feature_cols: Feature column names.
        pt_sl: [profit_take_multiplier, stop_loss_multiplier] for barriers.

    Returns (signals_df, diagnostics_dict).
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for both barrier thresholds AND vol filter
    vol = close.pct_change().dropna().ewm(span=VOL_SPAN, min_periods=VOL_MIN_PERIODS).std()

    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    if (end_date - start_date).days < 180:
        return pd.DataFrame(), {"label_balance": None, "train_sizes": [], "filtered_events": 0}

    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame(), {"label_balance": None, "train_sizes": [], "filtered_events": 0}

    retrain_every = 1
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    if len(ohlcv) > 1:
        median_gap = ohlcv.index.to_series().diff().dropna().median()
        embargo_td = median_gap * fwd_bars
    else:
        embargo_td = pd.Timedelta(hours=2)

    # Pre-compute triple-barrier labels for the ENTIRE close series.
    # This is expensive but done once. Each walk-forward window then
    # simply slices the pre-computed labels.
    print(f"    Computing triple-barrier labels (pt_sl={pt_sl})...")
    tb_start = time.time()

    # Use all bar timestamps as events (every bar is a potential trade entry)
    all_events = close.index
    tb_labels = get_triple_barrier_labels(
        close=close,
        t_events=all_events,
        pt_sl=pt_sl,
        target=vol,
        num_bars_vertical=fwd_bars,
    )
    tb_elapsed = time.time() - tb_start
    n_total_events = len(all_events)
    n_labeled = len(tb_labels)
    n_filtered = n_total_events - n_labeled
    print(f"    -> {n_labeled:,} labeled / {n_total_events:,} events "
          f"({n_filtered:,} filtered) in {tb_elapsed:.1f}s")

    if len(tb_labels) == 0:
        return pd.DataFrame(), {"label_balance": None, "train_sizes": [], "filtered_events": n_filtered}

    rows = []
    cached_model = None
    cached_scaler = None
    last_retrain_idx = -999
    all_train_labels = []
    train_sizes = []

    for month_idx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        embargo_cutoff = ms - embargo_td
        train_cutoff = embargo_cutoff

        feat_cols = [c for c in feature_cols if c in features.columns]
        if len(feat_cols) < 8:
            continue

        feat = features[feat_cols].copy()

        # Per-window winsorization
        train_mask = feat.index < train_cutoff
        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                train_vals = feat.loc[train_mask, col]
                if len(train_vals.dropna()) > 10:
                    lo = train_vals.quantile(WINSOR_LO)
                    hi = train_vals.quantile(WINSOR_HI)
                    feat[col] = feat[col].clip(lo, hi)

        fc = feat.dropna()

        # Use triple-barrier labels instead of fixed-horizon
        y_tb = tb_labels.reindex(fc.index).dropna()
        # Also need fwd_ret for portfolio simulation on the test set
        fwd_ret_aligned = fwd_ret.reindex(fc.index).dropna()

        # Valid indices: must have BOTH triple-barrier label AND fwd_ret
        valid_train = fc.index.intersection(y_tb.index)
        valid_test_candidates = fc.index.intersection(fwd_ret_aligned.index)

        tr = valid_train[valid_train < train_cutoff]
        te = valid_test_candidates[(valid_test_candidates >= ms) & (valid_test_candidates <= me)]

        if len(tr) < 200 or len(te) < 10:
            continue

        # Vol filter on test set (same as baseline)
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
        y_train = y_tb.loc[tr].astype(int)  # Triple-barrier binary labels
        X_test = fc.loc[te]

        all_train_labels.extend(y_train.values.tolist())
        train_sizes.append(len(tr))

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

        # Signal generation (same as baseline — using predicted probabilities)
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
        return pd.DataFrame(), {
            "label_balance": None,
            "train_sizes": train_sizes,
            "filtered_events": n_filtered,
            "total_train_labels": len(all_train_labels),
        }

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name

    # Diagnostics
    label_arr = np.array(all_train_labels)
    label_balance = float(label_arr.mean()) if len(label_arr) > 0 else None

    diagnostics = {
        "label_balance": label_balance,
        "train_sizes": train_sizes,
        "filtered_events": n_filtered,
        "total_train_labels": len(label_arr),
    }

    return df, diagnostics


# =============================================================================
# PORTFOLIO SIMULATION (same as multi_approach_comparison.py)
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

        if dd > DELEVER_START:
            dd_ratio = (dd - DELEVER_START) / (MAX_DD - DELEVER_START)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        if use_prob:
            sig_abs = abs(signal_strengths[i])
            raw_size = equity * BASE_SIZING_PCT * sig_abs * size_mult
            raw_size = min(raw_size, equity * BASE_SIZING_PCT * MAX_SIGNAL_MULT * size_mult)
        else:
            raw_size = equity * BASE_SIZING_PCT * size_mult

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
# CONFIDENCE FILTER (percentile 70, same as baseline approach A)
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
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TRIPLE-BARRIER LABELING TEST")
    print("3 labeling approaches x 6 assets @ 15m")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Approaches:")
    for name, cfg in LABELING_APPROACHES.items():
        if cfg["type"] == "fixed":
            print(f"  {name}: y = (fwd_ret > 0).astype(int)")
        else:
            print(f"  {name}: PT={cfg['pt_sl'][0]}x vol, SL={cfg['pt_sl'][1]}x vol, VB={FWD_BARS} bars")
    print()

    total_start = time.time()

    # Collect signals per approach across all assets
    all_signals: Dict[str, List[pd.DataFrame]] = {name: [] for name in LABELING_APPROACHES}
    all_diagnostics: Dict[str, Dict[str, Any]] = {name: {} for name in LABELING_APPROACHES}

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {asset_name} ({symbol})...")
        asset_start = time.time()

        # Load and resample to 15m
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        feat_15m = compute_features(ohlcv_15m)
        print(f"  15m bars: {len(ohlcv_15m):,}")

        # --- Approach 1: Fixed-Horizon ---
        approach_name = "Fixed-horizon"
        print(f"\n  [{approach_name}]")
        t0 = time.time()
        sig_fixed, diag_fixed = generate_fixed_horizon_signals(
            ohlcv_15m, feat_15m, fwd_bars=FWD_BARS,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        # Apply percentile-70 confidence filter (same as approach A baseline)
        sig_fixed_filtered = apply_confidence_filter(sig_fixed, "percentile", 70)
        elapsed = time.time() - t0
        print(f"    -> {len(sig_fixed):,} raw -> {len(sig_fixed_filtered):,} filtered signals in {elapsed:.1f}s")
        if diag_fixed["label_balance"] is not None:
            print(f"    -> Label balance: {diag_fixed['label_balance']:.3f} (fraction positive)")
        all_signals[approach_name].append(sig_fixed_filtered)
        all_diagnostics[approach_name][asset_name] = diag_fixed

        # --- Approach 2: Triple-Barrier [1,1] ---
        approach_name = "Triple-barrier [1,1]"
        cfg = LABELING_APPROACHES[approach_name]
        print(f"\n  [{approach_name}]")
        t0 = time.time()
        sig_tb11, diag_tb11 = generate_triple_barrier_signals(
            ohlcv_15m, feat_15m, fwd_bars=FWD_BARS,
            asset_name=asset_name, feature_cols=FEATURES_14,
            pt_sl=cfg["pt_sl"],
        )
        sig_tb11_filtered = apply_confidence_filter(sig_tb11, "percentile", 70)
        elapsed = time.time() - t0
        print(f"    -> {len(sig_tb11):,} raw -> {len(sig_tb11_filtered):,} filtered signals in {elapsed:.1f}s")
        if diag_tb11["label_balance"] is not None:
            print(f"    -> Label balance: {diag_tb11['label_balance']:.3f} (fraction positive)")
        print(f"    -> Events filtered (no barrier): {diag_tb11['filtered_events']:,}")
        all_signals[approach_name].append(sig_tb11_filtered)
        all_diagnostics[approach_name][asset_name] = diag_tb11

        # --- Approach 3: Triple-Barrier [2,1] ---
        approach_name = "Triple-barrier [2,1]"
        cfg = LABELING_APPROACHES[approach_name]
        print(f"\n  [{approach_name}]")
        t0 = time.time()
        sig_tb21, diag_tb21 = generate_triple_barrier_signals(
            ohlcv_15m, feat_15m, fwd_bars=FWD_BARS,
            asset_name=asset_name, feature_cols=FEATURES_14,
            pt_sl=cfg["pt_sl"],
        )
        sig_tb21_filtered = apply_confidence_filter(sig_tb21, "percentile", 70)
        elapsed = time.time() - t0
        print(f"    -> {len(sig_tb21):,} raw -> {len(sig_tb21_filtered):,} filtered signals in {elapsed:.1f}s")
        if diag_tb21["label_balance"] is not None:
            print(f"    -> Label balance: {diag_tb21['label_balance']:.3f} (fraction positive)")
        print(f"    -> Events filtered (no barrier): {diag_tb21['filtered_events']:,}")
        all_signals[approach_name].append(sig_tb21_filtered)
        all_diagnostics[approach_name][asset_name] = diag_tb21

        elapsed_asset = time.time() - asset_start
        print(f"\n  {asset_name} complete in {elapsed_asset:.1f}s")

    # =========================================================================
    # Portfolio simulation per approach
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PORTFOLIO SIMULATIONS")
    print("=" * 80)

    results: Dict[str, Dict[str, Any]] = {}
    for approach_name in LABELING_APPROACHES:
        if not all_signals[approach_name]:
            print(f"  {approach_name}: no signals collected, skipping")
            continue

        combined = pd.concat(all_signals[approach_name]).sort_values("timestamp")
        print(f"  {approach_name}: {len(combined):,} total signals across all assets")

        sim = simulate_portfolio(combined)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                trades_per_day = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(
                    daily, STARTING_CAPITAL, sim["n_trades"], trades_per_day,
                )
                results[approach_name] = stats
                results[approach_name]["n_trades"] = sim["n_trades"]
                results[approach_name]["halted"] = sim["halted"]
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
    # Summary table
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print("=" * 80)
    header = (f"{'Labeling':<25} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
              f"{'Trades':>8} {'WinRate':>8} {'Label Bal':>10}")
    print(header)
    print("-" * len(header))

    for approach_name in LABELING_APPROACHES:
        if approach_name not in results:
            print(f"{approach_name:<25} {'--':>8} {'--':>9} {'--':>8} {'--':>8} {'--':>8} {'--':>10}")
            continue

        r = results[approach_name]

        # Compute aggregate label balance across all assets
        diags = all_diagnostics.get(approach_name, {})
        balances = [d["label_balance"] for d in diags.values() if d.get("label_balance") is not None]
        avg_balance = np.mean(balances) if balances else float("nan")

        print(f"{approach_name:<25} {r.get('sharpe', 0):>8.3f} "
              f"{r.get('total_return_pct', 0):>8.2f}% "
              f"{r.get('max_dd_pct', 0):>7.2f}% "
              f"{r.get('n_trades', 0):>8} "
              f"{r.get('win_rate_days', 0) * 100:>7.1f}% "
              f"{avg_balance:>9.1%}")

    # =========================================================================
    # Diagnostics summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("DIAGNOSTICS")
    print("=" * 80)

    for approach_name in LABELING_APPROACHES:
        diags = all_diagnostics.get(approach_name, {})
        if not diags:
            continue

        print(f"\n  {approach_name}:")
        for asset_name, d in diags.items():
            lb = d.get("label_balance")
            lb_str = f"{lb:.3f}" if lb is not None else "N/A"
            n_train = d.get("total_train_labels", sum(d.get("train_sizes", [])))
            n_filt = d.get("filtered_events", 0)
            print(f"    {asset_name}: label_balance={lb_str}, "
                  f"train_labels={n_train:,}, filtered_events={n_filt:,}")

    # =========================================================================
    # Save to JSON
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Prepare diagnostics for JSON (remove non-serializable items)
    diag_json = {}
    for approach_name, asset_diags in all_diagnostics.items():
        diag_json[approach_name] = {}
        for asset_name, d in asset_diags.items():
            diag_json[approach_name][asset_name] = {
                "label_balance": d.get("label_balance"),
                "total_train_labels": d.get("total_train_labels", sum(d.get("train_sizes", []))),
                "filtered_events": d.get("filtered_events", 0),
                "n_retrain_windows": len(d.get("train_sizes", [])),
            }

    output = {
        "meta": {
            "experiment": "triple_barrier_labeling_test",
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
            "fwd_bars": FWD_BARS,
            "vol_span": VOL_SPAN,
            "vol_min_periods": VOL_MIN_PERIODS,
            "features": FEATURES_14,
            "confidence_filter": "percentile-70",
        },
        "labeling_approaches": {
            name: cfg for name, cfg in LABELING_APPROACHES.items()
        },
        "results": results,
        "diagnostics": diag_json,
    }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/triple_barrier_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
