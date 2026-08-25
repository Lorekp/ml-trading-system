"""
Remaining Validation — Checklist items A10, B4, C3.

A10: Extended History (2022 Bear Market)
  Run baseline approach A (15m, 14 features, LogReg L1) on BTC and ETH using
  full 2020-2026 data. Report Sharpe per year, especially 2022 bear market.

B4: Monte Carlo Shuffle Test
  Shuffle direction column 1000 times, re-run portfolio simulation each time.
  Compute p-value: fraction of shuffled Sharpes that exceed actual Sharpe.

C3: Concurrent Position Handling Audit
  Analyze simulate_portfolio() for exposure limit enforcement.
  Find timestamps with multiple simultaneous signals, verify limits hold.
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
# CONFIGURATION — copied from multi_approach_comparison.py
# =============================================================================

CORE_ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "BNB": "BNBUSDT",
    "SUI": "SUIUSDT",
    "LINK": "LINKUSDT",
}

EXTENDED_ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}

CORE_DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
EXTENDED_DATA_DIR = Path("C:/Claude/ml-trading-system/data/homeserver-mirror/binance-1m")

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

# Feature list (14 features, approach A baseline)
FEATURES_14 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]

# Monte Carlo config
N_SHUFFLES = 1000


# =============================================================================
# INLINE TECHNICAL INDICATORS (copied from multi_approach_comparison.py)
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
    """AFML Ch 4: Average uniqueness for overlapping labels."""
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
# DATA LOADING
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
# RAW SIGNAL GENERATION — Walk-Forward
# =============================================================================

def generate_raw_signals(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    fwd_bars: int,
    asset_name: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Walk-forward monthly retrain with expanding window.

    Returns signals DataFrame with columns:
      timestamp, direction, fwd_return, confidence, probability,
      signal_strength, asset
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for vol filter
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

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

    for month_idx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)
            embargo_cutoff = ms - embargo_td
        else:
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
# CONFIDENCE FILTER
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
# PORTFOLIO SIMULATION (with audit hooks for C3)
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame,
                       audit: bool = False) -> Optional[Dict]:
    """Portfolio simulation with shared capital pool.

    If audit=True, also returns detailed per-trade audit information
    for concurrent position analysis.
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

    # Audit arrays
    if audit:
        skip_reasons = [""] * n  # Why was this signal skipped?
        exposure_at_signal = np.zeros(n)
        asset_exposure_at_signal = np.zeros(n)

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
            if audit:
                skip_reasons[i] = "halted"
                exposure_at_signal[i] = current_exposure
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        if dd >= MAX_DD:
            halted = True
            equity_arr[i] = equity
            dd_arr[i] = dd
            if audit:
                skip_reasons[i] = "dd_halt"
                exposure_at_signal[i] = current_exposure
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

        original_size = raw_size

        # Exposure limits
        asset = assets_arr[i]
        asset_exp = asset_exposure.get(asset, 0.0)

        if audit:
            exposure_at_signal[i] = current_exposure / equity * 100 if equity > 0 else 0.0
            asset_exposure_at_signal[i] = asset_exp / equity * 100 if equity > 0 else 0.0

        clipped_by_asset = False
        clipped_by_total = False

        if asset_exp + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - asset_exp, 0.0)
            clipped_by_asset = True

        if current_exposure + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - current_exposure, 0.0)
            clipped_by_total = True

        if raw_size < 1.0:
            equity_arr[i] = equity
            dd_arr[i] = dd
            if audit:
                if clipped_by_total:
                    skip_reasons[i] = "total_exposure_limit"
                elif clipped_by_asset:
                    skip_reasons[i] = "asset_exposure_limit"
                elif size_mult < 0.01:
                    skip_reasons[i] = "deleveraging"
                else:
                    skip_reasons[i] = "size_too_small"
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

        if audit:
            skip_reasons[i] = "executed"

    result = {
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

    if audit:
        result["skip_reasons"] = skip_reasons
        result["exposure_at_signal"] = exposure_at_signal
        result["asset_exposure_at_signal"] = asset_exposure_at_signal
        result["assets_arr"] = assets_arr

    return result


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


def compute_sharpe_from_daily(daily_pnl: pd.Series,
                              capital: float = STARTING_CAPITAL) -> float:
    """Compute annualized Sharpe from daily PnL series."""
    if daily_pnl is None or len(daily_pnl) < 5:
        return 0.0
    daily_ret = daily_pnl / capital
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    if std_r > 0:
        return float(mean_r / std_r * np.sqrt(365.25))
    return 0.0


# =============================================================================
# TEST 1: A10 — Extended History (2022 Bear Market)
# =============================================================================

def test_a10_extended_history() -> Dict[str, Any]:
    """Run approach A on BTC and ETH using full 2020-2026 extended data.

    Report Sharpe per year, with special attention to 2022 bear market.
    """
    print("=" * 80)
    print("TEST A10: EXTENDED HISTORY (2020-2026, Bear Market Validation)")
    print("=" * 80)

    results = {}

    for asset_name, symbol in EXTENDED_ASSETS.items():
        print(f"\n--- {asset_name} ({symbol}) ---")

        # Load extended data
        try:
            ohlcv_1m = load_binance_1m(symbol, EXTENDED_DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        # Resample to 15m
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        print(f"  15m bars: {len(ohlcv_15m):,}")

        # Compute features
        feat_15m = compute_features(ohlcv_15m)

        # Generate raw signals (walk-forward on full range)
        print(f"  Running walk-forward on full 2020-2026 range...")
        t0 = time.time()
        raw_signals = generate_raw_signals(
            ohlcv_15m, feat_15m, fwd_bars=4,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        elapsed = time.time() - t0
        print(f"  Generated {len(raw_signals):,} raw signals in {elapsed:.1f}s")

        if len(raw_signals) == 0:
            print(f"  No signals generated for {asset_name}")
            results[asset_name] = {"error": "no_signals"}
            continue

        # Apply percentile-70 confidence filter (approach A)
        filtered = apply_confidence_filter(raw_signals, "percentile", 70)
        print(f"  After confidence filter: {len(filtered):,} signals")

        if len(filtered) == 0:
            results[asset_name] = {"error": "no_filtered_signals"}
            continue

        # Compute Sharpe per year
        filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
        filtered["year"] = filtered["timestamp"].dt.year

        yearly_sharpes = {}
        for year in sorted(filtered["year"].unique()):
            year_signals = filtered[filtered["year"] == year].copy()
            if len(year_signals) < 20:
                yearly_sharpes[str(year)] = {"sharpe": None, "n_signals": len(year_signals),
                                              "note": "insufficient signals"}
                continue

            # Run portfolio sim on this year's signals
            sim = simulate_portfolio(year_signals)
            if sim and sim["n_trades"] >= 10:
                daily = compute_daily_pnl_series(sim)
                sharpe = compute_sharpe_from_daily(daily)
                n_trades = sim["n_trades"]
            else:
                sharpe = 0.0
                n_trades = sim["n_trades"] if sim else 0

            yearly_sharpes[str(year)] = {
                "sharpe": round(sharpe, 3),
                "n_signals": len(year_signals),
                "n_trades": n_trades,
            }
            label = " <-- BEAR MARKET" if year == 2022 else ""
            print(f"    {year}: Sharpe={sharpe:.3f}, signals={len(year_signals)}, "
                  f"trades={n_trades}{label}")

        # Also compute bear-market-specific Sharpe (May-Dec 2022)
        bear_mask = (
            (filtered["timestamp"] >= "2022-05-01") &
            (filtered["timestamp"] < "2023-01-01")
        )
        bear_signals = filtered[bear_mask].copy()
        if len(bear_signals) >= 20:
            sim_bear = simulate_portfolio(bear_signals)
            if sim_bear and sim_bear["n_trades"] >= 10:
                daily_bear = compute_daily_pnl_series(sim_bear)
                bear_sharpe = compute_sharpe_from_daily(daily_bear)
            else:
                bear_sharpe = 0.0
            yearly_sharpes["2022_may_dec"] = {
                "sharpe": round(bear_sharpe, 3),
                "n_signals": len(bear_signals),
                "n_trades": sim_bear["n_trades"] if sim_bear else 0,
            }
            print(f"    2022 May-Dec (BEAR): Sharpe={bear_sharpe:.3f}, "
                  f"signals={len(bear_signals)}")
        else:
            yearly_sharpes["2022_may_dec"] = {"sharpe": None, "n_signals": len(bear_signals),
                                               "note": "insufficient signals"}

        # Overall
        sim_all = simulate_portfolio(filtered)
        if sim_all and sim_all["n_trades"] >= 10:
            daily_all = compute_daily_pnl_series(sim_all)
            overall_sharpe = compute_sharpe_from_daily(daily_all)
        else:
            overall_sharpe = 0.0
        yearly_sharpes["overall"] = {
            "sharpe": round(overall_sharpe, 3),
            "n_signals": len(filtered),
            "n_trades": sim_all["n_trades"] if sim_all else 0,
        }
        print(f"    OVERALL: Sharpe={overall_sharpe:.3f}")

        results[asset_name] = yearly_sharpes

    return results


# =============================================================================
# TEST 2: B4 — Monte Carlo Shuffle Test
# =============================================================================

def test_b4_monte_carlo() -> Dict[str, Any]:
    """Monte Carlo shuffle test for approach A on 6 core assets.

    1. Run real model to get actual Sharpe
    2. Shuffle direction column 1000 times, re-run portfolio sim each time
    3. Compute p-value
    """
    print("\n" + "=" * 80)
    print("TEST B4: MONTE CARLO SHUFFLE TEST (1000 shuffles)")
    print("=" * 80)

    # Step 1: Generate real signals for all 6 core assets
    all_signals = []
    for asset_name, symbol in CORE_ASSETS.items():
        print(f"\n  Loading {asset_name}...")
        try:
            ohlcv_1m = load_binance_1m(symbol, CORE_DATA_DIR)
        except FileNotFoundError as e:
            print(f"    SKIPPING {asset_name}: {e}")
            continue

        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        feat_15m = compute_features(ohlcv_15m)

        raw = generate_raw_signals(
            ohlcv_15m, feat_15m, fwd_bars=4,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        if len(raw) > 0:
            filtered = apply_confidence_filter(raw, "percentile", 70)
            if len(filtered) > 0:
                all_signals.append(filtered)
                print(f"    {asset_name}: {len(filtered)} filtered signals")

    if not all_signals:
        print("  No signals generated!")
        return {"error": "no_signals"}

    combined = pd.concat(all_signals).sort_values("timestamp").reset_index(drop=True)
    print(f"\n  Total combined signals: {len(combined):,}")

    # Step 2: Run real portfolio simulation
    print("  Running real portfolio simulation...")
    sim_real = simulate_portfolio(combined)
    if not sim_real or sim_real["n_trades"] < 10:
        print("  Insufficient real trades!")
        return {"error": "insufficient_real_trades"}

    daily_real = compute_daily_pnl_series(sim_real)
    actual_sharpe = compute_sharpe_from_daily(daily_real)
    print(f"  Actual Sharpe: {actual_sharpe:.4f}")

    # Step 3: Shuffle directions 1000 times, re-run simulation each time
    print(f"  Running {N_SHUFFLES} shuffled simulations...")
    rng = np.random.RandomState(42)
    null_sharpes = []
    t0 = time.time()

    for i in range(N_SHUFFLES):
        shuffled = combined.copy()
        # Shuffle the direction column (preserves everything else: timing, sizing, assets)
        shuffled_dirs = shuffled["direction"].values.copy()
        rng.shuffle(shuffled_dirs)
        shuffled["direction"] = shuffled_dirs

        sim_shuffled = simulate_portfolio(shuffled)
        if sim_shuffled and sim_shuffled["n_trades"] >= 10:
            daily_shuffled = compute_daily_pnl_series(sim_shuffled)
            shuffled_sharpe = compute_sharpe_from_daily(daily_shuffled)
        else:
            shuffled_sharpe = 0.0

        null_sharpes.append(shuffled_sharpe)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_SHUFFLES - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{N_SHUFFLES} done ({elapsed:.1f}s, ETA {eta:.0f}s)")

    null_sharpes = np.array(null_sharpes)
    total_time = time.time() - t0

    # Step 4: Compute p-value
    p_value = float(np.mean(null_sharpes >= actual_sharpe))
    null_mean = float(np.mean(null_sharpes))
    null_std = float(np.std(null_sharpes))
    null_median = float(np.median(null_sharpes))
    null_p5 = float(np.percentile(null_sharpes, 5))
    null_p95 = float(np.percentile(null_sharpes, 95))

    significant = p_value < 0.01
    conclusion = "SIGNIFICANT (p < 0.01)" if significant else "NOT SIGNIFICANT (p >= 0.01)"

    print(f"\n  Monte Carlo Results:")
    print(f"    Actual Sharpe:   {actual_sharpe:.4f}")
    print(f"    Null mean:       {null_mean:.4f} (std {null_std:.4f})")
    print(f"    Null median:     {null_median:.4f}")
    print(f"    Null [5%, 95%]:  [{null_p5:.4f}, {null_p95:.4f}]")
    print(f"    p-value:         {p_value:.4f}")
    print(f"    Conclusion:      {conclusion}")
    print(f"    Runtime:         {total_time:.1f}s")

    return {
        "actual_sharpe": round(actual_sharpe, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "null_median": round(null_median, 4),
        "null_p5": round(null_p5, 4),
        "null_p95": round(null_p95, 4),
        "p_value": round(p_value, 4),
        "n_shuffles": N_SHUFFLES,
        "significant": significant,
        "conclusion": conclusion,
        "n_real_trades": sim_real["n_trades"],
        "n_total_signals": len(combined),
        "runtime_seconds": round(total_time, 1),
    }


# =============================================================================
# TEST 3: C3 — Concurrent Position Handling Audit
# =============================================================================

def test_c3_concurrent_positions() -> Dict[str, Any]:
    """Audit concurrent position handling in simulate_portfolio().

    1. Load signals from approach A on 6 assets
    2. Find timestamps where multiple assets signal simultaneously
    3. Verify exposure limits are correctly enforced
    4. Count how many signals are skipped due to exposure limits
    """
    print("\n" + "=" * 80)
    print("TEST C3: CONCURRENT POSITION HANDLING AUDIT")
    print("=" * 80)

    # Step 1: Generate signals for all 6 core assets
    all_signals = []
    for asset_name, symbol in CORE_ASSETS.items():
        print(f"\n  Loading {asset_name}...")
        try:
            ohlcv_1m = load_binance_1m(symbol, CORE_DATA_DIR)
        except FileNotFoundError as e:
            print(f"    SKIPPING {asset_name}: {e}")
            continue

        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        feat_15m = compute_features(ohlcv_15m)

        raw = generate_raw_signals(
            ohlcv_15m, feat_15m, fwd_bars=4,
            asset_name=asset_name, feature_cols=FEATURES_14,
        )
        if len(raw) > 0:
            filtered = apply_confidence_filter(raw, "percentile", 70)
            if len(filtered) > 0:
                all_signals.append(filtered)
                print(f"    {asset_name}: {len(filtered)} filtered signals")

    if not all_signals:
        print("  No signals generated!")
        return {"error": "no_signals"}

    combined = pd.concat(all_signals).sort_values("timestamp").reset_index(drop=True)
    print(f"\n  Total combined signals: {len(combined):,}")

    # Step 2: Analyze simultaneous signals
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    ts_counts = combined.groupby("timestamp").size()
    simultaneous_stats = {
        "max_simultaneous": int(ts_counts.max()),
        "mean_simultaneous": round(float(ts_counts.mean()), 2),
        "timestamps_with_multi": int((ts_counts > 1).sum()),
        "total_unique_timestamps": int(len(ts_counts)),
    }

    # Distribution of simultaneous signal counts
    count_dist = ts_counts.value_counts().sort_index()
    print(f"\n  Simultaneous signal distribution:")
    for n_sim, n_ts in count_dist.items():
        print(f"    {n_sim} signals at same time: {n_ts} occurrences")

    # Step 3: Run portfolio sim with audit mode
    print(f"\n  Running portfolio simulation with audit...")
    sim = simulate_portfolio(combined, audit=True)
    if not sim:
        return {"error": "simulation_failed"}

    # Step 4: Analyze exposure enforcement
    skip_reasons_arr = sim["skip_reasons"]
    exposure_arr = sim["exposure_at_signal"]
    n_total = len(combined)
    n_executed = sim["n_trades"]
    n_skipped = n_total - n_executed

    # Count skip reasons
    reason_counts = {}
    for reason in skip_reasons_arr:
        if reason and reason != "executed":
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    skipped_exposure_total = reason_counts.get("total_exposure_limit", 0)
    skipped_exposure_asset = reason_counts.get("asset_exposure_limit", 0)
    skipped_exposure = skipped_exposure_total + skipped_exposure_asset
    skipped_dd = reason_counts.get("dd_halt", 0) + reason_counts.get("halted", 0)
    skipped_delever = reason_counts.get("deleveraging", 0)
    skipped_size = reason_counts.get("size_too_small", 0)

    # Track actual exposure over time to find max
    # Re-simulate to track true exposure % at each point
    sub = combined.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)
    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = sub["signal_strength"].values.astype(np.float64)

    # Detailed exposure tracking
    equity = STARTING_CAPITAL
    peak_equity = equity
    current_exposure = 0.0
    asset_exp_track: dict = {}
    halted = False

    max_exposure_pct = 0.0
    max_per_asset_pct = 0.0
    exposure_violations = 0
    per_asset_violations = 0

    for i in range(n):
        if halted:
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= MAX_DD:
            halted = True
            continue

        if dd > DELEVER_START:
            dd_ratio = (dd - DELEVER_START) / (MAX_DD - DELEVER_START)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        sig_abs = abs(signal_strengths[i])
        raw_size = equity * BASE_SIZING_PCT * sig_abs * size_mult
        raw_size = min(raw_size, equity * BASE_SIZING_PCT * MAX_SIGNAL_MULT * size_mult)

        asset = assets_arr[i]
        a_exp = asset_exp_track.get(asset, 0.0)

        if a_exp + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - a_exp, 0.0)

        if current_exposure + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - current_exposure, 0.0)

        if raw_size < 1.0:
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
        asset_exp_track[asset] = asset_exp_track.get(asset, 0.0) * 0.9

        # Check exposure percentages
        total_exp_pct = (current_exposure / equity * 100) if equity > 0 else 0.0
        if total_exp_pct > max_exposure_pct:
            max_exposure_pct = total_exp_pct

        for a, e in asset_exp_track.items():
            a_pct = (e / equity * 100) if equity > 0 else 0.0
            if a_pct > max_per_asset_pct:
                max_per_asset_pct = a_pct

        # Check for violations
        if total_exp_pct > MAX_EXPOSURE * 100 + 0.01:  # small tolerance for floating point
            exposure_violations += 1
        for a, e in asset_exp_track.items():
            a_pct = (e / equity * 100) if equity > 0 else 0.0
            if a_pct > MAX_PER_ASSET * 100 + 0.01:
                per_asset_violations += 1

    # Print results
    signals_skipped_pct = (skipped_exposure / n_total * 100) if n_total > 0 else 0.0

    print(f"\n  CONCURRENT POSITION AUDIT RESULTS:")
    print(f"    Total signals:                {n_total}")
    print(f"    Executed trades:              {n_executed}")
    print(f"    Skipped (total):              {n_skipped}")
    print(f"      - Exposure limit (total):   {skipped_exposure_total}")
    print(f"      - Exposure limit (asset):   {skipped_exposure_asset}")
    print(f"      - DD halt:                  {skipped_dd}")
    print(f"      - Deleveraging:             {skipped_delever}")
    print(f"      - Size too small:           {skipped_size}")
    print(f"    Max simultaneous signals:     {simultaneous_stats['max_simultaneous']}")
    print(f"    Max observed total exposure:  {max_exposure_pct:.2f}%")
    print(f"    Max observed per-asset exp:   {max_per_asset_pct:.2f}%")
    print(f"    Total exposure violations:    {exposure_violations}")
    print(f"    Per-asset exposure violations: {per_asset_violations}")
    print(f"    PASS: {'YES' if exposure_violations == 0 and per_asset_violations == 0 else 'NO'}")

    return {
        "total_signals": n_total,
        "executed_trades": n_executed,
        "skipped_total": n_skipped,
        "skipped_exposure_total_limit": skipped_exposure_total,
        "skipped_exposure_asset_limit": skipped_exposure_asset,
        "skipped_dd_halt": skipped_dd,
        "skipped_deleveraging": skipped_delever,
        "skipped_size_too_small": skipped_size,
        "signals_skipped_exposure_pct": round(signals_skipped_pct, 2),
        "max_simultaneous_signals": simultaneous_stats["max_simultaneous"],
        "mean_simultaneous_signals": simultaneous_stats["mean_simultaneous"],
        "timestamps_with_multiple": simultaneous_stats["timestamps_with_multi"],
        "max_observed_exposure_pct": round(max_exposure_pct, 2),
        "max_observed_per_asset_pct": round(max_per_asset_pct, 2),
        "total_exposure_violations": exposure_violations,
        "per_asset_exposure_violations": per_asset_violations,
        "exposure_limit_enforced": exposure_violations == 0 and per_asset_violations == 0,
        "simultaneous_distribution": {str(k): int(v) for k, v in count_dist.items()},
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("REMAINING VALIDATION — Checklist items A10, B4, C3")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    total_start = time.time()
    results = {}

    # Test 1: A10 — Extended History
    print("\n")
    results["A10_extended_history"] = test_a10_extended_history()

    # Test 2: B4 — Monte Carlo Shuffle Test
    print("\n")
    results["B4_monte_carlo"] = test_b4_monte_carlo()

    # Test 3: C3 — Concurrent Position Audit
    print("\n")
    results["C3_concurrent_positions"] = test_c3_concurrent_positions()

    total_elapsed = time.time() - total_start

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # A10 summary
    print("\nA10 EXTENDED HISTORY:")
    a10 = results.get("A10_extended_history", {})
    for asset_name in ["BTC", "ETH"]:
        if asset_name in a10:
            asset_data = a10[asset_name]
            if isinstance(asset_data, dict) and "error" not in asset_data:
                for year_key in sorted(asset_data.keys()):
                    if year_key == "overall":
                        continue
                    year_data = asset_data[year_key]
                    sharpe = year_data.get("sharpe")
                    if sharpe is not None:
                        bear_label = " <-- THE critical number" if year_key == "2022" or year_key == "2022_may_dec" else ""
                        label = f"  {asset_name} {year_key}: Sharpe {sharpe:.3f}{bear_label}"
                        print(label)
                    else:
                        print(f"  {asset_name} {year_key}: insufficient data")
                overall = asset_data.get("overall", {})
                if overall:
                    print(f"  {asset_name} OVERALL: Sharpe {overall.get('sharpe', 'N/A')}")

    # B4 summary
    print("\nB4 MONTE CARLO:")
    b4 = results.get("B4_monte_carlo", {})
    if "error" not in b4:
        print(f"  Actual Sharpe: {b4.get('actual_sharpe', 'N/A')}")
        print(f"  Null mean: {b4.get('null_mean', 'N/A')} (std {b4.get('null_std', 'N/A')})")
        print(f"  p-value: {b4.get('p_value', 'N/A')}")
        print(f"  Conclusion: {b4.get('conclusion', 'N/A')}")
    else:
        print(f"  ERROR: {b4.get('error')}")

    # C3 summary
    print("\nC3 CONCURRENT POSITIONS:")
    c3 = results.get("C3_concurrent_positions", {})
    if "error" not in c3:
        print(f"  Max simultaneous signals: {c3.get('max_simultaneous_signals', 'N/A')}")
        print(f"  Signals skipped (exposure limit): {c3.get('skipped_exposure_total_limit', 0) + c3.get('skipped_exposure_asset_limit', 0)} "
              f"({c3.get('signals_skipped_exposure_pct', 0):.1f}%)")
        print(f"  Max observed exposure: {c3.get('max_observed_exposure_pct', 'N/A')}%")
        print(f"  Exposure violations: {c3.get('total_exposure_violations', 'N/A')} "
              f"({'PASS' if c3.get('exposure_limit_enforced') else 'FAIL'})")
    else:
        print(f"  ERROR: {c3.get('error')}")

    # =========================================================================
    # Save results
    # =========================================================================
    results["meta"] = {
        "experiment": "remaining_validation",
        "tests": ["A10_extended_history", "B4_monte_carlo", "C3_concurrent_positions"],
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(total_elapsed, 1),
    }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/remaining_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
