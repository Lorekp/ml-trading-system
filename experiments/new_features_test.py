"""
New Features Test — Hyperliquid derivatives (OI, Premium, Funding) as features.

Tests 4 feature sets on 6 core crypto assets at 15m:
  1. Baseline:  14 OHLCV features (control)
  2. +OI:       14 + oi_change_pct_1 + oi_change_pct_4 + premium_zscore = 17
  3. +Funding:  14 + funding_rolling_3 + funding_extreme + funding_trend = 17
  4. +All:      14 + all 6 derivative features = 20

Key design:
  - Same walk-forward methodology as multi_approach_comparison.py
  - ONE model config per feature set, same confidence filter (percentile 70)
  - All 4 feature sets use same walk-forward splits (fair comparison)
  - Date range: intersection of Binance 1m (2024-03+) and HL data (2023-05-20+)
    => 2024-03 to 2026-02

HL data: C:/Claude/ml-trading-system/data/hl-asset-ctxs/hl-asset-ctxs/
  Daily .csv.lz4 files, 1-minute snapshots for ALL coins.
  Columns: time, coin, funding, open_interest, premium, oracle_px, mark_px,
           mid_px, impact_bid_px, impact_ask_px
"""

import io
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

import lz4.frame
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "XRP": "XRPUSDT",
    "BNB": "BNBUSDT",
    "SUI": "SUIUSDT",
    "LINK": "LINKUSDT",
}

# Coin names as they appear in HL data
HL_COIN_MAP = {
    "BTC": "BTC",
    "ETH": "ETH",
    "XRP": "XRP",
    "BNB": "BNB",
    "SUI": "SUI",
    "LINK": "LINK",
}

DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
HL_DATA_DIR = Path("C:/Claude/ml-trading-system/data/hl-asset-ctxs/hl-asset-ctxs")

# Date range: intersection of Binance 1m and HL data
DATE_RANGE_START = "2024-03-01"
DATE_RANGE_END = "2026-02-28"

# Fixed model — no tuning (same as multi_approach_comparison)
MODEL_C = 0.1
MODEL_PENALTY = "l1"
MODEL_SOLVER = "saga"
MODEL_MAX_ITER = 1000
MODEL_SEED = 42

# Portfolio parameters (same as multi_approach_comparison)
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
CONFIDENCE_FILTER_MODE = "percentile"
CONFIDENCE_FILTER_THRESHOLD = 70

# Winsorization
WINSOR_LO = 0.01
WINSOR_HI = 0.99

# Signal discretization
SIGNAL_STEP_SIZE = 0.1

# Base 14 OHLCV features
FEATURES_14 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
]

# Derivative feature names
OI_FEATURES = ["oi_change_pct_1", "oi_change_pct_4", "premium_zscore"]
FUNDING_FEATURES = ["funding_rolling_3", "funding_extreme", "funding_trend"]
ALL_DERIV_FEATURES = OI_FEATURES + FUNDING_FEATURES

# Feature set definitions
FEATURE_SETS = {
    "Baseline (14 feat)": FEATURES_14,
    "+OI (17 feat)": FEATURES_14 + OI_FEATURES,
    "+Funding (17 feat)": FEATURES_14 + FUNDING_FEATURES,
    "+All (20 feat)": FEATURES_14 + ALL_DERIV_FEATURES,
}


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
# SIGNAL / SIZING HELPERS (from multi_approach_comparison.py)
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
# SAMPLE WEIGHTS — AFML Ch 4
# =============================================================================

def compute_sample_weights(timestamps: np.ndarray, fwd_bars: int) -> np.ndarray:
    """AFML Ch 4: Average uniqueness for overlapping labels. O(n)."""
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
# STATISTICS
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
# OHLCV FEATURE COMPUTATION (14 base features)
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
# HL DERIVATIVES LOADING
# =============================================================================

def _load_all_coins_batch(hl_data_dir: Path, start_date: str, end_date: str,
                          coins: list) -> Dict[str, pd.DataFrame]:
    """Batch-load HL derivatives for ALL coins at once. Much faster than per-coin loading.

    Reads each LZ4 file once, splits by coin, caches per-coin parquets.
    Subsequent runs load from cache instantly.
    """
    cache_dir = hl_data_dir.parent / "cache_15m"
    cache_dir.mkdir(exist_ok=True)

    # Check cache first
    cached = {}
    all_cached = True
    for coin in coins:
        cache_fp = cache_dir / f"{coin}_hl_1m.parquet"
        if cache_fp.exists():
            cached[coin] = pd.read_parquet(cache_fp)
        else:
            all_cached = False

    if all_cached:
        print(f"    HL data loaded from cache for {len(coins)} coins")
        return cached

    # Load all files, extract all coins at once
    print(f"    Loading HL data from LZ4 files (first time — will cache)...")
    coin_frames = {c: [] for c in coins}
    date_range = pd.date_range(start_date, end_date, freq='D')
    loaded = 0

    for date in date_range:
        fp = hl_data_dir / f"{date.strftime('%Y%m%d')}.csv.lz4"
        if not fp.exists():
            continue
        try:
            with open(fp, 'rb') as f:
                data = lz4.frame.decompress(f.read())
            df = pd.read_csv(io.BytesIO(data))
            for coin in coins:
                coin_df = df[df['coin'] == coin]
                if len(coin_df) > 0:
                    coin_frames[coin].append(coin_df)
            loaded += 1
            if loaded % 100 == 0:
                print(f"      {loaded}/{len(date_range)} files loaded...", flush=True)
        except Exception:
            continue

    print(f"    Loaded {loaded} files, processing...", flush=True)

    result = {}
    for coin in coins:
        if not coin_frames[coin]:
            continue
        df = pd.concat(coin_frames[coin])
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        numeric_cols = ['funding', 'open_interest', 'premium', 'oracle_px', 'mark_px']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[numeric_cols].dropna(how='all')
        result[coin] = df
        # Cache to parquet
        cache_fp = cache_dir / f"{coin}_hl_1m.parquet"
        df.to_parquet(cache_fp)

    print(f"    Cached {len(result)} coins to {cache_dir}")
    return result


def load_hl_derivatives(coin: str, start_date: str, end_date: str,
                        hl_data_dir: Path) -> Optional[pd.DataFrame]:
    """Load HL derivatives for one coin, return DataFrame with UTC DatetimeIndex.

    Uses batch loader with caching for efficiency.
    """
    # Try cache first
    cache_dir = hl_data_dir.parent / "cache_15m"
    cache_fp = cache_dir / f"{coin}_hl_1m.parquet"
    if cache_fp.exists():
        result = pd.read_parquet(cache_fp)
        # Filter date range
        if result.index.tz is None:
            result.index = result.index.tz_localize('UTC')
        mask = (result.index >= pd.Timestamp(start_date, tz='UTC')) & \
               (result.index <= pd.Timestamp(end_date, tz='UTC'))
        return result[mask] if mask.any() else None

    # No cache — caller should use _load_all_coins_batch() first
    print(f"    WARNING: No cache for {coin}, loading individually (slow)...")
    frames = []
    date_range = pd.date_range(start_date, end_date, freq='D')

    for date in date_range:
        fp = hl_data_dir / f"{date.strftime('%Y%m%d')}.csv.lz4"
        if not fp.exists():
            continue
        try:
            with open(fp, 'rb') as f:
                data = lz4.frame.decompress(f.read())
            df = pd.read_csv(io.BytesIO(data))
            df = df[df['coin'] == coin]
            if len(df) == 0:
                continue
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    result = pd.concat(frames)
    result['time'] = pd.to_datetime(result['time'], utc=True)
    result = result.set_index('time').sort_index()
    result = result[~result.index.duplicated(keep='first')]

    numeric_cols = ['funding', 'open_interest', 'premium', 'oracle_px', 'mark_px',
                    'mid_px', 'impact_bid_px', 'impact_ask_px']
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')

    print(f"    HL data for {coin}: {len(result):,} rows, "
          f"{result.index.min().date()} to {result.index.max().date()} "
          f"({loaded}/{n_dates} days loaded, {skipped} skipped)")

    return result[['funding', 'open_interest', 'premium', 'oracle_px', 'mark_px']]


# =============================================================================
# DERIVATIVE FEATURE COMPUTATION
# =============================================================================

def compute_derivative_features(hl_df: pd.DataFrame,
                                ohlcv_15m: pd.DataFrame) -> pd.DataFrame:
    """Compute OI, premium, funding features aligned to 15m bars.

    Args:
        hl_df: 1-minute HL derivatives data with columns:
               funding, open_interest, premium, oracle_px, mark_px
        ohlcv_15m: 15-minute OHLCV bars (used for alignment).

    Returns:
        DataFrame with 6 derivative features aligned to ohlcv_15m index:
          - oi_change_pct_1: OI percent change over 1 bar
          - oi_change_pct_4: OI percent change over 4 bars
          - premium_zscore: Premium z-score (30-bar rolling window)
          - funding_rolling_3: 3-bar rolling mean of funding rate
          - funding_extreme: Binary flag for extreme funding (>2 std)
          - funding_trend: Sign of funding change over 4 bars
    """
    # Resample to 15m (use last value, no lookahead)
    hl_15m = hl_df[['funding', 'open_interest', 'premium', 'oracle_px', 'mark_px']].resample('15min').last()

    # Align to OHLCV index via forward-fill
    hl_aligned = hl_15m.reindex(ohlcv_15m.index, method='ffill')

    feat = pd.DataFrame(index=ohlcv_15m.index)

    # --- OI features ---
    oi = hl_aligned['open_interest']
    feat['oi_change_pct_1'] = oi.pct_change(1)
    feat['oi_change_pct_4'] = oi.pct_change(4)

    # --- Premium z-score ---
    premium = (hl_aligned['mark_px'] - hl_aligned['oracle_px']) / hl_aligned['oracle_px'].replace(0, np.nan)
    prem_mean = premium.rolling(30, min_periods=10).mean()
    prem_std = premium.rolling(30, min_periods=10).std()
    feat['premium_zscore'] = (premium - prem_mean) / prem_std.replace(0, np.nan)

    # --- Funding features ---
    funding = hl_aligned['funding']
    feat['funding_rolling_3'] = funding.rolling(3, min_periods=1).mean()
    fund_mean = funding.rolling(50, min_periods=20).mean()
    fund_std = funding.rolling(50, min_periods=20).std()
    feat['funding_extreme'] = (funding.abs() > (fund_mean.abs() + 2 * fund_std)).astype(float)
    feat['funding_trend'] = np.sign(funding - funding.shift(4))

    # Cast to float32 for consistency
    for col in feat.columns:
        feat[col] = feat[col].astype(np.float32)

    return feat


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

    Returns (signals_df, coefficients_dict_or_None).
    """
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)

    # Volatility for vol filter
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()

    if (end_date - start_date).days < 180:
        return pd.DataFrame(), None

    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"),
        freq="M",
    )

    if len(all_months) == 0:
        return pd.DataFrame(), None

    # Retrain optimization for large datasets
    retrain_every = 1
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    # Embargo
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
        return pd.DataFrame(), coefficients

    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = asset_name
    return df, coefficients


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
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("NEW FEATURES TEST — Hyperliquid Derivatives as Features")
    print("4 feature sets x 6 assets @ 15m")
    print(f"Date range: {DATE_RANGE_START} to {DATE_RANGE_END}")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total_start = time.time()

    # Storage: feature_set_name -> list of filtered signal DataFrames
    all_signals: Dict[str, List[pd.DataFrame]] = {name: [] for name in FEATURE_SETS}
    # Per-asset coefficients for each feature set
    all_coefficients: Dict[str, Dict[str, Dict]] = {name: {} for name in FEATURE_SETS}
    # Per-asset signal counts
    per_asset_results: Dict[str, Dict[str, Any]] = {}
    # Track derivative feature coverage
    deriv_coverage: Dict[str, Dict[str, Any]] = {}

    # Pre-load ALL HL derivatives at once (batch, with caching)
    hl_coins = list(HL_COIN_MAP.values())
    print(f"Pre-loading HL derivatives for {len(hl_coins)} coins...")
    hl_batch = _load_all_coins_batch(HL_DATA_DIR, DATE_RANGE_START, DATE_RANGE_END, hl_coins)
    print(f"HL data ready: {len(hl_batch)} coins loaded\n")

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {asset_name} ({symbol})...")
        asset_start = time.time()

        # 1. Load Binance 1m data
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIR)
        except FileNotFoundError as e:
            print(f"  SKIPPING {asset_name}: {e}")
            continue

        # 2. Resample to 15m
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

        # 3. Filter to date range
        start_ts = pd.Timestamp(DATE_RANGE_START, tz="UTC")
        end_ts = pd.Timestamp(DATE_RANGE_END, tz="UTC")
        ohlcv_15m = ohlcv_15m[(ohlcv_15m.index >= start_ts) & (ohlcv_15m.index <= end_ts)]
        print(f"  15m bars in date range: {len(ohlcv_15m):,}")

        if len(ohlcv_15m) < 1000:
            print(f"  SKIPPING {asset_name}: insufficient data in date range")
            continue

        # 4. Compute base 14 OHLCV features
        feat_14 = compute_features(ohlcv_15m)

        # 5. Get HL derivatives from pre-loaded batch
        hl_coin = HL_COIN_MAP[asset_name]
        hl_df = hl_batch.get(hl_coin)
        if hl_df is not None:
            # Filter to date range
            if hl_df.index.tz is None:
                hl_df.index = hl_df.index.tz_localize('UTC')
            hl_df = hl_df[(hl_df.index >= start_ts) & (hl_df.index <= end_ts)]
            print(f"  HL derivatives for {hl_coin}: {len(hl_df):,} rows")

        deriv_feat = None
        if hl_df is not None and len(hl_df) > 100:
            deriv_feat = compute_derivative_features(hl_df, ohlcv_15m)
            # Report coverage
            n_total = len(ohlcv_15m)
            n_non_nan = deriv_feat.dropna().shape[0]
            coverage_pct = n_non_nan / n_total * 100 if n_total > 0 else 0
            deriv_coverage[asset_name] = {
                "total_15m_bars": n_total,
                "derivative_bars": n_non_nan,
                "coverage_pct": round(coverage_pct, 1),
                "hl_rows": len(hl_df),
            }
            print(f"  Derivative features: {n_non_nan:,}/{n_total:,} bars "
                  f"({coverage_pct:.1f}% coverage)")
        else:
            print(f"  WARNING: No derivative features for {asset_name}")
            deriv_coverage[asset_name] = {
                "total_15m_bars": len(ohlcv_15m),
                "derivative_bars": 0,
                "coverage_pct": 0.0,
                "hl_rows": 0,
            }

        # 6. Build feature DataFrames for each feature set
        feature_dfs = {}
        for fs_name, fs_cols in FEATURE_SETS.items():
            if fs_name == "Baseline (14 feat)":
                feature_dfs[fs_name] = feat_14
            else:
                # Need derivative features
                if deriv_feat is None:
                    print(f"    SKIPPING {fs_name}: no derivative data")
                    feature_dfs[fs_name] = None
                    continue
                # Combine base + derivative features
                deriv_cols_needed = [c for c in fs_cols if c not in FEATURES_14]
                available_deriv_cols = [c for c in deriv_cols_needed if c in deriv_feat.columns]
                if len(available_deriv_cols) < len(deriv_cols_needed):
                    missing = set(deriv_cols_needed) - set(available_deriv_cols)
                    print(f"    WARNING: Missing derivative columns: {missing}")
                combined_feat = pd.concat([feat_14, deriv_feat[available_deriv_cols]], axis=1)
                # Drop rows with NaN derivative features (start of rolling windows)
                combined_feat = combined_feat.dropna(subset=available_deriv_cols)
                feature_dfs[fs_name] = combined_feat

        # 7. Generate raw signals for each feature set
        per_asset_results[asset_name] = {}
        for fs_name, fs_cols in FEATURE_SETS.items():
            feat_df = feature_dfs.get(fs_name)
            if feat_df is None:
                print(f"    {fs_name}: SKIPPED (no data)")
                per_asset_results[asset_name][fs_name] = {"n_signals": 0}
                continue

            print(f"    {fs_name}: training model ({len(feat_df):,} feature rows, {len(fs_cols)} features)...")
            t0 = time.time()

            # Use the OHLCV aligned to the feature index (may be smaller for deriv sets)
            ohlcv_aligned = ohlcv_15m.reindex(feat_df.index)

            raw_signals, coefs = generate_raw_signals(
                ohlcv_aligned, feat_df, fwd_bars=4,
                asset_name=asset_name, feature_cols=fs_cols,
                collect_coefficients=True,
            )

            # Apply confidence filter (percentile 70, same as baseline approach A)
            filtered = apply_confidence_filter(
                raw_signals, CONFIDENCE_FILTER_MODE, CONFIDENCE_FILTER_THRESHOLD,
            )

            elapsed_fs = time.time() - t0
            n_raw = len(raw_signals) if raw_signals is not None else 0
            n_filt = len(filtered) if filtered is not None else 0
            print(f"      -> {n_raw:,} raw -> {n_filt:,} filtered signals in {elapsed_fs:.1f}s")

            if filtered is not None and len(filtered) > 0:
                all_signals[fs_name].append(filtered)

            if coefs:
                all_coefficients[fs_name][asset_name] = coefs

            per_asset_results[asset_name][fs_name] = {
                "n_raw_signals": n_raw,
                "n_filtered_signals": n_filt,
                "long_pct": round(float((filtered["direction"] > 0).mean()) * 100, 1)
                    if n_filt > 0 else 0,
                "mean_confidence": round(float(filtered["confidence"].mean()), 4)
                    if n_filt > 0 else 0,
            }

        elapsed = time.time() - asset_start
        print(f"  {asset_name} complete in {elapsed:.1f}s")

    # =========================================================================
    # Portfolio simulations for each feature set
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("PORTFOLIO SIMULATIONS")
    print("=" * 80)

    results: Dict[str, Dict[str, Any]] = {}
    for fs_name in FEATURE_SETS:
        if not all_signals[fs_name]:
            print(f"  {fs_name}: no signals collected, skipping")
            continue

        combined = pd.concat(all_signals[fs_name]).sort_values("timestamp")
        print(f"  {fs_name}: {len(combined):,} total signals across all assets")

        sim = simulate_portfolio(combined)
        if sim and sim["n_trades"] >= 20:
            daily = compute_daily_pnl_series(sim)
            if daily is not None:
                n_days = len(daily)
                trades_per_day = sim["n_trades"] / max(n_days, 1)
                stats = compute_full_stats(
                    daily, STARTING_CAPITAL, sim["n_trades"], trades_per_day,
                )
                results[fs_name] = stats
                results[fs_name]["n_trades"] = sim["n_trades"]
                results[fs_name]["halted"] = sim["halted"]
                print(f"    -> Sharpe={stats.get('sharpe', 0):.3f}, "
                      f"Return={stats.get('total_return_pct', 0):.2f}%, "
                      f"MaxDD={stats.get('max_dd_pct', 0):.2f}%, "
                      f"Trades={sim['n_trades']}, "
                      f"WinRate={stats.get('win_rate_days', 0) * 100:.1f}%")
            else:
                print(f"    -> insufficient daily PnL data")
        else:
            n_trades = sim["n_trades"] if sim else 0
            print(f"    -> insufficient trades ({n_trades})")

    # =========================================================================
    # L1 Coefficients for derivative features
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("L1 COEFFICIENTS — NEW DERIVATIVE FEATURES")
    print("=" * 80)

    deriv_coef_summary: Dict[str, Dict[str, float]] = {}
    for fs_name in ["+OI (17 feat)", "+Funding (17 feat)", "+All (20 feat)"]:
        if fs_name not in all_coefficients or not all_coefficients[fs_name]:
            continue

        print(f"\n  {fs_name}:")
        avg_coefs = {}
        for asset_name, month_coefs in all_coefficients[fs_name].items():
            last_month = sorted(month_coefs.keys())[-1]
            last_coefs = month_coefs[last_month]
            # Only show derivative features
            deriv_cols = [c for c in last_coefs if c not in FEATURES_14]
            if deriv_cols:
                print(f"    {asset_name} (last retrain: {last_month}):")
                for feat_name in deriv_cols:
                    coef_val = last_coefs[feat_name]
                    print(f"      {feat_name:>25s}: {coef_val:+.6f}")
                    avg_coefs.setdefault(feat_name, []).append(coef_val)

        if avg_coefs:
            print(f"\n    AVERAGE across assets:")
            fs_avg = {}
            for feat_name in sorted(avg_coefs.keys(), key=lambda x: abs(np.mean(avg_coefs[x])), reverse=True):
                mean_val = np.mean(avg_coefs[feat_name])
                print(f"      {feat_name:>25s}: {mean_val:+.6f}")
                fs_avg[feat_name] = round(float(mean_val), 6)
            deriv_coef_summary[fs_name] = fs_avg

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Build coefficient strings for the summary line
    def coef_str(fs_name: str) -> str:
        """Format derivative coefficient values for summary."""
        if fs_name not in deriv_coef_summary:
            return "—"
        coefs = deriv_coef_summary[fs_name]
        parts = []
        name_map = {
            "oi_change_pct_1": "oi_pct1",
            "oi_change_pct_4": "oi_pct4",
            "premium_zscore": "prem_z",
            "funding_rolling_3": "fund_r3",
            "funding_extreme": "fund_ext",
            "funding_trend": "fund_trend",
        }
        for full_name, short_name in name_map.items():
            if full_name in coefs:
                parts.append(f"{short_name}={coefs[full_name]:+.4f}")
        return ", ".join(parts) if parts else "—"

    header = (f"{'Feature Set':<25} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
              f"{'Trades':>8} {'WinRate':>8} {'New Features L1 Coefs'}")
    print(header)
    print("-" * 110)

    for fs_name in FEATURE_SETS:
        if fs_name in results:
            r = results[fs_name]
            cs = coef_str(fs_name) if fs_name != "Baseline (14 feat)" else "—"
            print(f"{fs_name:<25} {r.get('sharpe', 0):>8.3f} "
                  f"{r.get('total_return_pct', 0):>8.2f}% "
                  f"{r.get('max_dd_pct', 0):>7.2f}% "
                  f"{r.get('n_trades', 0):>8} "
                  f"{r.get('win_rate_days', 0) * 100:>7.1f}% "
                  f"  {cs}")
        else:
            print(f"{fs_name:<25} {'N/A':>8} {'N/A':>9} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>8}   —")

    # =========================================================================
    # Derivative feature coverage summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("DERIVATIVE DATA COVERAGE")
    print("=" * 80)
    for asset_name, cov in deriv_coverage.items():
        print(f"  {asset_name}: {cov['derivative_bars']:,}/{cov['total_15m_bars']:,} bars "
              f"({cov['coverage_pct']:.1f}%), {cov['hl_rows']:,} HL rows")

    # =========================================================================
    # Save results
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\nTotal runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    output = {
        "meta": {
            "experiment": "new_features_test",
            "description": "Test Hyperliquid derivatives (OI, premium, funding) as features",
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(total_elapsed, 1),
            "assets": list(ASSETS.keys()),
            "n_assets": len(ASSETS),
            "date_range": {"start": DATE_RANGE_START, "end": DATE_RANGE_END},
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
            "confidence_filter_mode": CONFIDENCE_FILTER_MODE,
            "confidence_filter_threshold": CONFIDENCE_FILTER_THRESHOLD,
            "winsor_lo": WINSOR_LO,
            "winsor_hi": WINSOR_HI,
            "signal_step_size": SIGNAL_STEP_SIZE,
        },
        "feature_sets": {
            name: cols for name, cols in FEATURE_SETS.items()
        },
        "results": results,
        "derivative_coverage": deriv_coverage,
        "per_asset": per_asset_results,
        "l1_coefficients_summary": deriv_coef_summary,
        "l1_coefficients_per_asset": {
            fs_name: {
                asset: {
                    month: {k: round(v, 6) for k, v in coefs.items()}
                    for month, coefs in month_coefs.items()
                }
                for asset, month_coefs in asset_coefs.items()
            }
            for fs_name, asset_coefs in all_coefficients.items()
            if asset_coefs
        },
    }

    output_path = Path("C:/Claude/ml-trading-system/v4-results/new_features_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 80}")
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
