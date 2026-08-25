"""HISTORICAL EXPERIMENT — its output is not execution validation.

Execution sensitivity study for a historical directional model.

Context: Our best model (LogReg L1 C=0.1, 19 features, no funding_zscore) achieves
Sharpe 1.78 with idealized execution (100% fill, 1bp flat slippage, no funding costs).
This script applies modeled conditions; it does not provide independent fills.

Architecture: Reuses the entire precompute-once infrastructure from model_comparison.py.
Generates signals ONCE with LogReg L1 C=0.1, then simulates portfolio 9 times with
different execution parameters.

Scenarios:
  S1_IDEAL       100% fill, 1bp slippage, no funding   (baseline)
  S2_GOOD_FILL    85% fill, 1bp slippage, no funding
  S3_AVG_FILL     75% fill, 1bp slippage, no funding
  S4_BAD_FILL     60% fill, 1bp slippage, no funding
  S5_HIGH_SLIP   100% fill, 2bp slippage, no funding
  S6_ASSET_SLIP  100% fill, per-asset slippage, no funding
  S7_WITH_FUND   100% fill, 1bp slippage, with funding
  S8_REALISTIC    75% fill, per-asset slippage, with funding
  S9_PESSIMISTIC  60% fill, per-asset+1bp slippage, with funding
"""

import sys
import warnings
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

# Import AFML modules with inline fallback
try:
    from afml.fracdiff import frac_diff_ffd
    from afml.derivatives import funding_features, oi_features, impact_features
    from afml.features import corwin_schultz_spread, rolling_entropy
    from afml.bet_sizing import get_signal, discretize_signal, kelly_size
    from afml.evaluation import deflated_sharpe_ratio
    AFML_IMPORTED = True
except ImportError:
    AFML_IMPORTED = False
    print("[WARN] Could not import afml modules — using inline implementations", flush=True)

# Optional HMM for regime filter
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARN] hmmlearn not available — volatility percentile fallback", flush=True)


# =============================================================================
# INLINE AFML FALLBACKS
# =============================================================================

if not AFML_IMPORTED:

    def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
        weights = [1.0]
        k = 1
        while True:
            w_next = -weights[-1] / k * (d - k + 1)
            if abs(w_next) < threshold:
                break
            weights.append(w_next)
            k += 1
        return np.array(weights[::-1]).reshape(-1, 1)

    def frac_diff_ffd(series: pd.DataFrame, d: float, threshold: float = 1e-5) -> pd.DataFrame:
        weights = get_weights_ffd(d, threshold)
        width = len(weights) - 1
        result = {}
        for col in series.columns:
            col_data = series[[col]].ffill().dropna()
            ffd_values = pd.Series(dtype=float)
            for i in range(width, len(col_data)):
                window = col_data.iloc[i - width: i + 1]
                ffd_values[col_data.index[i]] = np.dot(weights.T, window.values)[0, 0]
            result[col] = ffd_values
        return pd.concat(result, axis=1)

    def funding_features(funding_rate, window_short=50, window_long=200):
        features = pd.DataFrame(index=funding_rate.index)
        features["funding"] = funding_rate
        mean = funding_rate.rolling(window_short).mean()
        std = funding_rate.rolling(window_short).std()
        features["funding_zscore"] = (funding_rate - mean) / std.replace(0, np.nan)
        return features

    def oi_features(open_interest, close, window=50):
        features = pd.DataFrame(index=open_interest.index)
        oi_pct = open_interest.pct_change(5)
        features["oi_change_5"] = oi_pct
        return features

    def impact_features(impact_bid, impact_ask, mid_price):
        features = pd.DataFrame(index=mid_price.index)
        spread = (impact_ask - impact_bid) / mid_price.replace(0, np.nan)
        features["impact_spread"] = spread
        return features

    def corwin_schultz_spread(high, low, window=20):
        log_hl_sq = np.log(high / low) ** 2
        beta = log_hl_sq.rolling(2).sum().rolling(window).mean()
        h2 = high.rolling(2).max()
        l2 = low.rolling(2).min()
        gamma = np.log(h2 / l2) ** 2
        denom = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        alpha = alpha.clip(lower=0)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        return spread

    def rolling_entropy(close, window=50, n_bins=5):
        returns = close.pct_change().dropna()
        result = pd.Series(index=returns.index, dtype=float)
        for i in range(window, len(returns)):
            wr = returns.iloc[i - window: i].values
            quantiles = np.percentile(wr[~np.isnan(wr)], np.linspace(0, 100, n_bins + 1))
            quantiles[0] = -np.inf
            quantiles[-1] = np.inf
            indices = np.digitize(wr, quantiles[1:])
            msg = "".join(str(int(x)) for x in indices)
            dictionary = set()
            complexity = 0
            ii = 0
            while ii < len(msg):
                for jj in range(ii + 1, len(msg) + 1):
                    substr = msg[ii:jj]
                    if substr not in dictionary:
                        dictionary.add(substr)
                        complexity += 1
                        ii = jj
                        break
                else:
                    break
            result.iloc[i] = complexity / len(msg) if len(msg) > 0 else 0
        return result

    def get_signal(probability, num_classes=2):
        p = np.asarray(probability, dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        z = (p - 1.0 / num_classes) / np.sqrt(p * (1.0 - p))
        signal = 2.0 * norm.cdf(z) - 1.0
        return float(signal) if np.ndim(probability) == 0 else signal

    def discretize_signal(signal, step_size=0.1):
        return np.round(np.asarray(signal) / step_size) * step_size

    def kelly_size(win_prob, win_return, loss_return):
        if loss_return == 0:
            return 0.0
        b = win_return / abs(loss_return)
        q = 1.0 - win_prob
        return (b * win_prob - q) / b

    def deflated_sharpe_ratio(observed_sr, num_trials, num_observations,
                              skewness=0.0, kurtosis=3.0, sr_variance=None):
        if sr_variance is None:
            sr_variance = 1.0
        gamma = 0.5772
        sr_0 = np.sqrt(sr_variance) * (
            (1 - gamma) * norm.ppf(1 - 1.0 / num_trials)
            + gamma * norm.ppf(1 - 1.0 / (num_trials * np.e))
        )
        sigma_sr = np.sqrt(
            (1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr ** 2)
            / (num_observations - 1)
        )
        if sigma_sr == 0:
            return 0.0
        return float(norm.cdf((observed_sr - sr_0) / sigma_sr))


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
HL_CSV_DIR = Path("C:/Claude/ml-trading-system/v3-results")
OUTPUT_DIR = Path("C:/Claude/ml-trading-system/v4-results")
RESULTS_FILE = OUTPUT_DIR / "execution_reality_results.json"

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT",
}

FFD_D_STARS = {
    "BTC": 0.20, "ETH": 0.10, "XRP": 0.30,
    "BNB": 0.10, "SUI": 0.20, "LINK": 0.10,
}

BAR_TF = "30min"
FWD_BARS = 4
ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")

# Portfolio parameters
STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001  # 1 bps (baseline)
MAX_EXPOSURE = 0.60
MAX_PER_ASSET = 0.20
BASE_SIZING_PCT = 0.02

VOL_FILTER_QUANTILE = 0.50
CONFIDENCE_FILTER_PERCENTILE = 70

WINSOR_LO = 0.01
WINSOR_HI = 0.99
FFD_THRESHOLD = 1e-4
SIGNAL_STEP_SIZE = 0.1
MAX_SIGNAL_MULT = 2.0
DSR_NUM_TRIALS = 4095

EMBARGO_BARS = 4
EMBARGO_TIMEDELTA = pd.Timedelta(hours=2)

HMM_N_STATES = 3

# 19 features — funding_zscore DROPPED (proven noise)
BASE_19_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14",
    "bar_range", "vol_zscore", "rsi_14", "stoch_rsi", "macd_hist",
    "bb_pct_b", "atr_14", "dist_swing_high", "dist_swing_low",
    "dist_nearest_level", "ffd", "cs_spread", "entropy",
    "oi_change_5", "impact_spread",
]

# =============================================================================
# EXECUTION SCENARIO PARAMETERS
# =============================================================================

ASSET_SLIPPAGE = {
    "BTC": 0.00005,   # 0.5 bps (very liquid)
    "ETH": 0.00005,   # 0.5 bps
    "XRP": 0.0001,    # 1.0 bps
    "BNB": 0.0001,    # 1.0 bps
    "SUI": 0.0002,    # 2.0 bps (thinner book)
    "LINK": 0.0002,   # 2.0 bps
}

# Default funding rate when historical data unavailable (10 bps / 8h, typical BTC)
DEFAULT_FUNDING_RATE = 0.0001

SCENARIOS = [
    {
        "name": "S1_IDEAL",
        "fill_rate": 1.0,
        "slippage_mode": "flat",
        "slippage_bps": 1.0,
        "funding": False,
        "description": "Current backtest (baseline)",
    },
    {
        "name": "S2_GOOD_FILL",
        "fill_rate": 0.85,
        "slippage_mode": "flat",
        "slippage_bps": 1.0,
        "funding": False,
        "description": "Good maker fill rate",
    },
    {
        "name": "S3_AVG_FILL",
        "fill_rate": 0.75,
        "slippage_mode": "flat",
        "slippage_bps": 1.0,
        "funding": False,
        "description": "Average maker fill rate",
    },
    {
        "name": "S4_BAD_FILL",
        "fill_rate": 0.60,
        "slippage_mode": "flat",
        "slippage_bps": 1.0,
        "funding": False,
        "description": "Bad fill rate (fast markets)",
    },
    {
        "name": "S5_HIGH_SLIP",
        "fill_rate": 1.0,
        "slippage_mode": "flat",
        "slippage_bps": 2.0,
        "funding": False,
        "description": "Higher slippage estimate",
    },
    {
        "name": "S6_ASSET_SLIP",
        "fill_rate": 1.0,
        "slippage_mode": "per_asset",
        "slippage_bps": None,  # uses ASSET_SLIPPAGE dict
        "funding": False,
        "description": "BTC/ETH: 0.5bp, XRP/BNB: 1bp, SUI/LINK: 2bp",
    },
    {
        "name": "S7_WITH_FUND",
        "fill_rate": 1.0,
        "slippage_mode": "flat",
        "slippage_bps": 1.0,
        "funding": True,
        "description": "Add 8-hourly funding payments",
    },
    {
        "name": "S8_REALISTIC",
        "fill_rate": 0.75,
        "slippage_mode": "per_asset",
        "slippage_bps": None,
        "funding": True,
        "description": "Combined realistic scenario",
    },
    {
        "name": "S9_PESSIMISTIC",
        "fill_rate": 0.60,
        "slippage_mode": "per_asset_plus",
        "slippage_bps": 1.0,  # extra 1bp added to per-asset
        "funding": True,
        "description": "Worst realistic case",
    },
]


# =============================================================================
# SAMPLE WEIGHTS — AFML Ch 4: Average Uniqueness
# =============================================================================

def compute_sample_weights(timestamps: np.ndarray, fwd_bars: int = FWD_BARS) -> np.ndarray:
    """AFML Ch 4: Average uniqueness for overlapping labels."""
    n = len(timestamps)
    weights = np.ones(n, dtype=np.float64)

    for i in range(n):
        t_start = timestamps[i]
        t_end = timestamps[min(i + fwd_bars, n - 1)]
        concurrent = np.sum((timestamps >= t_start) & (timestamps <= t_end))
        weights[i] = 1.0 / max(concurrent, 1)

    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w

    return weights


# =============================================================================
# DATA LOADING (module-level caches)
# =============================================================================

_ohlcv_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
_hl_cache: Dict[str, Optional[pd.DataFrame]] = {}
_funding_cache: Dict[str, pd.Series] = {}  # raw funding rates per asset


def load_ohlcv(symbol: str, tf: str = BAR_TF) -> pd.DataFrame:
    """Load 1m Binance CSVs, resample to target timeframe. Cached."""
    key = (symbol, tf)
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]

    d = DATA_DIR / SYMBOLS[symbol]
    if not d.exists():
        raise FileNotFoundError(f"No data directory: {d}")

    dfs = []
    for f in sorted(d.glob("*.csv")):
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "count", "tbv", "tqv", "ignore"]
        if first.replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f)
            df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    result = combined.resample(tf).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])

    _ohlcv_cache[key] = result
    return result


def load_hl_derivatives(symbol: str, tf: str = BAR_TF) -> Optional[pd.DataFrame]:
    """Load pre-extracted HL derivatives CSV. Cached."""
    if symbol in _hl_cache:
        return _hl_cache[symbol]

    csv_path = HL_CSV_DIR / f"hl_{symbol}.csv"
    if not csv_path.exists():
        print(f"    [HL] No CSV for {symbol}: {csv_path}", flush=True)
        _hl_cache[symbol] = None
        return None

    combined = pd.read_csv(csv_path)
    combined["time"] = pd.to_datetime(combined["time"], utc=True)
    combined = combined.set_index("time")
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Cache raw funding rates at 1-minute resolution before resampling
    if "funding" in combined.columns:
        funding_raw = combined["funding"].dropna()
        if len(funding_raw) > 0:
            _funding_cache[symbol] = funding_raw

    keep_cols = [c for c in ["funding", "open_interest", "mid_px",
                             "impact_bid_px", "impact_ask_px"]
                 if c in combined.columns]
    if not keep_cols:
        _hl_cache[symbol] = None
        return None

    combined = combined[keep_cols]
    resampled = combined.resample(tf).last().ffill()

    _hl_cache[symbol] = resampled
    return resampled


def preload_all_data():
    """Load all OHLCV and HL data upfront into caches."""
    for symbol in ASSETS:
        try:
            load_ohlcv(symbol, BAR_TF)
        except FileNotFoundError:
            print(f"  [WARN] No OHLCV data for {symbol}", flush=True)
        try:
            load_hl_derivatives(symbol, BAR_TF)
        except Exception as e:
            print(f"  [WARN] HL load failed for {symbol}: {e}", flush=True)


# =============================================================================
# PRECOMPUTED FEATURE CACHES
# =============================================================================

_ffd_cache: Dict[str, pd.Series] = {}
_full_features_cache: Dict[str, pd.DataFrame] = {}
_vol_cache: Dict[str, pd.Series] = {}
_hmm_regime_cache: Dict[str, pd.Series] = {}


def precompute_ffd(symbol: str, close: pd.Series, d_star: float):
    """Compute FFD ONCE for the full series and cache it."""
    if symbol in _ffd_cache:
        return
    try:
        log_close = np.log(close).to_frame("close")
        ffd_result = frac_diff_ffd(log_close, d_star, threshold=FFD_THRESHOLD)
        if len(ffd_result) > 0:
            _ffd_cache[symbol] = ffd_result.iloc[:, 0]
            print(f"    [FFD] {symbol}: d*={d_star}, {len(ffd_result)} values", flush=True)
    except Exception as e:
        print(f"    [FFD] {symbol}: failed ({e})", flush=True)


def _fit_hmm_regime_full(close, vol, volume_z, n_states=HMM_N_STATES):
    """Fit HMM on (return_10bar, vol, volume_z) on full data."""
    ret10 = close.pct_change(10)
    feat_df = pd.DataFrame({
        "ret10": ret10,
        "vol": vol,
        "vol_z": volume_z,
    }).dropna()

    if len(feat_df) < 100:
        return pd.Series(dtype=int)

    if HMM_AVAILABLE:
        try:
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            hmm.fit(feat_df.values)
            regimes = hmm.predict(feat_df.values)
            return pd.Series(regimes, index=feat_df.index, name="regime")
        except Exception:
            pass

    # Fallback: volatility percentile regime
    vol_aligned = vol.reindex(feat_df.index)
    q33 = vol_aligned.quantile(0.33)
    q67 = vol_aligned.quantile(0.67)
    regimes = pd.Series(1, index=feat_df.index, name="regime")
    regimes[vol_aligned < q33] = 0
    regimes[vol_aligned > q67] = 2
    return regimes


def precompute_all_features():
    """Precompute ALL features for all assets on the FULL series.

    19 features (funding_zscore dropped):
    ret_1, ret_3, ret_7, ret_14, bar_range, vol_zscore, rsi_14,
    stoch_rsi, macd_hist, bb_pct_b, atr_14, dist_swing_high, dist_swing_low,
    dist_nearest_level, ffd, cs_spread, entropy, oi_change_5, impact_spread.

    Also precomputes: ewm volatility (for vol filter), HMM regimes.
    """
    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            continue

        t_sym = time.time()
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]

        f = pd.DataFrame(index=ohlcv.index)

        # --- Returns ---
        for lb in [1, 3, 7, 14]:
            f[f"ret_{lb}"] = close.pct_change(lb)

        # --- Technical indicators ---
        f["bar_range"] = (high - low) / close
        f["vol_zscore"] = (
            (volume - volume.rolling(30).mean())
            / volume.rolling(30).std().replace(0, np.nan)
        )
        f["rsi_14"] = rsi(close, 14)
        f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
        f["macd_hist"] = macd(close)["macd_hist"]
        f["bb_pct_b"] = bollinger_pct_b(close, 20)
        f["atr_14"] = atr(high, low, close, 14)

        sw = swing_levels(high, low, lookback=20)
        f["dist_swing_high"] = sw["dist_swing_high"]
        f["dist_swing_low"] = sw["dist_swing_low"]
        f["dist_nearest_level"] = sw["dist_nearest_level"]

        # --- FFD ---
        if symbol in _ffd_cache:
            f["ffd"] = _ffd_cache[symbol].reindex(f.index)
        else:
            try:
                log_close = np.log(close).to_frame("close")
                d_star = FFD_D_STARS.get(symbol, 0.4)
                ffd_result = frac_diff_ffd(log_close, d_star, threshold=FFD_THRESHOLD)
                if len(ffd_result) > 0:
                    f["ffd"] = ffd_result.iloc[:, 0].reindex(f.index)
            except Exception:
                pass

        # --- Corwin-Schultz spread ---
        try:
            cs = corwin_schultz_spread(high, low, window=20)
            f["cs_spread"] = cs.reindex(f.index)
        except Exception:
            pass

        # --- Rolling entropy ---
        try:
            ent = rolling_entropy(close, window=50, n_bins=5)
            f["entropy"] = ent.reindex(f.index)
        except Exception:
            pass

        # --- HL Derivatives (NO funding_zscore) ---
        hl_data = _hl_cache.get(symbol)
        if hl_data is not None:
            if "open_interest" in hl_data.columns:
                try:
                    oi = oi_features(
                        hl_data["open_interest"],
                        close.reindex(hl_data.index, method="ffill"),
                    )
                    if "oi_change_5" in oi.columns:
                        f["oi_change_5"] = oi["oi_change_5"].reindex(f.index)
                except Exception:
                    pass

            if "impact_bid_px" in hl_data.columns and "impact_ask_px" in hl_data.columns:
                try:
                    mid_col = "mid_px" if "mid_px" in hl_data.columns else None
                    if mid_col:
                        imp = impact_features(
                            hl_data["impact_bid_px"],
                            hl_data["impact_ask_px"],
                            hl_data[mid_col],
                        )
                    else:
                        mid = (hl_data["impact_bid_px"] + hl_data["impact_ask_px"]) / 2
                        imp = impact_features(
                            hl_data["impact_bid_px"],
                            hl_data["impact_ask_px"],
                            mid,
                        )
                    if "impact_spread" in imp.columns:
                        f["impact_spread"] = imp["impact_spread"].reindex(f.index)
                except Exception:
                    pass

        _full_features_cache[symbol] = f

        # --- Volatility for vol filter ---
        _vol_cache[symbol] = close.pct_change().dropna().ewm(span=50).std()

        # --- HMM regime on full data ---
        try:
            vol_series = _vol_cache[symbol]
            vol_z_full = (
                (volume - volume.rolling(30).mean())
                / volume.rolling(30).std().replace(0, np.nan)
            )
            regime_full = _fit_hmm_regime_full(close, vol_series, vol_z_full)
            if len(regime_full) > 0:
                _hmm_regime_cache[symbol] = regime_full
        except Exception:
            pass

        elapsed = time.time() - t_sym
        print(
            f"    [Features] {symbol}: {len(f.columns)} cols, {len(f)} rows ({elapsed:.1f}s)",
            flush=True,
        )


# =============================================================================
# FEATURE SLICING (per walk-forward window)
# =============================================================================

def _get_features_for_window(symbol: str, train_end: pd.Timestamp) -> pd.DataFrame:
    """Slice precomputed features and apply per-window return winsorization.

    Uses 19 features (no funding_zscore).
    """
    base = _full_features_cache[symbol]

    cols = [c for c in BASE_19_FEATURES if c in base.columns]
    f = base[cols].copy()

    # Per-window winsorization of returns using training-period quantiles
    train_mask = f.index < train_end
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in f.columns:
            train_vals = f.loc[train_mask, col]
            if len(train_vals.dropna()) > 10:
                lo = train_vals.quantile(WINSOR_LO)
                hi = train_vals.quantile(WINSOR_HI)
                f[col] = f[col].clip(lo, hi)

    return f


# =============================================================================
# SIGNAL GENERATION — LogReg L1 C=0.1 ONLY
# =============================================================================

def _generate_signals_for_asset(symbol: str) -> List[Tuple]:
    """Walk-forward signal generation for one asset with LogReg L1 C=0.1.

    Returns list of (timestamp, symbol, direction, fwd_return, confidence,
                     probability, signal_strength) tuples.
    """
    if symbol not in _full_features_cache:
        return []

    ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
    if ohlcv is None:
        return []

    close = ohlcv["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = _vol_cache.get(symbol)
    full_regime = _hmm_regime_cache.get(symbol)

    rows = []

    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        # Embargo: train cutoff is 2h before test start
        train_cutoff = ms - EMBARGO_TIMEDELTA

        # Slice features with per-window winsorization (19 features, no funding_zscore)
        feat = _get_features_for_window(symbol, train_cutoff)

        # Add regime feature + find worst regime
        worst_regime = None
        if full_regime is not None:
            regime_col = full_regime.reindex(feat.index)
            feat["regime"] = regime_col.fillna(1).astype(int)

            train_regimes = full_regime[full_regime.index < train_cutoff]
            if len(train_regimes) > 50:
                train_fwd = fwd_ret.reindex(train_regimes.index).dropna()
                common = train_regimes.index.intersection(train_fwd.index)
                if len(common) > 50:
                    regime_returns = pd.DataFrame({
                        "regime": train_regimes.loc[common],
                        "fwd_ret": train_fwd.loc[common],
                    }).groupby("regime")["fwd_ret"].mean()
                    worst_regime = regime_returns.idxmin()

        # Clean: drop rows with any NaN feature
        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        # Train/test split
        tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 500 or len(te) < 20:
            continue

        # Vol filter: top 50% volatility bars in test
        if vol is not None:
            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 50:
                continue
            vt = tv.quantile(VOL_FILTER_QUANTILE)
            hv = (vol.reindex(te) > vt).fillna(False)
            te = te[hv.values]
        else:
            continue

        if len(te) < 10:
            continue

        X_train = fc.loc[tr]
        y_train_binary = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        # Sample weights
        train_timestamps = X_train.index.values.astype(np.int64)
        sample_w = compute_sample_weights(train_timestamps, fwd_bars=FWD_BARS)

        # Build and fit LogReg L1 C=0.1
        try:
            model = LogisticRegression(
                C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42,
            )

            scaler = StandardScaler()
            X_tr_arr = scaler.fit_transform(X_train)
            X_te_arr = scaler.transform(X_test)

            model.fit(X_tr_arr, y_train_binary, sample_weight=sample_w)

            yp = model.predict_proba(X_te_arr)[:, 1]

        except Exception:
            continue

        # Confidence filter (top 30%)
        conf = np.abs(yp - 0.5)
        if len(conf) > 10:
            conf_threshold = np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
            conf_mask = conf >= conf_threshold
        else:
            conf_mask = np.ones(len(conf), dtype=bool)

        # Regime suppression (suppress worst regime trades)
        regime_mask = np.ones(len(yp), dtype=bool)
        if worst_regime is not None and "regime" in X_test.columns:
            regime_mask = X_test["regime"].values != worst_regime

        # Combined filter
        combined_mask = conf_mask & regime_mask

        yp_masked = yp[combined_mask]
        conf_masked = conf[combined_mask]
        te_masked = te[combined_mask]
        fwd_ret_masked = fwd_ret.reindex(te_masked)

        if len(yp_masked) == 0:
            continue

        directions = np.where(yp_masked > 0.5, 1.0, -1.0)

        # Signal strength for sizing
        signals = get_signal(yp_masked, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        for j in range(len(yp_masked)):
            idx = te_masked[j]
            fr = fwd_ret_masked.iloc[j] if hasattr(fwd_ret_masked, "iloc") else fwd_ret_masked.loc[idx]
            if np.isfinite(fr):
                rows.append((
                    idx, symbol, directions[j],
                    float(fr),
                    float(conf_masked[j]),
                    float(yp_masked[j]),
                    float(signals_disc[j]) if np.ndim(signals_disc) > 0 else float(signals_disc),
                ))

    return rows


def generate_all_signals(max_workers: int = 3) -> pd.DataFrame:
    """Generate walk-forward signals for all assets in parallel (LogReg L1 C=0.1)."""
    all_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for symbol in ASSETS:
            fut = pool.submit(_generate_signals_for_asset, symbol)
            futures[fut] = symbol

        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"    {symbol}: {len(rows)} signals", flush=True)
            except Exception as e:
                print(f"    {symbol}: ERROR — {e}", flush=True)
                traceback.print_exc()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "asset", "direction", "fwd_return",
        "confidence", "probability", "signal_strength",
    ])
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# =============================================================================
# FUNDING COST MODEL
# =============================================================================

def _get_funding_rate(symbol: str, timestamp: pd.Timestamp) -> float:
    """Look up historical funding rate for a symbol at a timestamp.

    Funding data is from HL CSVs at ~1min resolution.
    If exact timestamp not found, use nearest available.
    If no data at all for symbol, use DEFAULT_FUNDING_RATE.
    """
    if symbol not in _funding_cache:
        return DEFAULT_FUNDING_RATE

    funding_series = _funding_cache[symbol]
    if len(funding_series) == 0:
        return DEFAULT_FUNDING_RATE

    # Find nearest funding rate
    try:
        ts = pd.Timestamp(timestamp)
        if ts in funding_series.index:
            val = funding_series.loc[ts]
            if np.isfinite(val):
                return float(val)

        # Use nearest timestamp within 4 hours
        idx = funding_series.index.get_indexer([ts], method="nearest")[0]
        if 0 <= idx < len(funding_series):
            val = funding_series.iloc[idx]
            if np.isfinite(val):
                return float(val)
    except Exception:
        pass

    # Fallback: median funding rate for this asset
    median_rate = funding_series.median()
    if np.isfinite(median_rate):
        return float(median_rate)

    return DEFAULT_FUNDING_RATE


def _compute_funding_cost(timestamp: pd.Timestamp, symbol: str,
                          direction: float, hold_hours: float = 2.0) -> float:
    """Estimate funding payment during hold period.

    Funding settles every 8h (00:00, 08:00, 16:00 UTC).
    If our hold period [t, t+hold_hours] spans a settlement, we pay/receive funding.
    Direction: +1 (long) pays positive funding, -1 (short) receives positive funding.

    Returns the funding cost as a fraction of position size (positive = cost).
    """
    try:
        ts = pd.Timestamp(timestamp)
        t_hour = ts.hour + ts.minute / 60.0

        settlements = [0.0, 8.0, 16.0, 24.0]  # include 24 for wrap-around

        t_end = t_hour + hold_hours

        for s in settlements:
            if t_hour < s <= t_end:
                # This settlement falls within our hold period
                funding_rate = _get_funding_rate(symbol, ts)
                # Long pays positive funding, short receives it
                # Cost is positive when we pay
                return direction * funding_rate

        # Handle wrap-around: if t_end > 24, check 0-hour settlement
        if t_end > 24.0:
            t_end_wrapped = t_end - 24.0
            for s in [0.0, 8.0]:
                if s <= t_end_wrapped:
                    funding_rate = _get_funding_rate(symbol, ts)
                    return direction * funding_rate
    except Exception:
        pass

    return 0.0


# =============================================================================
# PORTFOLIO SIMULATION — WITH EXECUTION PARAMETERS
# =============================================================================

def simulate_portfolio_with_execution(
    signals_df: pd.DataFrame,
    scenario: dict,
) -> Optional[dict]:
    """Portfolio simulation with execution reality parameters.

    Parameters from scenario:
    - fill_rate: probability of fill (< 1.0 means random drops)
    - slippage_mode: 'flat', 'per_asset', or 'per_asset_plus'
    - slippage_bps: flat slippage in bps (or extra bps for per_asset_plus)
    - funding: whether to apply funding costs
    """
    if len(signals_df) < 20:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = sub["signal_strength"].values.astype(np.float64)

    fill_rate = scenario["fill_rate"]
    slippage_mode = scenario["slippage_mode"]
    slippage_bps_flat = scenario.get("slippage_bps", 1.0)
    use_funding = scenario["funding"]

    # Generate fill mask (reproducible per scenario)
    if fill_rate < 1.0:
        # Use scenario name hash as seed for reproducibility across runs
        seed = hash(scenario["name"]) % (2**31)
        rng = np.random.RandomState(seed)
        fill_mask = rng.random(n) < fill_rate
    else:
        fill_mask = np.ones(n, dtype=bool)

    equity_arr = np.empty(n)
    pnl_arr = np.zeros(n)
    dd_arr = np.empty(n)
    size_arr = np.zeros(n)
    skipped = np.ones(n, dtype=bool)
    funding_costs_arr = np.zeros(n)

    equity = STARTING_CAPITAL
    peak_equity = equity
    halted = False

    current_exposure = 0.0
    asset_exposure: dict = {}

    total_funding_paid = 0.0

    for i in range(n):
        if halted:
            equity_arr[i] = equity
            dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            continue

        # Check fill
        if not fill_mask[i]:
            # Trade not filled — 0 PnL, skip
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
        sig_abs = abs(signal_strengths[i])
        raw_size = equity * BASE_SIZING_PCT * sig_abs * size_mult
        raw_size = min(raw_size, equity * BASE_SIZING_PCT * MAX_SIGNAL_MULT * size_mult)

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

        # Determine slippage for this trade
        if slippage_mode == "flat":
            slip = slippage_bps_flat * 0.0001  # convert bps to fraction
        elif slippage_mode == "per_asset":
            slip = ASSET_SLIPPAGE.get(asset, 0.0001)
        elif slippage_mode == "per_asset_plus":
            slip = ASSET_SLIPPAGE.get(asset, 0.0001) + slippage_bps_flat * 0.0001
        else:
            slip = SLIPPAGE

        # Execute trade
        trade_return = directions[i] * fwd_returns[i] - slip

        # Apply funding cost if enabled
        funding_cost = 0.0
        if use_funding:
            ts = pd.Timestamp(timestamps[i])
            funding_cost = _compute_funding_cost(ts, asset, directions[i], hold_hours=2.0)
            trade_return -= abs(funding_cost)  # funding is always a drag in expected value
            total_funding_paid += abs(funding_cost) * raw_size

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
        funding_costs_arr[i] = abs(funding_cost) * raw_size
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
        "total_funding_paid": total_funding_paid,
        "funding_costs": funding_costs_arr,
    }


# =============================================================================
# METRICS (extended with edge_bps)
# =============================================================================

def compute_metrics(sim: Optional[dict], label: str = "") -> Optional[dict]:
    """Compute standard quant metrics from simulation output."""
    if sim is None or sim["n_trades"] < 20:
        return None

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
    sizes = sim["sizes"][active]
    timestamps = sim["timestamps"]
    n_trades = sim["n_trades"]

    t0 = pd.Timestamp(timestamps[active][0])
    t1 = pd.Timestamp(timestamps[active][-1])
    days = max((t1 - t0).total_seconds() / 86400, 1)
    years = days / 365.25
    trades_per_day = n_trades / days

    total_ret = sim["final_equity"] / STARTING_CAPITAL - 1
    cagr = (
        (sim["final_equity"] / STARTING_CAPITAL) ** (1.0 / years) - 1
        if years > 0
        else 0
    )
    max_dd = float(sim["dd"].max())

    ts_series = pd.Series(pnls, index=pd.to_datetime(timestamps[active]))
    daily = ts_series.resample("1D").sum()
    daily = daily[daily.index >= t0]
    daily_nonzero = daily[daily != 0]

    if len(daily_nonzero) < 5:
        return None

    mean_d = daily_nonzero.mean()
    std_d = daily_nonzero.std()
    down_d = daily_nonzero[daily_nonzero < 0]
    down_std = down_d.std() if len(down_d) > 1 else std_d

    sharpe = mean_d / std_d * np.sqrt(365.25) if std_d > 0 else 0.0
    sortino = mean_d / down_std * np.sqrt(365.25) if down_std > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    win_trades = int((pnls > 0).sum())
    win_rate_t = win_trades / n_trades
    win_days = int((daily_nonzero > 0).sum())
    win_rate_d = win_days / len(daily_nonzero)

    gains = pnls[pnls > 0].sum()
    losses_sum = abs(pnls[pnls < 0].sum())
    pf = gains / losses_sum if losses_sum > 0 else 99.0

    # Edge in basis points: mean PnL per trade / mean position size * 10000
    edge_bps = 0.0
    if len(sizes) > 0 and sizes.mean() > 0:
        edge_bps = (pnls.mean() / sizes.mean()) * 10000

    # DSR
    sharpe_unann = mean_d / std_d if std_d > 0 else 0.0
    num_obs = len(daily_nonzero)
    skewness = float(daily_nonzero.skew()) if len(daily_nonzero) > 3 else 0.0
    kurt = float(daily_nonzero.kurtosis()) + 3.0 if len(daily_nonzero) > 4 else 3.0
    dsr = deflated_sharpe_ratio(
        observed_sr=sharpe_unann,
        num_trials=DSR_NUM_TRIALS,
        num_observations=num_obs,
        skewness=skewness,
        kurtosis=kurt,
    )

    return {
        "label": label,
        "n_trades": n_trades,
        "trades_per_day": round(trades_per_day, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "final_equity": round(sim["final_equity"], 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_trades": round(win_rate_t, 4),
        "win_rate_days": round(win_rate_d, 4),
        "profit_factor": round(min(pf, 99.0), 3),
        "mean_daily_pnl": round(mean_d, 2),
        "halted": sim["halted"],
        "dsr": round(dsr, 4),
        "edge_bps": round(edge_bps, 2),
        "total_funding_paid": round(sim.get("total_funding_paid", 0.0), 2),
    }


# =============================================================================
# BREAK-EVEN FILL RATE COMPUTATION
# =============================================================================

def find_breakeven_fill_rate(signals_df: pd.DataFrame) -> float:
    """Binary search for the fill rate where Sharpe crosses 0.

    Uses flat 1bp slippage, no funding (isolating fill rate effect).
    """
    lo, hi = 0.10, 1.0
    max_iter = 20

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        scenario = {
            "name": f"breakeven_{mid:.3f}",
            "fill_rate": mid,
            "slippage_mode": "flat",
            "slippage_bps": 1.0,
            "funding": False,
        }
        sim = simulate_portfolio_with_execution(signals_df, scenario)
        m = compute_metrics(sim, f"fill={mid:.3f}")
        if m is None:
            # Not enough trades — fill rate too low
            lo = mid
            continue

        if m["sharpe"] > 0:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) < 0.005:
            break

    return round((lo + hi) / 2.0, 3)


def find_breakeven_fill_rate_realistic(signals_df: pd.DataFrame) -> float:
    """Binary search for fill rate where Sharpe crosses 0 under realistic conditions.

    Uses per-asset slippage + funding (isolating fill rate with other costs present).
    """
    lo, hi = 0.10, 1.0
    max_iter = 20

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        scenario = {
            "name": f"breakeven_real_{mid:.3f}",
            "fill_rate": mid,
            "slippage_mode": "per_asset",
            "slippage_bps": None,
            "funding": True,
        }
        sim = simulate_portfolio_with_execution(signals_df, scenario)
        m = compute_metrics(sim, f"fill={mid:.3f}")
        if m is None:
            lo = mid
            continue

        if m["sharpe"] > 0:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) < 0.005:
            break

    return round((lo + hi) / 2.0, 3)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    print("=" * 100, flush=True)
    print("EXECUTION REALITY TEST — LogReg L1 C=0.1 under realistic execution", flush=True)
    print("=" * 100, flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD: {MAX_DD*100:.0f}%, "
          f"Delever: {DELEVER_START*100:.1f}%", flush=True)
    print(f"Sizing: {BASE_SIZING_PCT*100:.0f}%/trade, MaxExposure: {MAX_EXPOSURE*100:.0f}%, "
          f"MaxPerAsset: {MAX_PER_ASSET*100:.0f}%", flush=True)
    print(f"Bars: {BAR_TF}, Fwd: {FWD_BARS} bars ({FWD_BARS*30}m)", flush=True)
    print(f"Embargo: {EMBARGO_BARS} bars ({EMBARGO_TIMEDELTA})", flush=True)
    print(f"Assets: {ASSETS}", flush=True)
    print(f"HMM: {HMM_AVAILABLE}", flush=True)
    print(f"Features: {len(BASE_19_FEATURES)} (funding_zscore dropped)", flush=True)
    print(f"Scenarios: {len(SCENARIOS)}", flush=True)
    print("=" * 100, flush=True)

    # =========================================================================
    # PHASE 0: Load data and precompute features
    # =========================================================================
    print("\n[PHASE 0] Loading data and precomputing features...", flush=True)
    t0 = time.time()

    preload_all_data()
    print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)

    # Report funding data availability
    print(f"  Funding data available for: {list(_funding_cache.keys())}", flush=True)
    for sym, fs in _funding_cache.items():
        print(f"    {sym}: {len(fs)} rates, median={fs.median():.6f}, "
              f"range=[{fs.min():.6f}, {fs.max():.6f}]", flush=True)

    print("  Precomputing FFD...", flush=True)
    t_ffd = time.time()
    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is not None:
            precompute_ffd(symbol, ohlcv["close"], FFD_D_STARS.get(symbol, 0.4))
    print(f"  FFD done in {time.time()-t_ffd:.1f}s", flush=True)

    print("  Precomputing all features (technicals, HL derivs, HMM)...", flush=True)
    t_feat = time.time()
    precompute_all_features()
    print(f"  Features done in {time.time()-t_feat:.1f}s", flush=True)

    phase0_time = time.time() - t0
    print(f"\n  Phase 0 total: {phase0_time:.1f}s", flush=True)

    # =========================================================================
    # PHASE 1: Generate signals ONCE (the slow part)
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("[PHASE 1] Generating signals — LogReg L1 C=0.1 (walk-forward)...", flush=True)
    print(f"{'='*100}", flush=True)

    t_sig = time.time()
    signals_df = generate_all_signals(max_workers=3)
    sig_time = time.time() - t_sig

    print(f"\n  Signals generated: {len(signals_df)} total in {sig_time:.1f}s", flush=True)
    if len(signals_df) > 0:
        per_asset = signals_df.groupby("asset").size()
        for asset, count in per_asset.items():
            print(f"    {asset}: {count} signals", flush=True)

    if len(signals_df) < 100:
        print("\n[FATAL] Too few signals generated. Cannot proceed.", flush=True)
        sys.exit(1)

    # =========================================================================
    # PHASE 2: Simulate 9 execution scenarios (the fast part)
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print(f"[PHASE 2] Simulating {len(SCENARIOS)} execution scenarios...", flush=True)
    print(f"{'='*100}", flush=True)

    scenario_results = []

    for idx, scenario in enumerate(SCENARIOS):
        t_sc = time.time()
        print(f"\n  [{idx+1}/{len(SCENARIOS)}] {scenario['name']}: {scenario['description']}...",
              flush=True)

        sim = simulate_portfolio_with_execution(signals_df, scenario)
        m = compute_metrics(sim, scenario["name"])

        elapsed = time.time() - t_sc

        if m:
            # Build fill/slip/funding display strings
            fill_str = f"{scenario['fill_rate']*100:.0f}%"
            if scenario['slippage_mode'] == 'flat':
                slip_str = f"{scenario.get('slippage_bps', 1.0):.1f}bp"
            elif scenario['slippage_mode'] == 'per_asset':
                slip_str = "per-asset"
            else:
                slip_str = f"per-asset+{scenario.get('slippage_bps', 1.0):.0f}bp"
            fund_str = "yes" if scenario['funding'] else "no"

            print(f"    Fill={fill_str}, Slip={slip_str}, Funding={fund_str}", flush=True)
            print(f"    Sharpe={m['sharpe']:.3f}, Return={m['total_return_pct']:+.2f}%, "
                  f"MaxDD={m['max_dd_pct']:.2f}%, Trades={m['n_trades']}, "
                  f"Edge={m['edge_bps']:.1f}bps ({elapsed:.1f}s)", flush=True)
            if scenario['funding'] and m['total_funding_paid'] > 0:
                print(f"    Funding paid: ${m['total_funding_paid']:.2f}", flush=True)

            scenario_results.append({
                "scenario": scenario,
                "metrics": m,
            })
        else:
            print(f"    INSUFFICIENT DATA ({elapsed:.1f}s)", flush=True)
            scenario_results.append({
                "scenario": scenario,
                "metrics": None,
            })

    # =========================================================================
    # PHASE 3: Break-even fill rate analysis
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("[PHASE 3] Break-even fill rate analysis...", flush=True)
    print(f"{'='*100}", flush=True)

    t_be = time.time()
    print("  Finding break-even fill rate (flat 1bp slippage, no funding)...", flush=True)
    breakeven_flat = find_breakeven_fill_rate(signals_df)
    print(f"    Break-even (flat): {breakeven_flat*100:.1f}% fill rate", flush=True)

    print("  Finding break-even fill rate (per-asset slippage + funding)...", flush=True)
    breakeven_realistic = find_breakeven_fill_rate_realistic(signals_df)
    print(f"    Break-even (realistic): {breakeven_realistic*100:.1f}% fill rate", flush=True)

    be_time = time.time() - t_be
    print(f"  Break-even analysis done in {be_time:.1f}s", flush=True)

    # =========================================================================
    # RESULTS: Summary table
    # =========================================================================
    total_time = time.time() - overall_start

    print(f"\n{'='*100}", flush=True)
    print("=== EXECUTION REALITY TEST ===", flush=True)
    print(f"Best model: LogReg L1 C=0.1 (19 features, no funding_zscore)", flush=True)
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"{'='*100}", flush=True)

    hdr = (
        f"{'Scenario':<20} {'FillRate':>8} {'Slippage':>10} {'Funding':>8} "
        f"{'Sharpe':>7} {'Return%':>9} {'MaxDD%':>7} {'Trades':>7} {'Edge_bps':>9}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for sr in scenario_results:
        sc = sr["scenario"]
        m = sr["metrics"]

        fill_str = f"{sc['fill_rate']*100:.0f}%"
        if sc['slippage_mode'] == 'flat':
            slip_str = f"{sc.get('slippage_bps', 1.0):.1f}bp"
        elif sc['slippage_mode'] == 'per_asset':
            slip_str = "per-asset"
        else:
            slip_str = f"pa+{sc.get('slippage_bps', 1.0):.0f}bp"
        fund_str = "yes" if sc['funding'] else "no"

        if m:
            print(
                f"{sc['name']:<20} {fill_str:>8} {slip_str:>10} {fund_str:>8} "
                f"{m['sharpe']:>7.2f} {m['total_return_pct']:>+8.2f}% "
                f"{m['max_dd_pct']:>7.2f} {m['n_trades']:>7} {m['edge_bps']:>8.1f}",
                flush=True,
            )
        else:
            print(
                f"{sc['name']:<20} {fill_str:>8} {slip_str:>10} {fund_str:>8} "
                f"{'N/A':>7} {'N/A':>9} {'N/A':>7} {'N/A':>7} {'N/A':>9}",
                flush=True,
            )

    # =========================================================================
    # VERDICT
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'='*100}", flush=True)

    # Find S1 (baseline) and S8 (realistic)
    s1_m = next((sr["metrics"] for sr in scenario_results
                 if sr["scenario"]["name"] == "S1_IDEAL"), None)
    s8_m = next((sr["metrics"] for sr in scenario_results
                 if sr["scenario"]["name"] == "S8_REALISTIC"), None)
    s9_m = next((sr["metrics"] for sr in scenario_results
                 if sr["scenario"]["name"] == "S9_PESSIMISTIC"), None)

    if s1_m:
        print(f"\n  Baseline Sharpe (S1 ideal):        {s1_m['sharpe']:.2f}", flush=True)
    if s8_m:
        print(f"  Realistic Sharpe (S8):             {s8_m['sharpe']:.2f}", flush=True)
        if s1_m:
            degradation = (1 - s8_m['sharpe'] / s1_m['sharpe']) * 100 if s1_m['sharpe'] != 0 else 0
            print(f"  Sharpe degradation (ideal->real):  {degradation:.1f}%", flush=True)
    if s9_m:
        print(f"  Pessimistic Sharpe (S9):           {s9_m['sharpe']:.2f}", flush=True)

    print(f"\n  Break-even fill rate (flat slip):    {breakeven_flat*100:.1f}%", flush=True)
    print(f"  Break-even fill rate (realistic):    {breakeven_realistic*100:.1f}%", flush=True)

    # Viability verdict
    viable = False
    if s8_m and s8_m["sharpe"] > 0.5:
        print("\n  >> The system IS VIABLE under realistic execution (Sharpe > 0.5).", flush=True)
        viable = True
        if s8_m["sharpe"] > 1.0:
            print("  >> STRONG: Sharpe > 1.0 even at realistic conditions.", flush=True)
        if s8_m["sharpe"] > 1.5:
            print("  >> EXCELLENT: Sharpe > 1.5 at realistic conditions — production-ready.", flush=True)
    elif s8_m and s8_m["sharpe"] > 0:
        print("\n  >> The system is MARGINAL under realistic execution (Sharpe < 0.5).", flush=True)
        print("  >> Needs execution optimization (better fills, reduced slippage) to be viable.", flush=True)
    elif s8_m:
        print("\n  >> The system is NOT VIABLE under realistic execution (Sharpe < 0).", flush=True)
        print("  >> The edge does not survive transaction costs.", flush=True)
    else:
        print("\n  >> Cannot determine viability — insufficient data for realistic scenario.", flush=True)

    # Sensitivity analysis
    print(f"\n  Sensitivity analysis:", flush=True)

    # Fill rate sensitivity
    fill_scenarios = [sr for sr in scenario_results
                      if sr["scenario"]["slippage_mode"] == "flat"
                      and sr["scenario"].get("slippage_bps", 1.0) == 1.0
                      and not sr["scenario"]["funding"]
                      and sr["metrics"] is not None]
    if len(fill_scenarios) >= 2:
        fill_sharpes = [(sr["scenario"]["fill_rate"], sr["metrics"]["sharpe"])
                        for sr in fill_scenarios]
        fill_sharpes.sort(key=lambda x: x[0], reverse=True)
        sharpe_100 = next((s for f, s in fill_sharpes if f == 1.0), None)
        sharpe_60 = next((s for f, s in fill_sharpes if f == 0.60), None)
        if sharpe_100 is not None and sharpe_60 is not None:
            sharpe_per_10pct = (sharpe_100 - sharpe_60) / 4.0
            print(f"    Fill rate: ~{sharpe_per_10pct:.3f} Sharpe per 10% fill improvement", flush=True)

    # Slippage sensitivity
    s1_sharpe = s1_m["sharpe"] if s1_m else None
    s5_m = next((sr["metrics"] for sr in scenario_results
                 if sr["scenario"]["name"] == "S5_HIGH_SLIP"), None)
    if s1_sharpe is not None and s5_m is not None:
        slip_sensitivity = s1_sharpe - s5_m["sharpe"]
        print(f"    Slippage: {slip_sensitivity:.3f} Sharpe lost per extra 1bp", flush=True)

    # Funding impact
    s7_m = next((sr["metrics"] for sr in scenario_results
                 if sr["scenario"]["name"] == "S7_WITH_FUND"), None)
    if s1_sharpe is not None and s7_m is not None:
        fund_impact = s1_sharpe - s7_m["sharpe"]
        print(f"    Funding: {fund_impact:.3f} Sharpe impact from funding costs", flush=True)

    # =========================================================================
    # Save results
    # =========================================================================
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "version": "execution_reality_v1",
        "total_runtime_s": round(total_time, 1),
        "model": "LogReg L1 C=0.1",
        "n_features": len(BASE_19_FEATURES),
        "features": BASE_19_FEATURES,
        "total_signals": len(signals_df),
        "signals_per_asset": signals_df.groupby("asset").size().to_dict() if len(signals_df) > 0 else {},
        "config": {
            "starting_capital": STARTING_CAPITAL,
            "max_dd": MAX_DD,
            "max_exposure": MAX_EXPOSURE,
            "max_per_asset": MAX_PER_ASSET,
            "base_sizing_pct": BASE_SIZING_PCT,
            "bar_tf": BAR_TF,
            "fwd_bars": FWD_BARS,
            "embargo_bars": EMBARGO_BARS,
            "assets": ASSETS,
            "months": [str(m) for m in ALL_MONTHS],
            "asset_slippage_bps": {k: v * 10000 for k, v in ASSET_SLIPPAGE.items()},
        },
        "funding_data": {
            sym: {
                "n_rates": len(fs),
                "median": round(float(fs.median()), 8),
                "mean": round(float(fs.mean()), 8),
            }
            for sym, fs in _funding_cache.items()
        },
        "scenarios": [
            {
                "name": sr["scenario"]["name"],
                "description": sr["scenario"]["description"],
                "fill_rate": sr["scenario"]["fill_rate"],
                "slippage_mode": sr["scenario"]["slippage_mode"],
                "slippage_bps": sr["scenario"].get("slippage_bps"),
                "funding": sr["scenario"]["funding"],
                "metrics": sr["metrics"],
            }
            for sr in scenario_results
        ],
        "breakeven": {
            "flat_slippage_no_funding": breakeven_flat,
            "per_asset_slippage_with_funding": breakeven_realistic,
        },
        "verdict": {
            "baseline_sharpe": s1_m["sharpe"] if s1_m else None,
            "realistic_sharpe": s8_m["sharpe"] if s8_m else None,
            "pessimistic_sharpe": s9_m["sharpe"] if s9_m else None,
            "viable": viable,
            "breakeven_fill_rate_flat": breakeven_flat,
            "breakeven_fill_rate_realistic": breakeven_realistic,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_FILE}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"DONE. Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"{'='*100}", flush=True)
