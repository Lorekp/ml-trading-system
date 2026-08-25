"""
Tier 2 Verification — 12 Tests.

Deeper robustness checks beyond Tier 1: distribution stability, rolling vs expanding,
asset subset sensitivity, confidence/vol threshold sensitivity, timeframe sensitivity,
position flip cost, flash crash behavior, FFD d* stability, cross-exchange alignment,
data completeness, and probabilistic Sharpe ratio.

Model: LogReg L1 C=0.1 with StandardScaler, 19 features (no funding_zscore),
4-bar embargo, sample weights, vol filter 50%, confidence filter 70th pctile,
HMM regime suppression.
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
from scipy.stats import norm, linregress
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

# Import AFML modules with inline fallback
try:
    from afml.fracdiff import frac_diff_ffd, find_min_d
    from afml.derivatives import funding_features, oi_features, impact_features
    from afml.features import corwin_schultz_spread, rolling_entropy
    from afml.bet_sizing import get_signal, discretize_signal
    from afml.evaluation import deflated_sharpe_ratio
    AFML_IMPORTED = True
except ImportError:
    AFML_IMPORTED = False
    print("[WARN] Could not import afml modules -- using inline implementations", flush=True)

# Optional HMM for regime filter
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARN] hmmlearn not available -- volatility percentile fallback", flush=True)


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

    def find_min_d(series, d_range=None, threshold=1e-5, adf_pvalue=0.05, adf_maxlag=1):
        from statsmodels.tsa.stattools import adfuller
        if d_range is None:
            d_range = np.linspace(0, 1, 21)
        log_prices = np.log(series).to_frame("price")
        records = []
        for d in d_range:
            if d == 0:
                ffd = log_prices.dropna()
            else:
                ffd = frac_diff_ffd(log_prices, d, threshold)
            if len(ffd) < 20:
                records.append({"d": d, "adf_stat": np.nan, "p_value": 1.0, "correlation": np.nan})
                continue
            ffd_values = ffd.iloc[:, 0].dropna()
            try:
                adf_result = adfuller(ffd_values, maxlag=adf_maxlag, regression="c", autolag=None)
                adf_stat = adf_result[0]
                p_value = adf_result[1]
            except Exception:
                adf_stat = np.nan
                p_value = 1.0
            common_idx = log_prices.index.intersection(ffd.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(
                    log_prices.loc[common_idx, "price"].values,
                    ffd.loc[common_idx].iloc[:, 0].values,
                )[0, 1]
            else:
                corr = np.nan
            records.append({"d": d, "adf_stat": adf_stat, "p_value": p_value, "correlation": corr})
        results_df = pd.DataFrame(records)
        stationary = results_df[results_df["p_value"] < adf_pvalue]
        d_star = float(stationary["d"].min()) if len(stationary) > 0 else None
        return {"d_star": d_star, "results": results_df}

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
SLIPPAGE = 0.0001  # 1 bps
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

# Embargo: 4 bars * 30 min = 2 hours gap
EMBARGO_BARS = 4
EMBARGO_TIMEDELTA = pd.Timedelta(hours=2)

# HMM regime parameters
HMM_N_STATES = 3

# 19 features -- funding_zscore DROPPED (proven noise)
BASE_19_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14",
    "bar_range", "vol_zscore", "rsi_14", "stoch_rsi", "macd_hist",
    "bb_pct_b", "atr_14", "dist_swing_high", "dist_swing_low",
    "dist_nearest_level", "ffd", "cs_spread", "entropy",
    "oi_change_5", "impact_spread",
]


# =============================================================================
# SAMPLE WEIGHTS -- AFML Ch 4: Average Uniqueness
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
        _hl_cache[symbol] = None
        return None

    combined = pd.read_csv(csv_path)
    combined["time"] = pd.to_datetime(combined["time"], utc=True)
    combined = combined.set_index("time")
    for col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    keep_cols = [c for c in ["funding", "open_interest", "mid_px",
                             "impact_bid_px", "impact_ask_px", "mark_px"]
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

_full_features_cache: Dict[str, pd.DataFrame] = {}
_vol_cache: Dict[str, pd.Series] = {}
_hmm_regime_cache: Dict[str, pd.Series] = {}
_vol_regime_cache: Dict[str, pd.Series] = {}


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


def _compute_vol_regime(vol):
    """Simple vol percentile regime: 0=low (<33rd), 1=medium, 2=high (>67th)."""
    q33 = vol.quantile(0.33)
    q67 = vol.quantile(0.67)
    regime = pd.Series(1, index=vol.index, name="vol_regime")
    regime[vol < q33] = 0
    regime[vol > q67] = 2
    return regime


def _compute_features_for_ohlcv(symbol, ohlcv, d_star=None):
    """Compute all 19 features for given OHLCV data.

    This is extracted to allow re-use for different timeframes (Test 6).
    """
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
    if d_star is None:
        d_star = FFD_D_STARS.get(symbol, 0.4)
    try:
        log_close = np.log(close).to_frame("close")
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

    return f


def precompute_all_features():
    """Precompute ALL 19 features for all assets on the FULL series.

    Also precomputes: ewm volatility (for vol filter), HMM regimes,
    simple vol percentile regimes.
    """
    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            continue

        t_sym = time.time()
        close = ohlcv["close"]
        volume = ohlcv["volume"]

        f = _compute_features_for_ohlcv(symbol, ohlcv)
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

        # --- Simple vol percentile regime ---
        try:
            vol_series = _vol_cache[symbol]
            _vol_regime_cache[symbol] = _compute_vol_regime(vol_series)
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

def _get_features_for_window(
    symbol: str,
    train_end: pd.Timestamp,
    features_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Slice precomputed features and apply per-window return winsorization."""
    cache = features_cache if features_cache is not None else _full_features_cache
    base = cache[symbol]

    cols = [c for c in BASE_19_FEATURES if c in base.columns]
    f = base[cols].copy()

    # Per-window winsorization of returns using training-period quantiles
    train_mask = f.index < train_end
    winsor_info = {"n_train": int(train_mask.sum()), "train_end": str(train_end),
                   "used_test_data": False, "quantiles": {}}
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in f.columns:
            train_vals = f.loc[train_mask, col]
            if len(train_vals.dropna()) > 10:
                lo = train_vals.quantile(WINSOR_LO)
                hi = train_vals.quantile(WINSOR_HI)
                f[col] = f[col].clip(lo, hi)
                winsor_info["quantiles"][col] = {"lo": float(lo), "hi": float(hi)}

    return f, winsor_info


# =============================================================================
# SIGNAL GENERATION -- Walk-Forward with LogReg L1 C=0.1
# =============================================================================

def _generate_signals_for_asset(
    symbol: str,
    use_hmm: bool = True,
    use_vol_regime: bool = False,
    skip_confidence_filter: bool = False,
    vol_filter_quantile: float = VOL_FILTER_QUANTILE,
    features_cache: Optional[Dict[str, pd.DataFrame]] = None,
    vol_cache: Optional[Dict[str, pd.Series]] = None,
    ohlcv_cache: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    bar_tf: str = BAR_TF,
    fwd_bars: int = FWD_BARS,
    embargo_bars: int = EMBARGO_BARS,
    rolling_window_months: Optional[int] = None,
) -> Tuple[List[Tuple], Dict[str, np.ndarray], Dict[str, dict], Dict[str, dict]]:
    """Walk-forward signal generation for one asset using LogReg L1 C=0.1.

    Extended with options for Tier 2 tests:
    - skip_confidence_filter: if True, pass all predictions (for Test 4)
    - vol_filter_quantile: override vol filter threshold (for Test 5)
    - features_cache/vol_cache/ohlcv_cache: use alternate caches (for Test 6)
    - rolling_window_months: if set, use rolling window instead of expanding (for Test 2)
    """
    fc_cache = features_cache if features_cache is not None else _full_features_cache
    vc_cache = vol_cache if vol_cache is not None else _vol_cache
    oc_cache = ohlcv_cache if ohlcv_cache is not None else _ohlcv_cache

    if symbol not in fc_cache:
        return [], {}, {}, {}

    ohlcv = oc_cache.get((symbol, bar_tf))
    if ohlcv is None:
        return [], {}, {}, {}

    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
    vol = vc_cache.get(symbol)

    embargo_td = pd.Timedelta(minutes=30 * embargo_bars) if bar_tf == "30min" else pd.Timedelta(hours=embargo_bars)

    # Precomputed regime for this symbol
    full_regime = _hmm_regime_cache.get(symbol) if use_hmm else None
    vol_regime = _vol_regime_cache.get(symbol) if use_vol_regime else None

    rows = []
    monthly_coefs = {}
    boundary_audit = {}
    winsor_audit = {}

    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        month_str = str(month)

        # Embargo: train cutoff is embargo_bars before test start
        train_cutoff = ms - embargo_td

        # Rolling window: restrict training start
        if rolling_window_months is not None:
            train_start = ms - pd.DateOffset(months=rolling_window_months)
            if train_start.tzinfo is None:
                train_start = train_start.tz_localize("UTC")
        else:
            train_start = None

        # Slice features with per-window winsorization (19 features)
        feat, w_info = _get_features_for_window(symbol, train_cutoff, features_cache=fc_cache)
        winsor_audit[month_str] = w_info

        # Add regime feature + find worst regime
        worst_regime = None
        if use_hmm and full_regime is not None:
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

        elif use_vol_regime and vol_regime is not None:
            regime_col = vol_regime.reindex(feat.index)
            feat["regime"] = regime_col.fillna(1).astype(int)
            worst_regime = 0

        # Clean: drop rows with any NaN feature
        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        # Train/test split
        if train_start is not None:
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        else:
            tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 500 or len(te) < 20:
            continue

        # Record boundary audit
        boundary_audit[month_str] = {
            "last_train_ts": tr[-1],
            "first_test_ts": te[0],
        }

        # Vol filter: apply with given quantile
        if vol is not None and vol_filter_quantile > 0:
            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 50:
                continue
            vt = tv.quantile(vol_filter_quantile)
            hv = (vol.reindex(te) > vt).fillna(False)
            te = te[hv.values]
        elif vol is None and vol_filter_quantile > 0:
            continue

        if len(te) < 10:
            continue

        X_train = fc.loc[tr]
        y_train_binary = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        # Sample weights
        train_timestamps = X_train.index.values.astype(np.int64)
        sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars)

        # LogReg L1 C=0.1 with StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        lr_model = LogisticRegression(
            C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42,
        )
        lr_model.fit(X_tr_scaled, y_train_binary, sample_weight=sample_w)
        yp = lr_model.predict_proba(X_te_scaled)[:, 1]

        # Store coefficients
        feature_names = list(X_train.columns)
        coefs = lr_model.coef_[0]
        monthly_coefs[month_str] = {
            feature_names[k]: float(coefs[k]) for k in range(len(feature_names))
        }

        # Confidence filter (top 30%) -- unless skipped
        conf = np.abs(yp - 0.5)
        if skip_confidence_filter:
            conf_mask = np.ones(len(conf), dtype=bool)
        else:
            if len(conf) > 10:
                conf_threshold = np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
                conf_mask = conf >= conf_threshold
            else:
                conf_mask = np.ones(len(conf), dtype=bool)

        # Regime suppression
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
        signals = get_signal(yp_masked, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        for j in range(len(yp_masked)):
            idx = te_masked[j]
            fr = fwd_ret_masked.iloc[j] if hasattr(fwd_ret_masked, "iloc") else fwd_ret_masked.loc[idx]
            if np.isfinite(fr):
                sig_val = float(signals_disc[j]) if np.ndim(signals_disc) > 0 else float(signals_disc)
                rows.append((
                    idx, symbol, directions[j],
                    float(fr),
                    float(conf_masked[j]),
                    float(yp_masked[j]),
                    sig_val,
                    month_str,
                ))

    return rows, monthly_coefs, boundary_audit, winsor_audit


def generate_all_signals(
    use_hmm: bool = True,
    use_vol_regime: bool = False,
    max_workers: int = 3,
    skip_confidence_filter: bool = False,
    vol_filter_quantile: float = VOL_FILTER_QUANTILE,
    features_cache: Optional[Dict[str, pd.DataFrame]] = None,
    vol_cache: Optional[Dict[str, pd.Series]] = None,
    ohlcv_cache: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
    bar_tf: str = BAR_TF,
    fwd_bars: int = FWD_BARS,
    embargo_bars: int = EMBARGO_BARS,
    rolling_window_months: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict, dict, dict]:
    """Generate walk-forward signals for all assets in parallel."""
    all_rows = []
    all_monthly_coefs = {}
    all_boundary_audit = {}
    all_winsor_audit = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for symbol in ASSETS:
            fut = pool.submit(
                _generate_signals_for_asset,
                symbol, use_hmm, use_vol_regime,
                skip_confidence_filter,
                vol_filter_quantile,
                features_cache,
                vol_cache,
                ohlcv_cache,
                bar_tf,
                fwd_bars,
                embargo_bars,
                rolling_window_months,
            )
            futures[fut] = symbol

        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                rows, m_coefs, b_audit, w_audit = fut.result()
                all_rows.extend(rows)
                for month_str, coefs in m_coefs.items():
                    all_monthly_coefs.setdefault(month_str, {})[symbol] = coefs
                for month_str, info in b_audit.items():
                    all_boundary_audit.setdefault(month_str, {})[symbol] = info
                for month_str, info in w_audit.items():
                    all_winsor_audit.setdefault(month_str, {})[symbol] = info
            except Exception as e:
                print(f"    {symbol}: ERROR -- {e}", flush=True)
                traceback.print_exc()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "asset", "direction", "fwd_return",
        "confidence", "probability", "signal_strength", "month",
    ])
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df, all_monthly_coefs, all_boundary_audit, all_winsor_audit


# =============================================================================
# PORTFOLIO SIMULATION -- identical to tier1_verification.py
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame, slippage: float = SLIPPAGE) -> Optional[dict]:
    """Portfolio simulation with shared capital pool."""
    if len(signals_df) < 20:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = (
        sub["signal_strength"].values.astype(np.float64)
        if "signal_strength" in sub.columns
        else None
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
        trade_return = directions[i] * fwd_returns[i] - slippage
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

    # Compute metrics
    active = ~skipped
    n_trades = int(active.sum())
    if n_trades < 5:
        return None

    pnls = pnl_arr[active]
    active_ts = timestamps[active]

    t0 = pd.Timestamp(active_ts[0])
    t1 = pd.Timestamp(active_ts[-1])
    days = max((t1 - t0).total_seconds() / 86400, 1)
    years = days / 365.25

    total_ret = equity / STARTING_CAPITAL - 1
    cagr = (equity / STARTING_CAPITAL) ** (1.0 / years) - 1 if years > 0 else 0.0
    max_dd = float(dd_arr.max())

    ts_series = pd.Series(pnls, index=pd.to_datetime(active_ts))
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

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_trades": round(win_rate_t, 4),
        "win_rate_days": round(win_rate_d, 4),
        "profit_factor": round(min(pf, 99.0), 3),
        "n_trades": n_trades,
        "halted": halted,
        "final_equity": round(equity, 2),
        "n_days": len(daily_nonzero),
        "daily_returns": daily_nonzero,  # pd.Series for DSR computation
        "accuracy": round(float(np.mean(
            (directions[active] > 0) == (fwd_returns[active] > 0)
        )), 4),
    }


# =============================================================================
# TEST 1: Prediction Distribution Stability (E2)
# =============================================================================

def test_prediction_distribution_stability(signals_df):
    """Compare model probability distributions between first 7 and last 7 months.
    KL divergence between binned distributions.
    PASS: KL < 0.1
    """
    if len(signals_df) < 100:
        return False, {"error": "insufficient signals"}

    probs = signals_df["probability"].values
    months = signals_df["month"].values

    unique_months = sorted(set(months))
    n_months = len(unique_months)
    mid = n_months // 2

    first_half_months = set(unique_months[:mid])
    second_half_months = set(unique_months[mid:])

    first_probs = probs[np.isin(months, list(first_half_months))]
    second_probs = probs[np.isin(months, list(second_half_months))]

    if len(first_probs) < 50 or len(second_probs) < 50:
        return False, {"error": "insufficient signals in one half"}

    # Bin into 50 bins from 0 to 1
    n_bins = 50
    bins = np.linspace(0, 1, n_bins + 1)
    eps = 1e-10

    hist_first, _ = np.histogram(first_probs, bins=bins, density=True)
    hist_second, _ = np.histogram(second_probs, bins=bins, density=True)

    # Normalize to proper distributions
    hist_first = hist_first / hist_first.sum() + eps
    hist_second = hist_second / hist_second.sum() + eps

    # KL divergence: sum(p * log(p/q))
    kl_div = float(np.sum(hist_first * np.log(hist_first / hist_second)))

    passed = kl_div < 0.1
    return passed, {
        "kl_divergence": round(kl_div, 6),
        "n_first_half": len(first_probs),
        "n_second_half": len(second_probs),
        "first_half_months": sorted(first_half_months),
        "second_half_months": sorted(second_half_months),
        "first_mean_prob": round(float(np.mean(first_probs)), 4),
        "second_mean_prob": round(float(np.mean(second_probs)), 4),
    }


# =============================================================================
# TEST 2: Rolling vs Expanding Window (E3)
# =============================================================================

def test_rolling_vs_expanding(expanding_sharpe):
    """Run walk-forward with 6-month ROLLING window and compare to expanding.
    PASS: |rolling_sharpe - expanding_sharpe| < 0.3
    """
    print("    Generating signals with 6-month rolling window...", flush=True)
    signals_rolling, _, _, _ = generate_all_signals(
        use_hmm=True, use_vol_regime=False,
        rolling_window_months=6,
    )
    sim_rolling = simulate_portfolio(signals_rolling)
    rolling_sharpe = sim_rolling["sharpe"] if sim_rolling else 0.0

    delta = abs(rolling_sharpe - expanding_sharpe)
    passed = delta < 0.3

    return passed, {
        "expanding_sharpe": round(expanding_sharpe, 3),
        "rolling_sharpe": round(rolling_sharpe, 3),
        "delta": round(delta, 3),
        "n_signals_rolling": len(signals_rolling),
        "n_trades_rolling": sim_rolling["n_trades"] if sim_rolling else 0,
    }


# =============================================================================
# TEST 3: Asset Subset Sensitivity (E4)
# =============================================================================

def test_asset_subset_sensitivity(signals_df):
    """Filter signal set to asset subsets and compare Sharpes.
    Subsets: SUI only, SUI+BTC, SUI+BTC+BNB, All 6
    PASS: Sharpe improves or stays flat as assets added.
    """
    subsets = {
        "SUI_only": ["SUI"],
        "SUI_BTC": ["SUI", "BTC"],
        "SUI_BTC_BNB": ["SUI", "BTC", "BNB"],
        "all_6": ASSETS,
    }

    results = {}
    sharpes = []
    for label, asset_list in subsets.items():
        sub_signals = signals_df[signals_df["asset"].isin(asset_list)]
        sim = simulate_portfolio(sub_signals)
        s = sim["sharpe"] if sim else 0.0
        n = sim["n_trades"] if sim else 0
        sharpes.append(s)
        results[label] = {
            "assets": asset_list,
            "sharpe": round(s, 3),
            "n_trades": n,
            "n_signals": len(sub_signals),
        }

    # PASS: monotonic improvement (each subset >= previous)
    monotonic = all(sharpes[i] <= sharpes[i + 1] + 0.05 for i in range(len(sharpes) - 1))
    passed = monotonic

    return passed, results


# =============================================================================
# TEST 4: Confidence Threshold Sensitivity (E5)
# =============================================================================

def test_confidence_threshold_sensitivity(signals_no_conf_df):
    """Apply different confidence thresholds post-hoc.
    Thresholds: 0th (no filter), 50th, 60th, 70th, 80th, 90th percentile.
    PASS: Monotonic improvement as threshold increases.
    """
    thresholds = [0, 50, 60, 70, 80, 90]
    results = {}
    sharpes = []

    for pctile in thresholds:
        if pctile == 0:
            filtered = signals_no_conf_df.copy()
        else:
            conf = signals_no_conf_df["confidence"].values
            if len(conf) > 10:
                thresh = np.percentile(conf, pctile)
                mask = conf >= thresh
                filtered = signals_no_conf_df[mask].copy()
            else:
                filtered = signals_no_conf_df.copy()

        sim = simulate_portfolio(filtered)
        s = sim["sharpe"] if sim else 0.0
        n = sim["n_trades"] if sim else 0
        sharpes.append(s)
        results[f"p{pctile}"] = {
            "percentile": pctile,
            "sharpe": round(s, 3),
            "n_trades": n,
            "n_signals": len(filtered),
        }

    # PASS: Monotonic improvement (each threshold >= previous, with tolerance)
    monotonic = all(sharpes[i] <= sharpes[i + 1] + 0.05 for i in range(len(sharpes) - 1))
    passed = monotonic

    return passed, {"thresholds": results, "sharpes": [round(s, 3) for s in sharpes]}


# =============================================================================
# TEST 5: Vol Filter Threshold Sensitivity (E6)
# =============================================================================

def test_vol_filter_sensitivity():
    """Run walk-forward with different vol filter quantiles.
    Quantiles: 0.0 (no filter), 0.3, 0.4, 0.5 (current), 0.6, 0.7
    PASS: Sharpe improves with stricter vol filter.
    """
    quantiles = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}
    sharpes = []

    for q in quantiles:
        print(f"    Vol filter q={q:.1f}...", flush=True)
        signals_q, _, _, _ = generate_all_signals(
            use_hmm=True, use_vol_regime=False,
            vol_filter_quantile=q,
        )
        sim = simulate_portfolio(signals_q)
        s = sim["sharpe"] if sim else 0.0
        n = sim["n_trades"] if sim else 0
        sharpes.append(s)
        results[f"q{q:.1f}"] = {
            "quantile": q,
            "sharpe": round(s, 3),
            "n_trades": n,
            "n_signals": len(signals_q),
        }

    # PASS: Monotonic improvement (with tolerance)
    monotonic = all(sharpes[i] <= sharpes[i + 1] + 0.05 for i in range(len(sharpes) - 1))
    passed = monotonic

    return passed, {"quantiles": results, "sharpes": [round(s, 3) for s in sharpes]}


# =============================================================================
# TEST 6: Timeframe Sensitivity (E8)
# =============================================================================

def test_timeframe_sensitivity(baseline_sharpe):
    """Resample to 1H bars and run full walk-forward.
    PASS: |1H_sharpe - 30min_sharpe| < 0.3
    """
    print("    Resampling data to 1H and computing features...", flush=True)
    t_start = time.time()

    # Build 1H OHLCV cache
    ohlcv_1h_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    features_1h_cache: Dict[str, pd.DataFrame] = {}
    vol_1h_cache: Dict[str, pd.Series] = {}

    for symbol in ASSETS:
        ohlcv_30m = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv_30m is None:
            continue

        # Resample 30min to 1H
        ohlcv_1h = ohlcv_30m.resample("1h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["open"])
        ohlcv_1h_cache[(symbol, "1h")] = ohlcv_1h

        # Compute features at 1H resolution
        f = _compute_features_for_ohlcv(symbol, ohlcv_1h)
        features_1h_cache[symbol] = f

        # Volatility at 1H
        close_1h = ohlcv_1h["close"]
        vol_1h_cache[symbol] = close_1h.pct_change().dropna().ewm(span=50).std()

    print(f"    1H data prepared in {time.time()-t_start:.1f}s", flush=True)
    print("    Generating 1H walk-forward signals...", flush=True)

    # Run walk-forward at 1H: FWD_BARS=2 (2x1H=2H), EMBARGO=2 bars (2x1H=2H)
    signals_1h, _, _, _ = generate_all_signals(
        use_hmm=True, use_vol_regime=False,
        features_cache=features_1h_cache,
        vol_cache=vol_1h_cache,
        ohlcv_cache=ohlcv_1h_cache,
        bar_tf="1h",
        fwd_bars=2,
        embargo_bars=2,
    )
    sim_1h = simulate_portfolio(signals_1h)
    sharpe_1h = sim_1h["sharpe"] if sim_1h else 0.0

    delta = abs(sharpe_1h - baseline_sharpe)
    passed = delta < 0.3

    return passed, {
        "sharpe_30min": round(baseline_sharpe, 3),
        "sharpe_1h": round(sharpe_1h, 3),
        "delta": round(delta, 3),
        "n_signals_1h": len(signals_1h),
        "n_trades_1h": sim_1h["n_trades"] if sim_1h else 0,
    }


# =============================================================================
# TEST 7: Position Flip Cost (C2)
# =============================================================================

def test_position_flip_cost(signals_df, baseline_sharpe):
    """Identify direction flips on same asset and re-simulate with 2x slippage.
    PASS: Sharpe impact < 0.2
    """
    if len(signals_df) < 20:
        return False, {"error": "insufficient signals"}

    sorted_signals = signals_df.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    n_flips = 0
    n_total = len(sorted_signals)
    flip_indices = set()

    for i in range(1, len(sorted_signals)):
        if (sorted_signals.loc[i, "asset"] == sorted_signals.loc[i - 1, "asset"]
                and sorted_signals.loc[i, "direction"] != sorted_signals.loc[i - 1, "direction"]):
            n_flips += 1
            flip_indices.add(i)

    flip_rate = n_flips / n_total if n_total > 0 else 0

    # Re-simulate with 2x slippage on flip trades
    flip_signals = signals_df.copy()
    flip_signals = flip_signals.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    # Mark flips
    is_flip = np.zeros(len(flip_signals), dtype=bool)
    for i in range(1, len(flip_signals)):
        if (flip_signals.loc[i, "asset"] == flip_signals.loc[i - 1, "asset"]
                and flip_signals.loc[i, "direction"] != flip_signals.loc[i - 1, "direction"]):
            is_flip[i] = True

    # Simulate with 2x slippage for flips
    sim_flip = simulate_portfolio(flip_signals, slippage=SLIPPAGE * 2)
    flip_sharpe = sim_flip["sharpe"] if sim_flip else 0.0

    sharpe_impact = baseline_sharpe - flip_sharpe
    passed = sharpe_impact < 0.2

    return passed, {
        "n_flips": n_flips,
        "n_total": n_total,
        "flip_rate": round(flip_rate, 4),
        "baseline_sharpe": round(baseline_sharpe, 3),
        "flip_cost_sharpe": round(flip_sharpe, 3),
        "sharpe_impact": round(sharpe_impact, 3),
    }


# =============================================================================
# TEST 8: Flash Crash Behavior (C6)
# =============================================================================

def test_flash_crash_behavior(signals_df):
    """Find 5 worst BTC 30-min returns and check model behavior.
    PASS: Model not systematically long during crashes (< 60% wrong-sided).
    """
    btc_ohlcv = _ohlcv_cache.get(("BTC", BAR_TF))
    if btc_ohlcv is None:
        return False, {"error": "no BTC OHLCV data"}

    close = btc_ohlcv["close"]
    bar_returns = close.pct_change()

    # OOS period only (Jan 2025 - Feb 2026)
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    oos_end = pd.Timestamp("2026-03-01", tz="UTC")
    oos_returns = bar_returns[(bar_returns.index >= oos_start) & (bar_returns.index <= oos_end)]

    # Find 5 worst bars
    worst_5 = oos_returns.nsmallest(5)

    crash_events = []
    n_wrong_sided = 0
    n_had_signal = 0

    for crash_ts, crash_ret in worst_5.items():
        # Check if model had signals at this bar or the preceding bar
        crash_ts_pd = pd.Timestamp(crash_ts)
        preceding_ts = crash_ts_pd - pd.Timedelta(minutes=30)

        nearby_signals = signals_df[
            (signals_df["timestamp"] >= preceding_ts) &
            (signals_df["timestamp"] <= crash_ts_pd)
        ]

        had_signal = len(nearby_signals) > 0
        if had_signal:
            n_had_signal += 1
            # Check if any signal was wrong-sided (long during crash)
            wrong_sided = (nearby_signals["direction"] > 0).any()
            if wrong_sided:
                n_wrong_sided += 1
        else:
            wrong_sided = False

        event = {
            "timestamp": str(crash_ts),
            "btc_return_pct": round(float(crash_ret) * 100, 3),
            "model_had_signal": had_signal,
            "n_signals": len(nearby_signals),
        }
        if had_signal:
            event["signal_assets"] = nearby_signals["asset"].tolist()
            event["signal_directions"] = nearby_signals["direction"].tolist()
            event["wrong_sided"] = bool(wrong_sided)

        crash_events.append(event)

    wrong_rate = n_wrong_sided / max(n_had_signal, 1)
    passed = wrong_rate < 0.60

    return passed, {
        "crash_events": crash_events,
        "n_had_signal": n_had_signal,
        "n_wrong_sided": n_wrong_sided,
        "wrong_rate": round(wrong_rate, 3),
    }


# =============================================================================
# TEST 9: FFD d* Stability (D4)
# =============================================================================

def test_ffd_d_stability():
    """Compute FFD d* using only first 6 months and compare to hardcoded values.
    PASS: All within +/- 0.05
    """
    from statsmodels.tsa.stattools import adfuller

    # Use first 6 months of data (training-only)
    cutoff = pd.Timestamp("2025-07-01", tz="UTC")

    results = {}
    all_within = True

    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            results[symbol] = {"error": "no data"}
            all_within = False
            continue

        close = ohlcv["close"]
        close_train = close[close.index < cutoff]

        if len(close_train) < 500:
            results[symbol] = {"error": "insufficient training data", "n_bars": len(close_train)}
            all_within = False
            continue

        try:
            d_star_result = find_min_d(
                close_train,
                d_range=np.arange(0.0, 1.01, 0.05),
                threshold=FFD_THRESHOLD,
                adf_pvalue=0.05,
                adf_maxlag=1,
            )
            d_star_computed = d_star_result["d_star"]
        except Exception as e:
            results[symbol] = {"error": str(e)}
            all_within = False
            continue

        d_star_hardcoded = FFD_D_STARS[symbol]
        if d_star_computed is not None:
            diff = abs(d_star_computed - d_star_hardcoded)
            within = diff <= 0.05
            if not within:
                all_within = False
        else:
            diff = None
            within = False
            all_within = False

        results[symbol] = {
            "d_star_computed": round(d_star_computed, 3) if d_star_computed is not None else None,
            "d_star_hardcoded": d_star_hardcoded,
            "diff": round(diff, 3) if diff is not None else None,
            "within_tolerance": within,
            "n_train_bars": len(close_train),
        }

    passed = all_within
    return passed, results


# =============================================================================
# TEST 10: Cross-Exchange Data Alignment (D5)
# =============================================================================

def test_cross_exchange_alignment():
    """Compare Binance close and HL mid_px/mark_px correlation at 30min.
    PASS: Correlation > 0.999 for all assets.
    """
    results = {}
    all_above = True

    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        hl = _hl_cache.get(symbol)

        if ohlcv is None:
            results[symbol] = {"error": "no Binance data"}
            all_above = False
            continue

        if hl is None:
            results[symbol] = {"error": "no HL data"}
            all_above = False
            continue

        binance_close = ohlcv["close"]

        # Use mid_px or mark_px from HL
        if "mid_px" in hl.columns:
            hl_price = hl["mid_px"]
            price_col = "mid_px"
        elif "mark_px" in hl.columns:
            hl_price = hl["mark_px"]
            price_col = "mark_px"
        else:
            results[symbol] = {"error": "no mid_px or mark_px in HL data"}
            all_above = False
            continue

        # Align on common index
        common = binance_close.index.intersection(hl_price.index)
        if len(common) < 100:
            results[symbol] = {"error": "insufficient overlapping bars", "n_common": len(common)}
            all_above = False
            continue

        binance_vals = binance_close.loc[common].dropna()
        hl_vals = hl_price.loc[common].dropna()
        common2 = binance_vals.index.intersection(hl_vals.index)

        if len(common2) < 100:
            results[symbol] = {"error": "insufficient valid overlapping bars"}
            all_above = False
            continue

        corr = float(np.corrcoef(binance_vals.loc[common2].values, hl_vals.loc[common2].values)[0, 1])

        if corr < 0.999:
            all_above = False

        results[symbol] = {
            "correlation": round(corr, 6),
            "hl_price_col": price_col,
            "n_common_bars": len(common2),
            "above_threshold": corr >= 0.999,
        }

    passed = all_above
    return passed, results


# =============================================================================
# TEST 11: Data Completeness (D6)
# =============================================================================

def test_data_completeness():
    """Check for missing bars in OOS period.
    PASS: < 0.1% missing for all assets.
    """
    oos_start = pd.Timestamp("2025-01-01", tz="UTC")
    oos_end = pd.Timestamp("2026-02-28 23:30:00", tz="UTC")

    # Expected number of 30-min bars
    expected_bars = int((oos_end - oos_start).total_seconds() / (30 * 60)) + 1

    results = {}
    all_below = True

    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            results[symbol] = {"error": "no data"}
            all_below = False
            continue

        oos_data = ohlcv[(ohlcv.index >= oos_start) & (ohlcv.index <= oos_end)]
        actual_bars = len(oos_data)
        missing = max(expected_bars - actual_bars, 0)
        missing_rate = missing / expected_bars if expected_bars > 0 else 0

        if missing_rate >= 0.001:  # 0.1%
            all_below = False

        results[symbol] = {
            "expected_bars": expected_bars,
            "actual_bars": actual_bars,
            "missing_bars": missing,
            "missing_rate_pct": round(missing_rate * 100, 4),
            "below_threshold": missing_rate < 0.001,
        }

    passed = all_below
    return passed, results


# =============================================================================
# TEST 12: Probabilistic Sharpe Ratio (B2)
# =============================================================================

def test_probabilistic_sharpe_ratio(daily_returns):
    """PSR = Phi(sqrt(N-1) * (SR - SR*) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2))
    Where SR* = 0 (testing if true SR > 0).
    PASS: PSR > 0.95
    """
    if daily_returns is None or len(daily_returns) < 10:
        return False, {"error": "insufficient daily returns"}

    N = len(daily_returns)
    sr_hat = float(daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0.0
    sr_star = 0.0  # Testing SR > 0

    gamma3 = float(daily_returns.skew())  # Skewness
    gamma4 = float(daily_returns.kurtosis()) + 3.0  # Regular kurtosis (pandas gives excess)

    # PSR formula
    denominator = np.sqrt(1 - gamma3 * sr_hat + (gamma4 - 1) / 4 * sr_hat ** 2)
    if denominator == 0:
        psr = 0.0
    else:
        z_score = np.sqrt(N - 1) * (sr_hat - sr_star) / denominator
        psr = float(norm.cdf(z_score))

    # Also compute annualized Sharpe for reference
    sr_ann = sr_hat * np.sqrt(365.25)

    passed = psr > 0.95
    return passed, {
        "psr": round(psr, 6),
        "daily_sharpe": round(sr_hat, 6),
        "annualized_sharpe": round(sr_ann, 3),
        "sr_star": sr_star,
        "skewness": round(gamma3, 4),
        "kurtosis": round(gamma4, 4),
        "N_days": N,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    overall_start = time.time()

    print("=" * 60, flush=True)
    print("TIER 2 VERIFICATION -- 12 Tests", flush=True)
    print("=" * 60, flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Model: LogReg L1 C=0.1 + StandardScaler", flush=True)
    print(f"Features: {len(BASE_19_FEATURES)} (no funding_zscore)", flush=True)
    print(f"Embargo: {EMBARGO_BARS} bars ({EMBARGO_TIMEDELTA})", flush=True)
    print(f"Assets: {ASSETS}", flush=True)
    print(f"Months: {ALL_MONTHS[0]}-{ALL_MONTHS[-1]} ({len(ALL_MONTHS)})", flush=True)
    print("=" * 60, flush=True)

    # =========================================================================
    # Phase 0: Load data, precompute features
    # =========================================================================
    print("\n[Phase 0] Loading data + precomputing features...", flush=True)
    t0 = time.time()
    preload_all_data()
    precompute_all_features()
    t_phase0 = time.time() - t0
    print(f"  Phase 0 done in {t_phase0:.1f}s", flush=True)

    # =========================================================================
    # Phase 1: Generate walk-forward signals (baseline with HMM + all filters)
    # =========================================================================
    print("\n[Phase 1] Generating baseline walk-forward signals (LogReg L1 C=0.1 + HMM)...", flush=True)
    t1 = time.time()
    signals_df, all_monthly_coefs, all_boundary_audit, all_winsor_audit = generate_all_signals(
        use_hmm=True, use_vol_regime=False,
    )
    t_phase1 = time.time() - t1
    print(f"  Generated {len(signals_df)} signals across {len(ASSETS)} assets in {t_phase1:.1f}s", flush=True)

    # Full portfolio sim (baseline)
    sim_full = simulate_portfolio(signals_df)
    baseline_sharpe = sim_full["sharpe"] if sim_full else 0.0
    if sim_full:
        print(f"  Baseline Sharpe: {sim_full['sharpe']:.3f}, Trades: {sim_full['n_trades']}, "
              f"Return: {sim_full['total_return_pct']:+.2f}%", flush=True)
    else:
        print("  WARNING: Full portfolio simulation returned None!", flush=True)

    # Also generate signals WITHOUT confidence filter for Test 4
    print("\n[Phase 1b] Generating signals without confidence filter (for Test 4)...", flush=True)
    t1b = time.time()
    signals_no_conf_df, _, _, _ = generate_all_signals(
        use_hmm=True, use_vol_regime=False,
        skip_confidence_filter=True,
    )
    print(f"  Generated {len(signals_no_conf_df)} unfiltered signals in {time.time()-t1b:.1f}s", flush=True)

    # =========================================================================
    # Phase 2: Run 12 tests
    # =========================================================================
    print("\n[Phase 2] Running 12 verification tests...", flush=True)
    print("=" * 60, flush=True)

    test_results = {}
    n_passed = 0
    n_failed = 0

    # ------ Test 1: Prediction Distribution Stability (E2) ------
    print(f"\n[Test  1/12] Prediction Distribution Stability (E2)...", flush=True)
    t_test = time.time()
    try:
        passed_1, res_1 = test_prediction_distribution_stability(signals_df)
        kl = res_1.get("kl_divergence", 0)
        print(f"  KL divergence: {kl:.6f}", flush=True)
        print(f"  First half mean prob: {res_1.get('first_mean_prob', 0):.4f}, "
              f"Second half: {res_1.get('second_mean_prob', 0):.4f}", flush=True)
        status = "PASS" if passed_1 else "FAIL"
        print(f"  >> [{status}] KL < 0.1 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["1_prediction_distribution"] = {"passed": passed_1, "results": res_1}
        if passed_1:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["1_prediction_distribution"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 2: Rolling vs Expanding Window (E3) ------
    print(f"\n[Test  2/12] Rolling vs Expanding Window (E3)...", flush=True)
    t_test = time.time()
    try:
        passed_2, res_2 = test_rolling_vs_expanding(baseline_sharpe)
        exp_s = res_2.get("expanding_sharpe", 0)
        roll_s = res_2.get("rolling_sharpe", 0)
        delta = res_2.get("delta", 0)
        print(f"  Expanding: Sharpe={exp_s:.3f}, Rolling: Sharpe={roll_s:.3f}, Delta={delta:.3f}", flush=True)
        status = "PASS" if passed_2 else "FAIL"
        print(f"  >> [{status}] |delta| < 0.3 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["2_rolling_vs_expanding"] = {"passed": passed_2, "results": res_2}
        if passed_2:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["2_rolling_vs_expanding"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 3: Asset Subset Sensitivity (E4) ------
    print(f"\n[Test  3/12] Asset Subset Sensitivity (E4)...", flush=True)
    t_test = time.time()
    try:
        passed_3, res_3 = test_asset_subset_sensitivity(signals_df)
        for label, info in res_3.items():
            print(f"  {label:>15}: Sharpe={info['sharpe']:>6.3f}, n_trades={info['n_trades']:>5}", flush=True)
        status = "PASS" if passed_3 else "FAIL"
        print(f"  >> [{status}] Sharpe improves as assets added ({time.time()-t_test:.1f}s)", flush=True)
        test_results["3_asset_subset"] = {"passed": passed_3, "results": res_3}
        if passed_3:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["3_asset_subset"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 4: Confidence Threshold Sensitivity (E5) ------
    print(f"\n[Test  4/12] Confidence Threshold Sensitivity (E5)...", flush=True)
    t_test = time.time()
    try:
        passed_4, res_4 = test_confidence_threshold_sensitivity(signals_no_conf_df)
        for key, info in res_4.get("thresholds", {}).items():
            print(f"  {key:>5}: Sharpe={info['sharpe']:>6.3f}, n_trades={info['n_trades']:>5}", flush=True)
        status = "PASS" if passed_4 else "FAIL"
        print(f"  >> [{status}] Monotonic improvement with stricter threshold ({time.time()-t_test:.1f}s)", flush=True)
        test_results["4_confidence_threshold"] = {"passed": passed_4, "results": res_4}
        if passed_4:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["4_confidence_threshold"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 5: Vol Filter Threshold Sensitivity (E6) ------
    print(f"\n[Test  5/12] Vol Filter Threshold Sensitivity (E6)...", flush=True)
    t_test = time.time()
    try:
        passed_5, res_5 = test_vol_filter_sensitivity()
        for key, info in res_5.get("quantiles", {}).items():
            print(f"  {key:>5}: Sharpe={info['sharpe']:>6.3f}, n_trades={info['n_trades']:>5}", flush=True)
        status = "PASS" if passed_5 else "FAIL"
        print(f"  >> [{status}] Sharpe improves with stricter vol filter ({time.time()-t_test:.1f}s)", flush=True)
        test_results["5_vol_filter_sensitivity"] = {"passed": passed_5, "results": res_5}
        if passed_5:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["5_vol_filter_sensitivity"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 6: Timeframe Sensitivity (E8) ------
    print(f"\n[Test  6/12] Timeframe Sensitivity (E8)...", flush=True)
    t_test = time.time()
    try:
        passed_6, res_6 = test_timeframe_sensitivity(baseline_sharpe)
        s30 = res_6.get("sharpe_30min", 0)
        s1h = res_6.get("sharpe_1h", 0)
        delta = res_6.get("delta", 0)
        print(f"  30min: Sharpe={s30:.3f}, 1H: Sharpe={s1h:.3f}, Delta={delta:.3f}", flush=True)
        status = "PASS" if passed_6 else "FAIL"
        print(f"  >> [{status}] |1H - 30min| < 0.3 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["6_timeframe_sensitivity"] = {"passed": passed_6, "results": res_6}
        if passed_6:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["6_timeframe_sensitivity"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 7: Position Flip Cost (C2) ------
    print(f"\n[Test  7/12] Position Flip Cost (C2)...", flush=True)
    t_test = time.time()
    try:
        passed_7, res_7 = test_position_flip_cost(signals_df, baseline_sharpe)
        n_flips = res_7.get("n_flips", 0)
        flip_rate = res_7.get("flip_rate", 0)
        impact = res_7.get("sharpe_impact", 0)
        print(f"  Flips: {n_flips}/{res_7.get('n_total', 0)} ({flip_rate:.1%})", flush=True)
        print(f"  Baseline Sharpe: {res_7.get('baseline_sharpe', 0):.3f}, "
              f"With flip cost: {res_7.get('flip_cost_sharpe', 0):.3f}, "
              f"Impact: {impact:.3f}", flush=True)
        status = "PASS" if passed_7 else "FAIL"
        print(f"  >> [{status}] Sharpe impact < 0.2 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["7_position_flip_cost"] = {"passed": passed_7, "results": res_7}
        if passed_7:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["7_position_flip_cost"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 8: Flash Crash Behavior (C6) ------
    print(f"\n[Test  8/12] Flash Crash Behavior (C6)...", flush=True)
    t_test = time.time()
    try:
        passed_8, res_8 = test_flash_crash_behavior(signals_df)
        for event in res_8.get("crash_events", []):
            ret_str = f"{event['btc_return_pct']:+.3f}%"
            had_sig = event.get("model_had_signal", False)
            wrong = event.get("wrong_sided", "N/A")
            print(f"  {event['timestamp'][:19]}: BTC {ret_str}, "
                  f"had_signal={had_sig}, wrong_sided={wrong}", flush=True)
        wrong_rate = res_8.get("wrong_rate", 0)
        print(f"  Wrong-sided rate: {wrong_rate:.1%} (n_had_signal={res_8.get('n_had_signal', 0)})", flush=True)
        status = "PASS" if passed_8 else "FAIL"
        print(f"  >> [{status}] < 60% wrong-sided ({time.time()-t_test:.1f}s)", flush=True)
        test_results["8_flash_crash"] = {"passed": passed_8, "results": res_8}
        if passed_8:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["8_flash_crash"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 9: FFD d* Stability (D4) ------
    print(f"\n[Test  9/12] FFD d* Stability (D4)...", flush=True)
    t_test = time.time()
    try:
        passed_9, res_9 = test_ffd_d_stability()
        for symbol, info in res_9.items():
            if "error" in info:
                print(f"  {symbol:>4}: ERROR -- {info['error']}", flush=True)
            else:
                d_comp = info.get("d_star_computed", "N/A")
                d_hard = info.get("d_star_hardcoded", "N/A")
                diff = info.get("diff", "N/A")
                within = info.get("within_tolerance", False)
                print(f"  {symbol:>4}: computed={d_comp}, hardcoded={d_hard}, "
                      f"diff={diff}, within={within}", flush=True)
        status = "PASS" if passed_9 else "FAIL"
        print(f"  >> [{status}] All d* within +/-0.05 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["9_ffd_d_stability"] = {"passed": passed_9, "results": res_9}
        if passed_9:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["9_ffd_d_stability"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 10: Cross-Exchange Data Alignment (D5) ------
    print(f"\n[Test 10/12] Cross-Exchange Data Alignment (D5)...", flush=True)
    t_test = time.time()
    try:
        passed_10, res_10 = test_cross_exchange_alignment()
        for symbol, info in res_10.items():
            if "error" in info:
                print(f"  {symbol:>4}: ERROR -- {info['error']}", flush=True)
            else:
                corr = info.get("correlation", 0)
                n_bars = info.get("n_common_bars", 0)
                above = info.get("above_threshold", False)
                print(f"  {symbol:>4}: corr={corr:.6f}, n_bars={n_bars}, above_0.999={above}", flush=True)
        status = "PASS" if passed_10 else "FAIL"
        print(f"  >> [{status}] All correlations > 0.999 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["10_cross_exchange"] = {"passed": passed_10, "results": res_10}
        if passed_10:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["10_cross_exchange"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 11: Data Completeness (D6) ------
    print(f"\n[Test 11/12] Data Completeness (D6)...", flush=True)
    t_test = time.time()
    try:
        passed_11, res_11 = test_data_completeness()
        for symbol, info in res_11.items():
            if "error" in info:
                print(f"  {symbol:>4}: ERROR -- {info['error']}", flush=True)
            else:
                expected = info.get("expected_bars", 0)
                actual = info.get("actual_bars", 0)
                missing_pct = info.get("missing_rate_pct", 0)
                below = info.get("below_threshold", False)
                print(f"  {symbol:>4}: {actual}/{expected} bars, "
                      f"missing={missing_pct:.4f}%, ok={below}", flush=True)
        status = "PASS" if passed_11 else "FAIL"
        print(f"  >> [{status}] < 0.1% missing for all ({time.time()-t_test:.1f}s)", flush=True)
        test_results["11_data_completeness"] = {"passed": passed_11, "results": res_11}
        if passed_11:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["11_data_completeness"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # ------ Test 12: Probabilistic Sharpe Ratio (B2) ------
    print(f"\n[Test 12/12] Probabilistic Sharpe Ratio (B2)...", flush=True)
    t_test = time.time()
    try:
        daily_rets = sim_full["daily_returns"] if sim_full else None
        passed_12, res_12 = test_probabilistic_sharpe_ratio(daily_rets)
        psr = res_12.get("psr", 0)
        sr = res_12.get("annualized_sharpe", 0)
        skew = res_12.get("skewness", 0)
        kurt = res_12.get("kurtosis", 0)
        n_days = res_12.get("N_days", 0)
        print(f"  PSR={psr:.6f}, Ann.Sharpe={sr:.3f}, Skew={skew:.4f}, "
              f"Kurt={kurt:.4f}, N={n_days}", flush=True)
        status = "PASS" if passed_12 else "FAIL"
        print(f"  >> [{status}] PSR > 0.95 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["12_psr"] = {"passed": passed_12, "results": res_12}
        if passed_12:
            n_passed += 1
        else:
            n_failed += 1
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["12_psr"] = {"passed": False, "error": str(e)}
        n_failed += 1

    # =========================================================================
    # Phase 3: Summary + JSON
    # =========================================================================
    total_time = time.time() - overall_start

    print("\n" + "=" * 60, flush=True)
    print(f"SUMMARY: {n_passed}/12 PASSED, {n_failed}/12 FAILED", flush=True)
    print("=" * 60, flush=True)

    failed_tests = [k for k, v in test_results.items() if not v.get("passed", False)]
    passed_tests = [k for k, v in test_results.items() if v.get("passed", False)]

    if failed_tests:
        print(f"FAILED: {failed_tests}", flush=True)
    if passed_tests:
        print(f"PASSED: {passed_tests}", flush=True)

    if n_passed == 12:
        print("\n*** ALL 12 TESTS PASSED -- TIER 2 VERIFIED ***", flush=True)
    elif n_passed >= 10:
        print(f"\n*** {n_passed}/12 PASSED -- CLOSE, address remaining failures ***", flush=True)
    else:
        print(f"\n*** {n_passed}/12 PASSED -- NEEDS WORK ***", flush=True)

    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"  Phase 0 (data+features): {t_phase0:.0f}s", flush=True)
    print(f"  Phase 1 (signals):       {t_phase1:.0f}s", flush=True)
    print(f"  Phase 2 (tests):         {total_time - t_phase0 - t_phase1:.0f}s", flush=True)

    # =========================================================================
    # Save JSON results
    # =========================================================================

    # Make results JSON-serializable
    def make_serializable(obj):
        """Recursively convert non-serializable types."""
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, pd.Series):
            return {str(k): make_serializable(v) for k, v in obj.to_dict().items()}
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj

    json_results = make_serializable(test_results)

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "LogReg_L1_C0.1",
        "features": BASE_19_FEATURES,
        "n_assets": len(ASSETS),
        "assets": ASSETS,
        "n_months": len(ALL_MONTHS),
        "n_signals": len(signals_df),
        "baseline_sharpe": baseline_sharpe,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "failed_tests": failed_tests,
        "passed_tests": passed_tests,
        "total_runtime_s": round(total_time, 1),
        "tests": json_results,
    }

    out_path = OUTPUT_DIR / "tier2_verification_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}", flush=True)
    except Exception as e:
        print(f"\n[WARN] Failed to save JSON: {e}", flush=True)

    print(f"Completed: {datetime.now()}", flush=True)
