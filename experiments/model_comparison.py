"""
Model Comparison — Rigorous comparison of ML model families on direction prediction.

Context: validation_honest.py showed:
  - Signal is real (Z=3.13 vs random)
  - Logistic regression matches XGBoost (both ~Sharpe 1.7)
  - Signal appears linear
  - funding_zscore is noise — DROPPED here

Tests 17 configs across 9 model types:
  Linear: LogReg L2 (3), L1 (2), ElasticNet (2), Ridge (1)
  Tree:   XGBoost (3), LightGBM (2), RandomForest (2)
  Other:  LinearSVC (1), KNN (1)

Architecture: precompute-once (reused from validation_honest.py), 19 features.
Methodology: 4-bar embargo, AFML sample uniqueness weights, vol filter, confidence
filter, HMM regime filter, probability-based sizing.
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
import xgboost as xgb
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

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

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] LightGBM not available — skipping LGB configs", flush=True)

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
RESULTS_FILE = OUTPUT_DIR / "model_comparison_results.json"

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
DSR_NUM_TRIALS = 4095

# Embargo: 4 bars * 30 min = 2 hours gap
EMBARGO_BARS = 4
EMBARGO_TIMEDELTA = pd.Timedelta(hours=2)

# HMM regime parameters
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
            # funding_zscore deliberately omitted — proven noise

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
# MODEL CONFIG DEFINITIONS
# =============================================================================

def _build_model_configs() -> List[dict]:
    """Build list of all model configurations to test."""
    configs = []

    # --- LogisticRegression L2 (3 configs) ---
    for C_val in [0.01, 1.0, 100.0]:
        configs.append({
            "name": f"LogReg_L2_C{C_val}",
            "family": "linear",
            "config_str": f"C={C_val},L2",
            "needs_scaling": True,
            "has_predict_proba": True,
            "supports_sample_weight": True,
            "build_fn": lambda C=C_val: LogisticRegression(
                C=C, penalty="l2", max_iter=1000, solver="lbfgs", random_state=42,
            ),
        })

    # --- LogisticRegression L1 (2 configs) ---
    for C_val in [0.1, 1.0]:
        configs.append({
            "name": f"LogReg_L1_C{C_val}",
            "family": "linear",
            "config_str": f"C={C_val},L1",
            "needs_scaling": True,
            "has_predict_proba": True,
            "supports_sample_weight": True,
            "is_l1": True,
            "build_fn": lambda C=C_val: LogisticRegression(
                C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42,
            ),
        })

    # --- LogisticRegression ElasticNet (2 configs) ---
    for C_val, l1r in [(1.0, 0.5), (0.1, 0.3)]:
        configs.append({
            "name": f"LogReg_EN_C{C_val}_r{l1r}",
            "family": "linear",
            "config_str": f"C={C_val},EN,r={l1r}",
            "needs_scaling": True,
            "has_predict_proba": True,
            "supports_sample_weight": True,
            "build_fn": lambda C=C_val, ratio=l1r: LogisticRegression(
                C=C, penalty="elasticnet", l1_ratio=ratio, solver="saga",
                max_iter=1000, random_state=42,
            ),
        })

    # --- RidgeClassifier (1 config) ---
    configs.append({
        "name": "Ridge_a1.0",
        "family": "linear",
        "config_str": "alpha=1.0",
        "needs_scaling": True,
        "has_predict_proba": False,
        "supports_sample_weight": True,
        "build_fn": lambda: RidgeClassifier(alpha=1.0),
    })

    # --- XGBoost (3 configs) ---
    xgb_base = dict(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    xgb_configs = [
        ("XGB_A", "d3,lr0.05,n100", dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                                           min_child_weight=50, subsample=0.7, colsample_bytree=0.7)),
        ("XGB_B", "d2,lr0.03,n50", dict(max_depth=2, learning_rate=0.03, n_estimators=50,
                                          min_child_weight=100, subsample=0.8, colsample_bytree=0.8)),
        ("XGB_C", "d4,lr0.1,n200", dict(max_depth=4, learning_rate=0.1, n_estimators=200,
                                          min_child_weight=30, subsample=0.6, colsample_bytree=0.6)),
    ]
    for name, cfg_str, params in xgb_configs:
        merged = {**xgb_base, **params}
        configs.append({
            "name": name,
            "family": "tree",
            "config_str": cfg_str,
            "needs_scaling": False,
            "has_predict_proba": True,
            "supports_sample_weight": True,
            "build_fn": lambda p=merged: xgb.XGBClassifier(**p),
        })

    # --- LightGBM (2 configs) ---
    if LGB_AVAILABLE:
        lgb_configs = [
            ("LGB_A", "d3,lr0.05,n100", dict(
                max_depth=3, learning_rate=0.05, n_estimators=100,
                min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                verbose=-1, random_state=42,
            )),
            ("LGB_B", "d2,lr0.03,n50", dict(
                max_depth=2, learning_rate=0.03, n_estimators=50,
                min_child_weight=100, verbose=-1, random_state=42,
            )),
        ]
        for name, cfg_str, params in lgb_configs:
            configs.append({
                "name": name,
                "family": "tree",
                "config_str": cfg_str,
                "needs_scaling": False,
                "has_predict_proba": True,
                "supports_sample_weight": True,
                "build_fn": lambda p=params: lgb.LGBMClassifier(**p),
            })

    # --- Random Forest (2 configs) ---
    rf_configs = [
        ("RF_A", "n100,d3,msl50", dict(n_estimators=100, max_depth=3, min_samples_leaf=50, random_state=42)),
        ("RF_B", "n200,d5,msl30", dict(n_estimators=200, max_depth=5, min_samples_leaf=30, random_state=42)),
    ]
    for name, cfg_str, params in rf_configs:
        configs.append({
            "name": name,
            "family": "tree",
            "config_str": cfg_str,
            "needs_scaling": False,
            "has_predict_proba": True,
            "supports_sample_weight": True,
            "build_fn": lambda p=params: RandomForestClassifier(**p),
        })

    # --- LinearSVC (1 config) ---
    configs.append({
        "name": "LinearSVC",
        "family": "other",
        "config_str": "C=1.0",
        "needs_scaling": True,
        "has_predict_proba": False,
        "supports_sample_weight": True,
        "build_fn": lambda: LinearSVC(C=1.0, max_iter=5000, random_state=42),
    })

    # --- KNN (1 config) ---
    configs.append({
        "name": "KNN_50",
        "family": "other",
        "config_str": "k=50,dist",
        "needs_scaling": True,
        "has_predict_proba": True,
        "supports_sample_weight": False,
        "build_fn": lambda: KNeighborsClassifier(n_neighbors=50, weights="distance"),
    })

    return configs


# =============================================================================
# SIGNAL GENERATION — PER MODEL CONFIG
# =============================================================================

def _generate_signals_for_asset(
    symbol: str,
    model_config: dict,
) -> List[Tuple]:
    """Walk-forward signal generation for one asset with given model config.

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

    needs_scaling = model_config["needs_scaling"]
    has_proba = model_config["has_predict_proba"]
    supports_sw = model_config["supports_sample_weight"]

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
        sample_w = None
        if supports_sw:
            train_timestamps = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=FWD_BARS)

        # Build and fit model
        try:
            model = model_config["build_fn"]()

            if needs_scaling:
                scaler = StandardScaler()
                X_tr_arr = scaler.fit_transform(X_train)
                X_te_arr = scaler.transform(X_test)
            else:
                X_tr_arr = X_train.values
                X_te_arr = X_test.values

            # Fit with sample weights where supported
            if supports_sw and sample_w is not None:
                model.fit(X_tr_arr, y_train_binary, sample_weight=sample_w)
            else:
                model.fit(X_tr_arr, y_train_binary)

            # Predict probabilities
            if has_proba:
                yp = model.predict_proba(X_te_arr)[:, 1]
            else:
                # Models without predict_proba: use decision_function + sigmoid
                if hasattr(model, "decision_function"):
                    df_vals = model.decision_function(X_te_arr)
                    yp = 1.0 / (1.0 + np.exp(-df_vals))
                else:
                    # Fallback: predict binary, map to 0.4/0.6
                    preds = model.predict(X_te_arr)
                    yp = np.where(preds == 1, 0.6, 0.4)

        except Exception as e:
            # Model failed for this month — skip
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
        if has_proba:
            signals = get_signal(yp_masked, num_classes=2)
            signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)
        else:
            # For models without predict_proba: fixed signal strength of 0.5
            # (Ridge, LinearSVC use sigmoid-converted proba for direction
            #  but fixed sizing since proba calibration is unreliable)
            signals_disc = np.full(len(yp_masked), 0.5)

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


def generate_signals_for_config(
    model_config: dict,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Generate walk-forward signals for all assets in parallel for one model config."""
    all_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for symbol in ASSETS:
            fut = pool.submit(_generate_signals_for_asset, symbol, model_config)
            futures[fut] = symbol

        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
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
# PORTFOLIO SIMULATION — identical to validation_honest.py
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame) -> Optional[dict]:
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


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(sim: Optional[dict], label: str = "") -> Optional[dict]:
    """Compute standard quant metrics from simulation output."""
    if sim is None or sim["n_trades"] < 20:
        return None

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
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

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

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
        "dsr_num_obs": num_obs,
        "daily_skewness": round(skewness, 3),
        "daily_kurtosis": round(kurt, 3),
    }


# =============================================================================
# L1 FEATURE IMPORTANCE EXTRACTION
# =============================================================================

def _extract_l1_importance(model_config: dict) -> Optional[dict]:
    """Train L1 LogReg on full available data and extract feature coefficients.

    Returns dict with feature names as keys, coefficient magnitudes as values,
    plus a list of zeroed-out features.
    """
    # Gather all data across assets
    all_X = []
    all_y = []

    for symbol in ASSETS:
        if symbol not in _full_features_cache:
            continue
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            continue

        close = ohlcv["close"]
        fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)

        feat = _get_features_for_window(symbol, pd.Timestamp("2026-01-01", tz="UTC"))
        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        if len(valid) < 100:
            continue

        X = fc.loc[valid]
        y = (y_all.loc[valid] > 0).astype(int)
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        return None

    X_combined = pd.concat(all_X)
    y_combined = pd.concat(all_y)

    # Remove regime column if present
    feat_cols = [c for c in X_combined.columns if c != "regime"]
    X_combined = X_combined[feat_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    model = model_config["build_fn"]()
    model.fit(X_scaled, y_combined)

    coefs = model.coef_[0]
    feature_importance = {}
    zeroed_features = []

    for i, feat_name in enumerate(feat_cols):
        feature_importance[feat_name] = round(float(abs(coefs[i])), 6)
        if abs(coefs[i]) < 1e-6:
            zeroed_features.append(feat_name)

    return {
        "coefficients": {feat_cols[i]: round(float(coefs[i]), 6) for i in range(len(feat_cols))},
        "abs_importance": feature_importance,
        "zeroed_features": zeroed_features,
        "n_active": len(feat_cols) - len(zeroed_features),
        "n_total": len(feat_cols),
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    print("=" * 100, flush=True)
    print("MODEL COMPARISON — 17 configs, 9 model types, 19 features (no funding_zscore)", flush=True)
    print("=" * 100, flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD: {MAX_DD*100:.0f}%, "
          f"Delever: {DELEVER_START*100:.1f}%", flush=True)
    print(f"Sizing: {BASE_SIZING_PCT*100:.0f}%/trade, MaxExposure: {MAX_EXPOSURE*100:.0f}%, "
          f"MaxPerAsset: {MAX_PER_ASSET*100:.0f}%", flush=True)
    print(f"Slippage: {SLIPPAGE*10000:.0f}bp | Bars: {BAR_TF}, Fwd: {FWD_BARS} bars "
          f"({FWD_BARS*30}m)", flush=True)
    print(f"Embargo: {EMBARGO_BARS} bars ({EMBARGO_TIMEDELTA})", flush=True)
    print(f"Assets: {ASSETS}", flush=True)
    print(f"FFD d*: {FFD_D_STARS}", flush=True)
    print(f"Months: {ALL_MONTHS[0]}-{ALL_MONTHS[-1]} ({len(ALL_MONTHS)} months)", flush=True)
    print(f"HMM: {HMM_AVAILABLE}, LightGBM: {LGB_AVAILABLE}", flush=True)
    print(f"Features: {len(BASE_19_FEATURES)} (funding_zscore dropped)", flush=True)
    print("=" * 100, flush=True)

    # =========================================================================
    # PHASE 0: Load data and precompute features
    # =========================================================================
    print("\n[PHASE 0] Loading data and precomputing features...", flush=True)
    t0 = time.time()

    preload_all_data()
    print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)

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
    # PHASE 1: Run all model configs
    # =========================================================================
    model_configs = _build_model_configs()
    n_configs = len(model_configs)

    print(f"\n{'='*100}", flush=True)
    print(f"[PHASE 1] Running {n_configs} model configurations...", flush=True)
    print(f"{'='*100}", flush=True)

    config_metrics: List[Optional[dict]] = []
    config_names: List[str] = []

    # Track L1 configs for feature importance extraction
    l1_configs = []

    for idx, cfg in enumerate(model_configs):
        t_cfg = time.time()
        label = f"{cfg['name']} ({cfg['config_str']})"
        print(f"\n  [Config {idx+1}/{n_configs}] {label}...", flush=True)

        # Track L1 configs
        if cfg.get("is_l1", False):
            l1_configs.append(cfg)

        try:
            signals = generate_signals_for_config(cfg, max_workers=3)
            n_signals = len(signals)

            sim = simulate_portfolio(signals)
            m = compute_metrics(sim, label)
            config_metrics.append(m)
            config_names.append(cfg["name"])

            elapsed = time.time() - t_cfg

            if m:
                print(
                    f"    Sharpe: {m['sharpe']:.3f}, "
                    f"Sortino: {m['sortino']:.3f}, "
                    f"Return: {m['total_return_pct']:+.2f}%, "
                    f"MaxDD: {m['max_dd_pct']:.2f}%, "
                    f"WR: {m['win_rate_trades']*100:.1f}%, "
                    f"Trd/day: {m['trades_per_day']:.1f}, "
                    f"[{n_signals} sigs, {elapsed:.0f}s]",
                    flush=True,
                )
            else:
                print(f"    INSUFFICIENT DATA ({n_signals} signals, {elapsed:.0f}s)", flush=True)

        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            traceback.print_exc()
            config_metrics.append(None)
            config_names.append(cfg["name"])

    # =========================================================================
    # PHASE 2: L1 Feature Importance
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("[PHASE 2] L1 Feature Importance Analysis...", flush=True)
    print(f"{'='*100}", flush=True)

    l1_importance_results = {}
    for l1_cfg in l1_configs:
        print(f"  Extracting coefficients for {l1_cfg['name']}...", flush=True)
        try:
            imp = _extract_l1_importance(l1_cfg)
            if imp:
                l1_importance_results[l1_cfg["name"]] = imp
                print(f"    Active features: {imp['n_active']}/{imp['n_total']}", flush=True)
                if imp["zeroed_features"]:
                    print(f"    ZEROED (noise): {imp['zeroed_features']}", flush=True)
                else:
                    print(f"    No features zeroed — all contribute", flush=True)

                # Print top features by absolute importance
                sorted_imp = sorted(
                    imp["abs_importance"].items(), key=lambda x: x[1], reverse=True
                )
                print(f"    Top 10 by |coef|:", flush=True)
                for feat_name, abs_val in sorted_imp[:10]:
                    coef_val = imp["coefficients"][feat_name]
                    print(f"      {feat_name:<25} coef={coef_val:+.6f}  |coef|={abs_val:.6f}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)

    # =========================================================================
    # RESULTS: Ranked comparison table
    # =========================================================================
    total_time = time.time() - overall_start

    print(f"\n{'='*100}", flush=True)
    print(f"=== MODEL COMPARISON (19 features, no funding_zscore, embargo+weights) ===", flush=True)
    print(f"=== Total runtime: {total_time:.0f}s ({total_time/60:.1f}m) ===", flush=True)
    print(f"{'='*100}", flush=True)

    # Build ranked list (only configs with valid metrics)
    ranked = []
    for idx, cfg in enumerate(model_configs):
        m = config_metrics[idx]
        if m is not None:
            ranked.append((cfg, m))

    # Sort by Sharpe descending
    ranked.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    hdr = (
        f"{'Rank':>4}  {'Model':<20} {'Config':<20} {'Sharpe':>7} {'Sortino':>8} "
        f"{'MaxDD%':>7} {'WinRate':>8} {'Trd/day':>8} {'Return%':>9} {'DSR':>6}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for rank, (cfg, m) in enumerate(ranked, 1):
        print(
            f"{rank:>4}  {cfg['name']:<20} {cfg['config_str']:<20} "
            f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
            f"{m['max_dd_pct']:>7.2f} {m['win_rate_trades']*100:>7.1f}% "
            f"{m['trades_per_day']:>8.1f} {m['total_return_pct']:>+8.2f}% "
            f"{m['dsr']:>6.3f}",
            flush=True,
        )

    # Failed configs
    failed = [
        cfg["name"] for idx, cfg in enumerate(model_configs) if config_metrics[idx] is None
    ]
    if failed:
        print(f"\nFailed/insufficient data: {failed}", flush=True)

    # =========================================================================
    # ANALYSIS: Best linear vs best tree
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("ANALYSIS", flush=True)
    print(f"{'='*100}", flush=True)

    best_linear = None
    best_tree = None
    best_other = None

    for cfg, m in ranked:
        if cfg["family"] == "linear" and best_linear is None:
            best_linear = (cfg, m)
        elif cfg["family"] == "tree" and best_tree is None:
            best_tree = (cfg, m)
        elif cfg["family"] == "other" and best_other is None:
            best_other = (cfg, m)

    print("\n1. Best Linear vs Best Tree:", flush=True)
    if best_linear and best_tree:
        bl_cfg, bl_m = best_linear
        bt_cfg, bt_m = best_tree
        delta = bl_m["sharpe"] - bt_m["sharpe"]
        print(f"   Best linear: {bl_cfg['name']} ({bl_cfg['config_str']}) — Sharpe {bl_m['sharpe']:.3f}", flush=True)
        print(f"   Best tree:   {bt_cfg['name']} ({bt_cfg['config_str']}) — Sharpe {bt_m['sharpe']:.3f}", flush=True)
        print(f"   Delta (linear - tree): {delta:+.3f}", flush=True)
        if delta > 0.1:
            print("   >> VERDICT: Linear models outperform trees. Signal is predominantly linear.", flush=True)
        elif delta < -0.1:
            print("   >> VERDICT: Trees outperform linear. Non-linear patterns matter.", flush=True)
        else:
            print("   >> VERDICT: Comparable. Signal is linear (trees add no value).", flush=True)
    else:
        print("   >> Cannot compare — insufficient results in one or both families.", flush=True)

    print("\n2. Regularization Sensitivity (L2 family):", flush=True)
    l2_results = [(cfg, m) for cfg, m in ranked if cfg["name"].startswith("LogReg_L2")]
    if len(l2_results) >= 2:
        for cfg, m in l2_results:
            print(f"   {cfg['name']:<20} {cfg['config_str']:<15} Sharpe={m['sharpe']:.3f}", flush=True)
        best_l2 = l2_results[0]
        worst_l2 = l2_results[-1]
        spread = best_l2[1]["sharpe"] - worst_l2[1]["sharpe"]
        print(f"   Spread (best-worst): {spread:.3f}", flush=True)
        if spread > 0.3:
            print("   >> Regularization matters significantly. Model is sensitive to C.", flush=True)
        else:
            print("   >> Regularization has minor impact. Signal is robust.", flush=True)
    else:
        print("   >> Insufficient L2 results.", flush=True)

    print("\n3. L1 Feature Selection (which features are noise?):", flush=True)
    if l1_importance_results:
        for l1_name, imp in l1_importance_results.items():
            print(f"   {l1_name}:", flush=True)
            print(f"     Active: {imp['n_active']}/{imp['n_total']} features", flush=True)
            if imp["zeroed_features"]:
                print(f"     DROPPED by L1 (noise): {imp['zeroed_features']}", flush=True)
            else:
                print(f"     All features retained (none zeroed)", flush=True)
            # Show bottom 5 by importance
            sorted_imp = sorted(imp["abs_importance"].items(), key=lambda x: x[1])
            print(f"     Weakest 5 features:", flush=True)
            for feat_name, abs_val in sorted_imp[:5]:
                print(f"       {feat_name:<25} |coef|={abs_val:.6f}", flush=True)
    else:
        print("   >> No L1 results available.", flush=True)

    print("\n4. XGBoost Depth Sensitivity:", flush=True)
    xgb_results = [(cfg, m) for cfg, m in ranked if cfg["name"].startswith("XGB_")]
    if len(xgb_results) >= 2:
        for cfg, m in xgb_results:
            print(f"   {cfg['name']:<20} {cfg['config_str']:<20} Sharpe={m['sharpe']:.3f}", flush=True)
        best_xgb = xgb_results[0]
        worst_xgb = xgb_results[-1]
        print(f"   Best: {best_xgb[0]['name']}, Worst: {worst_xgb[0]['name']}, "
              f"Spread: {best_xgb[1]['sharpe'] - worst_xgb[1]['sharpe']:.3f}", flush=True)
    else:
        print("   >> Insufficient XGBoost results.", flush=True)

    if best_other:
        print(f"\n5. KNN/SVC Reference:", flush=True)
        bo_cfg, bo_m = best_other
        print(f"   {bo_cfg['name']}: Sharpe={bo_m['sharpe']:.3f} "
              f"(expected: poor for non-parametric on this data)", flush=True)

    # =========================================================================
    # Save results
    # =========================================================================
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "version": "model_comparison_v1",
        "total_runtime_s": round(total_time, 1),
        "config": {
            "starting_capital": STARTING_CAPITAL,
            "max_dd": MAX_DD,
            "slippage": SLIPPAGE,
            "max_exposure": MAX_EXPOSURE,
            "max_per_asset": MAX_PER_ASSET,
            "base_sizing_pct": BASE_SIZING_PCT,
            "bar_tf": BAR_TF,
            "fwd_bars": FWD_BARS,
            "embargo_bars": EMBARGO_BARS,
            "assets": ASSETS,
            "months": [str(m) for m in ALL_MONTHS],
            "ffd_d_stars": FFD_D_STARS,
            "hmm_available": HMM_AVAILABLE,
            "lgb_available": LGB_AVAILABLE,
            "n_features": len(BASE_19_FEATURES),
            "features": BASE_19_FEATURES,
        },
        "hl_data_available": {s: (_hl_cache.get(s) is not None) for s in ASSETS},
        "rankings": [
            {
                "rank": rank,
                "model": cfg["name"],
                "family": cfg["family"],
                "config": cfg["config_str"],
                **m,
            }
            for rank, (cfg, m) in enumerate(ranked, 1)
        ],
        "failed_configs": failed,
        "l1_feature_importance": l1_importance_results,
        "analysis": {
            "best_linear": best_linear[0]["name"] if best_linear else None,
            "best_linear_sharpe": best_linear[1]["sharpe"] if best_linear else None,
            "best_tree": best_tree[0]["name"] if best_tree else None,
            "best_tree_sharpe": best_tree[1]["sharpe"] if best_tree else None,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_FILE}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"DONE. Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"{'='*100}", flush=True)
