"""
Honest Validation — The Most Important Script.

Answers the question: does the ML model add ANY value at all?

Compares 5 variants through the SAME portfolio simulator:
  A. XGB_FIXED     — XGBoost with embargo + sample weights + filters
  B. XGB_NO_FUNDING — same as A, minus funding_zscore
  C. LOGISTIC       — LogisticRegression baseline (same features/pipeline)
  D. RANDOM         — random probabilities, same portfolio management (10 seeds)
  E. XGB_ORIGINAL   — current v4_tier2 system (no embargo, no sample weights)

Three validation fixes vs v4_tier2:
  1. EMBARGO: 4-bar gap between train end and test start (prevents label overlap)
  2. SAMPLE WEIGHTS: AFML Ch4 average uniqueness for overlapping 4-bar labels
  3. Both fixes combined in A/B/C; absent in D (irrelevant) and E (intentionally)

Architecture: precompute-once, slice-per-window (same as v4_tier2).
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
RESULTS_FILE = OUTPUT_DIR / "honest_validation_results.json"

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

# Portfolio parameters — identical to v4_tier2
STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001  # 1 bps
MAX_EXPOSURE = 0.60
MAX_PER_ASSET = 0.20
BASE_SIZING_PCT = 0.02

VOL_FILTER_QUANTILE = 0.50
CONFIDENCE_FILTER_PERCENTILE = 70

XGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    min_child_weight=50,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)

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

# Random baseline seeds
RANDOM_SEEDS = list(range(42, 52))  # 42..51, 10 seeds


# =============================================================================
# SAMPLE WEIGHTS — AFML Ch 4: Average Uniqueness
# =============================================================================

def compute_sample_weights(timestamps: np.ndarray, fwd_bars: int = FWD_BARS) -> np.ndarray:
    """AFML Ch 4: Average uniqueness for overlapping labels.

    For each label i spanning [t_i, t_i + fwd_bars], count how many other
    labels overlap that interval. Weight = 1 / concurrency. Normalized to
    mean = 1 so XGBoost interprets them as relative importance.
    """
    n = len(timestamps)
    weights = np.ones(n, dtype=np.float64)

    for i in range(n):
        t_start = timestamps[i]
        t_end = timestamps[min(i + fwd_bars, n - 1)]
        # Count labels whose start falls within [t_start, t_end]
        concurrent = np.sum((timestamps >= t_start) & (timestamps <= t_end))
        weights[i] = 1.0 / max(concurrent, 1)

    # Normalize to mean = 1
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w

    return weights


# =============================================================================
# DATA LOADING (module-level caches — shared across threads)
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
    """Precompute ALL 20 features for all assets on the FULL series.

    Features: ret_1, ret_3, ret_7, ret_14, bar_range, vol_zscore, rsi_14,
    stoch_rsi, macd_hist, bb_pct_b, atr_14, dist_swing_high, dist_swing_low,
    dist_nearest_level, ffd, cs_spread, entropy, funding_zscore, oi_change_5,
    impact_spread.

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

        # --- Returns (un-winsorized; winsorization applied per window) ---
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

        # --- HL Derivatives ---
        hl_data = _hl_cache.get(symbol)
        if hl_data is not None:
            if "funding" in hl_data.columns:
                try:
                    ff = funding_features(hl_data["funding"])
                    if "funding_zscore" in ff.columns:
                        f["funding_zscore"] = ff["funding_zscore"].reindex(f.index)
                except Exception:
                    pass

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

# The 20 base features used by ALL model variants (except NO_FUNDING drops one)
BASE_20_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14",
    "bar_range", "vol_zscore", "rsi_14", "stoch_rsi", "macd_hist",
    "bb_pct_b", "atr_14", "dist_swing_high", "dist_swing_low",
    "dist_nearest_level", "ffd", "cs_spread", "entropy",
    "funding_zscore", "oi_change_5", "impact_spread",
]


def _get_features_for_window(
    symbol: str,
    train_end: pd.Timestamp,
    drop_funding: bool = False,
) -> pd.DataFrame:
    """Slice precomputed features and apply per-window return winsorization.

    Args:
        symbol: Asset symbol.
        train_end: Timestamp marking end of training period.
        drop_funding: If True, exclude funding_zscore (Variant B).

    Returns:
        DataFrame with features, returns winsorized per window.
    """
    base = _full_features_cache[symbol]

    # Select only the 20 base features (ignore cross-asset, extra OI)
    cols = [c for c in BASE_20_FEATURES if c in base.columns]
    if drop_funding:
        cols = [c for c in cols if c != "funding_zscore"]

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
# SIGNAL GENERATION — PER VARIANT
# =============================================================================

def _generate_signals_for_asset(
    symbol: str,
    variant: str,
    random_seed: Optional[int] = None,
) -> List[Tuple]:
    """Walk-forward signal generation for one asset.

    Args:
        symbol: Asset symbol.
        variant: One of "XGB_FIXED", "XGB_NO_FUNDING", "LOGISTIC", "RANDOM",
                 "XGB_ORIGINAL".
        random_seed: Seed for RANDOM variant (required if variant=="RANDOM").

    Returns:
        List of (timestamp, symbol, direction, fwd_return, confidence,
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

    # Determine variant behavior
    use_embargo = variant in ("XGB_FIXED", "XGB_NO_FUNDING", "LOGISTIC")
    use_sample_weights = variant in ("XGB_FIXED", "XGB_NO_FUNDING", "LOGISTIC")
    drop_funding = variant == "XGB_NO_FUNDING"
    use_hmm_regime = variant in ("XGB_FIXED", "XGB_NO_FUNDING", "XGB_ORIGINAL")
    is_random = variant == "RANDOM"
    is_logistic = variant == "LOGISTIC"

    # Precomputed regime for this symbol
    full_regime = _hmm_regime_cache.get(symbol)

    rows = []
    rng = np.random.RandomState(random_seed) if is_random else None

    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        # --- FIX 1: EMBARGO ---
        # For fixed variants: train on data strictly before (month_start - embargo)
        # For original: train on data before month_start (no gap)
        if use_embargo:
            train_cutoff = ms - EMBARGO_TIMEDELTA
        else:
            train_cutoff = ms

        # Slice features with per-window winsorization
        feat = _get_features_for_window(symbol, train_cutoff, drop_funding=drop_funding)

        # Add regime feature if appropriate
        worst_regime = None
        if use_hmm_regime and full_regime is not None:
            regime_col = full_regime.reindex(feat.index)
            feat["regime"] = regime_col.fillna(1).astype(int)

            # Find worst regime from training data only
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

        # --- FIX 2: SAMPLE WEIGHTS ---
        sample_w = None
        if use_sample_weights:
            # Compute weights based on training timestamps
            train_timestamps = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=FWD_BARS)

        # --- MODEL FITTING & PREDICTION ---
        if is_random:
            # RANDOM baseline: uniform random probabilities in [0.45, 0.55]
            yp = rng.uniform(0.45, 0.55, size=len(X_test))

        elif is_logistic:
            # LOGISTIC baseline: standardized features, L2 regularization
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train)
            X_te_scaled = scaler.transform(X_test)

            lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            if sample_w is not None:
                lr_model.fit(X_tr_scaled, y_train_binary, sample_weight=sample_w)
            else:
                lr_model.fit(X_tr_scaled, y_train_binary)
            yp = lr_model.predict_proba(X_te_scaled)[:, 1]

        else:
            # XGBoost (FIXED or ORIGINAL)
            model = xgb.XGBClassifier(**XGB_PARAMS)
            if sample_w is not None:
                model.fit(X_train, y_train_binary, sample_weight=sample_w)
            else:
                model.fit(X_train, y_train_binary)
            yp = model.predict_proba(X_test)[:, 1]

        # --- Confidence filter (top 30%) ---
        conf = np.abs(yp - 0.5)
        if len(conf) > 10:
            conf_threshold = np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
            conf_mask = conf >= conf_threshold
        else:
            conf_mask = np.ones(len(conf), dtype=bool)

        # --- Regime suppression (suppress worst regime trades) ---
        regime_mask = np.ones(len(yp), dtype=bool)
        if use_hmm_regime and worst_regime is not None and "regime" in X_test.columns:
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
                rows.append((
                    idx, symbol, directions[j],
                    float(fr),
                    float(conf_masked[j]),
                    float(yp_masked[j]),
                    float(signals_disc[j]) if np.ndim(signals_disc) > 0 else float(signals_disc),
                ))

    return rows


def generate_signals(
    variant: str,
    random_seed: Optional[int] = None,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Generate walk-forward signals for all assets in parallel.

    Args:
        variant: One of "XGB_FIXED", "XGB_NO_FUNDING", "LOGISTIC", "RANDOM",
                 "XGB_ORIGINAL".
        random_seed: Only used when variant == "RANDOM".
        max_workers: Thread pool size.

    Returns:
        DataFrame with columns: timestamp, asset, direction, fwd_return,
        confidence, probability, signal_strength.
    """
    t_start = time.time()

    all_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for symbol in ASSETS:
            fut = pool.submit(
                _generate_signals_for_asset,
                symbol, variant, random_seed,
            )
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

    elapsed = time.time() - t_start
    return df


# =============================================================================
# PORTFOLIO SIMULATION — identical to v4_tier2
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame) -> Optional[dict]:
    """Portfolio simulation with shared capital pool.

    Same logic as v4_tier2: probability-based sizing, exposure limits,
    dynamic deleveraging, DD halt.
    """
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
# MAIN EXPERIMENT
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    print("=" * 100, flush=True)
    print("HONEST VALIDATION — Does the Model Add Any Value?", flush=True)
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
    print(f"Random seeds: {RANDOM_SEEDS}", flush=True)
    print(f"HMM: {HMM_AVAILABLE}", flush=True)
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
    # PHASE 1-5: Run all variants
    # =========================================================================

    VARIANTS = [
        ("XGB_ORIGINAL",  "XGB_ORIGINAL (v4, no fixes)"),
        ("XGB_FIXED",     "XGB_FIXED (embargo+weights)"),
        ("XGB_NO_FUNDING", "XGB_NO_FUNDING (no funding_zscore)"),
        ("LOGISTIC",      "LOGISTIC baseline"),
        ("RANDOM",        "RANDOM baseline"),
    ]

    variant_metrics: Dict[str, Optional[dict]] = {}
    all_results: Dict[str, dict] = {
        "timestamp": datetime.now().isoformat(),
        "version": "honest_validation_v1",
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
            "random_seeds": RANDOM_SEEDS,
        },
        "hl_data_available": {s: (_hl_cache.get(s) is not None) for s in ASSETS},
    }

    for idx, (variant_key, variant_label) in enumerate(VARIANTS):
        phase_num = idx + 1
        print(f"\n{'='*80}", flush=True)
        print(f"[PHASE {phase_num}/5] {variant_label}", flush=True)
        print(f"{'='*80}", flush=True)

        t_var = time.time()

        if variant_key == "RANDOM":
            # --- Run 10 seeds, average metrics ---
            print(f"  Running {len(RANDOM_SEEDS)} random seeds...", flush=True)
            seed_sharpes = []
            seed_sortinos = []
            seed_maxdds = []
            seed_winrates = []
            seed_returns = []
            seed_trades = []
            seed_metrics_list = []

            for seed_idx, seed in enumerate(RANDOM_SEEDS):
                signals = generate_signals("RANDOM", random_seed=seed, max_workers=3)
                sim = simulate_portfolio(signals)
                m = compute_metrics(sim, f"RANDOM_seed{seed}")

                if m is not None:
                    seed_sharpes.append(m["sharpe"])
                    seed_sortinos.append(m["sortino"])
                    seed_maxdds.append(m["max_dd_pct"])
                    seed_winrates.append(m["win_rate_trades"])
                    seed_returns.append(m["total_return_pct"])
                    seed_trades.append(m["trades_per_day"])
                    seed_metrics_list.append(m)

                print(
                    f"    Seed {seed}: "
                    + (f"Sharpe={m['sharpe']:.3f}, Ret={m['total_return_pct']:+.2f}%"
                       if m else "N/A"),
                    flush=True,
                )

            if seed_sharpes:
                # Build averaged metrics for RANDOM
                avg_m = {
                    "label": "RANDOM baseline (mean of 10 seeds)",
                    "n_trades": int(np.mean([m["n_trades"] for m in seed_metrics_list])),
                    "trades_per_day": round(np.mean(seed_trades), 2),
                    "total_return_pct": round(np.mean(seed_returns), 2),
                    "cagr_pct": round(
                        np.mean([m["cagr_pct"] for m in seed_metrics_list]), 2
                    ),
                    "final_equity": round(
                        np.mean([m["final_equity"] for m in seed_metrics_list]), 2
                    ),
                    "sharpe": round(np.mean(seed_sharpes), 3),
                    "sortino": round(np.mean(seed_sortinos), 3),
                    "calmar": round(
                        np.mean([m["calmar"] for m in seed_metrics_list]), 3
                    ),
                    "max_dd_pct": round(np.mean(seed_maxdds), 2),
                    "win_rate_trades": round(np.mean(seed_winrates), 4),
                    "win_rate_days": round(
                        np.mean([m["win_rate_days"] for m in seed_metrics_list]), 4
                    ),
                    "profit_factor": round(
                        np.mean([m["profit_factor"] for m in seed_metrics_list]), 3
                    ),
                    "mean_daily_pnl": round(
                        np.mean([m["mean_daily_pnl"] for m in seed_metrics_list]), 2
                    ),
                    "halted": any(m["halted"] for m in seed_metrics_list),
                    "dsr": round(
                        np.mean([m["dsr"] for m in seed_metrics_list]), 4
                    ),
                    "dsr_num_obs": int(
                        np.mean([m["dsr_num_obs"] for m in seed_metrics_list])
                    ),
                    "daily_skewness": round(
                        np.mean([m["daily_skewness"] for m in seed_metrics_list]), 3
                    ),
                    "daily_kurtosis": round(
                        np.mean([m["daily_kurtosis"] for m in seed_metrics_list]), 3
                    ),
                    # Extra: standard deviations across seeds
                    "_sharpe_std": round(np.std(seed_sharpes), 3),
                    "_return_std": round(np.std(seed_returns), 2),
                    "_seeds_run": len(seed_sharpes),
                }
                variant_metrics["RANDOM"] = avg_m
                all_results["random_per_seed"] = [
                    {k: v for k, v in m.items()} for m in seed_metrics_list
                ]

                print(
                    f"\n  RANDOM SUMMARY: Sharpe={avg_m['sharpe']:.3f} "
                    f"(std={avg_m['_sharpe_std']:.3f}), "
                    f"Return={avg_m['total_return_pct']:+.2f}% "
                    f"(std={avg_m['_return_std']:.2f}%)",
                    flush=True,
                )
            else:
                variant_metrics["RANDOM"] = None
                print("  RANDOM: all seeds failed", flush=True)

        else:
            # --- Standard single-run variant ---
            signals = generate_signals(variant_key, max_workers=3)
            n_signals = len(signals)
            print(f"  {n_signals} total signals generated", flush=True)

            sim = simulate_portfolio(signals)
            m = compute_metrics(sim, variant_label)
            variant_metrics[variant_key] = m

            if m:
                print(
                    f"  Sharpe: {m['sharpe']:.3f}, "
                    f"Return: {m['total_return_pct']:+.2f}%, "
                    f"MaxDD: {m['max_dd_pct']:.2f}%, "
                    f"WR: {m['win_rate_trades']:.4f}, "
                    f"PF: {m['profit_factor']:.3f}, "
                    f"DSR: {m['dsr']:.4f}",
                    flush=True,
                )
            else:
                print("  [WARN] Insufficient data for metrics", flush=True)

        elapsed = time.time() - t_var
        print(f"  Phase time: {elapsed:.0f}s", flush=True)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("=== HONEST VALIDATION RESULTS ===", flush=True)
    print(f"{'='*100}", flush=True)

    # Header
    hdr = f"{'Variant':<28} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WinRate':>8} {'Trd/day':>8} {'Return%':>9} {'DSR':>6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for vkey, vlabel in VARIANTS:
        m = variant_metrics.get(vkey)
        if m is None:
            print(f"{vlabel:<28} {'N/A':>7} {'N/A':>8} {'N/A':>7} {'N/A':>8} {'N/A':>8} {'N/A':>9} {'N/A':>6}", flush=True)
            continue

        sharpe_str = f"{m['sharpe']:.2f}"
        sortino_str = f"{m['sortino']:.2f}"
        maxdd_str = f"{m['max_dd_pct']:.2f}"
        wr_str = f"{m['win_rate_trades']*100:.1f}%"
        tpd_str = f"{m['trades_per_day']:.1f}"
        ret_str = f"{m['total_return_pct']:+.2f}%"
        dsr_str = f"{m['dsr']:.3f}"

        # Append std for RANDOM
        if vkey == "RANDOM" and "_sharpe_std" in m:
            sharpe_str += f"+-{m['_sharpe_std']:.2f}"

        print(
            f"{vlabel:<28} {sharpe_str:>7} {sortino_str:>8} {maxdd_str:>7} "
            f"{wr_str:>8} {tpd_str:>8} {ret_str:>9} {dsr_str:>6}",
            flush=True,
        )

    # =========================================================================
    # KEY QUESTIONS
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("KEY QUESTIONS:", flush=True)
    print(f"{'='*100}", flush=True)

    m_orig = variant_metrics.get("XGB_ORIGINAL")
    m_fixed = variant_metrics.get("XGB_FIXED")
    m_nofund = variant_metrics.get("XGB_NO_FUNDING")
    m_logistic = variant_metrics.get("LOGISTIC")
    m_random = variant_metrics.get("RANDOM")

    # Q1: Does embargo + weights change Sharpe?
    print("\n1. Does embargo+weights change Sharpe? (ORIGINAL vs FIXED)", flush=True)
    if m_orig and m_fixed:
        delta = m_fixed["sharpe"] - m_orig["sharpe"]
        print(f"   ORIGINAL Sharpe: {m_orig['sharpe']:.3f}", flush=True)
        print(f"   FIXED Sharpe:    {m_fixed['sharpe']:.3f}", flush=True)
        print(f"   Delta:           {delta:+.3f}", flush=True)
        if delta < -0.3:
            print("   >> VERDICT: Embargo+weights REDUCES Sharpe significantly.", flush=True)
            print("      The original Sharpe was likely inflated by label leakage.", flush=True)
        elif abs(delta) < 0.15:
            print("   >> VERDICT: Minimal change. Leakage was not a major factor.", flush=True)
        else:
            print("   >> VERDICT: Embargo+weights IMPROVES Sharpe. Model is robust.", flush=True)
    else:
        print("   >> Cannot compare — one or both variants failed.", flush=True)

    # Q2: Is funding_zscore real?
    print("\n2. Is funding_zscore adding real signal? (FIXED vs NO_FUNDING)", flush=True)
    if m_fixed and m_nofund:
        delta = m_fixed["sharpe"] - m_nofund["sharpe"]
        print(f"   FIXED Sharpe:       {m_fixed['sharpe']:.3f}", flush=True)
        print(f"   NO_FUNDING Sharpe:  {m_nofund['sharpe']:.3f}", flush=True)
        print(f"   Delta:              {delta:+.3f}", flush=True)
        if delta > 0.15:
            print("   >> VERDICT: funding_zscore adds meaningful signal (+{:.3f} Sharpe).".format(delta), flush=True)
        elif delta < -0.15:
            print("   >> VERDICT: funding_zscore HURTS — model is better without it.", flush=True)
        else:
            print("   >> VERDICT: Marginal. funding_zscore is noise at this level.", flush=True)
    else:
        print("   >> Cannot compare — one or both variants failed.", flush=True)

    # Q3: Is XGBoost complexity needed?
    print("\n3. Is XGBoost complexity needed? (FIXED vs LOGISTIC)", flush=True)
    if m_fixed and m_logistic:
        delta = m_fixed["sharpe"] - m_logistic["sharpe"]
        print(f"   FIXED (XGB) Sharpe:    {m_fixed['sharpe']:.3f}", flush=True)
        print(f"   LOGISTIC Sharpe:       {m_logistic['sharpe']:.3f}", flush=True)
        print(f"   Delta:                 {delta:+.3f}", flush=True)
        if delta > 0.2:
            print("   >> VERDICT: XGBoost adds value over linear. Non-linear patterns exist.", flush=True)
        elif abs(delta) < 0.1:
            print("   >> VERDICT: No benefit from XGBoost. Signal is linear or absent.", flush=True)
        else:
            print("   >> VERDICT: Small benefit from XGBoost. Marginal non-linearity.", flush=True)
    else:
        print("   >> Cannot compare — one or both variants failed.", flush=True)

    # Q4: Does the MODEL add anything vs RANDOM?
    print("\n4. Does the MODEL add anything? (FIXED vs RANDOM) — THE CRITICAL TEST", flush=True)
    if m_fixed and m_random:
        delta = m_fixed["sharpe"] - m_random["sharpe"]
        random_std = m_random.get("_sharpe_std", 0)
        print(f"   FIXED (XGB) Sharpe:   {m_fixed['sharpe']:.3f}", flush=True)
        print(f"   RANDOM Sharpe:        {m_random['sharpe']:.3f} (std={random_std:.3f})", flush=True)
        print(f"   Delta:                {delta:+.3f}", flush=True)

        # Statistical significance: is delta > 2 * random_std?
        if random_std > 0:
            z = delta / random_std
            print(f"   Z-score vs random:    {z:.2f}", flush=True)
        else:
            z = float("inf") if delta > 0 else 0

        if delta < 0.1 or (random_std > 0 and z < 2.0):
            print("   >> VERDICT: MODEL ADDS NO VALUE. Portfolio management IS the edge.", flush=True)
            print("      The XGBoost model's signal is indistinguishable from random.", flush=True)
            print("      Implication: simplify. Use a rules-based system, not ML.", flush=True)
        elif z >= 2.0 and delta >= 0.3:
            print("   >> VERDICT: MODEL ADDS REAL VALUE. Signal is significantly above random.", flush=True)
        else:
            print("   >> VERDICT: Weak evidence of model value. Not conclusive.", flush=True)

        if m_random["sharpe"] > 0.5:
            print(f"\n   !! WARNING: Random baseline achieves Sharpe {m_random['sharpe']:.3f}.", flush=True)
            print("      This means portfolio management alone generates significant returns.", flush=True)
            print("      The model's true contribution is only the EXCESS over random.", flush=True)
    else:
        print("   >> Cannot compare — one or both variants failed.", flush=True)

    # =========================================================================
    # DETAILED TABLE
    # =========================================================================
    print(f"\n{'='*100}", flush=True)
    print("DETAILED METRICS", flush=True)
    print(f"{'='*100}", flush=True)

    metrics_to_show = [
        ("Sharpe",            "sharpe",           ".3f"),
        ("Sortino",           "sortino",          ".3f"),
        ("Calmar",            "calmar",           ".3f"),
        ("Total Return %",    "total_return_pct",  "+.2f"),
        ("CAGR %",            "cagr_pct",         "+.2f"),
        ("MaxDD %",           "max_dd_pct",       ".2f"),
        ("Win Rate (trades)", "win_rate_trades",   ".4f"),
        ("Win Rate (days)",   "win_rate_days",     ".4f"),
        ("Profit Factor",     "profit_factor",    ".3f"),
        ("Trades",            "n_trades",         "d"),
        ("Trades/day",        "trades_per_day",   ".2f"),
        ("DSR",               "dsr",              ".4f"),
        ("Mean daily PnL",    "mean_daily_pnl",   ".2f"),
        ("Final Equity",      "final_equity",     ".2f"),
        ("Halted",            "halted",           ""),
    ]

    header = f"{'Metric':<22}"
    for vkey, vlabel in VARIANTS:
        short = vkey[:16]
        header += f" {short:>16}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for label, key, fmt in metrics_to_show:
        row = f"{label:<22}"
        for vkey, _ in VARIANTS:
            m = variant_metrics.get(vkey)
            if m is None:
                row += f" {'N/A':>16}"
            elif key == "halted":
                row += f" {'HALT' if m[key] else 'OK':>16}"
            else:
                val = m.get(key, float("nan"))
                try:
                    row += f" {format(val, fmt):>16}"
                except (ValueError, TypeError):
                    row += f" {str(val):>16}"
        print(row, flush=True)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    all_results["variants"] = {}
    for vkey, vlabel in VARIANTS:
        m = variant_metrics.get(vkey)
        all_results["variants"][vkey] = m if m else {"label": vlabel, "failed": True}

    total_time = time.time() - overall_start
    all_results["total_runtime_seconds"] = round(total_time, 1)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"Results saved to: {RESULTS_FILE}", flush=True)
    print(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"{'='*100}", flush=True)
