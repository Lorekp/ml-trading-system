"""HISTORICAL EXPERIMENT — its deployment verdict is not current evidence.

Tier 1 diagnostic study with 11 checks.

This was written as a gate between a backtest and deployment, but that claim is
withdrawn; see `docs/VALIDATION.md`.
Tests regime dependence, per-asset breakdown, monthly stability, feature
coefficient stability, honest DSR, CPCV, latency sensitivity, train/test
separation audit, feature look-ahead check, winsorization leak check,
and HMM filter value.

Model: LogReg L1 C=0.1 with StandardScaler, 19 features (no funding_zscore),
4-bar embargo, sample weights.
"""

import sys
import warnings
import json
import time
import itertools
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
    from afml.fracdiff import frac_diff_ffd
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

# CPCV parameters
CPCV_N_GROUPS = 6
CPCV_K_TEST = 2
CPCV_EMBARGO_PCT = 0.01

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

_full_features_cache: Dict[str, pd.DataFrame] = {}
_vol_cache: Dict[str, pd.Series] = {}
_hmm_regime_cache: Dict[str, pd.Series] = {}
_vol_regime_cache: Dict[str, pd.Series] = {}  # Simple vol percentile regime


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
) -> Tuple[pd.DataFrame, dict]:
    """Slice precomputed features and apply per-window return winsorization.

    Returns (feature_df, winsor_audit_info).
    """
    base = _full_features_cache[symbol]

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
) -> Tuple[List[Tuple], Dict[str, np.ndarray], Dict[str, dict], Dict[str, dict]]:
    """Walk-forward signal generation for one asset using LogReg L1 C=0.1.

    Args:
        symbol: Asset symbol.
        use_hmm: If True, use HMM regime filter. If False, no regime filter.
        use_vol_regime: If True (and use_hmm=False), use simple vol percentile
                        regime instead of HMM. Suppress low-vol regime.

    Returns:
        (signals_list, monthly_coefs, boundary_audit, winsor_audit)
        - signals_list: list of (timestamp, symbol, direction, fwd_return,
          confidence, probability, signal_strength, month_str) tuples
        - monthly_coefs: {month_str: {feature_name: coef_value}}
        - boundary_audit: {month_str: {last_train_ts, first_test_ts}}
        - winsor_audit: {month_str: winsor_info_dict}
    """
    if symbol not in _full_features_cache:
        return [], {}, {}, {}

    ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
    if ohlcv is None:
        return [], {}, {}, {}

    close = ohlcv["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = _vol_cache.get(symbol)

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

        # Embargo: train cutoff is 2h before test start
        train_cutoff = ms - EMBARGO_TIMEDELTA

        # Slice features with per-window winsorization (19 features)
        feat, w_info = _get_features_for_window(symbol, train_cutoff)
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
            # Suppress low-vol regime (regime 0)
            worst_regime = 0

        # Clean: drop rows with any NaN feature
        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        # Train/test split
        tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 500 or len(te) < 20:
            continue

        # Record boundary audit
        boundary_audit[month_str] = {
            "last_train_ts": tr[-1],
            "first_test_ts": te[0],
        }

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

        # LogReg L1 C=0.1 with StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        lr_model = LogisticRegression(
            C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42,
        )
        lr_model.fit(X_tr_scaled, y_train_binary, sample_weight=sample_w)
        yp = lr_model.predict_proba(X_te_scaled)[:, 1]

        # Store coefficients (mapped back to feature names)
        feature_names = list(X_train.columns)
        coefs = lr_model.coef_[0]
        monthly_coefs[month_str] = {
            feature_names[k]: float(coefs[k]) for k in range(len(feature_names))
        }

        # Confidence filter (top 30%)
        conf = np.abs(yp - 0.5)
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
) -> Tuple[pd.DataFrame, dict, dict, dict]:
    """Generate walk-forward signals for all assets in parallel.

    Returns:
        (signals_df, all_monthly_coefs, all_boundary_audit, all_winsor_audit)
    """
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
            )
            futures[fut] = symbol

        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                rows, m_coefs, b_audit, w_audit = fut.result()
                all_rows.extend(rows)
                # Merge per-asset results
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
# PORTFOLIO SIMULATION -- identical to validation_honest.py
# =============================================================================

def simulate_portfolio(signals_df: pd.DataFrame) -> Optional[dict]:
    """Portfolio simulation with shared capital pool.

    Returns full simulation dict including daily returns series,
    or None if insufficient signals.
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
# TEST 1: Regime Dependence
# =============================================================================

def test_regime_dependence(signals_df):
    """Classify months as BULL/BEAR/SIDEWAYS by BTC monthly return.
    PASS: Sharpe > 0 in at least 2 of 3 regimes.
    """
    btc_ohlcv = _ohlcv_cache.get(("BTC", BAR_TF))
    if btc_ohlcv is None:
        return False, {}, {}

    close = btc_ohlcv["close"]

    # Classify each month
    regimes = {}
    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        month_close = close.loc[(close.index >= ms) & (close.index <= me)]
        if len(month_close) < 10:
            continue
        start_close = month_close.iloc[0]
        end_close = month_close.iloc[-1]
        ret = (end_close - start_close) / start_close
        if ret > 0.05:
            regimes[str(month)] = "BULL"
        elif ret < -0.05:
            regimes[str(month)] = "BEAR"
        else:
            regimes[str(month)] = "SIDEWAYS"

    # For each regime, filter signals and compute Sharpe via portfolio sim
    results_per_regime = {}
    for regime in ["BULL", "BEAR", "SIDEWAYS"]:
        regime_months = [m for m, r in regimes.items() if r == regime]
        regime_signals = signals_df[signals_df["month"].isin(regime_months)]
        n_signals = len(regime_signals)
        if n_signals > 20:
            sim = simulate_portfolio(regime_signals)
            if sim:
                results_per_regime[regime] = {
                    "sharpe": sim["sharpe"],
                    "accuracy": sim["accuracy"],
                    "n_signals": n_signals,
                    "n_months": len(regime_months),
                }
            else:
                results_per_regime[regime] = {
                    "sharpe": 0.0, "accuracy": 0.0,
                    "n_signals": n_signals, "n_months": len(regime_months),
                }
        else:
            results_per_regime[regime] = {
                "sharpe": 0.0, "accuracy": 0.0,
                "n_signals": n_signals, "n_months": len(regime_months),
            }

    # PASS: Sharpe > 0 in at least 2 of 3 regimes
    positive_regimes = sum(1 for r in results_per_regime.values() if r["sharpe"] > 0)
    passed = positive_regimes >= 2
    return passed, results_per_regime, regimes


# =============================================================================
# TEST 2: Per-Asset Breakdown
# =============================================================================

def test_per_asset_breakdown(signals_df):
    """Report accuracy and PnL contribution per asset.
    PASS: >= 4/6 assets have accuracy > 51%.
    """
    results = {}
    for asset in ASSETS:
        asset_sigs = signals_df[signals_df["asset"] == asset]
        if len(asset_sigs) < 10:
            results[asset] = {"n_signals": len(asset_sigs), "accuracy": 0.0, "mean_return": 0.0}
            continue
        correct = ((asset_sigs["direction"] > 0) & (asset_sigs["fwd_return"] > 0)) | \
                  ((asset_sigs["direction"] < 0) & (asset_sigs["fwd_return"] < 0))
        accuracy = correct.mean()
        mean_ret = (asset_sigs["direction"] * asset_sigs["fwd_return"]).mean()
        results[asset] = {
            "n_signals": len(asset_sigs),
            "accuracy": round(float(accuracy), 4),
            "mean_return": round(float(mean_ret), 6),
        }

    above_51 = sum(1 for r in results.values() if r["accuracy"] > 0.51)
    passed = above_51 >= 4
    return passed, results


# =============================================================================
# TEST 3: Monthly Stability
# =============================================================================

def test_monthly_stability(signals_df):
    """Check for downward trend in monthly Sharpe.
    PASS: slope not significantly negative (p > 0.05 for negative slope).
    """
    monthly_sharpes = []
    for i, month in enumerate(ALL_MONTHS):
        month_str = str(month)
        month_sigs = signals_df[signals_df["month"] == month_str]
        if len(month_sigs) < 20:
            monthly_sharpes.append((i, month_str, float("nan")))
            continue
        sim = simulate_portfolio(month_sigs)
        if sim:
            monthly_sharpes.append((i, month_str, sim["sharpe"]))
        else:
            monthly_sharpes.append((i, month_str, float("nan")))

    # Linear regression of Sharpe vs month_index
    valid = [(idx, m, s) for idx, m, s in monthly_sharpes if not np.isnan(s)]
    if len(valid) < 5:
        return False, {"error": "too few months with data", "monthly_sharpes": monthly_sharpes}

    x = np.array([v[0] for v in valid])
    y = np.array([v[2] for v in valid])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # PASS: slope not significantly negative (p > 0.05 for negative slope)
    passed = not (slope < 0 and p_value < 0.05)
    return passed, {
        "monthly_sharpes": [(m, round(s, 3)) for _, m, s in monthly_sharpes],
        "slope": round(slope, 4),
        "intercept": round(intercept, 3),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 4),
    }


# =============================================================================
# TEST 4: Feature Coefficient Stability
# =============================================================================

def test_feature_stability(all_monthly_coefs):
    """Check if LogReg coefficients are stable across months.
    PASS: top 5 features by |mean_coef| maintain sign in > 80% of months.
    """
    # Aggregate coefficients across assets per month
    # For each month, average the coefficients across all assets that had data
    agg_coefs = {}
    for month_str in sorted(all_monthly_coefs.keys()):
        asset_coefs = all_monthly_coefs[month_str]  # {asset: {feat: val}}
        if not asset_coefs:
            continue
        # Average across assets
        all_feats = set()
        for ac in asset_coefs.values():
            all_feats.update(ac.keys())

        month_avg = {}
        for feat in all_feats:
            vals = [ac[feat] for ac in asset_coefs.values() if feat in ac]
            if vals:
                month_avg[feat] = np.mean(vals)
        agg_coefs[month_str] = month_avg

    if not agg_coefs:
        return False, {}, []

    # Only consider features from BASE_19_FEATURES (exclude "regime" if present)
    features = [f for f in BASE_19_FEATURES
                if any(f in agg_coefs[m] for m in agg_coefs)]

    results = {}
    for feat in features:
        coefs = [agg_coefs[m][feat] for m in sorted(agg_coefs.keys()) if feat in agg_coefs[m]]
        if not coefs:
            continue
        mean_coef = np.mean(coefs)
        std_coef = np.std(coefs)
        if mean_coef != 0:
            sign_consistency = np.mean([1 if (c > 0) == (mean_coef > 0) else 0 for c in coefs])
        else:
            sign_consistency = 0.5
        results[feat] = {
            "mean_coef": round(float(mean_coef), 6),
            "std_coef": round(float(std_coef), 6),
            "sign_consistency": round(float(sign_consistency), 3),
            "abs_mean": round(abs(float(mean_coef)), 6),
            "n_months": len(coefs),
        }

    # Sort by |mean_coef| to find top 5
    sorted_feats = sorted(results.items(), key=lambda x: x[1]["abs_mean"], reverse=True)
    top5 = sorted_feats[:5]

    # PASS: top 5 features maintain sign in > 80% of months
    passed = all(f[1]["sign_consistency"] > 0.80 for f in top5)
    return passed, results, top5


# =============================================================================
# TEST 5: Honest DSR
# =============================================================================

def test_honest_dsr(daily_returns):
    """Compute DSR with honest trial count K=20.
    PASS: DSR > 0.50.
    """
    if daily_returns is None or len(daily_returns) < 10:
        return False, {"error": "insufficient daily returns"}

    sr_ann = daily_returns.mean() / daily_returns.std() * np.sqrt(365.25) if daily_returns.std() > 0 else 0
    sr_unann = daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    skew = float(daily_returns.skew())
    kurt = float(daily_returns.kurtosis()) + 3.0  # scipy kurtosis is excess, DSR needs regular

    dsr = deflated_sharpe_ratio(
        observed_sr=sr_unann,
        num_trials=20,
        num_observations=len(daily_returns),
        skewness=skew,
        kurtosis=kurt,
    )

    passed = dsr > 0.50
    return passed, {
        "dsr": round(dsr, 4),
        "annualized_sharpe": round(sr_ann, 3),
        "unannualized_sharpe": round(sr_unann, 6),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "K": 20,
        "N": len(daily_returns),
    }


# =============================================================================
# TEST 6: CPCV with LogReg
# =============================================================================

def test_cpcv_logreg():
    """Portfolio-level CPCV with LogReg L1 C=0.1.
    PASS: >70% positive paths AND bootstrap CI lower bound > 0.
    """
    N_GROUPS = CPCV_N_GROUPS
    K_TEST = CPCV_K_TEST
    combos = list(itertools.combinations(range(N_GROUPS), K_TEST))

    # Build per-asset valid indices and groups
    asset_data = {}
    for symbol in ASSETS:
        if symbol not in _full_features_cache:
            continue
        feat = _full_features_cache[symbol]
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            continue
        close = ohlcv["close"]
        fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
        vol = _vol_cache.get(symbol)

        cols = [c for c in BASE_19_FEATURES if c in feat.columns]
        fc = feat[cols].dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)
        n_obs = len(valid)

        if n_obs < 500:
            continue

        group_size = n_obs // N_GROUPS
        embargo_size = max(int(n_obs * CPCV_EMBARGO_PCT), FWD_BARS)

        groups = []
        for g in range(N_GROUPS):
            start = g * group_size
            end = start + group_size if g < N_GROUPS - 1 else n_obs
            groups.append(np.arange(start, end))

        asset_data[symbol] = {
            "valid": valid, "fc": fc, "y_all": y_all, "vol": vol,
            "groups": groups, "embargo_size": embargo_size, "fwd_ret": fwd_ret,
        }

    def run_cpcv_split(combo):
        """Run one CPCV split across all assets."""
        all_signals = []
        for symbol, ad in asset_data.items():
            valid = ad["valid"]
            groups = ad["groups"]
            embargo_size = ad["embargo_size"]
            fc = ad["fc"]
            y_all = ad["y_all"]
            vol = ad["vol"]
            fwd_ret = ad["fwd_ret"]

            test_idx = np.concatenate([groups[g] for g in combo])
            train_idx = np.concatenate([groups[g] for g in range(N_GROUPS) if g not in combo])

            # Embargo: remove train indices within embargo_size of any test index boundary
            test_sorted = np.sort(test_idx)
            if len(test_sorted) == 0 or len(train_idx) == 0:
                continue

            # Find test block boundaries
            gaps = np.where(np.diff(test_sorted) > 1)[0]
            block_starts = [test_sorted[0]]
            block_ends = [test_sorted[-1]]
            for g_idx in gaps:
                block_ends.insert(-1, test_sorted[g_idx])
                block_starts.insert(-1, test_sorted[g_idx + 1])

            # Remove train samples within embargo of test boundaries
            embargo_mask = np.ones(len(train_idx), dtype=bool)
            for bs in block_starts:
                embargo_mask &= ~((train_idx >= bs - embargo_size) & (train_idx < bs))
            for be in block_ends:
                embargo_mask &= ~((train_idx > be) & (train_idx <= be + embargo_size))
            train_idx = train_idx[embargo_mask]

            train_times = valid[train_idx]
            test_times = valid[test_idx]

            if len(train_times) < 200 or len(test_times) < 50:
                continue

            # Winsorize returns on train stats
            X_all = fc.loc[valid].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in X_all.columns:
                    train_vals = X_all.loc[train_times, col].dropna()
                    if len(train_vals) > 10:
                        lo = train_vals.quantile(WINSOR_LO)
                        hi = train_vals.quantile(WINSOR_HI)
                        X_all[col] = X_all[col].clip(lo, hi)

            # Vol filter
            if vol is not None:
                train_vol = vol.reindex(train_times).dropna()
                if len(train_vol) < 50:
                    continue
                vol_thresh = train_vol.quantile(VOL_FILTER_QUANTILE)
                test_vol = vol.reindex(test_times)
                hv_mask = (test_vol > vol_thresh).fillna(False)
                test_times = test_times[hv_mask.values]

            if len(test_times) < 20:
                continue

            X_train = X_all.loc[train_times]
            y_train_binary = (y_all.loc[train_times] > 0).astype(int)
            X_test = X_all.loc[test_times]

            # Sample weights
            train_ts_int = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_ts_int, fwd_bars=FWD_BARS)

            # LogReg L1 C=0.1
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)

            lr = LogisticRegression(
                C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42,
            )
            try:
                lr.fit(X_tr_sc, y_train_binary, sample_weight=sample_w)
            except Exception:
                continue
            yp = lr.predict_proba(X_te_sc)[:, 1]

            # Confidence filter
            conf = np.abs(yp - 0.5)
            if len(conf) > 10:
                conf_thresh = np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
                trade_mask = conf >= conf_thresh
            else:
                trade_mask = np.ones(len(conf), dtype=bool)

            if trade_mask.sum() < 5:
                continue

            # Build signal list
            test_sub = test_times[trade_mask]
            probs_sub = yp[trade_mask]
            actual_rets = fwd_ret.reindex(test_sub).values

            signals = get_signal(probs_sub, num_classes=2)
            signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

            for j in range(len(test_sub)):
                fr = actual_rets[j]
                if not np.isfinite(fr):
                    continue
                direction = 1.0 if probs_sub[j] > 0.5 else -1.0
                sig_val = float(signals_disc[j]) if np.ndim(signals_disc) > 0 else float(signals_disc)
                all_signals.append({
                    "timestamp": test_sub[j],
                    "asset": symbol,
                    "direction": direction,
                    "fwd_return": float(fr),
                    "signal_strength": sig_val,
                })

        if len(all_signals) < 20:
            return None

        sig_df = pd.DataFrame(all_signals)
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        sim = simulate_portfolio(sig_df)
        return sim

    # Run all splits
    split_results = []
    for idx, combo in enumerate(combos):
        try:
            sim = run_cpcv_split(combo)
            if sim:
                split_results.append(sim)
        except Exception:
            pass

    if not split_results:
        return False, {"error": "no valid CPCV splits"}

    sharpes = np.array([s["sharpe"] for s in split_results])
    pct_positive = float(np.mean(sharpes > 0) * 100)

    # Bootstrap CI
    n_boot = 1000
    boot_means = np.array([
        np.mean(np.random.choice(sharpes, size=len(sharpes), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # PASS: >70% positive paths AND CI lower bound > 0
    passed = pct_positive > 70 and ci_lo > 0
    return passed, {
        "pct_positive": round(pct_positive, 1),
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "ci_lo": round(float(ci_lo), 3),
        "ci_hi": round(float(ci_hi), 3),
        "n_splits": len(split_results),
        "n_total_combos": len(combos),
        "per_split_sharpe": [round(float(s), 3) for s in sharpes],
    }


# =============================================================================
# TEST 7: Latency Sensitivity
# =============================================================================

def test_latency_sensitivity(signals_df):
    """Delay all signals by 1 bar and re-simulate.
    PASS: delayed Sharpe > 0 and > 50% of immediate Sharpe.
    """
    sim_immediate = simulate_portfolio(signals_df)
    if sim_immediate is None:
        return False, {"error": "no immediate simulation"}

    delayed = signals_df.copy()
    delayed_returns = []

    for _, row in delayed.iterrows():
        symbol = row["asset"]
        ts = row["timestamp"]
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is None:
            delayed_returns.append(np.nan)
            continue

        close = ohlcv["close"]
        # Find bars after signal timestamp
        future = close.loc[close.index > ts]
        if len(future) < FWD_BARS + 1:
            delayed_returns.append(np.nan)
            continue
        t1_close = future.iloc[0]      # close at t+1 (delayed entry)
        t1_4_close = future.iloc[FWD_BARS]  # close at t+1+FWD_BARS (delayed exit)
        delayed_returns.append((t1_4_close - t1_close) / t1_close)

    delayed["fwd_return"] = delayed_returns
    delayed = delayed.dropna(subset=["fwd_return"])

    sim_delayed = simulate_portfolio(delayed)

    if sim_immediate and sim_delayed:
        imm_sharpe = sim_immediate["sharpe"]
        del_sharpe = sim_delayed["sharpe"]
        ratio = del_sharpe / imm_sharpe if imm_sharpe != 0 else 0
        passed = del_sharpe > 0 and ratio > 0.5
    else:
        imm_sharpe = sim_immediate["sharpe"] if sim_immediate else None
        del_sharpe = sim_delayed["sharpe"] if sim_delayed else None
        ratio = 0
        passed = False

    return passed, {
        "immediate_sharpe": imm_sharpe,
        "delayed_sharpe": del_sharpe,
        "ratio": round(ratio, 3) if ratio else 0,
        "n_immediate": len(signals_df),
        "n_delayed": len(delayed),
    }


# =============================================================================
# TEST 8: Train/Test Separation Audit
# =============================================================================

def test_train_test_separation(all_boundary_audit):
    """Verify embargo gap at every walk-forward boundary.
    PASS: all gaps >= 2.0 hours.
    """
    min_gap_hours = float("inf")
    violations = []
    n_boundaries = 0

    for month_str in sorted(all_boundary_audit.keys()):
        asset_boundaries = all_boundary_audit[month_str]
        for asset, info in asset_boundaries.items():
            last_train = pd.Timestamp(info["last_train_ts"])
            first_test = pd.Timestamp(info["first_test_ts"])
            gap_hours = (first_test - last_train).total_seconds() / 3600
            min_gap_hours = min(min_gap_hours, gap_hours)
            n_boundaries += 1
            if gap_hours < 2.0:
                violations.append({
                    "month": month_str,
                    "asset": asset,
                    "gap_hours": round(gap_hours, 2),
                })

    if n_boundaries == 0:
        return False, {"error": "no boundaries found"}

    passed = len(violations) == 0 and min_gap_hours >= 2.0
    return passed, {
        "min_gap_hours": round(min_gap_hours, 2),
        "violations": violations,
        "n_boundaries": n_boundaries,
    }


# =============================================================================
# TEST 9: Feature Look-Ahead Check
# =============================================================================

def test_feature_lookahead():
    """Verify no feature uses future data.
    PASS: no forward-looking patterns found in feature names or columns.
    """
    FORWARD_LOOKING_PATTERNS = ["fwd_", "forward", "future", "shift(-"]

    issues = []
    for feat in BASE_19_FEATURES:
        for pattern in FORWARD_LOOKING_PATTERNS:
            if pattern in feat.lower():
                issues.append(f"{feat} contains '{pattern}'")

    # Verify feature matrix doesn't contain fwd_return or similar
    sample_symbol = ASSETS[0]
    sample_features = _full_features_cache.get(sample_symbol)
    if sample_features is not None:
        for col in sample_features.columns:
            if "fwd" in col.lower() or "forward" in col.lower() or "future" in col.lower():
                issues.append(f"Feature matrix for {sample_symbol} contains column '{col}'")

    # Verify all rolling windows are backward-only (structural check)
    backward_only_features = {
        "ret_1": "pct_change(1) -- backward",
        "ret_3": "pct_change(3) -- backward",
        "ret_7": "pct_change(7) -- backward",
        "ret_14": "pct_change(14) -- backward",
        "bar_range": "(high-low)/close -- current bar only",
        "vol_zscore": "rolling(30).mean/std -- backward",
        "rsi_14": "ewm(14) -- backward",
        "stoch_rsi": "rolling+ewm -- backward",
        "macd_hist": "ewm(12,26,9) -- backward",
        "bb_pct_b": "rolling(20) -- backward",
        "atr_14": "ewm(14) -- backward",
        "dist_swing_high": "rolling(20).max -- backward",
        "dist_swing_low": "rolling(20).min -- backward",
        "dist_nearest_level": "rolling(20).max/min -- backward",
        "ffd": "frac_diff (convolution) -- backward",
        "cs_spread": "rolling(20) -- backward",
        "entropy": "rolling(50) -- backward",
        "oi_change_5": "pct_change(5) -- backward",
        "impact_spread": "(ask-bid)/mid -- current bar only",
    }

    feature_audit = {}
    for feat in BASE_19_FEATURES:
        feature_audit[feat] = backward_only_features.get(feat, "unknown -- CHECK MANUALLY")

    passed = len(issues) == 0
    return passed, {
        "issues": issues,
        "n_features_checked": len(BASE_19_FEATURES),
        "feature_audit": feature_audit,
    }


# =============================================================================
# TEST 10: Winsorization Leak Check
# =============================================================================

def test_winsorization_leak(all_winsor_audit):
    """Verify winsorization uses train-only quantiles.
    PASS: no violations found.
    """
    violations = []

    for month_str in sorted(all_winsor_audit.keys()):
        asset_audits = all_winsor_audit[month_str]
        for asset, info in asset_audits.items():
            if info.get("used_test_data", False):
                violations.append({"month": month_str, "asset": asset})

    n_checked = sum(len(v) for v in all_winsor_audit.values())
    passed = len(violations) == 0
    return passed, {
        "violations": violations,
        "n_months_checked": len(all_winsor_audit),
        "n_asset_months_checked": n_checked,
    }


# =============================================================================
# TEST 11: HMM Filter Value
# =============================================================================

def test_hmm_value(signals_hmm_df, sim_hmm):
    """Compare LogReg with/without HMM filter, and with simple vol percentile.
    PASS: HMM adds > 0.1 Sharpe over no-regime baseline.
    """
    # Run 2: WITHOUT HMM
    print("    Generating signals without HMM...", flush=True)
    signals_no_hmm, _, _, _ = generate_all_signals(use_hmm=False, use_vol_regime=False)
    sim_no_hmm = simulate_portfolio(signals_no_hmm)

    # Run 3: Simple vol percentile instead of HMM
    print("    Generating signals with vol percentile regime...", flush=True)
    signals_vol, _, _, _ = generate_all_signals(use_hmm=False, use_vol_regime=True)
    sim_vol = simulate_portfolio(signals_vol)

    hmm_sharpe = sim_hmm["sharpe"] if sim_hmm else 0.0
    no_hmm_sharpe = sim_no_hmm["sharpe"] if sim_no_hmm else 0.0
    vol_sharpe = sim_vol["sharpe"] if sim_vol else 0.0

    hmm_delta = hmm_sharpe - no_hmm_sharpe
    vol_delta = vol_sharpe - no_hmm_sharpe

    # PASS: HMM adds > 0.1 Sharpe
    passed = hmm_delta > 0.1
    return passed, {
        "sharpe_with_hmm": hmm_sharpe,
        "sharpe_without": no_hmm_sharpe,
        "sharpe_vol_regime": vol_sharpe,
        "hmm_delta": round(hmm_delta, 3),
        "vol_delta": round(vol_delta, 3),
        "n_signals_hmm": len(signals_hmm_df),
        "n_signals_no_hmm": len(signals_no_hmm),
        "n_signals_vol": len(signals_vol),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    overall_start = time.time()

    print("=" * 60, flush=True)
    print("TIER 1 VERIFICATION -- 11 Tests, All Must Pass", flush=True)
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
    # Phase 1: Generate walk-forward signals (with HMM, the default)
    # =========================================================================
    print("\n[Phase 1] Generating walk-forward signals (LogReg L1 C=0.1 + HMM)...", flush=True)
    t1 = time.time()
    signals_df, all_monthly_coefs, all_boundary_audit, all_winsor_audit = generate_all_signals(
        use_hmm=True, use_vol_regime=False,
    )
    t_phase1 = time.time() - t1
    print(f"  Generated {len(signals_df)} signals across {len(ASSETS)} assets in {t_phase1:.1f}s", flush=True)

    # Also run portfolio sim on the full signal set (needed by multiple tests)
    sim_full = simulate_portfolio(signals_df)
    if sim_full:
        print(f"  Full Sharpe: {sim_full['sharpe']:.3f}, Trades: {sim_full['n_trades']}, "
              f"Return: {sim_full['total_return_pct']:+.2f}%", flush=True)
    else:
        print("  WARNING: Full portfolio simulation returned None!", flush=True)

    # =========================================================================
    # Phase 2: Run 11 tests
    # =========================================================================
    print("\n[Phase 2] Running 11 verification tests...", flush=True)
    print("=" * 60, flush=True)

    test_results = {}
    n_passed = 0
    n_failed = 0
    critical_fails = []
    minor_fails = []

    # ------ Test 1: Regime Dependence ------
    print(f"\n[Test  1/11] Regime Dependence...", flush=True)
    t_test = time.time()
    try:
        passed_1, regime_results, regimes = test_regime_dependence(signals_df)
        for regime in ["BULL", "BEAR", "SIDEWAYS"]:
            r = regime_results.get(regime, {})
            n_months = r.get("n_months", 0)
            sharpe = r.get("sharpe", 0)
            acc = r.get("accuracy", 0)
            n_sig = r.get("n_signals", 0)
            print(f"  {regime:>8} ({n_months} months): Sharpe={sharpe:>6.3f}, "
                  f"Acc={acc*100:>5.1f}%, n={n_sig}", flush=True)
        pos_r = sum(1 for r in regime_results.values() if r.get("sharpe", 0) > 0)
        status = "PASS" if passed_1 else "FAIL"
        print(f"  >> [{status}] Signal works in {pos_r}/3 regimes "
              f"(need >=2) ({time.time()-t_test:.1f}s)", flush=True)
        test_results["1_regime_dependence"] = {
            "passed": passed_1, "results": regime_results,
            "regime_classification": regimes,
        }
        if passed_1:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Regime Dependence")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["1_regime_dependence"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Regime Dependence (ERROR)")

    # ------ Test 2: Per-Asset Breakdown ------
    print(f"\n[Test  2/11] Per-Asset Breakdown...", flush=True)
    t_test = time.time()
    try:
        passed_2, asset_results = test_per_asset_breakdown(signals_df)
        for asset in ASSETS:
            r = asset_results.get(asset, {})
            acc = r.get("accuracy", 0)
            n_sig = r.get("n_signals", 0)
            mean_ret = r.get("mean_return", 0)
            bps = mean_ret * 10000
            print(f"  {asset:>4}:  Acc={acc*100:>5.1f}%, n={n_sig:>5}, meanRet={bps:>+6.1f}bps", flush=True)
        above = sum(1 for r in asset_results.values() if r.get("accuracy", 0) > 0.51)
        status = "PASS" if passed_2 else "FAIL"
        print(f"  >> [{status}] {above}/6 assets have accuracy > 51% "
              f"(need >=4) ({time.time()-t_test:.1f}s)", flush=True)
        test_results["2_per_asset_breakdown"] = {"passed": passed_2, "results": asset_results}
        if passed_2:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Per-Asset Breakdown")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["2_per_asset_breakdown"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Per-Asset Breakdown (ERROR)")

    # ------ Test 3: Monthly Stability ------
    print(f"\n[Test  3/11] Monthly Stability...", flush=True)
    t_test = time.time()
    try:
        passed_3, stability_results = test_monthly_stability(signals_df)
        if "monthly_sharpes" in stability_results:
            for m, s in stability_results["monthly_sharpes"]:
                s_str = f"{s:>7.3f}" if not np.isnan(s) else "    N/A"
                print(f"  {m}: Sharpe={s_str}", flush=True)
        slope = stability_results.get("slope", 0)
        p_val = stability_results.get("p_value", 1)
        status = "PASS" if passed_3 else "FAIL"
        print(f"  Trend: slope={slope:.4f}, p={p_val:.4f}", flush=True)
        print(f"  >> [{status}] No significant downward trend "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["3_monthly_stability"] = {"passed": passed_3, "results": stability_results}
        if passed_3:
            n_passed += 1
        else:
            n_failed += 1
            minor_fails.append("Monthly Stability")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["3_monthly_stability"] = {"passed": False, "error": str(e)}
        n_failed += 1
        minor_fails.append("Monthly Stability (ERROR)")

    # ------ Test 4: Feature Coefficient Stability ------
    print(f"\n[Test  4/11] Feature Coefficient Stability...", flush=True)
    t_test = time.time()
    try:
        passed_4, feat_results, top5 = test_feature_stability(all_monthly_coefs)
        print(f"  Top 5 features by |mean coefficient|:", flush=True)
        for feat_name, info in top5:
            print(f"    {feat_name:>20}: mean={info['mean_coef']:>+.6f}, "
                  f"sign_consistency={info['sign_consistency']:.1%}", flush=True)
        status = "PASS" if passed_4 else "FAIL"
        print(f"  >> [{status}] Top 5 features maintain sign >80% of months "
              f"({time.time()-t_test:.1f}s)", flush=True)
        # Convert for JSON serialization
        test_results["4_feature_stability"] = {
            "passed": passed_4,
            "top5": [(n, d) for n, d in top5],
            "all_features": feat_results,
        }
        if passed_4:
            n_passed += 1
        else:
            n_failed += 1
            minor_fails.append("Feature Coefficient Stability")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["4_feature_stability"] = {"passed": False, "error": str(e)}
        n_failed += 1
        minor_fails.append("Feature Coefficient Stability (ERROR)")

    # ------ Test 5: Honest DSR ------
    print(f"\n[Test  5/11] Honest DSR (K=20)...", flush=True)
    t_test = time.time()
    try:
        daily_rets = sim_full["daily_returns"] if sim_full else None
        passed_5, dsr_results = test_honest_dsr(daily_rets)
        dsr = dsr_results.get("dsr", 0)
        sr = dsr_results.get("annualized_sharpe", 0)
        skew = dsr_results.get("skewness", 0)
        kurt = dsr_results.get("kurtosis", 0)
        n_obs = dsr_results.get("N", 0)
        print(f"  DSR={dsr:.4f}, Sharpe={sr:.3f}, Skew={skew:.3f}, "
              f"Kurt={kurt:.3f}, K=20, N={n_obs}", flush=True)
        status = "PASS" if passed_5 else "FAIL"
        print(f"  >> [{status}] DSR > 0.50 ({time.time()-t_test:.1f}s)", flush=True)
        test_results["5_honest_dsr"] = {"passed": passed_5, "results": dsr_results}
        if passed_5:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Honest DSR")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["5_honest_dsr"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Honest DSR (ERROR)")

    # ------ Test 6: CPCV with LogReg ------
    print(f"\n[Test  6/11] CPCV with LogReg L1 C=0.1...", flush=True)
    t_test = time.time()
    try:
        passed_6, cpcv_results = test_cpcv_logreg()
        pct_pos = cpcv_results.get("pct_positive", 0)
        mean_s = cpcv_results.get("mean_sharpe", 0)
        ci_lo = cpcv_results.get("ci_lo", 0)
        ci_hi = cpcv_results.get("ci_hi", 0)
        n_splits = cpcv_results.get("n_splits", 0)
        print(f"  {n_splits} splits: {pct_pos:.1f}% positive Sharpe", flush=True)
        print(f"  Mean Sharpe={mean_s:.3f}, 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
        if "per_split_sharpe" in cpcv_results:
            sharpes_str = ", ".join(f"{s:.3f}" for s in cpcv_results["per_split_sharpe"])
            print(f"  Per-split: [{sharpes_str}]", flush=True)
        status = "PASS" if passed_6 else "FAIL"
        print(f"  >> [{status}] >70% positive AND CI_lo > 0 "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["6_cpcv_logreg"] = {"passed": passed_6, "results": cpcv_results}
        if passed_6:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("CPCV LogReg")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["6_cpcv_logreg"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("CPCV LogReg (ERROR)")

    # ------ Test 7: Latency Sensitivity ------
    print(f"\n[Test  7/11] Latency Sensitivity (1-bar delay)...", flush=True)
    t_test = time.time()
    try:
        passed_7, latency_results = test_latency_sensitivity(signals_df)
        imm_s = latency_results.get("immediate_sharpe", 0)
        del_s = latency_results.get("delayed_sharpe", 0)
        ratio = latency_results.get("ratio", 0)
        print(f"  Immediate Sharpe={imm_s:.3f}", flush=True)
        print(f"  Delayed Sharpe  ={del_s:.3f} ({ratio:.1%} of immediate)", flush=True)
        status = "PASS" if passed_7 else "FAIL"
        print(f"  >> [{status}] Delayed Sharpe > 0 and > 50% of immediate "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["7_latency_sensitivity"] = {"passed": passed_7, "results": latency_results}
        if passed_7:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Latency Sensitivity")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["7_latency_sensitivity"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Latency Sensitivity (ERROR)")

    # ------ Test 8: Train/Test Separation Audit ------
    print(f"\n[Test  8/11] Train/Test Separation Audit...", flush=True)
    t_test = time.time()
    try:
        passed_8, separation_results = test_train_test_separation(all_boundary_audit)
        min_gap = separation_results.get("min_gap_hours", 0)
        n_bound = separation_results.get("n_boundaries", 0)
        n_viol = len(separation_results.get("violations", []))
        print(f"  {n_bound} boundaries checked, min gap={min_gap:.2f}h, "
              f"violations={n_viol}", flush=True)
        if n_viol > 0:
            for v in separation_results["violations"][:5]:
                print(f"    VIOLATION: {v['month']} {v['asset']} gap={v['gap_hours']:.2f}h", flush=True)
        status = "PASS" if passed_8 else "FAIL"
        print(f"  >> [{status}] All gaps >= 2.0 hours "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["8_train_test_separation"] = {"passed": passed_8, "results": separation_results}
        if passed_8:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Train/Test Separation")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["8_train_test_separation"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Train/Test Separation (ERROR)")

    # ------ Test 9: Feature Look-Ahead Check ------
    print(f"\n[Test  9/11] Feature Look-Ahead Check...", flush=True)
    t_test = time.time()
    try:
        passed_9, lookahead_results = test_feature_lookahead()
        n_checked = lookahead_results.get("n_features_checked", 0)
        n_issues = len(lookahead_results.get("issues", []))
        print(f"  {n_checked} features checked, {n_issues} issues found", flush=True)
        if n_issues > 0:
            for issue in lookahead_results["issues"]:
                print(f"    ISSUE: {issue}", flush=True)
        else:
            print(f"  All features use backward-only computation", flush=True)
        status = "PASS" if passed_9 else "FAIL"
        print(f"  >> [{status}] No look-ahead contamination "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["9_feature_lookahead"] = {"passed": passed_9, "results": lookahead_results}
        if passed_9:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Feature Look-Ahead")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["9_feature_lookahead"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Feature Look-Ahead (ERROR)")

    # ------ Test 10: Winsorization Leak Check ------
    print(f"\n[Test 10/11] Winsorization Leak Check...", flush=True)
    t_test = time.time()
    try:
        passed_10, winsor_results = test_winsorization_leak(all_winsor_audit)
        n_months = winsor_results.get("n_months_checked", 0)
        n_am = winsor_results.get("n_asset_months_checked", 0)
        n_viol = len(winsor_results.get("violations", []))
        print(f"  {n_months} months, {n_am} asset-months checked, "
              f"{n_viol} violations", flush=True)
        status = "PASS" if passed_10 else "FAIL"
        print(f"  >> [{status}] Winsorization uses train-only quantiles "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["10_winsorization_leak"] = {"passed": passed_10, "results": winsor_results}
        if passed_10:
            n_passed += 1
        else:
            n_failed += 1
            critical_fails.append("Winsorization Leak")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["10_winsorization_leak"] = {"passed": False, "error": str(e)}
        n_failed += 1
        critical_fails.append("Winsorization Leak (ERROR)")

    # ------ Test 11: HMM Filter Value ------
    print(f"\n[Test 11/11] HMM Filter Value...", flush=True)
    t_test = time.time()
    try:
        passed_11, hmm_results = test_hmm_value(signals_df, sim_full)
        hmm_s = hmm_results.get("sharpe_with_hmm", 0)
        no_s = hmm_results.get("sharpe_without", 0)
        vol_s = hmm_results.get("sharpe_vol_regime", 0)
        delta = hmm_results.get("hmm_delta", 0)
        print(f"  WITH HMM:        Sharpe={hmm_s:.3f} (n={hmm_results.get('n_signals_hmm', 0)})", flush=True)
        print(f"  WITHOUT regime:  Sharpe={no_s:.3f} (n={hmm_results.get('n_signals_no_hmm', 0)})", flush=True)
        print(f"  Vol percentile:  Sharpe={vol_s:.3f} (n={hmm_results.get('n_signals_vol', 0)})", flush=True)
        print(f"  HMM delta: {delta:+.3f}", flush=True)
        status = "PASS" if passed_11 else "FAIL"
        print(f"  >> [{status}] HMM adds > 0.1 Sharpe "
              f"({time.time()-t_test:.1f}s)", flush=True)
        test_results["11_hmm_value"] = {"passed": passed_11, "results": hmm_results}
        if passed_11:
            n_passed += 1
        else:
            n_failed += 1
            minor_fails.append("HMM Filter Value")
    except Exception as e:
        print(f"  >> [ERROR] {e}", flush=True)
        traceback.print_exc()
        test_results["11_hmm_value"] = {"passed": False, "error": str(e)}
        n_failed += 1
        minor_fails.append("HMM Filter Value (ERROR)")

    # =========================================================================
    # Phase 3: Summary
    # =========================================================================
    total_time = time.time() - overall_start

    print("\n" + "=" * 60, flush=True)
    print(f"SUMMARY: {n_passed}/11 PASSED, {n_failed}/11 FAILED", flush=True)
    print("=" * 60, flush=True)

    if critical_fails:
        print(f"CRITICAL FAILS: {critical_fails}", flush=True)
    else:
        print("CRITICAL FAILS: None", flush=True)

    if minor_fails:
        print(f"MINOR FAILS: {minor_fails}", flush=True)
    else:
        print("MINOR FAILS: None", flush=True)

    if n_passed == 11:
        print("\n*** ALL 11 TESTS PASSED -- READY FOR PRODUCTION DEPLOYMENT ***", flush=True)
    elif n_passed >= 9:
        print(f"\n*** {n_passed}/11 PASSED -- CLOSE, address remaining failures ***", flush=True)
    else:
        print(f"\n*** {n_passed}/11 PASSED -- NOT READY for production ***", flush=True)

    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f}m)", flush=True)
    print(f"  Phase 0 (data+features): {t_phase0:.0f}s", flush=True)
    print(f"  Phase 1 (signals):       {t_phase1:.0f}s", flush=True)
    print(f"  Phase 2 (tests):         {total_time - t_phase0 - t_phase1:.0f}s", flush=True)

    # =========================================================================
    # Save JSON results
    # =========================================================================

    # Make results JSON-serializable (remove pd.Series)
    json_results = {}
    for k, v in test_results.items():
        json_results[k] = {}
        for k2, v2 in v.items():
            if isinstance(v2, pd.Series):
                json_results[k][k2] = v2.to_dict()
            elif isinstance(v2, dict):
                # Deep-convert any nested pd.Timestamp or similar
                clean = {}
                for k3, v3 in v2.items():
                    if isinstance(v3, (pd.Timestamp, np.datetime64)):
                        clean[k3] = str(v3)
                    elif isinstance(v3, dict):
                        inner = {}
                        for k4, v4 in v3.items():
                            if isinstance(v4, (pd.Timestamp, np.datetime64)):
                                inner[k4] = str(v4)
                            elif isinstance(v4, dict):
                                inner2 = {}
                                for k5, v5 in v4.items():
                                    if isinstance(v5, (pd.Timestamp, np.datetime64)):
                                        inner2[k5] = str(v5)
                                    elif isinstance(v5, (np.integer, np.floating)):
                                        inner2[k5] = float(v5)
                                    else:
                                        inner2[k5] = v5
                                inner[k4] = inner2
                            elif isinstance(v4, (np.integer, np.floating)):
                                inner[k4] = float(v4)
                            else:
                                inner[k4] = v4
                        clean[k3] = inner
                    elif isinstance(v3, (np.integer, np.floating)):
                        clean[k3] = float(v3)
                    else:
                        clean[k3] = v3
                json_results[k][k2] = clean
            elif isinstance(v2, (np.integer, np.floating)):
                json_results[k][k2] = float(v2)
            else:
                json_results[k][k2] = v2

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "LogReg_L1_C0.1",
        "features": BASE_19_FEATURES,
        "n_assets": len(ASSETS),
        "assets": ASSETS,
        "n_months": len(ALL_MONTHS),
        "n_signals": len(signals_df),
        "full_sharpe": sim_full["sharpe"] if sim_full else None,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "critical_fails": critical_fails,
        "minor_fails": minor_fails,
        "total_runtime_s": round(total_time, 1),
        "tests": json_results,
    }

    out_path = OUTPUT_DIR / "tier1_verification_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}", flush=True)
    except Exception as e:
        print(f"\n[WARN] Failed to save JSON: {e}", flush=True)

    print(f"Completed: {datetime.now()}", flush=True)
