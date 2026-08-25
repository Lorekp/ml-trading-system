"""
Direction V4 -- Tier 2 Improvements (FAST version).

Tests ALL Tier 2 improvements against the Tier 1 combined baseline:
  1. Tier 1 Combined (baseline)      -- 20 features, prob sizing, walk-forward
  2. +Regime (HMM 3-state)           -- add regime feature + suppress worst regime
  3. +MetaLabel                      -- meta-labeler filters losing trades
  4. +Ensemble (XGB+LGB)             -- average XGB & LightGBM probabilities
  5. +CrossAsset (BTC features)      -- BTC return/vol for altcoins
  6. ALL Combined                    -- all Tier 2 improvements together

Architecture: Precompute-once, slice-per-window.
  - ALL features (FFD, entropy, CS spread, technicals, HL derivatives) computed
    ONCE on the full series per asset in Phase 0.
  - Walk-forward loop only does: slice cached features, winsorize per window,
    train XGBoost (2s), predict.
  - Target: <5 minutes total (vs 2+ hours original).
"""

import sys
import warnings
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

# Import AFML modules with inline fallback
try:
    from afml.fracdiff import frac_diff_ffd
    from afml.derivatives import funding_features, oi_features, impact_features
    from afml.features import corwin_schultz_spread, rolling_entropy
    from afml.bet_sizing import get_signal, discretize_signal, kelly_size
    from afml.evaluation import deflated_sharpe_ratio
    from afml.importance import mean_decrease_impurity
    AFML_IMPORTED = True
except ImportError:
    AFML_IMPORTED = False
    print("[WARN] Could not import afml modules - using inline implementations", flush=True)

# Optional imports for Tier 2
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[WARN] hmmlearn not available - using volatility percentile regime fallback", flush=True)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] lightgbm not available - ensemble will use XGB only", flush=True)

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

# =============================================================================
# INLINE AFML IMPLEMENTATIONS (fallback)
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

    def mean_decrease_impurity(model, feature_names):
        imp = model.feature_importances_
        result = pd.DataFrame({"mean": imp}, index=feature_names)
        result["mean"] /= result["mean"].sum()
        result["std"] = 0.0
        return result.sort_values("mean", ascending=False)


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
HL_CSV_DIR = Path("C:/Claude/ml-trading-system/v3-results")
OUTPUT_DIR = Path("C:/Claude/ml-trading-system/v4-results")
RESULTS_FILE = OUTPUT_DIR / "tier2_results.json"

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT",
}

# Hardcoded FFD d* values (stable, skip ADF search)
FFD_D_STARS = {
    "BTC": 0.20, "ETH": 0.10, "XRP": 0.30,
    "BNB": 0.10, "SUI": 0.20, "LINK": 0.10,
}

BAR_TF = "30min"
FWD_BARS = 4
ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")

STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001
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

LGB_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    min_child_weight=50,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="binary",
    metric="binary_logloss",
    verbose=-1,
    random_state=42,
    n_jobs=1,
)

WINSOR_LO = 0.01
WINSOR_HI = 0.99
FFD_THRESHOLD = 1e-4
IMPORTANCE_BOTTOM_N = 3
DSR_NUM_TRIALS = 4095
SIGNAL_STEP_SIZE = 0.1
MAX_SIGNAL_MULT = 2.0

# HMM regime parameters
HMM_N_STATES = 3


# =============================================================================
# DATA LOADING (module-level cache for pickling in ProcessPoolExecutor)
# =============================================================================

_ohlcv_cache: dict = {}
_hl_cache: dict = {}


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

    keep_cols = [c for c in ["funding", "open_interest", "mid_px", "impact_bid_px", "impact_ask_px"]
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
# PRECOMPUTED FEATURE CACHES (the key speed optimization)
# =============================================================================

_ffd_cache: dict = {}
_full_features_cache: dict = {}
_btc_cross_cache: dict = {}
_vol_cache: dict = {}
_hmm_regime_cache: dict = {}
_extra_oi_cache: dict = {}


def precompute_ffd(symbol: str, close: pd.Series, d_star: float):
    """Compute FFD ONCE for the full series and cache it."""
    if symbol in _ffd_cache:
        return
    try:
        log_close = np.log(close).to_frame("close")
        ffd_result = frac_diff_ffd(log_close, d_star, threshold=FFD_THRESHOLD)
        if len(ffd_result) > 0:
            _ffd_cache[symbol] = ffd_result.iloc[:, 0]
            print(f"    [FFD] {symbol}: precomputed d*={d_star}, {len(ffd_result)} values", flush=True)
    except Exception as e:
        print(f"    [FFD] {symbol}: failed ({e})", flush=True)


def precompute_all_features():
    """Precompute ALL features for all assets on the FULL series.

    This is the key optimization: compute expensive features (FFD, entropy,
    CS spread, technicals, HL derivatives) ONCE, then slice per walk-forward
    window. Only winsorization of returns is deferred to per-window.

    Populates:
      _full_features_cache[symbol] - DataFrame with all raw features (un-winsorized returns)
      _btc_cross_cache - dict with 'btc_ret_1' and 'btc_vol_z' Series
      _vol_cache[symbol] - ewm volatility Series for vol filtering
      _hmm_regime_cache[symbol] - full-sample HMM regime Series
      _extra_oi_cache[symbol] - DataFrame with extra OI features
    """
    # --- BTC cross-asset features (precompute once) ---
    try:
        btc_ohlcv = _ohlcv_cache.get(("BTC", BAR_TF))
        if btc_ohlcv is not None:
            btc_close = btc_ohlcv["close"]
            btc_volume = btc_ohlcv["volume"]
            btc_ret_1 = btc_close.pct_change(1).shift(1)
            btc_vol_mean = btc_volume.rolling(30).mean()
            btc_vol_std = btc_volume.rolling(30).std().replace(0, np.nan)
            btc_vol_z = ((btc_volume - btc_vol_mean) / btc_vol_std).shift(1)
            _btc_cross_cache["btc_ret_1"] = btc_ret_1
            _btc_cross_cache["btc_vol_z"] = btc_vol_z
            print("    [BTC Cross] Precomputed BTC cross-asset features", flush=True)
    except Exception as e:
        print(f"    [BTC Cross] Failed: {e}", flush=True)

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

        # --- Raw returns (UN-winsorized -- winsorization applied per window) ---
        for lb in [1, 3, 7, 14]:
            f[f"ret_{lb}"] = close.pct_change(lb)

        # --- Technical indicators (computed once, no train_end dependency) ---
        f["bar_range"] = (high - low) / close
        f["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
        f["rsi_14"] = rsi(close, 14)
        f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
        f["macd_hist"] = macd(close)["macd_hist"]
        f["bb_pct_b"] = bollinger_pct_b(close, 20)
        f["atr_14"] = atr(high, low, close, 14)

        sw = swing_levels(high, low, lookback=20)
        f["dist_swing_high"] = sw["dist_swing_high"]
        f["dist_swing_low"] = sw["dist_swing_low"]
        f["dist_nearest_level"] = sw["dist_nearest_level"]

        # --- FFD (from precomputed cache) ---
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

        # --- Corwin-Schultz spread (computed once) ---
        try:
            cs = corwin_schultz_spread(high, low, window=20)
            f["cs_spread"] = cs.reindex(f.index)
        except Exception:
            pass

        # --- Rolling entropy (computed once -- the big win, 3.5s instead of 3.5s*14) ---
        try:
            ent = rolling_entropy(close, window=50, n_bins=5)
            f["entropy"] = ent.reindex(f.index)
        except Exception:
            pass

        # --- HL Derivatives features (computed once) ---
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
                    oi = oi_features(hl_data["open_interest"], close.reindex(hl_data.index, method="ffill"))
                    if "oi_change_5" in oi.columns:
                        f["oi_change_5"] = oi["oi_change_5"].reindex(f.index)
                except Exception:
                    pass

            if "impact_bid_px" in hl_data.columns and "impact_ask_px" in hl_data.columns:
                try:
                    mid_col = "mid_px" if "mid_px" in hl_data.columns else None
                    if mid_col:
                        imp = impact_features(hl_data["impact_bid_px"], hl_data["impact_ask_px"], hl_data[mid_col])
                    else:
                        mid = (hl_data["impact_bid_px"] + hl_data["impact_ask_px"]) / 2
                        imp = impact_features(hl_data["impact_bid_px"], hl_data["impact_ask_px"], mid)
                    if "impact_spread" in imp.columns:
                        f["impact_spread"] = imp["impact_spread"].reindex(f.index)
                except Exception:
                    pass

        # --- BTC cross-asset features (for non-BTC) ---
        if symbol != "BTC" and "btc_ret_1" in _btc_cross_cache:
            f["btc_ret_1"] = _btc_cross_cache["btc_ret_1"].reindex(f.index, method="ffill")
            f["btc_vol_z"] = _btc_cross_cache["btc_vol_z"].reindex(f.index, method="ffill")

        _full_features_cache[symbol] = f

        # --- Precompute volatility for vol filter ---
        _vol_cache[symbol] = close.pct_change().dropna().ewm(span=50).std()

        # --- Precompute HMM regime on full data ---
        try:
            vol_series = _vol_cache[symbol]
            vol_z_full = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
            regime_full = _fit_hmm_regime_full(close, vol_series, vol_z_full)
            if len(regime_full) > 0:
                _hmm_regime_cache[symbol] = regime_full
        except Exception:
            pass

        # --- Precompute extra OI features ---
        if hl_data is not None and "open_interest" in hl_data.columns:
            try:
                oi_series = hl_data["open_interest"]
                extra_oi = pd.DataFrame(index=f.index)
                oi_change_1h = oi_series.pct_change(2)
                extra_oi["oi_change_1h"] = oi_change_1h.reindex(f.index)
                oi_change_4h = oi_series.pct_change(8)
                extra_oi["oi_change_4h"] = oi_change_4h.reindex(f.index)
                price_change = close.pct_change(2)
                oi_change_aligned = oi_change_1h.reindex(f.index)
                oi_price_div = (np.sign(oi_change_aligned) != np.sign(price_change)).astype(float)
                extra_oi["oi_price_divergence"] = oi_price_div
                _extra_oi_cache[symbol] = extra_oi
            except Exception:
                pass

        elapsed = time.time() - t_sym
        print(f"    [Features] {symbol}: {len(f.columns)} features, {len(f)} rows ({elapsed:.1f}s)", flush=True)


def _fit_hmm_regime_full(close, vol, volume_z, n_states=HMM_N_STATES):
    """Fit HMM on (return_10bar, vol, volume_z) on full data. Returns regime labels."""
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


# =============================================================================
# FAST WALK-FORWARD: slice from cache + winsorize per window
# =============================================================================

def _get_features_for_window(symbol, train_end, variant):
    """Slice precomputed features and apply per-window winsorization.

    This replaces build_expanded_features_with_tier2() in the inner loop.
    Cost: microseconds (slice + clip) instead of ~14 seconds (full recompute).
    """
    base_features = _full_features_cache[symbol]

    # Determine which columns to include based on variant
    use_crossasset = variant in ("crossasset", "all_combined")
    use_extra_oi = variant in ("all_combined",)

    # Start with base columns (always present)
    cols_to_use = [c for c in base_features.columns]

    # Remove cross-asset columns if not needed
    if not use_crossasset:
        cols_to_use = [c for c in cols_to_use if c not in ("btc_ret_1", "btc_vol_z")]

    # Add extra OI columns if needed
    if use_extra_oi and symbol in _extra_oi_cache:
        extra_oi = _extra_oi_cache[symbol]
        # We merge extra OI into the slice below
    else:
        extra_oi = None

    f = base_features[cols_to_use].copy()

    # Merge extra OI features
    if extra_oi is not None:
        for col in extra_oi.columns:
            f[col] = extra_oi[col]

    # --- Per-window winsorization of returns (the only train-dependent computation) ---
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
# SIGNAL GENERATION -- TIER 2 VARIANTS (module-level for ProcessPoolExecutor)
# =============================================================================

def _generate_signals_tier2_for_asset(
    symbol: str,
    d_star: float,
    variant: str,
) -> list:
    """Generate walk-forward signals for a single asset with Tier 2 improvements.

    FAST version: slices precomputed features instead of recomputing per window.

    variant is one of:
      "tier1"         -- Tier 1 combined baseline (expanded features, prob sizing)
      "regime"        -- add HMM regime feature
      "metalabel"     -- filter via meta-labeler
      "ensemble"      -- XGB+LGB ensemble
      "crossasset"    -- add BTC features for altcoins
      "all_combined"  -- all Tier 2 improvements together

    Returns list of tuples:
      (timestamp, symbol, direction, fwd_return, confidence, probability, signal_disc)
    """
    if symbol not in _full_features_cache:
        return []

    ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
    if ohlcv is None:
        return []

    close = ohlcv["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = _vol_cache.get(symbol)

    # Determine which Tier 2 additions are active for this variant
    use_regime = variant in ("regime", "all_combined")
    use_metalabel = variant in ("metalabel", "all_combined")
    use_ensemble = variant in ("ensemble", "all_combined")

    # Precomputed regime for this symbol (if available)
    full_regime = _hmm_regime_cache.get(symbol)

    rows = []

    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        # Slice precomputed features + winsorize per window (microseconds)
        feat = _get_features_for_window(symbol, ms, variant)

        # T2.1: Add regime column from precomputed full-sample regime
        worst_regime = None
        if use_regime and full_regime is not None:
            regime_col = full_regime.reindex(feat.index)
            feat["regime"] = regime_col.fillna(1).astype(int)

            # Find worst regime using only train data (before ms)
            train_regimes = full_regime[full_regime.index < ms]
            if len(train_regimes) > 50:
                train_fwd = fwd_ret.reindex(train_regimes.index).dropna()
                common = train_regimes.index.intersection(train_fwd.index)
                if len(common) > 50:
                    regime_returns = pd.DataFrame({
                        "regime": train_regimes.loc[common],
                        "fwd_ret": train_fwd.loc[common],
                    }).groupby("regime")["fwd_ret"].mean()
                    worst_regime = regime_returns.idxmin()

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        tr = valid[valid < ms]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 500 or len(te) < 20:
            continue

        # Vol filter (top 50% vol bars)
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
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]
        y_test_actual = y_all.loc[te]

        # --- Train primary XGBoost model ---
        model_xgb = xgb.XGBClassifier(**XGB_PARAMS)
        model_xgb.fit(X_train, y_train)
        yp_xgb = model_xgb.predict_proba(X_test)[:, 1]

        # --- T2.3: Ensemble with LightGBM ---
        if use_ensemble and LGB_AVAILABLE:
            try:
                model_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
                model_lgb.fit(X_train, y_train)
                yp_lgb = model_lgb.predict_proba(X_test)[:, 1]

                if CB_AVAILABLE:
                    try:
                        model_cb = cb.CatBoostClassifier(
                            max_depth=3,
                            learning_rate=0.05,
                            iterations=100,
                            verbose=0,
                            random_seed=42,
                        )
                        model_cb.fit(X_train, y_train)
                        yp_cb = model_cb.predict_proba(X_test)[:, 1]
                        yp = (yp_xgb + yp_lgb + yp_cb) / 3.0
                    except Exception:
                        yp = (yp_xgb + yp_lgb) / 2.0
                else:
                    yp = (yp_xgb + yp_lgb) / 2.0
            except Exception:
                yp = yp_xgb
        else:
            yp = yp_xgb

        # --- T2.2: Meta-labeling ---
        meta_mask = np.ones(len(yp), dtype=bool)
        if use_metalabel:
            try:
                # Generate primary predictions on training data for meta-label training
                yp_train = model_xgb.predict_proba(X_train)[:, 1]
                train_directions = np.where(yp_train > 0.5, 1.0, -1.0)
                train_fwd_rets = y_all.loc[tr].values
                train_trade_rets = train_directions * train_fwd_rets - SLIPPAGE

                # Meta-labels: was the primary prediction profitable?
                meta_labels = (train_trade_rets > 0).astype(int)

                # Meta-features: original features + primary probability
                meta_X_train = X_train.copy()
                meta_X_train["primary_prob"] = yp_train

                meta_X_test = X_test.copy()
                meta_X_test["primary_prob"] = yp

                # Train meta-labeler
                meta_model = xgb.XGBClassifier(
                    max_depth=2,
                    learning_rate=0.05,
                    n_estimators=50,
                    min_child_weight=50,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=43,
                )
                meta_model.fit(meta_X_train, meta_labels)
                meta_proba = meta_model.predict_proba(meta_X_test)[:, 1]

                # Only trade when meta-labeler P(correct) > 0.5
                meta_mask = meta_proba > 0.5
            except Exception:
                meta_mask = np.ones(len(yp), dtype=bool)

        # --- Confidence filter (top 30%) ---
        conf = np.abs(yp - 0.5)
        if len(conf) > 10:
            conf_threshold = np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
            conf_mask = conf >= conf_threshold
        else:
            conf_mask = np.ones(len(conf), dtype=bool)

        # --- Regime suppression (suppress worst regime trades) ---
        regime_mask = np.ones(len(yp), dtype=bool)
        if use_regime and worst_regime is not None and "regime" in X_test.columns:
            regime_mask = X_test["regime"].values != worst_regime

        # Combined filter: confidence AND meta-label AND regime
        combined_mask = conf_mask & meta_mask & regime_mask

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


def generate_signals_tier2(
    variant: str,
    variant_name: str,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Generate walk-forward signals for all assets in parallel.

    With precomputed features, each asset's walk-forward loop is just
    XGBoost training + slicing, so it's safe and fast to parallelize.
    """
    print(f"\n  [{variant_name}] Generating walk-forward signals (threaded, {max_workers} workers)...", flush=True)
    t_start = time.time()

    all_rows = []

    # ThreadPoolExecutor: shares memory (caches accessible), XGBoost releases GIL
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for symbol in ASSETS:
            d_star = FFD_D_STARS.get(symbol, 0.4)
            fut = pool.submit(_generate_signals_tier2_for_asset, symbol, d_star, variant)
            futures[fut] = symbol

        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                print(f"    {symbol}: {len(rows)} signals", flush=True)
            except Exception as e:
                print(f"    {symbol}: ERROR - {e}", flush=True)
                traceback.print_exc()

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "asset", "direction", "fwd_return",
        "confidence", "probability", "signal_strength",
    ])
    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    elapsed = time.time() - t_start
    print(f"  [{variant_name}] {len(df)} total signals in {elapsed:.0f}s", flush=True)
    return df


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def simulate_portfolio(
    signals_df: pd.DataFrame,
    sizing_mode: str = "prob",
    base_pct: float = BASE_SIZING_PCT,
) -> Optional[dict]:
    """Portfolio simulation with shared capital pool.

    Default sizing_mode is 'prob' (probability-based) since Tier 1 combined uses it.
    """
    if len(signals_df) < 20:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = sub["signal_strength"].values.astype(np.float64) if "signal_strength" in sub.columns else None

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

    use_prob = (sizing_mode == "prob" and signal_strengths is not None)

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
            raw_size = equity * base_pct * sig_abs * size_mult
            raw_size = min(raw_size, equity * base_pct * MAX_SIGNAL_MULT * size_mult)
        else:
            raw_size = equity * base_pct * size_mult

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

        current_exposure = min(current_exposure + raw_size * 0.1, equity * MAX_EXPOSURE)
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
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(sim: Optional[dict], label: str = "") -> Optional[dict]:
    """Compute comprehensive quant metrics from simulation result."""
    if sim is None or sim["n_trades"] < 20:
        return None

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
    equity = sim["equity"]
    n_trades = sim["n_trades"]
    timestamps = sim["timestamps"]

    t0 = pd.Timestamp(timestamps[active][0])
    t1 = pd.Timestamp(timestamps[active][-1])
    days = max((t1 - t0).total_seconds() / 86400, 1)
    years = days / 365.25
    trades_per_day = n_trades / days

    total_ret = sim["final_equity"] / STARTING_CAPITAL - 1
    cagr = (sim["final_equity"] / STARTING_CAPITAL) ** (1.0 / years) - 1 if years > 0 else 0

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

    monthly = ts_series.resample("ME").sum()
    months_pos = int((monthly > 0).sum())
    months_total = int((monthly != 0).sum())

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    kelly_f_star = kelly_size(win_rate_t, avg_win, avg_loss) if avg_loss != 0 else 0.0
    half_kelly = kelly_f_star / 2.0

    sharpe_unannn = mean_d / std_d if std_d > 0 else 0.0
    num_obs = len(daily_nonzero)
    skewness = float(daily_nonzero.skew()) if len(daily_nonzero) > 3 else 0.0
    kurt = float(daily_nonzero.kurtosis()) + 3.0 if len(daily_nonzero) > 4 else 3.0
    dsr = deflated_sharpe_ratio(
        observed_sr=sharpe_unannn,
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
        "months_positive": months_pos,
        "months_total": months_total,
        "mean_daily_pnl": round(mean_d, 2),
        "halted": sim["halted"],
        "kelly_f_star": round(kelly_f_star, 4),
        "half_kelly": round(half_kelly, 4),
        "dsr": round(dsr, 4),
        "dsr_num_trials": DSR_NUM_TRIALS,
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

    print("=" * 110, flush=True)
    print("DIRECTION V4 -- TIER 2 IMPROVEMENTS (FAST)", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD: {MAX_DD*100:.0f}%, Delever from: {DELEVER_START*100:.1f}%", flush=True)
    print(f"Base sizing: {BASE_SIZING_PCT*100:.0f}%/trade, MaxExposure: {MAX_EXPOSURE*100:.0f}%, "
          f"MaxPerAsset: {MAX_PER_ASSET*100:.0f}%", flush=True)
    print(f"Slippage: {SLIPPAGE*10000:.0f}bp | Bars: {BAR_TF}, Fwd: {FWD_BARS} bars ({FWD_BARS*30}m)", flush=True)
    print(f"Assets: {ASSETS}", flush=True)
    print(f"FFD d* (hardcoded): {FFD_D_STARS}", flush=True)
    print(f"Walk-forward months: {ALL_MONTHS[0]}-{ALL_MONTHS[-1]} ({len(ALL_MONTHS)} months)", flush=True)
    print(f"DSR trials: {DSR_NUM_TRIALS}", flush=True)
    print(f"HMM available: {HMM_AVAILABLE}, LightGBM: {LGB_AVAILABLE}, CatBoost: {CB_AVAILABLE}", flush=True)
    print("=" * 110, flush=True)

    all_results: dict = {
        "timestamp": datetime.now().isoformat(),
        "version": "tier2_fast",
        "config": {
            "starting_capital": STARTING_CAPITAL,
            "max_dd": MAX_DD,
            "slippage": SLIPPAGE,
            "max_exposure": MAX_EXPOSURE,
            "max_per_asset": MAX_PER_ASSET,
            "base_sizing_pct": BASE_SIZING_PCT,
            "bar_tf": BAR_TF,
            "fwd_bars": FWD_BARS,
            "assets": ASSETS,
            "months": [str(m) for m in ALL_MONTHS],
            "ffd_d_stars": FFD_D_STARS,
            "hmm_available": HMM_AVAILABLE,
            "lgb_available": LGB_AVAILABLE,
            "cb_available": CB_AVAILABLE,
        },
    }

    # =====================================================================
    # PHASE 0: Load all data + precompute ALL features upfront
    # =====================================================================
    print("\n[0/6] Loading all data upfront...", flush=True)
    t0 = time.time()
    preload_all_data()
    print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)

    # Precompute FFD for ALL assets
    print("  Precomputing FFD per asset...", flush=True)
    t_ffd = time.time()
    for symbol in ASSETS:
        ohlcv = _ohlcv_cache.get((symbol, BAR_TF))
        if ohlcv is not None:
            d_star = FFD_D_STARS.get(symbol, 0.4)
            precompute_ffd(symbol, ohlcv["close"], d_star)
    print(f"  FFD precomputed in {time.time()-t_ffd:.1f}s", flush=True)

    # Precompute ALL features (entropy, CS spread, technicals, HL derivs, BTC cross, HMM)
    print("  Precomputing ALL features for all assets...", flush=True)
    t_feat = time.time()
    precompute_all_features()
    print(f"  All features precomputed in {time.time()-t_feat:.1f}s", flush=True)

    all_results["hl_data_available"] = {s: (_hl_cache.get(s) is not None) for s in ASSETS}

    # =====================================================================
    # Define the 6 variants to test
    # =====================================================================
    VARIANTS = [
        ("tier1",        "Tier1 Combined"),
        ("regime",       "+Regime(HMM)"),
        ("metalabel",    "+MetaLabel"),
        ("ensemble",     "+Ensemble(XGB+LGB)"),
        ("crossasset",   "+CrossAsset(BTC)"),
        ("all_combined", "ALL Combined"),
    ]

    variant_signals = {}
    variant_metrics = {}

    for idx, (variant_key, variant_name) in enumerate(VARIANTS):
        phase_num = idx + 1
        print(f"\n{'=' * 80}", flush=True)
        print(f"[{phase_num}/6] {variant_name}", flush=True)
        print(f"{'=' * 80}", flush=True)

        t_var = time.time()

        signals = generate_signals_tier2(variant_key, variant_name)
        variant_signals[variant_key] = signals

        sim = simulate_portfolio(signals, sizing_mode="prob")
        metrics = compute_metrics(sim, variant_name)
        variant_metrics[variant_key] = metrics

        if metrics:
            print(f"  Sharpe: {metrics['sharpe']:.3f}, "
                  f"Return: {metrics['total_return_pct']:+.2f}%, "
                  f"MaxDD: {metrics['max_dd_pct']:.2f}%, "
                  f"WR: {metrics['win_rate_trades']:.4f}, "
                  f"PF: {metrics['profit_factor']:.3f}, "
                  f"Kelly f*: {metrics['kelly_f_star']:.4f}, "
                  f"DSR: {metrics['dsr']:.4f}", flush=True)
        else:
            print("  [WARN] Insufficient data for metrics", flush=True)

        print(f"  Phase time: {time.time()-t_var:.0f}s", flush=True)

    # =====================================================================
    # COMPARISON TABLE
    # =====================================================================
    print(f"\n{'=' * 120}", flush=True)
    print("TIER 2 COMPARISON TABLE", flush=True)
    print(f"{'=' * 120}", flush=True)

    header = f"{'Metric':<22}"
    for _, vname in VARIANTS:
        header += f" {vname:>16}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    metrics_to_show = [
        ("Sharpe",            "sharpe",           ".3f"),
        ("Sortino",           "sortino",          ".3f"),
        ("Total Return %",    "total_return_pct",  "+.2f"),
        ("CAGR %",            "cagr_pct",         "+.2f"),
        ("MaxDD %",           "max_dd_pct",       ".2f"),
        ("Win Rate (trades)", "win_rate_trades",   ".4f"),
        ("Win Rate (days)",   "win_rate_days",     ".4f"),
        ("Profit Factor",     "profit_factor",    ".3f"),
        ("Trades",            "n_trades",         "d"),
        ("Trades/day",        "trades_per_day",   ".2f"),
        ("Months positive",   "months_positive",  "d"),
        ("DSR",               "dsr",              ".4f"),
        ("Kelly f*",          "kelly_f_star",     ".4f"),
        ("Half-Kelly",        "half_kelly",       ".4f"),
        ("Calmar",            "calmar",           ".3f"),
        ("Mean daily PnL",    "mean_daily_pnl",   ".2f"),
        ("Final Equity",      "final_equity",     ".2f"),
        ("Halted",            "halted",           ""),
    ]

    for label, key, fmt in metrics_to_show:
        row = f"{label:<22}"
        for vkey, _ in VARIANTS:
            m = variant_metrics.get(vkey)
            if m is None:
                row += f" {'N/A':>16}"
            elif key == "halted":
                row += f" {'HALT' if m[key] else 'OK':>16}"
            elif key == "months_positive":
                row += f" {str(m[key]) + '/' + str(m.get('months_total', '?')):>16}"
            else:
                val = m.get(key, float("nan"))
                row += f" {format(val, fmt):>16}"
        print(row, flush=True)

    # =====================================================================
    # DELTA vs TIER 1 BASELINE
    # =====================================================================
    baseline = variant_metrics.get("tier1")
    if baseline is not None:
        print(f"\n{'-' * 100}", flush=True)
        print("IMPROVEMENT vs TIER 1 BASELINE", flush=True)
        print(f"{'-' * 100}", flush=True)

        print(f"  {'Variant':<22} {'dSharpe':>10} {'dReturn%':>10} {'dMaxDD%':>10} {'dWR':>10} "
              f"{'dPF':>10} {'dDSR':>10}", flush=True)
        print(f"  {'-' * 82}", flush=True)

        for vkey, vname in VARIANTS:
            if vkey == "tier1":
                continue
            m = variant_metrics.get(vkey)
            if m is None:
                print(f"  {vname:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} "
                      f"{'N/A':>10} {'N/A':>10}", flush=True)
                continue

            d_sharpe = m["sharpe"] - baseline["sharpe"]
            d_ret = m["total_return_pct"] - baseline["total_return_pct"]
            d_maxdd = m["max_dd_pct"] - baseline["max_dd_pct"]
            d_wr = m["win_rate_trades"] - baseline["win_rate_trades"]
            d_pf = m["profit_factor"] - baseline["profit_factor"]
            d_dsr = m["dsr"] - baseline["dsr"]

            print(f"  {vname:<22} {d_sharpe:>+10.3f} {d_ret:>+10.2f} {d_maxdd:>+10.2f} "
                  f"{d_wr:>+10.4f} {d_pf:>+10.3f} {d_dsr:>+10.4f}", flush=True)

    # =====================================================================
    # WHICH IMPROVEMENT HELPED MOST
    # =====================================================================
    print(f"\n{'-' * 80}", flush=True)
    print("TIER 2 IMPROVEMENT RANKING (by Sharpe improvement vs baseline)", flush=True)
    print(f"{'-' * 80}", flush=True)

    tier2_only = []
    for vkey, vname in VARIANTS:
        if vkey == "tier1":
            continue
        m = variant_metrics.get(vkey)
        if m is not None and baseline is not None:
            tier2_only.append((vname, m["sharpe"] - baseline["sharpe"], m["sharpe"]))

    tier2_only.sort(key=lambda x: x[1], reverse=True)
    for rank, (vname, delta, absolute) in enumerate(tier2_only, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank}: {vname:<22} dSharpe={delta:+.3f}  (absolute={absolute:.3f}){marker}", flush=True)

    # =====================================================================
    # KELLY CRITERION
    # =====================================================================
    print(f"\n{'-' * 80}", flush=True)
    print("KELLY CRITERION (AFML Ch 10)", flush=True)
    print(f"{'-' * 80}", flush=True)

    print(f"  {'Variant':<22} {'f*':>8} {'f*/2':>8} {'Win Rate':>10} {'Recommendation':<30}", flush=True)
    print(f"  {'-' * 82}", flush=True)
    for vkey, vname in VARIANTS:
        m = variant_metrics.get(vkey)
        if m is None:
            print(f"  {vname:<22} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'Insufficient data':<30}", flush=True)
        else:
            f_star = m["kelly_f_star"]
            half_k = m["half_kelly"]
            wr = m["win_rate_trades"]
            if f_star <= 0:
                rec = "NO BET (negative edge)"
            elif half_k > BASE_SIZING_PCT * 3:
                rec = f"Half-Kelly: {half_k*100:.1f}% - aggressive"
            elif half_k < BASE_SIZING_PCT * 0.5:
                rec = f"Half-Kelly: {half_k*100:.1f}% - reduce size"
            else:
                rec = f"Half-Kelly: {half_k*100:.1f}% - reasonable"
            print(f"  {vname:<22} {f_star:>8.4f} {half_k:>8.4f} {wr:>10.4f} {rec:<30}", flush=True)

    # =====================================================================
    # DEFLATED SHARPE RATIO
    # =====================================================================
    print(f"\n{'-' * 80}", flush=True)
    print("DEFLATED SHARPE RATIO (AFML Ch 14, Bailey & LdP 2014)", flush=True)
    print(f"  K={DSR_NUM_TRIALS} trials tested", flush=True)
    print(f"{'-' * 80}", flush=True)

    print(f"  {'Variant':<22} {'Sharpe':>8} {'DSR':>8} {'T(days)':>8} {'Skew':>8} {'Kurt':>8} {'Pass?':>8}",
          flush=True)
    print(f"  {'-' * 70}", flush=True)
    for vkey, vname in VARIANTS:
        m = variant_metrics.get(vkey)
        if m is None:
            print(f"  {vname:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}",
                  flush=True)
        else:
            dsr_val = m["dsr"]
            sr_val = m["sharpe"]
            n_obs = m["dsr_num_obs"]
            skew = m["daily_skewness"]
            kurt_val = m["daily_kurtosis"]
            passed = "YES" if dsr_val > 0.95 else "NO"
            print(f"  {vname:<22} {sr_val:>8.3f} {dsr_val:>8.4f} {n_obs:>8d} {skew:>8.3f} "
                  f"{kurt_val:>8.3f} {passed:>8}", flush=True)

    # =====================================================================
    # SIGNAL COUNT SUMMARY
    # =====================================================================
    print(f"\n{'-' * 80}", flush=True)
    print("SIGNAL COUNTS", flush=True)
    print(f"{'-' * 80}", flush=True)

    for vkey, vname in VARIANTS:
        sigs = variant_signals.get(vkey)
        n_sigs = len(sigs) if sigs is not None else 0
        sim_trades = variant_metrics[vkey]["n_trades"] if variant_metrics.get(vkey) else 0
        filter_pct = (1 - sim_trades / n_sigs) * 100 if n_sigs > 0 else 0
        print(f"  {vname:<22} signals={n_sigs:>6}  traded={sim_trades:>6}  "
              f"filtered={filter_pct:.1f}%", flush=True)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print(f"\n{'=' * 80}", flush=True)
    print("Saving results...", flush=True)

    all_results["variants"] = {}
    for vkey, vname in VARIANTS:
        m = variant_metrics.get(vkey)
        all_results["variants"][vname] = m if m else {"error": "insufficient data"}

    all_results["signal_counts"] = {}
    for vkey, vname in VARIANTS:
        sigs = variant_signals.get(vkey)
        all_results["signal_counts"][vname] = len(sigs) if sigs is not None else 0

    # Improvement ranking
    all_results["tier2_ranking"] = []
    for vname, delta, absolute in tier2_only:
        all_results["tier2_ranking"].append({
            "variant": vname,
            "sharpe_delta": round(delta, 4),
            "sharpe_absolute": round(absolute, 4),
        })

    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved to: {RESULTS_FILE}", flush=True)
    except Exception as e:
        print(f"  [ERROR] Save failed: {e}", flush=True)
        traceback.print_exc()

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    total_elapsed = time.time() - overall_start

    print(f"\n{'=' * 110}", flush=True)
    print("TIER 2 EXPERIMENT COMPLETE", flush=True)
    print(f"Finished: {datetime.now()}", flush=True)
    print(f"Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"{'=' * 110}", flush=True)

    valid_variants = [(vname, variant_metrics[vkey])
                      for vkey, vname in VARIANTS
                      if variant_metrics.get(vkey) is not None]
    if valid_variants:
        best_name, best_m = max(valid_variants, key=lambda x: x[1]["sharpe"])
        print(f"\n  Best variant by Sharpe: {best_name} ({best_m['sharpe']:.3f})", flush=True)
        print(f"  Best variant details:", flush=True)
        print(f"    Return: {best_m['total_return_pct']:+.2f}%", flush=True)
        print(f"    MaxDD:  {best_m['max_dd_pct']:.2f}%", flush=True)
        print(f"    Kelly f*/2: {best_m['half_kelly']:.4f}", flush=True)
        print(f"    DSR:    {best_m['dsr']:.4f} ({'PASS' if best_m['dsr'] > 0.95 else 'FAIL'})", flush=True)

        if baseline is not None:
            delta_sharpe = best_m["sharpe"] - baseline["sharpe"]
            delta_ret = best_m["total_return_pct"] - baseline["total_return_pct"]
            print(f"\n  vs Tier 1 baseline:", flush=True)
            print(f"    Sharpe delta: {delta_sharpe:+.3f}", flush=True)
            print(f"    Return delta: {delta_ret:+.2f}pp", flush=True)

        if tier2_only:
            best_improvement = tier2_only[0]
            print(f"\n  Most helpful Tier 2 improvement: {best_improvement[0]} "
                  f"(Sharpe +{best_improvement[1]:.3f})", flush=True)
    else:
        print("\n  [WARN] No valid variant results. Check data availability.", flush=True)

    print(f"\n  Results file: {RESULTS_FILE}", flush=True)
    print(f"  Total time: {total_elapsed:.0f}s", flush=True)
