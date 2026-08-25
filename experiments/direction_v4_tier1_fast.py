"""
Direction V4 — Tier 1 Improvements (FAST version).

Optimized for speed vs direction_v4_tier1.py:
  1. Hardcoded FFD d* values (skip find_min_d ADF search entirely)
  2. Parallel signal generation across 6 assets via ProcessPoolExecutor
  3. Fast HL data loading from pre-extracted CSVs
  4. Module-level OHLCV cache (load each symbol ONCE)
  5. Simplified feature importance (XGBoost built-in only, no SFI per-feature models)
  6. Vectorized portfolio simulation where possible

Target: <3 minutes total (vs 15+ min original).

Same 4 configurations:
  1. BASELINE: v3 system (14 features, flat 2% sizing)
  2. EXPANDED FEATURES: 14 + 6 new features with importance validation
  3. PROBABILITY-BASED BET SIZING: AFML Ch 10 signal -> position size
  4. COMBINED: expanded features + probability sizing
"""

import sys
import warnings
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
RESULTS_FILE = OUTPUT_DIR / "tier1_results.json"

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT",
}

# OPTIMIZATION 1: Hardcoded FFD d* values from first run (stable, skip ADF search)
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

WINSOR_LO = 0.01
WINSOR_HI = 0.99
FFD_THRESHOLD = 1e-4
IMPORTANCE_BOTTOM_N = 3
DSR_NUM_TRIALS = 4095
SIGNAL_STEP_SIZE = 0.1
MAX_SIGNAL_MULT = 2.0


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
    """Load all OHLCV and HL data upfront into caches. Call before parallelization."""
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
# FEATURE BUILDING
# =============================================================================

def build_baseline_features(ohlcv: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """Build the v3 baseline feature set: 14 OHLCV + technical features."""
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    tm = ohlcv.index < train_end

    f = pd.DataFrame(index=ohlcv.index)

    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo = ret[tm].quantile(WINSOR_LO)
        hi = ret[tm].quantile(WINSOR_HI)
        f[f"ret_{lb}"] = ret.clip(lo, hi)

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

    return f


def build_expanded_features(
    ohlcv: pd.DataFrame,
    train_end: pd.Timestamp,
    hl_data: Optional[pd.DataFrame],
    d_star: float = 0.4,
) -> pd.DataFrame:
    """Build expanded feature set: 14 baseline + up to 6 new features."""
    f = build_baseline_features(ohlcv, train_end)
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    # FFD
    try:
        log_close = np.log(close).to_frame("close")
        ffd_result = frac_diff_ffd(log_close, d_star, threshold=FFD_THRESHOLD)
        if len(ffd_result) > 0:
            f["ffd"] = ffd_result.iloc[:, 0].reindex(f.index)
    except Exception:
        pass

    # Corwin-Schultz spread
    try:
        cs = corwin_schultz_spread(high, low, window=20)
        f["cs_spread"] = cs.reindex(f.index)
    except Exception:
        pass

    # Rolling entropy
    try:
        ent = rolling_entropy(close, window=50, n_bins=5)
        f["entropy"] = ent.reindex(f.index)
    except Exception:
        pass

    # HL Derivatives features
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

    return f


# =============================================================================
# FEATURE IMPORTANCE — FAST VERSION (MDI + MDA only, skip SFI per-feature models)
# =============================================================================

def run_fast_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list,
) -> dict:
    """Fast feature importance: MDI + MDA only (skip expensive SFI per-feature models).

    SFI trains N separate XGBoost models (one per feature) which is the slowest part.
    Instead, we use MDI (free from fitted model) + MDA (one permutation pass).
    Drop features bottom-N in BOTH methods.
    """
    n = len(X_train)
    split_idx = int(n * 0.8)

    X_fit = X_train.iloc[:split_idx]
    y_fit = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]

    if len(X_val) < 50 or len(X_fit) < 200:
        return {
            "keep_features": feature_names,
            "dropped_features": [],
            "rankings": {},
        }

    # MDI: fit model, extract Gini importances
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_fit, y_fit)

    mdi_imp = model.feature_importances_
    mdi_total = mdi_imp.sum()
    if mdi_total > 0:
        mdi_imp = mdi_imp / mdi_total
    mdi_rank = pd.Series(mdi_imp, index=feature_names).rank(ascending=True)

    # MDA: permutation importance on validation set
    from sklearn.metrics import log_loss
    base_proba = model.predict_proba(X_val)
    base_score = log_loss(y_val, base_proba)

    mda_imp = {}
    rng = np.random.RandomState(42)
    for feat in feature_names:
        X_perm = X_val.copy()
        X_perm[feat] = rng.permutation(X_perm[feat].values)
        perm_proba = model.predict_proba(X_perm)
        perm_score = log_loss(y_val, perm_proba)
        mda_imp[feat] = perm_score - base_score

    mda_rank = pd.Series(mda_imp).rank(ascending=True)

    # Drop if bottom-N in BOTH methods (relaxed from original triple)
    bottom_threshold = IMPORTANCE_BOTTOM_N
    dropped = []
    rankings = {}

    for feat in feature_names:
        r = {
            "mdi_rank": int(mdi_rank.get(feat, 0)),
            "mda_rank": int(mda_rank.get(feat, 0)),
            "sfi_rank": 0,  # Not computed in fast version
        }
        rankings[feat] = r

        if (r["mdi_rank"] <= bottom_threshold and
            r["mda_rank"] <= bottom_threshold):
            dropped.append(feat)

    keep = [f for f in feature_names if f not in dropped]

    return {
        "keep_features": keep,
        "dropped_features": dropped,
        "rankings": rankings,
    }


# =============================================================================
# SIGNAL GENERATION — PER-ASSET (module-level for ProcessPoolExecutor pickling)
# =============================================================================

def _generate_signals_for_asset(
    symbol: str,
    feature_mode: str,
    d_star: float,
    validated_features: Optional[list] = None,
) -> list:
    """Generate walk-forward signals for a single asset.

    Must be a module-level function for Windows ProcessPoolExecutor pickling.

    Args:
        symbol: Asset symbol (BTC, ETH, etc.)
        feature_mode: "baseline" or "expanded"
        d_star: FFD differentiation order
        validated_features: If not None, restrict expanded features to these columns.

    Returns:
        List of tuples: (timestamp, symbol, direction, fwd_return, confidence, probability, signal_disc)
    """
    try:
        ohlcv = load_ohlcv(symbol, BAR_TF)
    except FileNotFoundError:
        return []

    hl_data = None
    if feature_mode == "expanded":
        try:
            hl_data = load_hl_derivatives(symbol, BAR_TF)
        except Exception:
            pass

    close = ohlcv["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = close.pct_change().dropna().ewm(span=50).std()

    rows = []

    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")

        if feature_mode == "baseline":
            feat = build_baseline_features(ohlcv, ms)
        else:
            feat = build_expanded_features(ohlcv, ms, hl_data, d_star=d_star)
            if validated_features is not None:
                valid_cols = [c for c in validated_features if c in feat.columns]
                if valid_cols:
                    feat = feat[valid_cols]

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        tr = valid[valid < ms]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 500 or len(te) < 20:
            continue

        tv = vol.loc[tr].dropna()
        if len(tv) < 50:
            continue
        vt = tv.quantile(VOL_FILTER_QUANTILE)
        hv = (vol.reindex(te) > vt).fillna(False)
        te = te[hv.values]

        if len(te) < 10:
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(fc.loc[tr], (y_all.loc[tr] > 0).astype(int))

        yp = model.predict_proba(fc.loc[te])[:, 1]

        conf = np.abs(yp - 0.5)
        if len(conf) > 10:
            mask = conf >= np.percentile(conf, CONFIDENCE_FILTER_PERCENTILE)
        else:
            mask = np.ones(len(conf), dtype=bool)

        # Vectorized signal computation for all predictions in this month
        yp_masked = yp[mask]
        conf_masked = conf[mask]
        te_masked = te[mask]
        fwd_ret_masked = fwd_ret.reindex(te_masked)

        if len(yp_masked) == 0:
            continue

        directions = np.where(yp_masked > 0.5, 1.0, -1.0)
        signals = get_signal(yp_masked, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=SIGNAL_STEP_SIZE)

        for j in range(len(yp_masked)):
            idx = te_masked[j]
            fr = fwd_ret_masked.iloc[j] if hasattr(fwd_ret_masked, 'iloc') else fwd_ret_masked.loc[idx]
            if np.isfinite(fr):
                rows.append((
                    idx, symbol, directions[j],
                    float(fr),
                    float(conf_masked[j]),
                    float(yp_masked[j]),
                    float(signals_disc[j]) if np.ndim(signals_disc) > 0 else float(signals_disc),
                ))

    return rows


def generate_signals_parallel(
    feature_mode: str,
    variant_name: str,
    validated_features_per_asset: Optional[dict] = None,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Generate walk-forward signals for all assets in parallel.

    Uses ProcessPoolExecutor to run 6 assets simultaneously.

    Args:
        feature_mode: "baseline" or "expanded"
        variant_name: Label for logging.
        validated_features_per_asset: Dict of {symbol: [feature_names]} for expanded mode.
        max_workers: Number of parallel processes.

    Returns:
        DataFrame with columns: timestamp, asset, direction, fwd_return,
        confidence, probability, signal_strength.
    """
    print(f"\n  [{variant_name}] Generating walk-forward signals (parallel, {max_workers} workers)...", flush=True)
    t_start = time.time()

    all_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for symbol in ASSETS:
            d_star = FFD_D_STARS.get(symbol, 0.4)
            val_feats = None
            if validated_features_per_asset is not None:
                val_feats = validated_features_per_asset.get(symbol)

            future = executor.submit(
                _generate_signals_for_asset,
                symbol, feature_mode, d_star, val_feats,
            )
            futures[future] = symbol

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows = future.result()
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


def generate_signals_serial(
    feature_mode: str,
    variant_name: str,
    validated_features_per_asset: Optional[dict] = None,
) -> pd.DataFrame:
    """Serial fallback for signal generation (if multiprocessing fails)."""
    print(f"\n  [{variant_name}] Generating walk-forward signals (serial)...", flush=True)
    t_start = time.time()
    all_rows = []

    for symbol in ASSETS:
        sym_start = time.time()
        d_star = FFD_D_STARS.get(symbol, 0.4)
        val_feats = None
        if validated_features_per_asset is not None:
            val_feats = validated_features_per_asset.get(symbol)

        rows = _generate_signals_for_asset(symbol, feature_mode, d_star, val_feats)
        all_rows.extend(rows)
        print(f"    {symbol}: {len(rows)} signals ({time.time()-sym_start:.1f}s)", flush=True)

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
# PORTFOLIO SIMULATION (optimized with pre-sorted numpy arrays)
# =============================================================================

def simulate_portfolio(
    signals_df: pd.DataFrame,
    sizing_mode: str = "flat",
    base_pct: float = BASE_SIZING_PCT,
) -> Optional[dict]:
    """Portfolio simulation with shared capital pool.

    Optimized: pre-extracts all columns as numpy arrays, minimizes Python overhead.
    The per-trade loop is kept because of stateful equity/exposure tracking,
    but array access is via numpy (no pandas overhead per iteration).
    """
    if len(signals_df) < 20:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    # Pre-extract as numpy arrays (avoid pandas indexing in loop)
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
    print("DIRECTION V4 — TIER 1 IMPROVEMENTS (FAST)", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD: {MAX_DD*100:.0f}%, Delever from: {DELEVER_START*100:.1f}%", flush=True)
    print(f"Base sizing: {BASE_SIZING_PCT*100:.0f}%/trade, MaxExposure: {MAX_EXPOSURE*100:.0f}%, "
          f"MaxPerAsset: {MAX_PER_ASSET*100:.0f}%", flush=True)
    print(f"Slippage: {SLIPPAGE*10000:.0f}bp | Bars: {BAR_TF}, Fwd: {FWD_BARS} bars ({FWD_BARS*30}m)", flush=True)
    print(f"Assets: {ASSETS}", flush=True)
    print(f"FFD d* (hardcoded): {FFD_D_STARS}", flush=True)
    print(f"Walk-forward months: {ALL_MONTHS[0]}-{ALL_MONTHS[-1]} ({len(ALL_MONTHS)} months)", flush=True)
    print(f"DSR trials: {DSR_NUM_TRIALS}", flush=True)
    print("=" * 110, flush=True)

    all_results: dict = {
        "timestamp": datetime.now().isoformat(),
        "version": "fast",
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
        },
    }

    # =====================================================================
    # PHASE 0: Load all data upfront (one-time cost)
    # =====================================================================
    print("\n[0/5] Loading all data upfront...", flush=True)
    t0 = time.time()
    preload_all_data()
    print(f"  Data loaded in {time.time()-t0:.1f}s", flush=True)

    all_results["ffd_d_stars"] = FFD_D_STARS
    all_results["hl_data_available"] = {s: (_hl_cache.get(s) is not None) for s in ASSETS}

    # =====================================================================
    # PHASE 1: BASELINE — v3 system (14 features, flat sizing)
    # =====================================================================
    print("\n" + "=" * 80, flush=True)
    print("[1/5] BASELINE: v3 system (14 features, flat 2% sizing)", flush=True)
    print("=" * 80, flush=True)

    # Try parallel first, fall back to serial
    try:
        baseline_signals = generate_signals_parallel("baseline", "Baseline", max_workers=6)
    except Exception as e:
        print(f"  [WARN] Parallel failed ({e}), falling back to serial...", flush=True)
        baseline_signals = generate_signals_serial("baseline", "Baseline")

    baseline_sim = simulate_portfolio(baseline_signals, sizing_mode="flat")
    baseline_metrics = compute_metrics(baseline_sim, "Baseline(v3)")

    if baseline_metrics:
        print(f"  Sharpe: {baseline_metrics['sharpe']:.3f}, "
              f"Return: {baseline_metrics['total_return_pct']:+.2f}%, "
              f"MaxDD: {baseline_metrics['max_dd_pct']:.2f}%, "
              f"Kelly f*: {baseline_metrics['kelly_f_star']:.4f}, "
              f"DSR: {baseline_metrics['dsr']:.4f}", flush=True)

    # =====================================================================
    # PHASE 2: EXPANDED FEATURES (14->20, with fast importance validation)
    # =====================================================================
    print("\n" + "=" * 80, flush=True)
    print("[2/5] EXPANDED FEATURES (14->20, MDI+MDA fast validation)", flush=True)
    print("=" * 80, flush=True)

    importance_results: dict = {}
    validated_features_per_asset: dict = {}

    print("\n  Running fast importance validation per asset...", flush=True)
    t_imp = time.time()

    for symbol in ASSETS:
        try:
            ohlcv = load_ohlcv(symbol, BAR_TF)
        except FileNotFoundError:
            continue

        close = ohlcv["close"]
        fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)

        importance_month = ALL_MONTHS[6]  # July 2025
        ms = pd.Timestamp(importance_month.start_time, tz="UTC")

        d = FFD_D_STARS.get(symbol, 0.4)
        hl = _hl_cache.get(symbol)
        feat = build_expanded_features(ohlcv, ms, hl, d_star=d)
        fc = feat.dropna()

        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)
        tr = valid[valid < ms]

        if len(tr) < 500:
            print(f"    {symbol}: insufficient data for importance, keeping all features", flush=True)
            validated_features_per_asset[symbol] = list(feat.columns)
            importance_results[symbol] = {"keep": list(feat.columns), "dropped": [], "rankings": {}}
            continue

        X_tr = fc.loc[tr]
        y_tr = (y_all.loc[tr] > 0).astype(int)

        trip = run_fast_importance(X_tr, y_tr, list(feat.columns))
        validated_features_per_asset[symbol] = trip["keep_features"]
        importance_results[symbol] = {
            "keep": trip["keep_features"],
            "dropped": trip["dropped_features"],
            "rankings": trip["rankings"],
        }

        print(f"    {symbol}: {len(trip['keep_features'])}/{len(feat.columns)} features kept", flush=True)
        if trip["dropped_features"]:
            print(f"      Dropped: {trip['dropped_features']}", flush=True)

    print(f"  Feature importance done in {time.time()-t_imp:.1f}s", flush=True)
    all_results["feature_importance"] = importance_results

    # Generate expanded signals (parallel)
    try:
        expanded_signals = generate_signals_parallel(
            "expanded", "Expanded",
            validated_features_per_asset=validated_features_per_asset,
            max_workers=6,
        )
    except Exception as e:
        print(f"  [WARN] Parallel failed ({e}), falling back to serial...", flush=True)
        expanded_signals = generate_signals_serial(
            "expanded", "Expanded",
            validated_features_per_asset=validated_features_per_asset,
        )

    expanded_sim = simulate_portfolio(expanded_signals, sizing_mode="flat")
    expanded_metrics = compute_metrics(expanded_sim, "ExpandedFeat")

    if expanded_metrics:
        print(f"  Sharpe: {expanded_metrics['sharpe']:.3f}, "
              f"Return: {expanded_metrics['total_return_pct']:+.2f}%, "
              f"MaxDD: {expanded_metrics['max_dd_pct']:.2f}%, "
              f"Kelly f*: {expanded_metrics['kelly_f_star']:.4f}, "
              f"DSR: {expanded_metrics['dsr']:.4f}", flush=True)

    # =====================================================================
    # PHASE 3: PROBABILITY-BASED BET SIZING (AFML Ch 10)
    # =====================================================================
    print("\n" + "=" * 80, flush=True)
    print("[3/5] PROBABILITY-BASED BET SIZING (AFML Ch 10)", flush=True)
    print("=" * 80, flush=True)

    prob_sim = simulate_portfolio(baseline_signals, sizing_mode="prob")
    prob_metrics = compute_metrics(prob_sim, "ProbSizing")

    if prob_metrics:
        print(f"  Sharpe: {prob_metrics['sharpe']:.3f}, "
              f"Return: {prob_metrics['total_return_pct']:+.2f}%, "
              f"MaxDD: {prob_metrics['max_dd_pct']:.2f}%, "
              f"Kelly f*: {prob_metrics['kelly_f_star']:.4f}, "
              f"DSR: {prob_metrics['dsr']:.4f}", flush=True)

    # =====================================================================
    # PHASE 4: COMBINED (expanded features + probability sizing)
    # =====================================================================
    print("\n" + "=" * 80, flush=True)
    print("[4/5] COMBINED: Expanded Features + Probability Sizing", flush=True)
    print("=" * 80, flush=True)

    combined_sim = simulate_portfolio(expanded_signals, sizing_mode="prob")
    combined_metrics = compute_metrics(combined_sim, "Combined")

    if combined_metrics:
        print(f"  Sharpe: {combined_metrics['sharpe']:.3f}, "
              f"Return: {combined_metrics['total_return_pct']:+.2f}%, "
              f"MaxDD: {combined_metrics['max_dd_pct']:.2f}%, "
              f"Kelly f*: {combined_metrics['kelly_f_star']:.4f}, "
              f"DSR: {combined_metrics['dsr']:.4f}", flush=True)

    # =====================================================================
    # PHASE 5: COMPARISON TABLE + KELLY + DSR
    # =====================================================================
    print("\n" + "=" * 110, flush=True)
    print("[5/5] COMPARISON TABLE", flush=True)
    print("=" * 110, flush=True)

    variants = [
        ("Baseline(v3)", baseline_metrics),
        ("ExpandedFeat", expanded_metrics),
        ("ProbSizing", prob_metrics),
        ("Combined", combined_metrics),
    ]

    header = f"{'Metric':<22}"
    for name, _ in variants:
        header += f" {name:>14}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    metrics_to_show = [
        ("Sharpe", "sharpe", ".3f"),
        ("Sortino", "sortino", ".3f"),
        ("Total Return %", "total_return_pct", "+.2f"),
        ("MaxDD %", "max_dd_pct", ".2f"),
        ("Win Rate (trades)", "win_rate_trades", ".4f"),
        ("Win Rate (days)", "win_rate_days", ".4f"),
        ("Profit Factor", "profit_factor", ".3f"),
        ("Trades/day", "trades_per_day", ".2f"),
        ("Months positive", "months_positive", "d"),
        ("DSR", "dsr", ".4f"),
        ("Kelly f*", "kelly_f_star", ".4f"),
        ("Half-Kelly", "half_kelly", ".4f"),
        ("Calmar", "calmar", ".3f"),
        ("Mean daily PnL", "mean_daily_pnl", ".2f"),
        ("Halted", "halted", ""),
    ]

    for label, key, fmt in metrics_to_show:
        row = f"{label:<22}"
        for _, m in variants:
            if m is None:
                row += f" {'N/A':>14}"
            elif key == "halted":
                row += f" {'HALT' if m[key] else 'OK':>14}"
            elif key == "months_positive":
                row += f" {str(m[key]) + '/' + str(m.get('months_total', '?')):>14}"
            else:
                val = m.get(key, float("nan"))
                row += f" {format(val, fmt):>14}"
        print(row, flush=True)

    # Feature importance summary
    print("\n" + "-" * 80, flush=True)
    print("FEATURE IMPORTANCE - Fast Validation Summary (MDI+MDA)", flush=True)
    print("-" * 80, flush=True)

    for symbol in ASSETS:
        if symbol in importance_results:
            res = importance_results[symbol]
            kept = res["keep"]
            dropped = res["dropped"]
            n_total = len(kept) + len(dropped)
            print(f"  {symbol}: {len(kept)}/{n_total} features survived", flush=True)
            if dropped:
                print(f"    Dropped (bottom-{IMPORTANCE_BOTTOM_N} in MDI+MDA): {dropped}", flush=True)
            if res.get("rankings"):
                ranked = sorted(res["rankings"].items(),
                                key=lambda x: x[1].get("mdi_rank", 0), reverse=True)
                top5 = ranked[:5]
                top5_strs = []
                for r in top5:
                    name_str = r[0]
                    mdi_r = r[1].get("mdi_rank", 0)
                    mda_r = r[1].get("mda_rank", 0)
                    top5_strs.append(f"{name_str}(mdi={mdi_r},mda={mda_r})")
                print(f"    Top-5 MDI: {top5_strs}", flush=True)

    # Kelly criterion
    print("\n" + "-" * 80, flush=True)
    print("KELLY CRITERION (AFML Ch 10)", flush=True)
    print("-" * 80, flush=True)

    print(f"  {'Variant':<18} {'f*':>8} {'f*/2':>8} {'Win Rate':>10} {'Recommendation':<30}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for name, m in variants:
        if m is None:
            print(f"  {name:<18} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'Insufficient data':<30}", flush=True)
        else:
            f_star = m["kelly_f_star"]
            half_k = m["half_kelly"]
            wr = m["win_rate_trades"]
            if f_star <= 0:
                rec = "NO BET (negative edge)"
            elif half_k > BASE_SIZING_PCT * 3:
                rec = f"Half-Kelly suggests {half_k*100:.1f}% - aggressive"
            elif half_k < BASE_SIZING_PCT * 0.5:
                rec = f"Half-Kelly suggests {half_k*100:.1f}% - reduce size"
            else:
                rec = f"Half-Kelly suggests {half_k*100:.1f}% - reasonable"
            print(f"  {name:<18} {f_star:>8.4f} {half_k:>8.4f} {wr:>10.4f} {rec:<30}", flush=True)

    # DSR
    print("\n" + "-" * 80, flush=True)
    print("DEFLATED SHARPE RATIO (AFML Ch 14, Bailey & LdP 2014)", flush=True)
    print(f"  K={DSR_NUM_TRIALS} trials tested, computing DSR for each variant", flush=True)
    print("-" * 80, flush=True)

    print(f"  {'Variant':<18} {'Sharpe':>8} {'DSR':>8} {'T(days)':>8} {'Skew':>8} {'Kurt':>8} {'Pass?':>8}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for name, m in variants:
        if m is None:
            print(f"  {name:<18} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}", flush=True)
        else:
            dsr_val = m["dsr"]
            sr_val = m["sharpe"]
            n_obs = m["dsr_num_obs"]
            skew = m["daily_skewness"]
            kurt_val = m["daily_kurtosis"]
            passed = "YES" if dsr_val > 0.95 else "NO"
            print(f"  {name:<18} {sr_val:>8.3f} {dsr_val:>8.4f} {n_obs:>8d} {skew:>8.3f} "
                  f"{kurt_val:>8.3f} {passed:>8}", flush=True)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "=" * 80, flush=True)
    print("Saving results...", flush=True)

    all_results["variants"] = {}
    for name, m in variants:
        all_results["variants"][name] = m if m else {"error": "insufficient data"}

    all_results["signal_counts"] = {
        "baseline": len(baseline_signals),
        "expanded": len(expanded_signals),
    }

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

    print("\n" + "=" * 110, flush=True)
    print("TIER 1 EXPERIMENT COMPLETE (FAST)", flush=True)
    print(f"Finished: {datetime.now()}", flush=True)
    print(f"Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print("=" * 110, flush=True)

    valid_variants = [(name, m) for name, m in variants if m is not None]
    if valid_variants:
        best_name, best_m = max(valid_variants, key=lambda x: x[1]["sharpe"])
        print(f"\n  Best variant by Sharpe: {best_name} ({best_m['sharpe']:.3f})", flush=True)
        print(f"  Best variant details:", flush=True)
        print(f"    Return: {best_m['total_return_pct']:+.2f}%", flush=True)
        print(f"    MaxDD:  {best_m['max_dd_pct']:.2f}%", flush=True)
        print(f"    Kelly f*/2: {best_m['half_kelly']:.4f}", flush=True)
        print(f"    DSR:    {best_m['dsr']:.4f} ({'PASS' if best_m['dsr'] > 0.95 else 'FAIL'})", flush=True)

        if baseline_metrics:
            delta_sharpe = best_m["sharpe"] - baseline_metrics["sharpe"]
            delta_ret = best_m["total_return_pct"] - baseline_metrics["total_return_pct"]
            print(f"\n  vs Baseline:", flush=True)
            print(f"    Sharpe delta: {delta_sharpe:+.3f}", flush=True)
            print(f"    Return delta: {delta_ret:+.2f}pp", flush=True)
    else:
        print("\n  [WARN] No valid variant results. Check data availability.", flush=True)

    print(f"\n  Results file: {RESULTS_FILE}", flush=True)
    print(f"  Total time: {total_elapsed:.0f}s", flush=True)
