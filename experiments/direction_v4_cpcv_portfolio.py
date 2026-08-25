"""
Direction V4 -- Portfolio-Level CPCV.

Combines CPCV (AFML Ch 12) with realistic portfolio simulation.
For each of 15 CPCV splits (C(6,2)):
  1. Train XGBoost on 4 train groups per asset (6 assets)
  2. Generate signals on 2 test groups with vol + confidence filters
  3. Run FULL realistic portfolio sim on test signals with shared capital
  4. Compute daily Sharpe, Sortino, MaxDD, Calmar, return, win rates

Architecture: Precompute-once from tier2.py (features computed ONCE, sliced per split).
"""

import sys
import warnings
import json
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

try:
    from afml.fracdiff import frac_diff_ffd
    from afml.derivatives import funding_features, oi_features, impact_features
    from afml.features import corwin_schultz_spread, rolling_entropy
    AFML_IMPORTED = True
except ImportError:
    AFML_IMPORTED = False
    print("[WARN] Could not import afml modules - using inline implementations", flush=True)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# =============================================================================
# INLINE AFML FALLBACKS
# =============================================================================

if not AFML_IMPORTED:

    def get_weights_ffd(d, threshold=1e-5):
        weights = [1.0]
        k = 1
        while True:
            w_next = -weights[-1] / k * (d - k + 1)
            if abs(w_next) < threshold:
                break
            weights.append(w_next)
            k += 1
        return np.array(weights[::-1]).reshape(-1, 1)

    def frac_diff_ffd(series, d, threshold=1e-5):
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
        features["oi_change_5"] = open_interest.pct_change(5)
        return features

    def impact_features(impact_bid, impact_ask, mid_price):
        features = pd.DataFrame(index=mid_price.index)
        features["impact_spread"] = (impact_ask - impact_bid) / mid_price.replace(0, np.nan)
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


# =============================================================================
# CONSTANTS
# =============================================================================
DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
HL_CSV_DIR = Path("C:/Claude/ml-trading-system/v3-results")
OUTPUT_DIR = Path("C:/Claude/ml-trading-system/v4-results")

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

BAR_TF = "30min"
FWD_BARS = 4
FFD_THRESHOLD = 1e-3
FFD_D_STARS = {"BTC": 0.2, "ETH": 0.1, "XRP": 0.3, "BNB": 0.1, "SUI": 0.2, "LINK": 0.1}

WINSOR_LO = 0.01
WINSOR_HI = 0.99
VOL_FILTER_QUANTILE = 0.50
CONF_PERCENTILE = 70
SLIPPAGE = 0.0001

XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)

# CPCV parameters -- AFML Ch 12
N_GROUPS = 6
K_TEST = 2
EMBARGO_PCT = 0.01

# Portfolio simulation parameters
STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
BASE_SIZING_PCT = 0.02
MAX_EXPOSURE = 0.60
MAX_PER_ASSET = 0.20


# =============================================================================
# DATA LOADING + CACHING (from tier2.py)
# =============================================================================
_ohlcv_cache = {}
_hl_cache = {}


def load_ohlcv(symbol, tf=BAR_TF):
    key = (symbol, tf)
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]
    d = DATA_DIR / SYMBOLS[symbol]
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
    result = combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    _ohlcv_cache[key] = result
    return result


def load_hl(symbol):
    if symbol in _hl_cache:
        return _hl_cache[symbol]
    csv_path = HL_CSV_DIR / f"hl_{symbol}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        resampled = df.resample(BAR_TF).last().ffill()
        _hl_cache[symbol] = resampled
        return resampled
    _hl_cache[symbol] = None
    return None


# =============================================================================
# PRECOMPUTE FEATURES (from tier2.py / cpcv.py)
# =============================================================================
_full_features = {}
_vol_series = {}
_fwd_returns = {}


def precompute_features():
    """Compute ALL features once for all assets on the full time series."""
    for symbol in ASSETS:
        ohlcv = load_ohlcv(symbol)
        hl = load_hl(symbol)
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]

        f = pd.DataFrame(index=ohlcv.index)

        # Returns
        for lb in [1, 3, 7, 14]:
            f[f"ret_{lb}"] = close.pct_change(lb)

        # Technicals
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

        # FFD
        try:
            d_star = FFD_D_STARS.get(symbol, 0.4)
            log_close = np.log(close).to_frame("close")
            ffd = frac_diff_ffd(log_close, d_star, threshold=FFD_THRESHOLD)
            if len(ffd) > 0:
                f["ffd"] = ffd.iloc[:, 0].reindex(f.index)
        except Exception:
            pass

        # CS spread + entropy
        try:
            f["cs_spread"] = corwin_schultz_spread(high, low, window=20)
        except Exception:
            pass
        try:
            f["entropy"] = rolling_entropy(close, window=50, n_bins=5)
        except Exception:
            pass

        # HL derivatives
        if hl is not None:
            if "funding" in hl.columns:
                try:
                    ff = funding_features(hl["funding"])
                    if "funding_zscore" in ff.columns:
                        f["funding_zscore"] = ff["funding_zscore"].reindex(f.index)
                except Exception:
                    pass
            if "open_interest" in hl.columns:
                try:
                    oi = oi_features(hl["open_interest"], close.reindex(hl.index, method="ffill"))
                    if "oi_change_5" in oi.columns:
                        f["oi_change_5"] = oi["oi_change_5"].reindex(f.index)
                except Exception:
                    pass
            if "impact_bid_px" in hl.columns and "impact_ask_px" in hl.columns:
                try:
                    mid = hl.get("mid_px", (hl["impact_bid_px"] + hl["impact_ask_px"]) / 2)
                    imp = impact_features(hl["impact_bid_px"], hl["impact_ask_px"], mid)
                    if "impact_spread" in imp.columns:
                        f["impact_spread"] = imp["impact_spread"].reindex(f.index)
                except Exception:
                    pass

        _full_features[symbol] = f
        _vol_series[symbol] = close.pct_change().dropna().ewm(span=50).std()
        _fwd_returns[symbol] = close.pct_change(FWD_BARS).shift(-FWD_BARS)

        print(f"  {symbol}: {len(f.columns)} features, {len(f)} rows", flush=True)


# =============================================================================
# SIGNAL GENERATION (AFML Ch 10)
# =============================================================================

def get_signal(prob):
    """AFML Ch 10: probability -> bet signal strength."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    return 2.0 * norm.cdf(z) - 1.0


def generate_signals_for_split(symbol, train_indices, test_indices, embargo_size):
    """Train on train_indices, generate signals on test_indices for one asset.

    Returns list of (timestamp, asset, direction, fwd_return, signal_strength)
    or empty list if insufficient data.
    """
    features = _full_features[symbol]
    fwd_ret = _fwd_returns[symbol]
    vol = _vol_series[symbol]

    fc = features.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)

    # Map group indices to actual timestamps
    train_times = valid[train_indices]
    test_times = valid[test_indices]

    if len(train_times) < 200 or len(test_times) < 50:
        return []

    # Embargo: remove training samples near test boundaries
    if embargo_size > 0:
        test_sorted = test_times.sort_values()
        # Find contiguous test blocks and embargo around each boundary
        embargo_td = pd.Timedelta(minutes=30 * embargo_size)
        # For non-contiguous test groups, find each group's boundaries
        test_arr = test_sorted.values
        gaps = np.where(np.diff(test_arr.astype(np.int64)) > 2 * 30 * 60 * 1e9)[0]
        block_starts = [test_arr[0]]
        block_ends = [test_arr[-1]]
        for g in gaps:
            block_ends.insert(-1, test_arr[g])
            block_starts.insert(-1, test_arr[g + 1])

        embargo_mask = np.ones(len(train_times), dtype=bool)
        for bs, be in zip(block_starts, block_ends):
            # Match timezone of train_times (UTC-aware)
            bs_ts = pd.Timestamp(bs)
            be_ts = pd.Timestamp(be)
            if train_times.tz is not None:
                bs_ts = bs_ts.tz_localize(train_times.tz) if bs_ts.tz is None else bs_ts.tz_convert(train_times.tz)
                be_ts = be_ts.tz_localize(train_times.tz) if be_ts.tz is None else be_ts.tz_convert(train_times.tz)
            embargo_mask &= ~(
                (train_times >= bs_ts - embargo_td) &
                (train_times <= be_ts + embargo_td)
            )
        train_times = train_times[embargo_mask]

    if len(train_times) < 200:
        return []

    # Winsorize returns on train set statistics
    X_all = fc.loc[valid].copy()
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in X_all.columns:
            train_vals = X_all.loc[train_times, col].dropna()
            if len(train_vals) > 10:
                lo = train_vals.quantile(WINSOR_LO)
                hi = train_vals.quantile(WINSOR_HI)
                X_all[col] = X_all[col].clip(lo, hi)

    # Vol filter: only trade high-vol bars in test
    train_vol = vol.reindex(train_times).dropna()
    if len(train_vol) < 50:
        return []
    vol_thresh = train_vol.quantile(VOL_FILTER_QUANTILE)
    test_vol = vol.reindex(test_times)
    hv_mask = (test_vol > vol_thresh).fillna(False)
    test_times = test_times[hv_mask.values]

    if len(test_times) < 20:
        return []

    X_train = X_all.loc[train_times]
    y_train = (y_all.loc[train_times] > 0).astype(int)
    X_test = X_all.loc[test_times]

    # Train XGBoost
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confidence filter
    conf = np.abs(y_prob - 0.5)
    if len(conf) > 10:
        conf_thresh = np.percentile(conf, CONF_PERCENTILE)
        trade_mask = conf >= conf_thresh
    else:
        trade_mask = np.ones(len(conf), dtype=bool)

    if trade_mask.sum() < 5:
        return []

    # Build signal list
    signals = []
    test_sub = test_times[trade_mask]
    probs_sub = y_prob[trade_mask]
    actual_rets = fwd_ret.reindex(test_sub).values

    for i in range(len(test_sub)):
        ts = test_sub[i]
        prob = probs_sub[i]
        fwd_r = actual_rets[i]
        if not np.isfinite(fwd_r):
            continue
        direction = 1.0 if prob > 0.5 else -1.0
        sig = get_signal(prob)
        sig_disc = round(sig / 0.1) * 0.1  # Discretize to 0.1 steps
        sig_disc = max(abs(sig_disc), 0.1)  # Floor at 0.1
        signals.append((ts, symbol, direction, fwd_r, sig_disc))

    return signals


# =============================================================================
# PORTFOLIO SIMULATOR (per CPCV split)
# =============================================================================

def simulate_portfolio_for_split(signals_list):
    """Simulate realistic portfolio on signals from one CPCV split.

    signals_list: list of (timestamp, asset, direction, fwd_return, signal_strength)
    sorted by timestamp.

    Returns (final_equity, pnl_by_day, halted, trade_count, win_count).
    """
    if not signals_list:
        return STARTING_CAPITAL, {}, False, 0, 0

    equity = STARTING_CAPITAL
    peak = equity
    halted = False
    pnl_by_day = {}
    asset_exposure = {}
    total_exposure = 0.0
    trade_count = 0
    win_count = 0

    for ts, asset, direction, fwd_ret, signal_str in signals_list:
        if halted:
            continue

        dd = (peak - equity) / peak if peak > 0 else 0
        if dd >= MAX_DD:
            halted = True
            continue

        # Deleveraging
        if dd > DELEVER_START:
            size_mult = max(1.0 - (dd - DELEVER_START) / (MAX_DD - DELEVER_START), 0.0)
        else:
            size_mult = 1.0

        # Position size: equity * base_pct * |signal_strength| * delever_mult
        raw_size = equity * BASE_SIZING_PCT * abs(signal_str) * size_mult

        # Exposure limits with decay
        ae = asset_exposure.get(asset, 0.0) * 0.9
        asset_exposure[asset] = ae
        if ae + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - ae, 0)

        total_exposure *= 0.9
        if total_exposure + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - total_exposure, 0)

        if raw_size < 1.0:
            continue

        # Execute trade
        pnl = raw_size * (direction * fwd_ret - SLIPPAGE)
        equity += pnl
        if equity > peak:
            peak = equity

        asset_exposure[asset] = ae + raw_size
        total_exposure += raw_size

        trade_count += 1
        if pnl > 0:
            win_count += 1

        day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
        pnl_by_day[day] = pnl_by_day.get(day, 0.0) + pnl

    return equity, pnl_by_day, halted, trade_count, win_count


def compute_metrics(equity, pnl_by_day, halted, trade_count, win_count):
    """Compute performance metrics from simulation results."""
    if not pnl_by_day or trade_count < 5:
        return None

    daily = pd.Series(pnl_by_day).sort_index()
    total_ret = equity / STARTING_CAPITAL - 1.0

    # Daily Sharpe
    mean_d = daily.mean()
    std_d = daily.std()
    sharpe = mean_d / std_d * np.sqrt(365.25) if std_d > 0 else 0.0

    # Sortino (downside std)
    down = daily[daily < 0]
    down_std = down.std() if len(down) > 1 else std_d
    sortino = mean_d / down_std * np.sqrt(365.25) if down_std > 0 else 0.0

    # MaxDD from equity curve
    cumulative = daily.cumsum() + STARTING_CAPITAL
    running_max = cumulative.cummax()
    drawdowns = (running_max - cumulative) / running_max
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    # Calmar
    days_span = max((daily.index[-1] - daily.index[0]).days, 1)
    years = days_span / 365.25
    cagr = (equity / STARTING_CAPITAL) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Win rates
    wr_trades = win_count / trade_count if trade_count > 0 else 0.0
    wr_days = (daily > 0).sum() / len(daily) if len(daily) > 0 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_trades": round(wr_trades, 3),
        "win_rate_days": round(wr_days, 3),
        "trades": trade_count,
        "halted": halted,
        "final_equity": round(equity, 2),
        "n_days": len(daily),
    }


# =============================================================================
# CPCV SPLIT GENERATION (from cpcv.py)
# =============================================================================

def get_valid_indices_for_asset(symbol):
    """Get valid observation indices for one asset (features + fwd_return both non-NaN)."""
    features = _full_features[symbol]
    fwd_ret = _fwd_returns[symbol]
    fc = features.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    return valid


def generate_cpcv_splits():
    """Generate C(N_GROUPS, K_TEST) CPCV split definitions.

    Uses BTC's valid index count as reference for group sizing
    (all assets get sliced by the same group index boundaries).

    Returns list of (combo_tuple, test_group_indices, train_group_indices).
    """
    combos = list(itertools.combinations(range(N_GROUPS), K_TEST))
    return combos


def get_groups_for_asset(symbol):
    """Split valid indices for one asset into N chronological groups.

    Returns (valid_index, groups_list, embargo_size).
    """
    valid = get_valid_indices_for_asset(symbol)
    n_obs = len(valid)

    if n_obs < 500:
        return valid, [], 0

    group_size = n_obs // N_GROUPS
    embargo_size = int(n_obs * EMBARGO_PCT)

    groups = []
    for i in range(N_GROUPS):
        start = i * group_size
        end = start + group_size if i < N_GROUPS - 1 else n_obs
        groups.append(np.arange(start, end))

    return valid, groups, embargo_size


# =============================================================================
# RUN ONE CPCV SPLIT (all assets -> portfolio)
# =============================================================================

def run_one_split(combo, split_idx):
    """Run one CPCV split across all assets, then simulate portfolio.

    combo: tuple of test group indices, e.g. (0, 1)
    split_idx: integer label for this split

    Returns metrics dict or None.
    """
    all_signals = []

    for symbol in ASSETS:
        valid, groups, embargo_size = get_groups_for_asset(symbol)
        if not groups:
            continue

        # Build train/test index arrays
        test_idx = np.concatenate([groups[g] for g in combo])
        train_idx = np.concatenate([groups[g] for g in range(N_GROUPS) if g not in combo])

        signals = generate_signals_for_split(symbol, train_idx, test_idx, embargo_size)
        all_signals.extend(signals)

    if len(all_signals) < 20:
        return None

    # Sort all signals by timestamp for realistic chronological simulation
    all_signals.sort(key=lambda x: x[0])

    # Run portfolio simulation
    equity, pnl_by_day, halted, trade_count, win_count = simulate_portfolio_for_split(all_signals)

    metrics = compute_metrics(equity, pnl_by_day, halted, trade_count, win_count)
    if metrics is None:
        return None

    metrics["split_idx"] = split_idx
    metrics["test_groups"] = list(combo)
    metrics["n_signals_raw"] = len(all_signals)

    return metrics


def run_one_split_for_asset(args):
    """Thread-pool wrapper: generate signals for one (asset, combo) pair."""
    symbol, combo = args
    valid, groups, embargo_size = get_groups_for_asset(symbol)
    if not groups:
        return []
    test_idx = np.concatenate([groups[g] for g in combo])
    train_idx = np.concatenate([groups[g] for g in range(N_GROUPS) if g not in combo])
    return generate_signals_for_split(symbol, train_idx, test_idx, embargo_size)


def run_one_split_parallel(combo, split_idx):
    """Run one CPCV split with parallel asset signal generation, then simulate portfolio."""
    # Generate signals for all assets in parallel (ThreadPool for shared cache)
    args_list = [(symbol, combo) for symbol in ASSETS]
    all_signals = []

    with ThreadPoolExecutor(max_workers=len(ASSETS)) as pool:
        futures = [pool.submit(run_one_split_for_asset, a) for a in args_list]
        for fut in as_completed(futures):
            all_signals.extend(fut.result())

    if len(all_signals) < 20:
        return None

    # Sort chronologically
    all_signals.sort(key=lambda x: x[0])

    # Portfolio simulation
    equity, pnl_by_day, halted, trade_count, win_count = simulate_portfolio_for_split(all_signals)

    metrics = compute_metrics(equity, pnl_by_day, halted, trade_count, win_count)
    if metrics is None:
        return None

    metrics["split_idx"] = split_idx
    metrics["test_groups"] = list(combo)
    metrics["n_signals_raw"] = len(all_signals)

    return metrics


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================

def bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    values = np.array(values)
    n = len(values)
    if n < 3:
        return (values.min(), values.max())
    boot_means = np.array([
        np.mean(np.random.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return (round(lo, 3), round(hi, 3))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    n_combos = len(list(itertools.combinations(range(N_GROUPS), K_TEST)))

    print("=" * 100, flush=True)
    print("PORTFOLIO-LEVEL CPCV -- Combinatorial Purged CV with Realistic Portfolio Sim", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"N={N_GROUPS} groups, k={K_TEST} test -> C({N_GROUPS},{K_TEST}) = {n_combos} splits", flush=True)
    print(f"Assets: {', '.join(ASSETS)}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL:,.0f} | Sizing: {BASE_SIZING_PCT*100:.0f}% * |signal| | "
          f"MaxDD halt: {MAX_DD*100:.0f}% | Delever: {DELEVER_START*100:.1f}%", flush=True)
    print(f"Exposure: {MAX_EXPOSURE*100:.0f}% total, {MAX_PER_ASSET*100:.0f}% per asset | "
          f"Slippage: {SLIPPAGE*10000:.0f} bps", flush=True)
    print(f"Vol filter: top {(1-VOL_FILTER_QUANTILE)*100:.0f}% | "
          f"Conf filter: top {100-CONF_PERCENTILE}%", flush=True)
    print("=" * 100, flush=True)

    # =========================================================================
    # Phase 0: Load + precompute features ONCE
    # =========================================================================
    print("\n[1/3] Loading data + precomputing features...", flush=True)
    t0 = time.time()
    for s in ASSETS:
        load_ohlcv(s)
        load_hl(s)
    precompute_features()
    t_precompute = time.time() - t0
    print(f"  Precompute done in {t_precompute:.1f}s", flush=True)

    # Print data summary
    for symbol in ASSETS:
        valid = get_valid_indices_for_asset(symbol)
        print(f"  {symbol}: {len(valid)} valid observations", flush=True)

    # =========================================================================
    # Phase 1: Run all CPCV splits
    # =========================================================================
    combos = list(itertools.combinations(range(N_GROUPS), K_TEST))
    print(f"\n[2/3] Running {len(combos)} CPCV portfolio splits...", flush=True)

    split_results = []
    for idx, combo in enumerate(combos):
        t_split = time.time()
        metrics = run_one_split_parallel(combo, idx + 1)
        elapsed = time.time() - t_split

        if metrics is not None:
            split_results.append(metrics)
            status = "HALT" if metrics["halted"] else "OK"
            print(f"  Split {idx+1:>2}/{len(combos)} groups={set(combo)} | "
                  f"Sharpe={metrics['sharpe']:>6.3f} Sortino={metrics['sortino']:>6.3f} "
                  f"Ret={metrics['total_return_pct']:>+6.2f}% DD={metrics['max_dd_pct']:>5.2f}% "
                  f"WR={metrics['win_rate_trades']:.3f} Trades={metrics['trades']:>5} "
                  f"{status} ({elapsed:.1f}s)", flush=True)
        else:
            print(f"  Split {idx+1:>2}/{len(combos)} groups={set(combo)} | "
                  f"SKIP (insufficient signals) ({elapsed:.1f}s)", flush=True)

    t_splits = time.time() - t0 - t_precompute

    # =========================================================================
    # Phase 2: Results
    # =========================================================================
    print(f"\n[3/3] Results...", flush=True)
    print("=" * 110, flush=True)
    print("PORTFOLIO-LEVEL CPCV RESULTS -- Realistic Multi-Asset Simulation per Split", flush=True)
    print("=" * 110, flush=True)

    if not split_results:
        print("  No valid splits produced results!", flush=True)
        sys.exit(1)

    # Per-split table
    print(f"{'Split':>5}  {'Groups':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'Return':>8}  "
          f"{'MaxDD':>6}  {'Calmar':>7}  {'WR':>5}  {'Trades':>6}  {'Halt':>4}", flush=True)
    print("-" * 110, flush=True)

    sharpes = []
    sortinos = []
    returns = []
    max_dds = []
    calmars = []
    win_rates = []
    trade_counts = []

    for r in split_results:
        groups_str = "{" + ",".join(str(g) for g in r["test_groups"]) + "}"
        status = "HALT" if r["halted"] else "OK"
        print(f"{r['split_idx']:>5}  {groups_str:>8}  {r['sharpe']:>7.3f}  "
              f"{r['sortino']:>8.3f}  {r['total_return_pct']:>+7.2f}%  "
              f"{r['max_dd_pct']:>5.2f}%  {r['calmar']:>7.3f}  "
              f"{r['win_rate_trades']*100:>4.1f}%  {r['trades']:>6}  {status:>4}", flush=True)

        sharpes.append(r["sharpe"])
        sortinos.append(r["sortino"])
        returns.append(r["total_return_pct"])
        max_dds.append(r["max_dd_pct"])
        calmars.append(r["calmar"])
        win_rates.append(r["win_rate_trades"] * 100)
        trade_counts.append(r["trades"])

    sharpes = np.array(sharpes)
    sortinos = np.array(sortinos)
    returns = np.array(returns)
    max_dds = np.array(max_dds)
    calmars = np.array(calmars)

    # Aggregate summary
    print(f"\n{'='*80}", flush=True)
    print("AGGREGATE SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    print(f"\n  Sharpe Ratio:", flush=True)
    print(f"    Mean:       {sharpes.mean():.3f} +/- {sharpes.std():.3f}", flush=True)
    print(f"    Median:     {np.median(sharpes):.3f}", flush=True)
    print(f"    Min/Max:    [{sharpes.min():.3f}, {sharpes.max():.3f}]", flush=True)
    print(f"    % Positive: {(sharpes > 0).mean() * 100:.1f}%", flush=True)

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(sharpes)
    print(f"    Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)

    print(f"\n  Sortino Ratio:", flush=True)
    print(f"    Mean:       {sortinos.mean():.3f} +/- {sortinos.std():.3f}", flush=True)
    print(f"    Median:     {np.median(sortinos):.3f}", flush=True)

    print(f"\n  Return (%):", flush=True)
    print(f"    Mean:       {returns.mean():+.2f}% +/- {returns.std():.2f}%", flush=True)
    print(f"    Min/Max:    [{returns.min():+.2f}%, {returns.max():+.2f}%]", flush=True)
    print(f"    % Positive: {(returns > 0).mean() * 100:.1f}%", flush=True)

    print(f"\n  Max Drawdown (%):", flush=True)
    print(f"    Mean:       {max_dds.mean():.2f}%", flush=True)
    print(f"    Worst:      {max_dds.max():.2f}%", flush=True)

    print(f"\n  Calmar Ratio:", flush=True)
    print(f"    Mean:       {calmars.mean():.3f}", flush=True)
    print(f"    Median:     {np.median(calmars):.3f}", flush=True)

    print(f"\n  Win Rate (trades):", flush=True)
    print(f"    Mean:       {np.mean(win_rates):.1f}%", flush=True)

    print(f"\n  Trades per split:", flush=True)
    print(f"    Mean:       {np.mean(trade_counts):.0f}", flush=True)
    print(f"    Total:      {sum(trade_counts)}", flush=True)

    halted_count = sum(1 for r in split_results if r["halted"])
    print(f"\n  Halted splits: {halted_count}/{len(split_results)}", flush=True)

    # Comparison to walk-forward (load from existing results if available)
    print(f"\n{'='*80}", flush=True)
    print("COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)

    wf_sharpe = None
    try:
        wf_path = OUTPUT_DIR / "tier2_results.json"
        if wf_path.exists():
            with open(wf_path) as f:
                wf_data = json.load(f)
            # Look for the best tier2 result Sharpe
            if "results" in wf_data:
                for r in wf_data["results"]:
                    if r.get("sharpe"):
                        if wf_sharpe is None or r["sharpe"] > wf_sharpe:
                            wf_sharpe = r["sharpe"]
    except Exception:
        pass

    if wf_sharpe is not None:
        print(f"  Walk-forward Sharpe (best tier2): {wf_sharpe:.3f}", flush=True)
    else:
        print(f"  Walk-forward Sharpe: N/A (no tier2_results.json found)", flush=True)

    print(f"  CPCV Portfolio mean Sharpe:      {sharpes.mean():.3f}", flush=True)
    print(f"  CPCV Portfolio median Sharpe:     {np.median(sharpes):.3f}", flush=True)

    # Interpretation
    print(f"\n--- INTERPRETATION (AFML Ch 12) ---", flush=True)
    pct_pos = (sharpes > 0).mean() * 100
    if pct_pos >= 80:
        print(f"  {pct_pos:.0f}% of splits have positive Sharpe -> STRONG evidence of real portfolio edge", flush=True)
    elif pct_pos >= 60:
        print(f"  {pct_pos:.0f}% of splits have positive Sharpe -> MODERATE evidence (edge exists but variable)", flush=True)
    elif pct_pos >= 50:
        print(f"  {pct_pos:.0f}% of splits have positive Sharpe -> WEAK evidence (barely above random)", flush=True)
    else:
        print(f"  {pct_pos:.0f}% of splits have positive Sharpe -> NO evidence of portfolio edge", flush=True)

    print(f"  Portfolio simulation is MORE realistic than per-asset CPCV:", flush=True)
    print(f"    - Shared capital with exposure limits", flush=True)
    print(f"    - Signal-strength position sizing (AFML Ch 10)", flush=True)
    print(f"    - DD deleveraging + halt mechanism", flush=True)
    print(f"    - Cross-asset exposure constraints", flush=True)

    # =========================================================================
    # Save results
    # =========================================================================
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_groups": N_GROUPS,
            "k_test": K_TEST,
            "embargo_pct": EMBARGO_PCT,
            "assets": ASSETS,
            "starting_capital": STARTING_CAPITAL,
            "max_dd": MAX_DD,
            "delever_start": DELEVER_START,
            "base_sizing_pct": BASE_SIZING_PCT,
            "max_exposure": MAX_EXPOSURE,
            "max_per_asset": MAX_PER_ASSET,
            "slippage_bps": SLIPPAGE * 10000,
            "vol_filter_quantile": VOL_FILTER_QUANTILE,
            "conf_percentile": CONF_PERCENTILE,
            "bar_tf": BAR_TF,
            "fwd_bars": FWD_BARS,
        },
        "per_split": split_results,
        "aggregate": {
            "sharpe_mean": round(sharpes.mean(), 3),
            "sharpe_std": round(sharpes.std(), 3),
            "sharpe_median": round(float(np.median(sharpes)), 3),
            "sharpe_min": round(float(sharpes.min()), 3),
            "sharpe_max": round(float(sharpes.max()), 3),
            "sharpe_pct_positive": round(float((sharpes > 0).mean() * 100), 1),
            "sharpe_bootstrap_95ci": list(bootstrap_ci(sharpes)),
            "sortino_mean": round(sortinos.mean(), 3),
            "return_mean_pct": round(returns.mean(), 2),
            "return_pct_positive": round(float((returns > 0).mean() * 100), 1),
            "max_dd_mean_pct": round(max_dds.mean(), 2),
            "max_dd_worst_pct": round(float(max_dds.max()), 2),
            "calmar_mean": round(calmars.mean(), 3),
            "win_rate_mean": round(float(np.mean(win_rates)), 1),
            "total_trades": int(sum(trade_counts)),
            "halted_splits": halted_count,
            "valid_splits": len(split_results),
            "total_splits": len(combos),
        },
    }

    out_path = OUTPUT_DIR / "cpcv_portfolio_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    total_time = time.time() - t0
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Completed: {datetime.now()} ({total_time:.0f}s total, "
          f"precompute={t_precompute:.0f}s, splits={t_splits:.0f}s)", flush=True)
