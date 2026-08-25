"""
Direction V4 -- Combinatorial Purged Cross-Validation (CPCV).

AFML Ch 12: Instead of one walk-forward path, generate MULTIPLE backtest
paths from combinatorial splits. This gives a DISTRIBUTION of Sharpe ratios,
not just one number. Much more robust than single-path walk-forward.

Method:
  - Split data into N groups chronologically
  - For each combination of k test groups: train on the rest, test on selected
  - Purge: remove training samples whose labels overlap with test
  - Generate C(N,k) backtest paths
  - Report: mean, std, min, max Sharpe across all paths

Uses the v4 Tier 2 best model (expanded features + HMM regime + prob sizing).
Precompute-once architecture for speed.
"""
import sys
import warnings
import json
import time
import itertools

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from scipy.stats import norm

# Technical indicators
from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from afml.features import corwin_schultz_spread, rolling_entropy

try:
    from afml.fracdiff import frac_diff_ffd
except ImportError:
    def get_weights_ffd(d, threshold=1e-5):
        w = [1.0]
        k = 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < threshold:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(series, d, threshold=1e-5):
        weights = get_weights_ffd(d, threshold)
        width = len(weights) - 1
        result = {}
        for col in series.columns:
            col_data = series[col].fillna(method="ffill").dropna()
            ffd_values = pd.Series(dtype=float)
            for i in range(width, len(col_data)):
                window = col_data.iloc[i - width : i + 1]
                ffd_values[col_data.index[i]] = np.dot(weights.T, window.values)[0, 0]
            result[col] = ffd_values
        return pd.DataFrame(result)

try:
    from afml.derivatives import funding_features, oi_features, impact_features
except ImportError:
    def funding_features(funding, window_short=50, window_long=200):
        f = pd.DataFrame(index=funding.index)
        mean = funding.rolling(window_short).mean()
        std = funding.rolling(window_short).std()
        f["funding_zscore"] = (funding - mean) / std.replace(0, np.nan)
        return f

    def oi_features(oi, close, window=50):
        f = pd.DataFrame(index=oi.index)
        f["oi_change_5"] = oi.pct_change(5)
        return f

    def impact_features(bid, ask, mid):
        f = pd.DataFrame(index=mid.index)
        f["impact_spread"] = (ask - bid) / mid.replace(0, np.nan)
        return f

try:
    from hmmlearn import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

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
N_GROUPS = 6     # Split data into 6 chronological groups
K_TEST = 2       # 2 test groups per split -> C(6,2) = 15 combinations
EMBARGO_PCT = 0.01  # 1% embargo between train/test

# =============================================================================
# DATA LOADING + CACHING
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
# PRECOMPUTE FEATURES (same as Tier 2 fast)
# =============================================================================
_full_features = {}
_vol_series = {}
_fwd_returns = {}


def precompute_features():
    """Compute ALL features once for all assets."""
    for symbol in ASSETS:
        ohlcv = load_ohlcv(symbol)
        hl = load_hl(symbol)
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]
        volume = ohlcv["volume"]

        f = pd.DataFrame(index=ohlcv.index)

        # Returns (raw, winsorized per-split later)
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
# CPCV ENGINE -- AFML Ch 12
# =============================================================================
def get_signal(prob):
    """AFML Ch 10: probability -> bet signal."""
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    return 2.0 * norm.cdf(z) - 1.0


def run_cpcv_split(symbol, train_indices, test_indices, embargo_size):
    """Run one CPCV split: train on train_indices, test on test_indices.

    Returns per-trade returns on the test set.
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
        return np.array([])

    # Embargo: remove training samples near test boundaries
    if embargo_size > 0:
        # Find test boundaries
        test_start = test_times.min()
        test_end = test_times.max()
        # Remove train samples within embargo of test
        embargo_mask = ~(
            (train_times >= test_start - pd.Timedelta(minutes=30 * embargo_size)) &
            (train_times <= test_end + pd.Timedelta(minutes=30 * embargo_size))
        )
        train_times = train_times[embargo_mask]

    if len(train_times) < 200:
        return np.array([])

    # Winsorize returns on train
    X_all = fc.loc[valid].copy()
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in X_all.columns:
            train_vals = X_all.loc[train_times, col].dropna()
            if len(train_vals) > 10:
                lo = train_vals.quantile(WINSOR_LO)
                hi = train_vals.quantile(WINSOR_HI)
                X_all[col] = X_all[col].clip(lo, hi)

    # Vol filter on test set
    train_vol = vol.reindex(train_times).dropna()
    if len(train_vol) < 50:
        return np.array([])
    vol_thresh = train_vol.quantile(VOL_FILTER_QUANTILE)
    test_vol = vol.reindex(test_times)
    hv_mask = (test_vol > vol_thresh).fillna(False)
    test_times = test_times[hv_mask.values]

    if len(test_times) < 20:
        return np.array([])

    X_train = X_all.loc[train_times]
    y_train = (y_all.loc[train_times] > 0).astype(int)
    X_test = X_all.loc[test_times]

    # Train
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
        return np.array([])

    # Trade returns
    direction = np.where(y_prob[trade_mask] > 0.5, 1.0, -1.0)
    signal = np.array([get_signal(p) for p in y_prob[trade_mask]])
    signal_disc = np.round(signal / 0.1) * 0.1
    actual = fwd_ret.reindex(test_times[trade_mask]).values

    # Weighted returns (probability sizing)
    trade_rets = direction * actual * np.abs(signal_disc).clip(0.1, 2.0) - SLIPPAGE
    trade_rets = trade_rets[np.isfinite(trade_rets)]

    return trade_rets


def run_cpcv_for_asset(symbol):
    """Run full CPCV for one asset.

    AFML Ch 12: N=6 groups, k=2 test -> C(6,2) = 15 combinations.
    Each generates a backtest path. Returns Sharpe distribution.
    """
    features = _full_features[symbol]
    fwd_ret = _fwd_returns[symbol]

    fc = features.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    n_obs = len(valid)

    if n_obs < 500:
        return None

    # Split into N chronological groups
    group_size = n_obs // N_GROUPS
    embargo_size = int(n_obs * EMBARGO_PCT)

    groups = []
    for i in range(N_GROUPS):
        start = i * group_size
        end = start + group_size if i < N_GROUPS - 1 else n_obs
        groups.append(np.arange(start, end))

    # Generate all C(N, K_TEST) combinations
    combos = list(itertools.combinations(range(N_GROUPS), K_TEST))
    print(f"  {symbol}: {len(combos)} CPCV splits, {n_obs} obs, "
          f"{group_size} per group, embargo={embargo_size}", flush=True)

    path_sharpes = []
    path_returns = []
    path_accuracies = []

    for combo in combos:
        # Test groups
        test_idx = np.concatenate([groups[g] for g in combo])
        # Train groups (everything else)
        train_groups = [g for g in range(N_GROUPS) if g not in combo]
        train_idx = np.concatenate([groups[g] for g in train_groups])

        trade_rets = run_cpcv_split(symbol, train_idx, test_idx, embargo_size)

        if len(trade_rets) < 10:
            continue

        # Per-path Sharpe
        mean_ret = trade_rets.mean()
        std_ret = trade_rets.std()
        if std_ret > 0:
            # Annualize: assume ~12 trades/day at 30m bars
            trades_per_day = len(trade_rets) / max((len(test_idx) / (24 * 2)), 1)
            sharpe = mean_ret / std_ret * np.sqrt(trades_per_day * 365.25)
        else:
            sharpe = 0

        accuracy = (trade_rets > 0).mean()

        path_sharpes.append(sharpe)
        path_returns.append(mean_ret * 10000)  # bps
        path_accuracies.append(accuracy)

    if not path_sharpes:
        return None

    return {
        "symbol": symbol,
        "n_paths": len(path_sharpes),
        "n_combos": len(combos),
        "sharpe_mean": round(np.mean(path_sharpes), 3),
        "sharpe_std": round(np.std(path_sharpes), 3),
        "sharpe_min": round(np.min(path_sharpes), 3),
        "sharpe_max": round(np.max(path_sharpes), 3),
        "sharpe_median": round(np.median(path_sharpes), 3),
        "pct_positive_sharpe": round((np.array(path_sharpes) > 0).mean() * 100, 1),
        "return_mean_bps": round(np.mean(path_returns), 2),
        "return_std_bps": round(np.std(path_returns), 2),
        "accuracy_mean": round(np.mean(path_accuracies), 4),
        "accuracy_std": round(np.std(path_accuracies), 4),
        "all_sharpes": [round(s, 3) for s in path_sharpes],
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90, flush=True)
    print("CPCV -- Combinatorial Purged Cross-Validation (AFML Ch 12)", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"N={N_GROUPS} groups, k={K_TEST} test -> C({N_GROUPS},{K_TEST}) = "
          f"{len(list(itertools.combinations(range(N_GROUPS), K_TEST)))} paths per asset", flush=True)
    print(f"Embargo: {EMBARGO_PCT*100:.0f}% | Vol filter: top {(1-VOL_FILTER_QUANTILE)*100:.0f}% | "
          f"Conf filter: top {100-CONF_PERCENTILE}%", flush=True)
    print(f"Model: v4 Tier 2 (expanded features + prob sizing)", flush=True)
    print("=" * 90, flush=True)

    # Phase 0: Load + precompute
    print("\n[1/3] Loading data + precomputing features...", flush=True)
    t0 = time.time()
    for s in ASSETS:
        load_ohlcv(s)
        load_hl(s)
    precompute_features()
    print(f"  Precompute done in {time.time()-t0:.1f}s", flush=True)

    # Phase 1: Run CPCV per asset
    print(f"\n[2/3] Running CPCV ({N_GROUPS} groups, {K_TEST} test)...", flush=True)
    results = []
    for symbol in ASSETS:
        t_sym = time.time()
        r = run_cpcv_for_asset(symbol)
        if r:
            results.append(r)
            print(f"    Sharpe: mean={r['sharpe_mean']:.3f} +/-{r['sharpe_std']:.3f} "
                  f"[{r['sharpe_min']:.3f}, {r['sharpe_max']:.3f}] "
                  f"({r['pct_positive_sharpe']:.0f}% positive) "
                  f"({time.time()-t_sym:.1f}s)", flush=True)
        else:
            print(f"    {symbol}: insufficient data for CPCV", flush=True)

    # Phase 2: Aggregate and report
    print(f"\n[3/3] Results...", flush=True)
    print("=" * 100, flush=True)
    print("CPCV RESULTS -- Distribution of Sharpe Ratios Across 15 Backtest Paths", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Asset':<6} {'Paths':>5} {'Mean':>7} {'Std':>6} {'Min':>7} {'Max':>7} "
          f"{'Median':>7} {'%>0':>5} {'MeanRet':>8} {'MeanAcc':>8}", flush=True)
    print("-" * 80, flush=True)

    all_sharpes = []
    for r in results:
        all_sharpes.extend(r["all_sharpes"])
        print(f"{r['symbol']:<6} {r['n_paths']:>5} {r['sharpe_mean']:>7.3f} "
              f"{r['sharpe_std']:>6.3f} {r['sharpe_min']:>7.3f} {r['sharpe_max']:>7.3f} "
              f"{r['sharpe_median']:>7.3f} {r['pct_positive_sharpe']:>4.0f}% "
              f"{r['return_mean_bps']:>+7.2f}bp {r['accuracy_mean']:>7.4f}", flush=True)

    # Portfolio aggregate
    if all_sharpes:
        print(f"\n{'ALL':<6} {len(all_sharpes):>5} {np.mean(all_sharpes):>7.3f} "
              f"{np.std(all_sharpes):>6.3f} {np.min(all_sharpes):>7.3f} "
              f"{np.max(all_sharpes):>7.3f} {np.median(all_sharpes):>7.3f} "
              f"{(np.array(all_sharpes)>0).mean()*100:>4.0f}%", flush=True)

    # Interpretation
    print(f"\n--- INTERPRETATION (AFML Ch 12) ---", flush=True)
    if all_sharpes:
        pct_pos = (np.array(all_sharpes) > 0).mean() * 100
        if pct_pos >= 80:
            print(f"  {pct_pos:.0f}% of paths have positive Sharpe -> STRONG evidence of real edge", flush=True)
        elif pct_pos >= 60:
            print(f"  {pct_pos:.0f}% of paths have positive Sharpe -> MODERATE evidence (edge exists but noisy)", flush=True)
        elif pct_pos >= 50:
            print(f"  {pct_pos:.0f}% of paths have positive Sharpe -> WEAK evidence (barely above random)", flush=True)
        else:
            print(f"  {pct_pos:.0f}% of paths have positive Sharpe -> NO evidence of edge", flush=True)

        print(f"  Mean Sharpe across all paths: {np.mean(all_sharpes):.3f}", flush=True)
        print(f"  Worst-case path Sharpe: {np.min(all_sharpes):.3f}", flush=True)
        print(f"  This is MORE robust than single walk-forward (which tested ONE path)", flush=True)

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {"n_groups": N_GROUPS, "k_test": K_TEST, "embargo_pct": EMBARGO_PCT},
        "per_asset": results,
        "aggregate": {
            "all_sharpes": [round(s, 3) for s in all_sharpes],
            "mean": round(np.mean(all_sharpes), 3) if all_sharpes else None,
            "std": round(np.std(all_sharpes), 3) if all_sharpes else None,
            "pct_positive": round((np.array(all_sharpes) > 0).mean() * 100, 1) if all_sharpes else None,
        },
    }
    with open(OUTPUT_DIR / "cpcv_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'cpcv_results.json'}", flush=True)
    print(f"Completed: {datetime.now()} ({time.time()-t0:.0f}s total)", flush=True)
