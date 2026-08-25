"""
Direction V2 — Per-Timeframe Hyperparameter Grid Search.

For each promising timeframe (15m, 30m, 1H, 4H), search over:
  - XGBoost params (depth, lr, n_estimators, regularization)
  - CUSUM multiplier (event sampling rate)
  - Confidence threshold (top X%)

Uses PurgedKFold CV to select best params, then evaluates on held-out test.
Tracks all trials for DSR computation.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from scipy.stats import binom
from itertools import product
from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_DIR = Path("/home/user/ml-data/binance-1m")
OUTPUT_DIR = Path("/home/user/ml-data/v2-results")
TRAIN_CUTOFF = "2025-07-01"
RT_COST_BPS = 4.0

# Timeframes to search (focus on economically viable ones)
TIMEFRAMES = [
    ("15m", "15min"),
    ("30m", "30min"),
    ("1H",  "1h"),
    ("4H",  "4h"),
]

# Hyperparameter search space
PARAM_GRID = {
    "max_depth":        [2, 3, 4, 5, 6],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "n_estimators":     [50, 100, 200, 300, 500],
    "min_child_weight": [5, 10, 20, 50, 100],
    "subsample":        [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
}

# CUSUM multipliers to search
CUSUM_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0, 5.0]

# Number of random param combinations per (timeframe, cusum_mult)
N_RANDOM_SAMPLES = 30

# PurgedKFold settings
N_SPLITS = 5
PCT_EMBARGO = 0.01


# =============================================================================
# DATA LOADING + FEATURES (same as previous scripts)
# =============================================================================
def load_btc(tf_resample):
    btc_dir = DATA_DIR / "BTCUSDT"
    dfs = []
    for f in sorted(btc_dir.glob("*.csv")):
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
    return combined.resample(tf_resample).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


def build_features(ohlcv):
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    train_mask = ohlcv.index < TRAIN_CUTOFF
    features = pd.DataFrame(index=ohlcv.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo = ret[train_mask].quantile(0.01)
        hi = ret[train_mask].quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)
    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (
        (volume - volume.rolling(30).mean()) /
        volume.rolling(30).std().replace(0, np.nan)
    )
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    features["macd_hist"] = macd(close)["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]
    return features


# =============================================================================
# CUSUM + SAMPLE WEIGHTS (inlined from AFML)
# =============================================================================
def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


def cusum_filter(raw_series, threshold):
    events, s_pos, s_neg = [], 0.0, 0.0
    diff = raw_series.diff().dropna()
    for t in diff.index:
        if isinstance(threshold, pd.Series):
            h = threshold.get(t, None)
            if h is None or np.isnan(h):
                continue
        else:
            h = float(threshold)
        s_pos = max(0.0, s_pos + diff.loc[t])
        s_neg = min(0.0, s_neg + diff.loc[t])
        if s_neg < -h:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            events.append(t)
    return pd.DatetimeIndex(events)


def get_num_concurrent(close_idx, t1):
    t1 = t1.fillna(close_idx[-1])
    iloc_range = close_idx.searchsorted([t1.index.min(), t1.max()])
    count = pd.Series(0, index=close_idx[iloc_range[0]:iloc_range[1] + 1], dtype=float)
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.0
    return count


def get_avg_uniqueness(t1, num_conc):
    weights = pd.Series(index=t1.index, dtype=float)
    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            continue
        sl = num_conc.loc[t_in:t_out]
        weights.loc[t_in] = (1.0 / sl).mean() if len(sl) > 0 else 0.0
    return weights


def compute_sample_weights(t1, close, clf_last_w=0.5):
    num_conc = get_num_concurrent(close.index, t1)
    uniqueness = get_avg_uniqueness(t1, num_conc)
    # Time decay
    cum_w = uniqueness.sort_index().cumsum()
    if cum_w.iloc[-1] > 0:
        slope = (1.0 - clf_last_w) / cum_w.iloc[-1]
        const = 1.0 - slope * cum_w.iloc[-1]
        decay = const + slope * cum_w
        decay[decay < 0] = 0.0
    else:
        decay = pd.Series(1.0, index=uniqueness.index)
    combined = uniqueness * decay
    if combined.sum() > 0:
        combined = combined / combined.mean()
    else:
        combined = pd.Series(1.0, index=t1.index)
    return combined


# =============================================================================
# PURGED K-FOLD CV (inlined from AFML Ch 7)
# =============================================================================
def purged_kfold_cv(X, y, t1, sample_weights, xgb_params, n_splits=5, pct_embargo=0.01):
    """Run PurgedKFold CV and return mean accuracy across folds."""
    n = len(X)
    embargo_size = int(n * pct_embargo)
    fold_boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    fold_accs = []

    for i in range(n_splits):
        test_start = fold_boundaries[i]
        test_end = fold_boundaries[i + 1]
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) == 0:
            continue

        # Build train set: purge overlapping labels + embargo
        if isinstance(X.index, pd.DatetimeIndex):
            t_test_start = X.index[test_start]
            t_test_end = X.index[min(test_end - 1, n - 1)]
            train_idx = []
            for j in range(n):
                if j in test_idx:
                    continue
                if test_end <= j < test_end + embargo_size:
                    continue
                obs_start = X.index[j]
                obs_end = t1.iloc[j] if j < len(t1) else obs_start
                if pd.isna(obs_end):
                    obs_end = obs_start
                overlaps = (
                    (t_test_start <= obs_start <= t_test_end)
                    or (t_test_start <= obs_end <= t_test_end)
                    or (obs_start <= t_test_start and obs_end >= t_test_end)
                )
                if overlaps:
                    continue
                train_idx.append(j)
            train_idx = np.array(train_idx)
        else:
            train_idx = np.concatenate([
                np.arange(0, test_start),
                np.arange(min(test_end + embargo_size, n), n)
            ])

        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        w_tr = sample_weights.iloc[train_idx].values if sample_weights is not None else None
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        model = xgb.XGBClassifier(**xgb_params)
        fit_kw = {}
        if w_tr is not None:
            fit_kw["sample_weight"] = w_tr
        model.fit(X_tr, y_tr, **fit_kw)

        preds = model.predict(X_te)
        fold_accs.append((preds == y_te.values).mean())

    return np.mean(fold_accs) if fold_accs else 0.0


# =============================================================================
# RANDOM SEARCH
# =============================================================================
def random_xgb_params(rng):
    """Sample one random XGBoost parameter combination."""
    return {
        "max_depth": rng.choice(PARAM_GRID["max_depth"]),
        "learning_rate": rng.choice(PARAM_GRID["learning_rate"]),
        "n_estimators": int(rng.choice(PARAM_GRID["n_estimators"])),
        "min_child_weight": int(rng.choice(PARAM_GRID["min_child_weight"])),
        "subsample": rng.choice(PARAM_GRID["subsample"]),
        "colsample_bytree": rng.choice(PARAM_GRID["colsample_bytree"]),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }


def search_timeframe(tf_label, tf_resample):
    """Full hyperparameter + CUSUM multiplier search for one timeframe."""
    print(f"\n{'='*80}", flush=True)
    print(f"SEARCHING: {tf_label}", flush=True)
    print(f"{'='*80}", flush=True)

    ohlcv = load_btc(tf_resample)
    features = build_features(ohlcv)
    close = ohlcv["close"]
    vol = get_daily_vol(close, span=50)

    # Mean abs return for this timeframe
    test_ret = close.pct_change(1)
    test_ret_test = test_ret[test_ret.index >= TRAIN_CUTOFF]
    mean_abs_ret_bps = test_ret_test.abs().mean() * 10000
    breakeven = 0.5 + RT_COST_BPS / (2 * mean_abs_ret_bps)
    print(f"  Bars: {len(ohlcv)}, MeanAbsRet: {mean_abs_ret_bps:.1f} bps, Breakeven: {breakeven*100:.2f}%", flush=True)

    rng = np.random.RandomState(42)
    all_trials = []
    best_result = None
    best_cv_acc = 0.0

    for cusum_mult in CUSUM_MULTIPLIERS:
        # CUSUM events
        threshold = vol * close * cusum_mult
        cusum_events = cusum_filter(close, threshold)
        event_rate = len(cusum_events) / len(ohlcv)

        # Labels
        fwd_ret = close.pct_change(1).shift(-1)
        features_clean = features.dropna()
        valid_events = cusum_events.intersection(features_clean.index)
        y_all = fwd_ret.reindex(valid_events).dropna()
        valid_events = y_all.index
        y_binary = (y_all > 0).astype(int)

        train_mask = valid_events < TRAIN_CUTOFF
        test_mask = valid_events >= TRAIN_CUTOFF
        events_train = valid_events[train_mask]
        events_test = valid_events[test_mask]

        n_train = len(events_train)
        n_test = len(events_test)

        if n_train < 100 or n_test < 50:
            print(f"  CUSUM mult={cusum_mult}: {n_train} train, {n_test} test — SKIP (too few)", flush=True)
            continue

        print(f"  CUSUM mult={cusum_mult}: {n_train} train, {n_test} test, rate={event_rate:.3f}", flush=True)

        # Sample weights
        # t1 for 1-bar label: event + 1 bar
        freq_map = {"5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240}
        bar_minutes = freq_map.get(tf_resample, 15)
        t1_train = pd.Series(
            events_train + pd.Timedelta(minutes=bar_minutes),
            index=events_train,
        ).clip(upper=close.index[-1])
        weights_train = compute_sample_weights(t1_train, close, clf_last_w=0.5)

        X_train = features_clean.loc[events_train]
        y_train = y_binary.loc[events_train]
        X_test = features_clean.loc[events_test]
        y_test = y_binary.loc[events_test]

        # Random search over XGBoost params
        for trial_i in range(N_RANDOM_SAMPLES):
            params = random_xgb_params(rng)

            # PurgedKFold CV on train
            # t1_train already has the right index; reindex to match X_train
            t1_for_cv = t1_train.reindex(X_train.index)
            # Fill any missing t1 with event_time + 1 bar
            missing = t1_for_cv.isna()
            if missing.any():
                t1_for_cv[missing] = t1_for_cv.index[missing] + pd.Timedelta(minutes=bar_minutes)
            cv_acc = purged_kfold_cv(X_train, y_train, t1_for_cv, weights_train, params, N_SPLITS, PCT_EMBARGO)

            trial = {
                "tf": tf_label,
                "cusum_mult": cusum_mult,
                "params": {k: v for k, v in params.items() if k not in ("objective", "eval_metric", "tree_method", "random_state")},
                "cv_acc": round(float(cv_acc), 4),
                "n_train": n_train,
                "n_test": n_test,
            }
            all_trials.append(trial)

            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_result = {
                    "params": params,
                    "cusum_mult": cusum_mult,
                    "cv_acc": cv_acc,
                    "X_train": X_train, "y_train": y_train,
                    "X_test": X_test, "y_test": y_test,
                    "weights_train": weights_train,
                    "n_train": n_train, "n_test": n_test,
                }

        print(f"    Best CV so far: {best_cv_acc:.4f}", flush=True)

    if best_result is None:
        print(f"  NO valid configuration found for {tf_label}", flush=True)
        return None, all_trials

    # Evaluate best on held-out test
    print(f"\n  BEST: cusum_mult={best_result['cusum_mult']}, CV acc={best_result['cv_acc']:.4f}", flush=True)
    print(f"  Params: { {k:v for k,v in best_result['params'].items() if k not in ('objective','eval_metric','tree_method','random_state')} }", flush=True)

    model = xgb.XGBClassifier(**best_result["params"])
    model.fit(best_result["X_train"], best_result["y_train"],
              sample_weight=best_result["weights_train"].values)

    y_prob = model.predict_proba(best_result["X_test"])[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    y_test_vals = best_result["y_test"].values

    acc_all = float((y_pred == y_test_vals).mean())
    k = int((y_pred == y_test_vals).sum())
    n_test = len(y_test_vals)
    p_val = 1 - binom.cdf(k - 1, n_test, 0.5) if k > n_test / 2 else 1.0

    # Confidence filtered
    conf = np.abs(y_prob - 0.5)
    filtered = {}
    for label, pct_thresh in [("top30", 70), ("top20", 80), ("top10", 90), ("top5", 95)]:
        mask = conf >= np.percentile(conf, pct_thresh)
        n_f = int(mask.sum())
        if n_f >= 10:
            a = float((y_pred[mask] == y_test_vals[mask]).mean())
            edge = (a - 0.5) * 2 * mean_abs_ret_bps
            net = edge - RT_COST_BPS
            filtered[label] = {"acc": round(a, 4), "n": n_f, "edge": round(edge, 2), "net": round(net, 2)}
        else:
            filtered[label] = {"acc": None, "n": n_f, "edge": None, "net": None}

    edge_all = (acc_all - 0.5) * 2 * mean_abs_ret_bps
    net_all = edge_all - RT_COST_BPS

    result = {
        "tf": tf_label,
        "mean_abs_ret_bps": round(mean_abs_ret_bps, 2),
        "breakeven_acc": round(breakeven, 4),
        "best_cusum_mult": best_result["cusum_mult"],
        "best_cv_acc": round(best_result["cv_acc"], 4),
        "best_params": {k: v for k, v in best_result["params"].items()
                        if k not in ("objective", "eval_metric", "tree_method", "random_state")},
        "n_train": best_result["n_train"],
        "n_test": n_test,
        "acc_all": round(acc_all, 4),
        "p_value": float(p_val),
        "edge_all_bps": round(edge_all, 2),
        "net_all_bps": round(net_all, 2),
        "filtered": filtered,
        "n_trials": len(all_trials),
    }

    # Print results
    print(f"\n  TEST RESULTS ({tf_label}):", flush=True)
    print(f"    Accuracy (all):  {acc_all:.4f} (p={p_val:.4f}), edge={edge_all:.2f}, net={net_all:+.2f} bps", flush=True)
    for label, f in filtered.items():
        if f["acc"] is not None:
            prof = "PROFITABLE" if f["net"] > 0 else ""
            print(f"    Accuracy ({label:>5}): {f['acc']:.4f} (n={f['n']}), edge={f['edge']:.2f}, net={f['net']:+.2f} bps {prof}", flush=True)

    return result, all_trials


def main():
    print("=" * 80, flush=True)
    print("DIRECTION V2 — PER-TIMEFRAME HYPERPARAMETER GRID SEARCH", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Search: {N_RANDOM_SAMPLES} random XGB params x {len(CUSUM_MULTIPLIERS)} CUSUM mults per timeframe", flush=True)
    print(f"CV: PurgedKFold {N_SPLITS} splits, {PCT_EMBARGO*100:.0f}% embargo", flush=True)
    print("=" * 80, flush=True)

    all_results = []
    total_trials = []

    for tf_label, tf_resample in TIMEFRAMES:
        t0 = datetime.now()
        result, trials = search_timeframe(tf_label, tf_resample)
        elapsed = datetime.now() - t0
        print(f"  {tf_label} search took {elapsed}", flush=True)
        if result:
            all_results.append(result)
        total_trials.extend(trials)

    # Summary
    print("\n" + "=" * 100, flush=True)
    print("GRID SEARCH SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"Total trials: {len(total_trials)}", flush=True)
    print(f"\n{'TF':>4} {'MAbsRet':>8} {'BEAcc':>7} {'BestCV':>7} {'AccAll':>7} {'Net':>7} | "
          f"{'Acc30%':>7} {'Net30%':>7} {'Acc10%':>7} {'Net10%':>7} | {'p-val':>9} {'CusumM':>7}", flush=True)
    print("-" * 100, flush=True)

    for r in all_results:
        f30 = r["filtered"].get("top30", {})
        f10 = r["filtered"].get("top10", {})
        acc30 = f"{f30.get('acc', 0):.4f}" if f30.get("acc") else "    N/A"
        net30 = f"{f30.get('net', 0):+.2f}" if f30.get("net") is not None else "    N/A"
        acc10 = f"{f10.get('acc', 0):.4f}" if f10.get("acc") else "    N/A"
        net10 = f"{f10.get('net', 0):+.2f}" if f10.get("net") is not None else "    N/A"
        p_str = f"{r['p_value']:.2e}" if r['p_value'] < 0.01 else f"{r['p_value']:.4f}"
        print(f"{r['tf']:>4} {r['mean_abs_ret_bps']:>7.1f}bp {r['breakeven_acc']*100:>6.2f}% "
              f"{r['best_cv_acc']:>6.4f} {r['acc_all']:>6.4f} {r['net_all_bps']:>+6.2f} | "
              f"{acc30:>7} {net30:>7} {acc10:>7} {net10:>7} | {p_str:>9} {r['best_cusum_mult']:>7.1f}", flush=True)

    # Best params per timeframe
    print("\n--- BEST PARAMETERS PER TIMEFRAME ---", flush=True)
    for r in all_results:
        print(f"  {r['tf']}: cusum_mult={r['best_cusum_mult']}, {r['best_params']}", flush=True)

    # Profitability check
    print("\n--- PROFITABILITY CHECK ---", flush=True)
    any_profitable = False
    for r in all_results:
        for label in ["top30", "top20", "top10", "top5"]:
            f = r["filtered"].get(label, {})
            if f.get("net") is not None and f["net"] > 0:
                print(f"  PROFITABLE: {r['tf']} {label} = +{f['net']:.2f} bps "
                      f"(acc={f['acc']:.4f}, n={f['n']})", flush=True)
                any_profitable = True
        if r["net_all_bps"] > 0:
            print(f"  PROFITABLE: {r['tf']} ALL = +{r['net_all_bps']:.2f} bps "
                  f"(acc={r['acc_all']:.4f}, n={r['n_test']})", flush=True)
            any_profitable = True
    if not any_profitable:
        print("  No configuration is net-profitable after grid search.", flush=True)

    # DSR check
    n_total_trials = len(total_trials)
    print(f"\n  Total trials tested: {n_total_trials} (for DSR correction)", flush=True)

    # Save
    output_path = OUTPUT_DIR / "direction_v2_gridsearch.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "n_total_trials": n_total_trials,
            "search_config": {
                "n_random_samples": N_RANDOM_SAMPLES,
                "cusum_multipliers": CUSUM_MULTIPLIERS,
                "param_grid": {k: [float(x) if isinstance(x, (np.floating,)) else x for x in v]
                               for k, v in PARAM_GRID.items()},
            },
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}", flush=True)
    print(f"Completed: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
