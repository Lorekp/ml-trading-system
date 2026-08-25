"""Tier 0A: Feature Importance via MDA (AFML Ch8).

Mean Decrease Accuracy with purged K-fold on stacked 30-asset data.
Permutes each feature and measures OOS accuracy drop.
Fully parallelized — data loading AND MDA computation use all cores.
"""
import sys
sys.path.insert(0, "src")

import os
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _load_asset(args):
    """Load and prepare one asset's data (parallelized)."""
    asset, symbol, lookback_months = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, None

    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)

    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 500:
        return asset, None

    end = valid[-1]
    start = end - pd.DateOffset(months=lookback_months)
    mask = (valid >= start) & (valid <= end)
    tr = valid[mask]
    if len(tr) < 200:
        return asset, None

    sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
    return asset, {
        "X": fc.loc[tr].values,
        "y": (y_all.loc[tr] > 0).astype(int).values,
        "sw": sw,
        "n": len(tr),
    }


def _mda_single_feature(args):
    """Compute MDA for a single feature (parallelized across features)."""
    fi, feat_name, X_arr, y_arr, sw, fold_indices, n_permutations = args
    warnings.filterwarnings("ignore")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def _make_lr():
        try:
            return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
        except Exception:
            return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)

    # Baseline scores per fold
    baseline_scores = []
    for train_idx, test_idx in fold_indices:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_arr[train_idx])
        X_test = scaler.transform(X_arr[test_idx])
        model = _make_lr()
        model.fit(X_train, y_arr[train_idx], sample_weight=sw[train_idx])
        baseline_scores.append(model.score(X_test, y_arr[test_idx], sample_weight=sw[test_idx]))

    # Permuted scores
    all_drops = []
    for perm in range(n_permutations):
        fold_drops = []
        for fold_i, (train_idx, test_idx) in enumerate(fold_indices):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_arr[train_idx])
            X_test = scaler.transform(X_arr[test_idx])

            model = _make_lr()
            model.fit(X_train, y_arr[train_idx], sample_weight=sw[train_idx])

            X_test_perm = X_test.copy()
            rng = np.random.RandomState(42 + perm * 100 + fi)
            X_test_perm[:, fi] = rng.permutation(X_test_perm[:, fi])

            perm_score = model.score(X_test_perm, y_arr[test_idx], sample_weight=sw[test_idx])
            fold_drops.append(baseline_scores[fold_i] - perm_score)

        all_drops.append(np.mean(fold_drops))

    return feat_name, {
        "mean_drop": float(np.mean(all_drops)),
        "std_drop": float(np.std(all_drops)),
        "baseline_acc": float(np.mean(baseline_scores)),
    }


def main():
    print("=" * 70)
    print("TIER 0A: FEATURE IMPORTANCE (MDA, AFML Ch8)")
    print(f"  Purged K-fold, 5 permutations, {W} workers")
    print("  30 assets, 8-month training window")
    print("=" * 70)

    t0 = time.time()

    # Phase 1: Load data in parallel
    print("\n[1] Loading data from 30 assets (parallel)...", flush=True)
    all_X, all_y, all_sw = [], [], []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_load_asset, (a, s, 8)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, result = f.result()
            done += 1
            if result is not None:
                all_X.append(result["X"])
                all_y.append(result["y"])
                all_sw.append(result["sw"])
                print(f"  [{done:>2d}/30] {asset}: {result['n']:,d} bars", flush=True)
            else:
                print(f"  [{done:>2d}/30] {asset}: skipped", flush=True)

    X_arr = np.vstack(all_X)
    y_arr = np.concatenate(all_y)
    sw = np.concatenate(all_sw)
    cfg = load_config(Path("configs/default.yaml"))
    feature_names = cfg.features.feature_list
    print(f"  Total: {len(X_arr):,d} samples, {len(feature_names)} features", flush=True)

    # Phase 2: Prepare purged K-fold indices
    print("\n[2] Computing MDA (parallel across features)...", flush=True)
    n_splits = 5
    purge_bars = 5
    n_permutations = 5
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_indices = []
    for train_idx, test_idx in kf.split(X_arr):
        if len(train_idx) > purge_bars:
            train_idx = train_idx[:-purge_bars]
        fold_indices.append((train_idx, test_idx))

    # Phase 3: Run MDA in parallel (one process per feature)
    importances = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {}
        for fi, feat_name in enumerate(feature_names):
            args = (fi, feat_name, X_arr, y_arr, sw, fold_indices, n_permutations)
            futures[pool.submit(_mda_single_feature, args)] = feat_name

        for f in as_completed(futures):
            feat_name, result = f.result()
            importances[feat_name] = result
            drop = result["mean_drop"]
            print(f"  {feat_name}: MDA drop = {drop:+.5f}", flush=True)

    # Phase 4: Report
    baseline_acc = next(iter(importances.values()))["baseline_acc"]
    print(f"\n{'='*70}")
    print(f"RESULTS (baseline OOS accuracy: {baseline_acc:.4f})")
    print(f"{'='*70}")
    print(f"\n  {'Rank':<5s} {'Feature':<22s} {'MDA (drop)':>10s} {'Std':>8s} {'Verdict':>10s}")
    print(f"  {'-'*57}")

    sorted_feats = sorted(importances.items(), key=lambda x: -x[1]["mean_drop"])
    for rank, (feat_name, vals) in enumerate(sorted_feats, 1):
        drop = vals["mean_drop"]
        std = vals["std_drop"]
        if drop > 2 * std and drop > 0.001:
            verdict = "IMPORTANT"
        elif drop > std and drop > 0.0005:
            verdict = "useful"
        elif drop < -std:
            verdict = "HARMFUL?"
        else:
            verdict = "noise"
        print(f"  {rank:<5d} {feat_name:<22s} {drop:>10.5f} {std:>8.5f} {verdict:>10s}")

    # L1 coefficients for comparison
    print(f"\n  L1 coefficient magnitudes:")
    print(f"  {'Feature':<22s} {'|Coef|':>10s}")
    print(f"  {'-'*34}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    model = make_logreg()
    model.fit(X_scaled, y_arr, sample_weight=sw)
    coefs = np.abs(model.coef_[0])
    for idx in np.argsort(-coefs):
        print(f"  {feature_names[idx]:<22s} {coefs[idx]:>10.5f}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
