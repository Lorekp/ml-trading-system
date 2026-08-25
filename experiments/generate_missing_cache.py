"""Generate missing signal cache for undersampled (C, lb) combos.

Parallelized with ProcessPoolExecutor. Each (C, lb, window) task
takes ~90s. With 4 workers, total ~15 min for 60 new combos.
"""

import sys, os, time, warnings, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

CACHE_DIR = Path("data/optuna/signal_cache")
WINDOWS = [
    {"label": "W1", "test_start": "2024-07", "test_end": "2025-02"},
    {"label": "W2", "test_start": "2025-03", "test_end": "2025-10"},
    {"label": "W3", "test_start": "2025-11", "test_end": "2026-02"},
]

# C values we want for every lookback
TARGET_C = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
TARGET_LB = [6, 8, 10, 12, 14, 16]


def cache_key(C, lb, wl):
    return f"C{C:.5f}_lb{lb}_{wl}"


def needs_generation():
    """Find (C, lb, window) combos that aren't cached yet."""
    tasks = []
    for C in TARGET_C:
        for lb in TARGET_LB:
            for w in WINDOWS:
                key = cache_key(C, lb, w["label"])
                if not (CACHE_DIR / f"{key}.pkl").exists():
                    tasks.append((C, lb, w))
    return tasks


def generate_one(args):
    """Generate signals for one (C, lb, window) combo."""
    C, lb, window = args
    import sys, os, warnings, pickle
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
    warnings.filterwarnings("ignore")

    import numpy as np, pandas as pd, sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.asset_selection import compute_monthly_dollar_volume
    from afml.bet_sizing import get_signal, discretize_signal

    NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
    DATA_DIR = Path("data/binance-1m")
    TOP_N = 30; MAX_BARS = 8
    cfg = load_config(Path("configs/default.yaml"))

    def make_logreg(c):
        if NEW_SKLEARN:
            return LogisticRegression(C=c, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
        return LogisticRegression(C=c, penalty="l1", solver="saga", max_iter=1000, random_state=42)

    # Load data
    asset_data = {}; asset_volumes = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"): continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000: continue
            close = ohlcv_15m["close"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 3000: continue
            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
            asset_data[symbol] = {
                "close": close, "close_arr": ohlcv_15m["close"].values.astype(np.float64),
                "fc": fc, "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "timestamps": ohlcv_15m.index,
                "start_date": ohlcv_15m.index.min(), "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception: pass

    sample_idx = next(iter(asset_volumes.values())).index
    test_start = pd.Timestamp(window["test_start"] + "-01")
    end_date = max(d["end_date"] for d in asset_data.values())
    test_end = pd.Timestamp(window["test_end"] + "-28")
    if sample_idx.tz is not None:
        test_start = test_start.tz_localize(sample_idx.tz)
        test_end = test_end.tz_localize(sample_idx.tz)
    test_end = min(test_end, end_date)
    test_months = pd.period_range(start=test_start.to_period("M"), end=test_end.to_period("M"), freq="M")

    all_signals = []
    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None: ms = ms.tz_localize(sample_idx.tz)
        lookback_end = ms - pd.Timedelta(days=1)
        vol_candidates = {}
        for symbol, vol_s in asset_volumes.items():
            avail = vol_s.loc[:lookback_end]
            if len(avail) < 45: continue
            latest = avail.iloc[-1]
            if np.isnan(latest) or latest <= 0: continue
            if len(avail) >= 90:
                m90 = avail.iloc[-90:].median()
                if m90 > 0 and latest / m90 > 5.0: continue
            vol_candidates[symbol] = latest
        selected = [s for s, _ in sorted(vol_candidates.items(), key=lambda x: -x[1])[:TOP_N]]

        for symbol in selected:
            if symbol not in asset_data: continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]; n = len(close_arr)
            fc = ad["fc"]; y_all = ad["y_all"]; valid = ad["valid"]
            vol = ad["vol"]; timestamps_idx = ad["timestamps"]
            ms_local = ms; me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None); me_local = me_local.tz_localize(None)
            te_all = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vol_quantiles = {q: vol_train.quantile(q) for q in [0.0, 0.25, 0.40, 0.50, 0.60, 0.80]}
            test_vol = vol.reindex(te_all)
            train_start = ms_local - pd.DateOffset(months=lb)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200: continue
            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg(C)
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te_all])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception: continue
            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            asset_name = symbol.replace("USDT", "")
            for i in range(len(te_all)):
                bar_idx = timestamps_idx.get_loc(te_all[i])
                if bar_idx + MAX_BARS >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]
                rets = {k: d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, MAX_BARS + 1)}
                vol_val = float(test_vol.get(te_all[i], np.nan))
                all_signals.append({
                    "timestamp": te_all[i], "direction": d, "confidence": conf[i],
                    "probability": yp[i], "signal_strength": float(ss[i]),
                    "asset": asset_name, "vol_val": vol_val,
                    **{f"ret_{k}": rets[k] for k in range(1, MAX_BARS + 1)},
                    **{f"vol_q_{q}": vol_quantiles[q] for q in vol_quantiles},
                })

    df = pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True) if all_signals else pd.DataFrame()
    key = cache_key(C, lb, window["label"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
        pickle.dump(df, f)
    return key, len(df)


def main():
    t0 = time.time()
    tasks = needs_generation()
    # Deduplicate: each (C, lb) needs data loaded once, generates 3 windows
    unique_combos = set((C, lb) for C, lb, _ in tasks)

    print(f"Missing cache entries: {len(tasks)} tasks ({len(unique_combos)} unique C x lb combos)")
    print(f"Target: {len(TARGET_C)} C values x {len(TARGET_LB)} lookbacks x 3 windows = {len(TARGET_C)*len(TARGET_LB)*3}")
    print(f"Already cached: {len(TARGET_C)*len(TARGET_LB)*3 - len(tasks)}")

    if not tasks:
        print("Nothing to generate!")
        return

    N_WORKERS = min(8, os.cpu_count() or 4)
    print(f"\nGenerating with {N_WORKERS} workers...", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(generate_one, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                key, n_signals = future.result()
                if done % 5 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] {key}: {n_signals} signals", flush=True)
            except Exception as e:
                C, lb, w = futures[future]
                print(f"  [{done}/{len(tasks)}] C={C} lb={lb} {w['label']}: ERROR {e}", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
