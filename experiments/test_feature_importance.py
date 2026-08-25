"""Per-feature ablation test — PARALLELIZED.

Shuffles one feature at a time and measures Sharpe drop.
Runs all ablations in parallel across CPU cores.
"""

import sys
sys.path.insert(0, "src")

import os
os.environ["ML_CLOSE_SKIP_CV"] = "1"  # Skip PurgedKFold for speed

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LB = 8
SEARCH_START = "2022-01"
SEARCH_END = "2024-12"


def trade_sharpe(df):
    if df is None or len(df) < 20:
        return 0.0
    rets = df["direction"].values * df["fwd_return"].values
    s = rets.std()
    return float(rets.mean() / s * np.sqrt(365.25 * 6)) if s > 0 else 0.0


def period_filter(df, start, end=None):
    months = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m")
    mask = months >= start
    if end:
        mask &= months <= end
    return df.loc[mask].reset_index(drop=True)


def run_ablation(args):
    """Run one ablation in a subprocess."""
    label, shuffle_col, shuffle_all = args

    import sys
    sys.path.insert(0, "src")
    import os
    os.environ["ML_CLOSE_SKIP_CV"] = "1"

    import numpy as np
    from pathlib import Path
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features

    from signal_engine.research import close_walkforward as _mod
    generate_asset_signals = _mod.generate_asset_signals

    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    features_15m = compute_features(ohlcv_15m, cfg.features)

    regime_features = {}
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf) if rtf != "15min" else ohlcv_15m
        regime_features[rtf] = compute_features(rtf_ohlcv, cfg.features)

    np.random.seed(42)
    feat = features_15m.copy()
    if shuffle_all:
        for col in feat.columns:
            vals = feat[col].values.copy()
            np.random.shuffle(vals)
            feat[col] = vals
    elif shuffle_col is not None:
        vals = feat[shuffle_col].values.copy()
        np.random.shuffle(vals)
        feat[shuffle_col] = vals

    # Override LOOKBACKS to only compute the one we need
    _mod.LOOKBACKS = [LB]
    signals = generate_asset_signals(f"BTC_{label}", ohlcv_15m, feat, regime_features, cfg)
    df = signals.get(LB, pd.DataFrame())
    search = period_filter(df, SEARCH_START, SEARCH_END) if len(df) > 0 else pd.DataFrame()
    sr = trade_sharpe(search)
    return label, sr, len(search)


def main():
    t0 = time.time()

    from signal_engine.config import load_config
    cfg = load_config(Path("configs/default.yaml"))
    feature_cols = cfg.features.feature_list

    # Build task list
    tasks = [("BASELINE", None, False)]  # real features
    tasks.append(("ALL_SHUFFLED", None, True))  # all shuffled
    for feat in feature_cols:
        tasks.append((f"no_{feat}", feat, False))  # shuffle one

    workers = min(len(tasks), max(1, os.cpu_count() or 1))
    print(f"Running {len(tasks)} ablations with {workers} workers...")
    print()

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_ablation, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            label, sr, n = future.result()
            results[label] = (sr, n)
            print(f"  [{i:>2}/{len(tasks)}] {label}: Sharpe={sr:.3f} (n={n})", flush=True)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    sr_baseline = results.get("BASELINE", (0, 0))[0]
    sr_shuffled = results.get("ALL_SHUFFLED", (0, 0))[0]

    print(f"  Baseline (real features): Sharpe={sr_baseline:.3f}")
    print(f"  All shuffled:             Sharpe={sr_shuffled:.3f}")
    print()

    print(f"  {'Feature':<25s} {'Sharpe':>8s} {'Drop':>8s} {'Drop%':>8s}")
    print("  " + "-" * 55)

    ablations = []
    for feat in feature_cols:
        key = f"no_{feat}"
        sr, n = results.get(key, (0, 0))
        drop = sr_baseline - sr
        drop_pct = (drop / sr_baseline * 100) if sr_baseline != 0 else 0
        ablations.append((feat, sr, drop, drop_pct))

    ablations.sort(key=lambda x: x[2], reverse=True)
    for feat, sr, drop, drop_pct in ablations:
        marker = " ***" if drop_pct > 5 else ""
        print(f"  {feat:<25s} {sr:>8.3f} {drop:>+8.3f} {drop_pct:>+7.1f}%{marker}")

    print()
    significant = [a for a in ablations if a[3] > 5.0]
    if significant:
        print(f"  {len(significant)} features cause >5% Sharpe drop when shuffled.")
        print("  -> Features DO carry signal beyond the base rate.")
    else:
        print("  NO feature causes >5% Sharpe drop when shuffled.")
        print("  -> Model is NOT using feature content. Edge is pure base rate.")

    print(f"\n  Completed in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
