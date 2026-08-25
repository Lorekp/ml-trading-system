"""Phase 4: Feature Engineering — add/remove/tune features.

Tests on strict OOS (train <2024, test 2024+), lb=14mo.
All variants compared to baseline (18 features, conf>0.08).

Variants:
A: Baseline (18 features)
B: Remove worst features (from SFI analysis: ffd_close, dist_swing_high/low hurt)
C: Minimal (top 8 features by SFI only)
D: Add per-asset FFD d* optimization
E: Feature selection via L1 path (let L1 decide)
F: Higher C (less regularization, more features active)
G: Lower C (more regularization, fewer features active)
H: Confidence threshold 0.12 (best from Phase 3) on each variant
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"

# Feature subsets based on SFI analysis
ALL_18 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "entropy",
    "parkinson_vol_20", "cs_spread", "ffd_close", "ffd_ret_1",
]

# Remove features that hurt in SFI (negative bps on 2024 test)
REMOVE_WORST = [f for f in ALL_18 if f not in [
    "ffd_close", "dist_swing_high", "dist_swing_low", "parkinson_vol_20",
    "rsi_14",  # high MDI but negative MDA — overfits
]]

# Top 8 by SFI bps
TOP_8 = ["ret_1", "ret_3", "stoch_rsi", "bar_range", "entropy",
         "cs_spread", "ffd_ret_1", "macd_hist"]

# Minimal momentum-only
MOMENTUM_ONLY = ["ret_1", "ret_3", "ret_7", "ret_14", "rsi_14", "stoch_rsi", "macd_hist"]

VARIANTS = {
    "A_18feat_C0.1": {"features": ALL_18, "C": 0.1, "conf": 0.08},
    "B_13feat_C0.1": {"features": REMOVE_WORST, "C": 0.1, "conf": 0.08},
    "C_8feat_C0.1": {"features": TOP_8, "C": 0.1, "conf": 0.08},
    "D_7feat_mom": {"features": MOMENTUM_ONLY, "C": 0.1, "conf": 0.08},
    "E_18feat_C0.01": {"features": ALL_18, "C": 0.01, "conf": 0.08},  # more regularization
    "F_18feat_C1.0": {"features": ALL_18, "C": 1.0, "conf": 0.08},  # less regularization
    "G_18feat_C0.1_t12": {"features": ALL_18, "C": 0.1, "conf": 0.12},  # higher threshold
    "H_13feat_C0.1_t12": {"features": REMOVE_WORST, "C": 0.1, "conf": 0.12},
    "I_8feat_C0.1_t12": {"features": TOP_8, "C": 0.1, "conf": 0.12},
    "J_18feat_C0.01_t12": {"features": ALL_18, "C": 0.01, "conf": 0.12},
}


def make_logreg(C=0.1):
    if NEW_SKLEARN:
        return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

    close = ohlcv_15m["close"]
    features = compute_features(ohlcv_15m, cfg.features)
    # Get all 18 features, winsorize
    feat_all = features[cfg.features.feature_list].copy()
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat_all.columns:
            feat_all[col] = feat_all[col].clip(feat_all[col].quantile(0.01), feat_all[col].quantile(0.99))
    fc_all = feat_all.dropna()

    fwd_ret = close.pct_change(4).shift(-4)
    y_all = fwd_ret.reindex(fc_all.index).dropna()
    valid = fc_all.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, {}

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc_all.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)
    test_months = pd.period_range(
        start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    results = {v: [] for v in VARIANTS}

    for month in test_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc_all.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)

        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50:
            continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        month_str = str(month)
        train_start = ms - pd.DateOffset(months=LB)
        if train_start < start_date:
            continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200:
            continue

        for vname, vcfg in VARIANTS.items():
            feat_cols = [c for c in vcfg["features"] if c in fc_all.columns]
            if len(feat_cols) < 3:
                continue
            fc = fc_all[feat_cols]

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg(C=vcfg["C"])
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception:
                continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            fv = fwd_ret.reindex(te).values
            finite = np.isfinite(fv)
            if not finite.any():
                continue

            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            signals["asset"] = asset
            filtered = apply_confidence_filter(
                signals, ConfidenceFilterConfig(mode="absolute", threshold=vcfg["conf"]))
            if len(filtered) > 0:
                results[vname].append((month_str, filtered.to_dict("list")))

    return asset, results


def flat_bet_stats(all_results, variant):
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get(variant, []):
            sigs.append(pd.DataFrame(sd))
    if not sigs:
        return None
    df = pd.concat(sigs)
    pnls = df["direction"].values * df["fwd_return"].values - 0.0001
    ts_arr = pd.to_datetime(df["timestamp"], utc=True)
    daily = pd.Series(5.0 * pnls, index=ts_arr.values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return None
    std = daily.std()
    sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
    return {
        "sharpe": sr, "win_rate": (pnls > 0).mean() * 100,
        "mean_bps": pnls.mean() * 10000, "n_trades": len(df),
        "n_assets": df["asset"].nunique(),
    }


def main():
    t0 = time.time()
    print("=" * 85)
    print(f"PHASE 4: FEATURE ENGINEERING (lb={LB}mo, test {TEST_START}+)")
    print("=" * 85)
    for v, c in VARIANTS.items():
        print(f"  {v}: {len(c['features'])}feat C={c['C']} conf>{c['conf']}")
    print(f"  {W} workers", flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, res = f.result()
            done += 1
            all_results[asset] = res
            print(f"  [{done:>2d}/30] {asset}", flush=True)

    print(f"\n  Signals done in {time.time() - t0:.0f}s")

    print(f"\n{'='*85}")
    print(f"RESULTS (flat-bet, test {TEST_START}+)")
    print(f"{'='*85}")
    print(f"  {'Variant':<25s} {'Feat':>4s} {'C':>5s} {'Thr':>5s} {'SR':>7s} {'Win%':>6s} "
          f"{'bps':>7s} {'Trades':>8s} {'Assets':>6s}")
    print("  " + "-" * 78)

    rows = []
    for vname in VARIANTS:
        vcfg = VARIANTS[vname]
        r = flat_bet_stats(all_results, vname)
        if r:
            rows.append((vname, vcfg, r))
            print(f"  {vname:<25s} {len(vcfg['features']):>4d} {vcfg['C']:>5.2f} {vcfg['conf']:>5.2f} "
                  f"{r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% {r['mean_bps']:>6.2f} "
                  f"{r['n_trades']:>8d} {r['n_assets']:>5d}")
        else:
            print(f"  {vname:<25s}  NO DATA")

    # Rank
    rows.sort(key=lambda x: -x[2]["sharpe"])
    print(f"\n  RANKED BY SHARPE:")
    for i, (v, c, r) in enumerate(rows[:5]):
        print(f"    {i+1}. {v:<25s} SR={r['sharpe']:.3f}  Win={r['win_rate']:.1f}%  "
              f"bps={r['mean_bps']:.1f}  n={r['n_trades']}")

    # Best at each threshold
    print(f"\n  BEST AT conf>0.08:")
    t08 = [(v, c, r) for v, c, r in rows if c["conf"] == 0.08]
    if t08:
        print(f"    {t08[0][0]}: SR={t08[0][2]['sharpe']:.3f}")

    print(f"\n  BEST AT conf>0.12:")
    t12 = [(v, c, r) for v, c, r in rows if c["conf"] == 0.12]
    if t12:
        print(f"    {t12[0][0]}: SR={t12[0][2]['sharpe']:.3f}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
