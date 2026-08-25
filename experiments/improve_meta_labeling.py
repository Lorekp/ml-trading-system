"""Experiment 4: Meta-labeling retest with T1 config.

Train XGBoost to predict WHICH LogReg signals will be profitable.
Use prediction as sizing multiplier.

Phase 3 rejected meta-labeling with OLD config (C=0.1, lb=14, thr=0.08).
T1 config (C=0.02, lb=2, thr=0.10) has different confidence distributions —
worth retesting.
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd
from xgboost import XGBClassifier
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/optuna/signal_cache")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
C_STR = "0.02000"; LB = 2; THR = 0.10; FWD = 6; LEV = 5.0; DLV = 0.05; SIZING = 0.06


def load_all_signals():
    """Load T1 signals from all windows, concatenated."""
    all_fi = []
    for wl, cd in W:
        with open(CACHE_DIR / f"C{C_STR}_lb{LB}_{wl}.pkl", "rb") as f:
            raw = pickle.load(f)
        fi = raw[raw["confidence"] > THR].copy()
        fi["fwd_return"] = fi[f"ret_{FWD}"].values
        fi["window"] = wl
        fi["profitable"] = (fi["direction"] * fi["fwd_return"] > 0).astype(int)
        all_fi.append(fi)
    return pd.concat(all_fi, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_portfolio(signals, label):
    """Run portfolio sim on signals, return metrics."""
    tp = tc = tt = ttd = 0; mdd = mwd = 0.0
    per_window = []

    for wl, cd in W:
        fi = signals[signals["window"] == wl].copy()
        if len(fi) < 10:
            per_window.append({"wl": wl, "pnl_cal": 0})
            continue

        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=DLV,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=SIZING * LEV, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5:
            per_window.append({"wl": wl, "pnl_cal": 0})
            continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1:
            per_window.append({"wl": wl, "pnl_cal": 0})
            continue
        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / 10000.0
        pnl = sim["equity"][-1] - 10000
        per_window.append({"wl": wl, "pnl_cal": pnl / cd, "dd": abs(dd.min()), "wd": abs(dr.min())})
        tp += pnl; tc += cd; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min())); mwd = max(mwd, abs(dr.min()))

    if tc == 0: return None
    return {"pnl_cal": tp / tc, "d2t": int(1000 / (tp / tc)) if tp > 0 else 9999,
            "dd": mdd, "wd": mwd, "trades": tt, "per_window": per_window}


def apply_meta_labeling(all_signals, meta_features, meta_mode, meta_threshold=0.5):
    """Apply meta-labeling: train XGBoost per window, predict sizing multiplier."""
    result = all_signals.copy()
    windows = ["W1", "W2", "W3"]

    for i, wl in enumerate(windows):
        test_mask = result["window"] == wl

        # Train on all PRIOR windows
        if i == 0:
            # No prior data for W1 — use first half of W1 to train, second half to test
            w1_data = result[result["window"] == "W1"]
            mid = len(w1_data) // 2
            train_data = w1_data.iloc[:mid]
            # Don't modify W1 — no prior data to train on
            continue
        else:
            prior_windows = windows[:i]
            train_mask = result["window"].isin(prior_windows)
            train_data = result[train_mask]

        if len(train_data) < 100:
            continue

        # Features for meta-labeler
        feat_cols = [c for c in meta_features if c in train_data.columns]
        if len(feat_cols) < 3:
            continue

        X_train = train_data[feat_cols].fillna(0).values
        y_train = train_data["profitable"].values

        test_data = result[test_mask]
        X_test = test_data[feat_cols].fillna(0).values

        # Train XGBoost meta-labeler
        xgb = XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X_train, y_train)
        meta_prob = xgb.predict_proba(X_test)[:, 1]

        test_indices = result.index[test_mask]
        if meta_mode == "filter":
            for j, idx in enumerate(test_indices):
                if meta_prob[j] < meta_threshold:
                    result.at[idx, "signal_strength"] = 0.0
        elif meta_mode == "scale":
            for j, idx in enumerate(test_indices):
                result.at[idx, "signal_strength"] = result.at[idx, "signal_strength"] * meta_prob[j]

    # Remove zeroed-out signals
    result = result[result["signal_strength"] > 0].reset_index(drop=True)
    return result


def main():
    t0 = time.time()
    print("EXPERIMENT 4: META-LABELING RETEST (T1 CONFIG)")
    print("=" * 95)

    all_signals = load_all_signals()
    print(f"Loaded {len(all_signals)} signals across 3 windows")
    print(f"  Win rate: {all_signals['profitable'].mean()*100:.1f}%")
    print(f"  Confidence range: [{all_signals['confidence'].min():.4f}, {all_signals['confidence'].max():.4f}]")

    # Available features for meta-labeler
    meta_features_base = ["confidence", "probability", "signal_strength"]
    # Add return features from cache
    ret_features = [f"ret_{k}" for k in range(1, 9) if f"ret_{k}" in all_signals.columns]
    meta_features_full = meta_features_base + ret_features + ["vol_val"]

    print(f"  Meta features (base): {meta_features_base}")
    print(f"  Meta features (full): {len(meta_features_full)} features\n")

    print(f"{'Variant':<35s} {'$/cal':>7s} {'~Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Trades':>6s}  W1$/d  W2$/d  W3$/d")
    print("-" * 100)

    # Baseline
    r = run_portfolio(all_signals, "Baseline")
    if r:
        pw = {p["wl"]: p for p in r["per_window"]}
        print(f"{'Baseline (no meta)':<35s} ${r['pnl_cal']:>6.2f} {r['d2t']:>5d} {r['dd']*100:>5.1f}% "
              f"{r['wd']*100:>5.2f}% {r['trades']:>6d}  "
              f"${pw.get('W1',{}).get('pnl_cal',0):>5.2f} ${pw.get('W2',{}).get('pnl_cal',0):>5.2f} "
              f"${pw.get('W3',{}).get('pnl_cal',0):>5.2f}", flush=True)

    # Meta-labeling variants
    for label, features, mode, threshold in [
        ("Filter base feat thr=0.50", meta_features_base, "filter", 0.50),
        ("Filter base feat thr=0.55", meta_features_base, "filter", 0.55),
        ("Filter base feat thr=0.60", meta_features_base, "filter", 0.60),
        ("Filter full feat thr=0.50", meta_features_full, "filter", 0.50),
        ("Filter full feat thr=0.55", meta_features_full, "filter", 0.55),
        ("Scale base feat", meta_features_base, "scale", 0.50),
        ("Scale full feat", meta_features_full, "scale", 0.50),
    ]:
        modified = apply_meta_labeling(all_signals.copy(), features, mode, threshold)
        r = run_portfolio(modified, label)
        if r is None:
            print(f"{label:<35s} NO DATA")
            continue
        pw = {p["wl"]: p for p in r["per_window"]}
        ok = r["dd"] < 0.10 and r["wd"] < 0.05
        print(f"{label:<35s} ${r['pnl_cal']:>6.2f} {r['d2t']:>5d} {r['dd']*100:>5.1f}% "
              f"{r['wd']*100:>5.2f}% {r['trades']:>6d}  "
              f"${pw.get('W1',{}).get('pnl_cal',0):>5.2f} ${pw.get('W2',{}).get('pnl_cal',0):>5.2f} "
              f"${pw.get('W3',{}).get('pnl_cal',0):>5.2f}  "
              f"{'OK' if ok else 'FAIL'}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
