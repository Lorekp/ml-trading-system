"""Phase 3: Meta-Labeling.

Primary model (LogReg L1) predicts side {long/short}.
Secondary model (LogReg or XGBoost) predicts bet/skip {0, 1}.
Bet size = secondary model's predicted probability.

Strict OOS: train <2024, test 2024+.
Per-asset models, monthly retraining, lb=14mo (best from Phase 1).

Variants:
A: Baseline (no meta-labeling, lb=14)
B: Meta-label with LogReg secondary
C: Meta-label with XGBoost secondary
D: Baseline with optimized confidence thresholds
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
import xgboost as xgb
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
CONF_THRESHOLDS = [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

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
    if len(valid) < 3000:
        return asset, {}

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)

    test_months = pd.period_range(
        start=test_start_ts.to_period("M"),
        end=end_date.to_period("M"), freq="M")

    # Results: variant -> list of (month, signals_dict)
    results = {"A": [], "B": [], "C": [], "D": []}
    # D stores raw (unfiltered) signals for threshold sweep
    results_raw = []

    for month in test_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

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

        # === PRIMARY MODEL (same for all variants) ===
        try:
            sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(fc.loc[tr])
            primary = make_logreg()
            primary.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)

            # Primary predictions on TRAINING data (for meta-labeler training)
            yp_train = primary.predict_proba(Xtr)[:, 1]
            dirs_train = np.where(yp_train > 0.5, 1.0, -1.0)
            # Meta-label: did the primary model's direction agree with actual?
            actual_dir = np.sign(y_all.loc[tr].values)
            meta_labels = (dirs_train == actual_dir).astype(int)  # 1=correct, 0=wrong

            # Primary predictions on TEST data
            Xte = scaler.transform(fc.loc[te])
            yp_test = primary.predict_proba(Xte)[:, 1]
            dirs_test = np.where(yp_test > 0.5, 1.0, -1.0)
            conf_test = np.abs(yp_test - 0.5)
        except Exception:
            continue

        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any():
            continue

        # === VARIANT A: Baseline (confidence > 0.08) ===
        sigs_a = get_signal(yp_test, num_classes=2)
        sd_a = discretize_signal(sigs_a, step_size=0.1)
        ss_a = np.abs(sd_a) if np.ndim(sd_a) > 0 else np.full(len(yp_test), abs(float(sd_a)))
        signals_base = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs_test[finite],
            "fwd_return": fv[finite], "confidence": conf_test[finite],
            "probability": yp_test[finite], "signal_strength": ss_a[finite],
        })
        signals_base["asset"] = asset
        filtered_a = apply_confidence_filter(
            signals_base, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered_a) > 0:
            results["A"].append((month_str, filtered_a.to_dict("list")))

        # Store raw for threshold sweep (D)
        results_raw.append((month_str, signals_base.to_dict("list")))

        # === META-LABELER TRAINING ===
        # Features for meta-labeler: original features + primary probability
        Xtr_meta = np.column_stack([Xtr, yp_train])
        Xte_meta = np.column_stack([Xte, yp_test])

        # === VARIANT B: Meta-label with LogReg ===
        try:
            meta_lr = make_logreg()
            meta_lr.fit(Xtr_meta, meta_labels, sample_weight=sw)
            meta_prob_b = meta_lr.predict_proba(Xte_meta)[:, 1]

            # Bet size from meta-labeler probability
            ss_b = meta_prob_b  # probability of being correct = bet size
            signals_b = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs_test[finite],
                "fwd_return": fv[finite], "confidence": meta_prob_b[finite],
                "probability": yp_test[finite],
                "signal_strength": ss_b[finite],
            })
            signals_b["asset"] = asset
            # Filter: only trade when meta-labeler says P(correct) > 0.55
            filtered_b = signals_b[signals_b["confidence"] > 0.55]
            if len(filtered_b) > 0:
                results["B"].append((month_str, filtered_b.to_dict("list")))
        except Exception:
            pass

        # === VARIANT C: Meta-label with XGBoost ===
        try:
            meta_xgb = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
                random_state=42, n_jobs=1, verbosity=0,
                eval_metric="logloss", objective="binary:logistic",
            )
            meta_xgb.fit(Xtr_meta, meta_labels, sample_weight=sw, verbose=False)
            meta_prob_c = meta_xgb.predict_proba(Xte_meta)[:, 1]

            ss_c = meta_prob_c
            signals_c = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs_test[finite],
                "fwd_return": fv[finite], "confidence": meta_prob_c[finite],
                "probability": yp_test[finite],
                "signal_strength": ss_c[finite],
            })
            signals_c["asset"] = asset
            filtered_c = signals_c[signals_c["confidence"] > 0.55]
            if len(filtered_c) > 0:
                results["C"].append((month_str, filtered_c.to_dict("list")))
        except Exception:
            pass

    # === VARIANT D: Threshold sweep on raw signals ===
    for thr in CONF_THRESHOLDS:
        key = f"D_{thr:.2f}"
        results[key] = []
        for month_str, raw_dict in results_raw:
            df = pd.DataFrame(raw_dict)
            filtered = df[df["confidence"] > thr]
            if len(filtered) > 0:
                results[key].append((month_str, filtered.to_dict("list")))

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
    print("=" * 80)
    print(f"PHASE 3: META-LABELING (lb={LB}mo, test {TEST_START}+)")
    print(f"  A: Baseline (conf>0.08)")
    print(f"  B: Meta-label LogReg (P(correct)>0.55)")
    print(f"  C: Meta-label XGBoost (P(correct)>0.55)")
    print(f"  D: Confidence threshold sweep")
    print(f"  {W} workers")
    print("=" * 80, flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, res = f.result()
            done += 1
            all_results[asset] = res
            na = len(res.get("A", []))
            nb = len(res.get("B", []))
            nc = len(res.get("C", []))
            print(f"  [{done:>2d}/30] {asset}: A={na} B={nb} C={nc}", flush=True)

    print(f"\n  Signals done in {time.time() - t0:.0f}s")

    # ============================================================
    # FLAT-BET RESULTS
    # ============================================================
    print(f"\n{'='*80}")
    print(f"FLAT-BET RESULTS (test {TEST_START}+)")
    print(f"{'='*80}")
    print(f"  {'Variant':<35s} {'SR':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>8s} {'Assets':>6s}")
    print("  " + "-" * 68)

    for variant, label in [
        ("A", "A: Baseline (conf>0.08)"),
        ("B", "B: Meta-label LogReg"),
        ("C", "C: Meta-label XGBoost"),
    ]:
        r = flat_bet_stats(all_results, variant)
        if r:
            print(f"  {label:<35s} {r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% "
                  f"{r['mean_bps']:>6.2f} {r['n_trades']:>8d} {r['n_assets']:>5d}")
        else:
            print(f"  {label:<35s}  NO DATA")

    # Threshold sweep
    print(f"\n  --- Confidence Threshold Sweep ---")
    print(f"  {'Threshold':<12s} {'SR':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>8s}")
    print("  " + "-" * 42)
    for thr in CONF_THRESHOLDS:
        key = f"D_{thr:.2f}"
        r = flat_bet_stats(all_results, key)
        if r:
            print(f"  conf>{thr:<6.2f}   {r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% "
                  f"{r['mean_bps']:>6.02f} {r['n_trades']:>8d}")
        else:
            print(f"  conf>{thr:<6.2f}    NO DATA")

    # ============================================================
    # NCO PORTFOLIO
    # ============================================================
    print(f"\n{'='*80}")
    print(f"NCO PORTFOLIO (test {TEST_START}+)")
    print(f"{'='*80}")

    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    from signal_engine.stats import compute_full_stats

    for variant, label in [
        ("A", "A: Baseline"),
        ("B", "B: Meta-label LogReg"),
        ("C", "C: Meta-label XGBoost"),
    ]:
        sigs = []
        for ar in all_results.values():
            for ms, sd in ar.get(variant, []):
                sigs.append(pd.DataFrame(sd))
        if not sigs:
            print(f"  {label}: NO DATA")
            continue
        combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
        combined = orig.apply_nco_wf(combined)
        pcfg = orig.get_pcfg()
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 10:
            print(f"  {label}: SIM FAILED")
            continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 5:
            continue
        stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                                   sim["n_trades"] / len(dpnl))
        equity = pd.Series(sim["equity"][~sim["skipped"]],
                           index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((equity - equity.expanding().max()) / equity.expanding().max()).min() * 100
        pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
        print(f"  {label:<25s} SR={stats['sharpe']:>6.3f}  Sortino={stats.get('sortino',0):>6.3f}  "
              f"Calmar={stats.get('calmar',0):>6.3f}  Ret={stats['total_return_pct']:>6.1f}%  "
              f"MaxDD={dd:>6.2f}%  PnL/d=${pnl_day:>7.2f}  "
              f"Win={( dpnl>0).mean()*100:.1f}%  Trades={sim['n_trades']}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
