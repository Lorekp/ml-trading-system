"""Phase 4 prop firm analysis: best feature configs × {14mo, expanding} × threshold sweep."""

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

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
TEST_START = "2024-01"

ALL_18 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "entropy",
    "parkinson_vol_20", "cs_spread", "ffd_close", "ffd_ret_1",
]
TOP_8 = ["ret_1", "ret_3", "stoch_rsi", "bar_range", "entropy",
         "cs_spread", "ffd_ret_1", "macd_hist"]

# Configs: (label, features, C)
CONFIGS = [
    ("18f_C0.1", ALL_18, 0.1),
    ("8f_C0.1", TOP_8, 0.1),
    ("18f_C0.01", ALL_18, 0.01),
]
LOOKBACKS = [14, -1]  # 14mo and expanding
THRESHOLDS = [0.08, 0.10, 0.12, 0.14, 0.15, 0.18, 0.20, 0.25]


def make_logreg(C=0.1):
    if NEW_SKLEARN:
        return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config
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

    # key = (config_label, lb) -> list of (month, raw_signals_dict)
    results = {}
    for clabel, _, _ in CONFIGS:
        for lb in LOOKBACKS:
            results[(clabel, lb)] = []

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

        for lb in LOOKBACKS:
            if lb == -1:
                train_start = start_date
            else:
                train_start = ms - pd.DateOffset(months=lb)
                if train_start < start_date:
                    continue
            train_cutoff = ms - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200:
                continue

            for clabel, feat_list, C in CONFIGS:
                feat_cols = [c for c in feat_list if c in fc_all.columns]
                fc = fc_all[feat_cols]
                try:
                    sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(fc.loc[tr])
                    model = make_logreg(C=C)
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
                results[(clabel, lb)].append((month_str, signals.to_dict("list")))

    return asset, results


def analyze(all_results, clabel, lb, threshold, pcfg):
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get((clabel, lb), []):
            df = pd.DataFrame(sd)
            df = df[df["confidence"] > threshold]
            if len(df) > 0:
                sigs.append(df)
    if not sigs:
        return None
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    if len(combined) < 50:
        return None

    combined = orig.apply_nco_wf(combined)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))

    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    max_dd = dd.min() * 100
    worst_day = (dpnl / pcfg.starting_capital * 100).min()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)
    pnl_pct = pnl_day / pcfg.starting_capital * 100

    return {
        "sharpe": stats["sharpe"],
        "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0),
        "ret_pct": stats["total_return_pct"],
        "max_dd": max_dd,
        "worst_day": worst_day,
        "pnl_day": pnl_day,
        "pnl_pct": pnl_pct,
        "win_rate": (dpnl > 0).mean() * 100,
        "n_trades": sim["n_trades"],
        "n_days": stats["n_days"],
        "final_eq": stats["final_equity"],
    }


def main():
    t0 = time.time()
    print("=" * 95)
    print("PHASE 4: PROP FIRM ANALYSIS — Best Feature Configs")
    print(f"  Configs: {[c[0] for c in CONFIGS]}")
    print(f"  Lookbacks: 14mo + expanding")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Strict OOS: test {TEST_START}+, {W} workers")
    print("=" * 95, flush=True)

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

    pcfg = orig.get_pcfg()

    for lb in LOOKBACKS:
        lb_label = "EXPANDING" if lb == -1 else f"{lb}-MONTH"
        print(f"\n{'='*95}")
        print(f"  {lb_label}")
        print(f"{'='*95}")
        print(f"  {'Config':<15s} {'Thr':>5s} {'SR':>6s} {'Sort':>6s} {'Calm':>6s} "
              f"{'Ret%':>7s} {'MaxDD':>7s} {'WrstD':>7s} {'PnL/d':>8s} {'PnL%':>6s} "
              f"{'Win%':>5s} {'Trades':>7s}")
        print("  " + "-" * 92)

        for clabel, _, _ in CONFIGS:
            for thr in THRESHOLDS:
                r = analyze(all_results, clabel, lb, thr, pcfg)
                if r is None:
                    continue
                print(f"  {clabel:<15s} {thr:>5.2f} {r['sharpe']:>6.3f} {r['sortino']:>6.3f} "
                      f"{r['calmar']:>6.3f} {r['ret_pct']:>6.1f}% {r['max_dd']:>6.2f}% "
                      f"{r['worst_day']:>6.2f}% ${r['pnl_day']:>7.2f} {r['pnl_pct']:>5.3f} "
                      f"{r['win_rate']:>4.1f}% {r['n_trades']:>7d}")

    # Prop firm optimizer
    MAX_DD = 10.0
    MAX_DAILY = 5.0
    TARGET = 10.0

    print(f"\n{'='*95}")
    print(f"PROP FIRM OPTIMIZER ({MAX_DD}% DD, {MAX_DAILY}% daily, {TARGET}% target)")
    print(f"{'='*95}")
    print(f"  {'Config':<15s} {'LB':>5s} {'Thr':>5s} {'Lev':>5s} {'Days':>6s} {'LevDD':>7s} "
          f"{'LevDay':>7s} {'SR':>6s} {'Win%':>5s} {'Trades':>7s}")
    print("  " + "-" * 80)

    candidates = []
    for lb in LOOKBACKS:
        for clabel, _, _ in CONFIGS:
            for thr in THRESHOLDS:
                r = analyze(all_results, clabel, lb, thr, pcfg)
                if r is None or r["max_dd"] >= 0 or r["worst_day"] >= 0:
                    continue
                abs_dd = abs(r["max_dd"])
                abs_day = abs(r["worst_day"])
                if abs_dd < 0.1 or abs_day < 0.1:
                    continue
                lev_dd = (MAX_DD * 0.80) / abs_dd
                lev_day = (MAX_DAILY * 0.80) / abs_day
                safe_lev = min(lev_dd, lev_day, 100)
                safe_lev = max(1.0, safe_lev)
                lev_pnl_pct = r["pnl_pct"] * safe_lev
                days = TARGET / lev_pnl_pct if lev_pnl_pct > 0 else 999
                candidates.append({
                    "config": clabel, "lb": lb, "thr": thr,
                    "lev": safe_lev, "days": days,
                    "lev_dd": abs_dd * safe_lev,
                    "lev_day": abs_day * safe_lev,
                    "sharpe": r["sharpe"], "win": r["win_rate"],
                    "trades": r["n_trades"],
                })

    candidates.sort(key=lambda x: x["days"])
    for c in candidates[:15]:
        lb_s = "EXP" if c["lb"] == -1 else f"{c['lb']}mo"
        print(f"  {c['config']:<15s} {lb_s:>5s} {c['thr']:>5.2f} {c['lev']:>4.1f}x "
              f"{c['days']:>5.1f}d {c['lev_dd']:>6.2f}% {c['lev_day']:>6.2f}% "
              f"{c['sharpe']:>6.3f} {c['win']:>4.1f}% {c['trades']:>7d}")

    if candidates:
        best = candidates[0]
        lb_s = "EXP" if best["lb"] == -1 else f"{best['lb']}mo"
        print(f"\n  RECOMMENDED: {best['config']} lb={lb_s} thr={best['thr']:.2f}")
        print(f"    Leverage: {best['lev']:.1f}x")
        print(f"    Days to 10%: {best['days']:.0f}")
        print(f"    Levered DD: {best['lev_dd']:.1f}% / {MAX_DD}% limit")
        print(f"    Levered daily: {best['lev_day']:.1f}% / {MAX_DAILY}% limit")
        print(f"    Sharpe: {best['sharpe']:.3f}, Win: {best['win']:.0f}%")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
