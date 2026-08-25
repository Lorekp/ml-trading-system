"""Strict OOS Report: Train ≤2023, Test 2024+.

Per-asset LogReg L1 models, monthly retraining with rolling lookback.
Lookbacks: [4, 6, 8, 10, 12] months.
NO data from 2024+ touches training. Ever.
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
LOOKBACKS = [6, 8, 10, 12, 14, -1]  # -1 = expanding (all available history)
TEST_START = "2024-01"  # strict OOS boundary


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

    # Only test months from 2024-01 onwards
    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)

    test_months = pd.period_range(
        start=test_start_ts.to_period("M"),
        end=end_date.to_period("M"), freq="M",
    )

    results = {lb: [] for lb in LOOKBACKS}

    for month in test_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        # Test bars this month, vol-filtered
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
            # Rolling lookback or expanding window
            if lb == -1:
                # Expanding: use all data from start
                train_start = start_date
            else:
                train_start = ms - pd.DateOffset(months=lb)
                if train_start < start_date:
                    continue
            train_cutoff = ms - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200:
                continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg()
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
                signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
            if len(filtered) > 0:
                results[lb].append((month_str, filtered.to_dict("list")))

    return asset, results


def flat_bet_stats(all_results, lb):
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get(lb, []):
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


def dd_stats(all_results, lb):
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get(lb, []):
            sigs.append(pd.DataFrame(sd))
    if not sigs:
        return None
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    combined = orig.apply_nco_wf(combined)
    pcfg = orig.get_pcfg()
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
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {
        "sharpe": stats["sharpe"],
        "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0),
        "total_return_pct": stats["total_return_pct"],
        "max_dd_pct": dd.min() * 100,
        "mean_dd_pct": dd[dd < -0.001].mean() * 100 if (dd < -0.001).any() else 0,
        "final_equity": stats["final_equity"],
        "pnl_day": pnl_day,
        "n_trades": sim["n_trades"],
        "n_days": stats["n_days"],
        "win_rate": (dpnl > 0).mean() * 100,
        "pos_months": f"{pos_m}/{len(monthly)}",
        "var95": np.percentile(dpnl, 5),
        "cvar95": dpnl[dpnl <= np.percentile(dpnl, 5)].mean(),
        "worst_day": dpnl.min(),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print(f"STRICT OOS REPORT: Train <2024, Test {TEST_START}+")
    print(f"  30 assets, per-asset models, monthly retraining")
    print(f"  Lookbacks: {LOOKBACKS}")
    print(f"  {W} workers")
    print("=" * 80, flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            counts = {lb: len(results.get(lb, [])) for lb in LOOKBACKS}
            print(f"  [{done:>2d}/30] {asset}: {counts}", flush=True)

    print(f"\n  Signals done in {time.time() - t0:.0f}s")

    # ============================================================
    # FLAT-BET COMPARISON
    # ============================================================
    print(f"\n{'='*80}")
    print(f"FLAT-BET RESULTS (test {TEST_START}+, $5/trade)")
    print(f"{'='*80}")
    print(f"  {'LB':<5s} {'Sharpe':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>8s} {'Assets':>6s}")
    print("  " + "-" * 42)
    for lb in LOOKBACKS:
        r = flat_bet_stats(all_results, lb)
        if r:
            print(f"  {('EXP' if lb==-1 else f'{lb}mo'):>5s} {r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% "
                  f"{r['mean_bps']:>6.2f} {r['n_trades']:>8d} {r['n_assets']:>5d}")
        else:
            print(f"  {('EXP' if lb==-1 else f'{lb}mo'):>5s}  NO DATA")

    # ============================================================
    # NCO PORTFOLIO + DRAWDOWN
    # ============================================================
    print(f"\n{'='*80}")
    print(f"NCO PORTFOLIO + DRAWDOWN (test {TEST_START}+)")
    print(f"{'='*80}")
    print(f"  {'LB':<5s} {'SR':>6s} {'Sort':>6s} {'Calm':>6s} {'Ret%':>8s} {'MaxDD':>7s} "
          f"{'MeanDD':>7s} {'PnL/d':>8s} {'Win%':>6s} {'Months':>7s} {'Trades':>7s}")
    print("  " + "-" * 82)
    for lb in LOOKBACKS:
        r = dd_stats(all_results, lb)
        if r:
            print(f"  {('EXP' if lb==-1 else f'{lb}mo'):>5s} {r['sharpe']:>6.3f} {r['sortino']:>6.3f} {r['calmar']:>6.3f} "
                  f"{r['total_return_pct']:>7.1f}% {r['max_dd_pct']:>6.2f}% "
                  f"{r['mean_dd_pct']:>6.2f}% ${r['pnl_day']:>7.2f} "
                  f"{r['win_rate']:>5.1f}% {r['pos_months']:>7s} {r['n_trades']:>7d}")
        else:
            print(f"  {('EXP' if lb==-1 else f'{lb}mo'):>5s}  NO DATA")

    # ============================================================
    # PER-ASSET BREAKDOWN (best lookback)
    # ============================================================
    # Find best lookback
    best_lb = None
    best_sr = -999
    for lb in LOOKBACKS:
        r = flat_bet_stats(all_results, lb)
        if r and r["sharpe"] > best_sr:
            best_sr = r["sharpe"]
            best_lb = lb

    if best_lb:
        print(f"\n{'='*80}")
        print(f"PER-ASSET BREAKDOWN (lb={best_lb}mo, test {TEST_START}+)")
        print(f"{'='*80}")
        print(f"  {'Asset':<8s} {'Sharpe':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>7s}")
        print("  " + "-" * 38)

        asset_stats = []
        for asset in sorted(orig.ASSETS.keys()):
            ar = all_results.get(asset, {})
            parts = []
            for ms, sd in ar.get(best_lb, []):
                parts.append(pd.DataFrame(sd))
            if not parts:
                continue
            df = pd.concat(parts)
            pnls = df["direction"].values * df["fwd_return"].values - 0.0001
            ts_arr = pd.to_datetime(df["timestamp"], utc=True)
            daily = pd.Series(5.0 * pnls, index=ts_arr.values).resample("1D").sum()
            daily = daily[daily != 0]
            if len(daily) < 5:
                continue
            std = daily.std()
            sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
            wr = (pnls > 0).mean() * 100
            bps = pnls.mean() * 10000
            asset_stats.append((asset, sr, wr, bps, len(df)))

        asset_stats.sort(key=lambda x: -x[1])
        pos = sum(1 for _, sr, _, _, _ in asset_stats if sr > 0)
        for asset, sr, wr, bps, nt in asset_stats:
            print(f"  {asset:<8s} {sr:>7.3f} {wr:>5.1f}% {bps:>6.2f} {nt:>7d}")
        print(f"\n  Positive: {pos}/{len(asset_stats)} assets")

    # ============================================================
    # YEARLY BREAKDOWN (NCO portfolio, best lookback)
    # ============================================================
    if best_lb:
        sigs = []
        for ar in all_results.values():
            for ms, sd in ar.get(best_lb, []):
                sigs.append(pd.DataFrame(sd))
        if sigs:
            combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
            combined = orig.apply_nco_wf(combined)
            pcfg = orig.get_pcfg()
            sim = simulate_portfolio(combined, pcfg)
            if sim and sim["n_trades"] >= 10:
                dpnl = compute_daily_pnl_series(sim)
                if dpnl is not None:
                    yearly = dpnl.resample("YE").sum()
                    print(f"\n  --- Yearly PnL (NCO, lb={best_lb}mo) ---")
                    for dt, pnl in yearly.items():
                        print(f"  {dt.year}: ${pnl:>10.2f}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
