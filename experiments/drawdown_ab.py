"""Compare drawdowns: A (baseline) vs B (CUSUM sampling)."""
import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.filters import cusum_filter, get_daily_vol
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)

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
    from afml.filters import cusum_filter, get_daily_vol
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {"A": {8: []}, "B": {8: []}}

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
        return asset, {"A": {8: []}, "B": {8: []}}

    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    daily_vol = get_daily_vol(close, span=100)
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5
    results = {"A": {8: []}, "B": {8: []}}

    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"), freq="M")

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
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
        train_start = ms - pd.DateOffset(months=8)
        if train_start < start_date:
            continue
        train_cutoff = ms - embargo_td
        tr_all = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr_all) < 200:
            continue

        # CUSUM
        train_close = close.loc[train_start:train_cutoff]
        train_log = np.log(train_close)
        train_dv = daily_vol.reindex(train_close.index).dropna()
        tr_cusum = pd.DatetimeIndex([])
        if len(train_dv) > 50:
            cusum_ev = cusum_filter(train_log, train_dv * 1.5)
            tr_cusum = cusum_ev[cusum_ev.isin(valid)]
            tr_cusum = tr_cusum[(tr_cusum >= train_start) & (tr_cusum < train_cutoff)]

        def train_predict(tr_idx, variant):
            if len(tr_idx) < 200:
                return
            try:
                sw = compute_sample_weights(fc.loc[tr_idx].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr_idx])
                model = make_logreg()
                model.fit(Xtr, (y_all.loc[tr_idx] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception:
                return
            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            fv = fwd_ret.reindex(te).values
            finite = np.isfinite(fv)
            if not finite.any():
                return
            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            signals["asset"] = asset
            filtered = apply_confidence_filter(
                signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
            if len(filtered) > 0:
                results[variant][8].append((month_str, filtered.to_dict("list")))

        train_predict(tr_all, "A")
        if len(tr_cusum) >= 200:
            train_predict(tr_cusum, "B")

    return asset, results


def dd_report(label, stats, sim, dpnl):
    if stats is None:
        print(f"  {label}: NO DATA")
        return
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    peak = equity.expanding().max()
    dd = (equity - peak) / peak

    dd_periods = []
    in_dd = False
    dd_start = dd_trough = None
    for ts_i, dd_val in dd.items():
        if dd_val < -0.001 and not in_dd:
            in_dd = True; dd_start = ts_i; dd_trough = dd_val
        elif dd_val < -0.001 and in_dd:
            if dd_val < dd_trough:
                dd_trough = dd_val
        elif dd_val >= -0.001 and in_dd:
            in_dd = False
            dd_periods.append({"trough": dd_trough * 100, "duration": (ts_i - dd_start).days})
    if in_dd and dd_start:
        dd_periods.append({"trough": dd_trough * 100, "duration": (dd.index[-1] - dd_start).days})
    dd_periods.sort(key=lambda x: x["trough"])
    durations = [p["duration"] for p in dd_periods]

    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)

    print(f"\n  === {label} ===")
    print(f"  Sharpe: {stats['sharpe']:.3f}  Sortino: {stats.get('sortino', 0):.3f}  "
          f"Calmar: {stats.get('calmar', 0):.3f}")
    print(f"  Return: {stats['total_return_pct']:.1f}%  Final: ${stats['final_equity']:,.0f}  "
          f"PnL/day: ${pnl_day:.2f}")
    print(f"  Trades: {sim['n_trades']}  Days: {stats['n_days']}")
    dd_vals = dd[dd < -0.001]
    print(f"  Max DD: {dd.min()*100:.2f}%  Mean DD: {dd_vals.mean()*100:.2f}%  "
          f"P95 DD: {dd_vals.quantile(0.05)*100:.2f}%")
    if len(dd_periods) >= 3:
        print(f"  Top 3 DDs: {dd_periods[0]['trough']:.1f}%, {dd_periods[1]['trough']:.1f}%, "
              f"{dd_periods[2]['trough']:.1f}%")
    if durations:
        print(f"  Avg underwater: {np.mean(durations):.1f}d  Max: {max(durations)}d  "
              f"P90: {np.percentile(durations, 90):.0f}d")
    print(f"  Win rate: {(dpnl > 0).mean() * 100:.1f}%  "
          f"Pos months: {pos_m}/{len(monthly)} ({pos_m / len(monthly) * 100:.0f}%)")
    print(f"  VaR95: ${np.percentile(dpnl, 5):.2f}  "
          f"CVaR95: ${dpnl[dpnl <= np.percentile(dpnl, 5)].mean():.2f}")
    print(f"  Worst day: ${dpnl.min():.2f}  Best day: ${dpnl.max():.2f}")


def main():
    t0 = time.time()
    print("Generating A + B signals...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, res = f.result()
            done += 1
            all_results[asset] = res
            print(f"  [{done:>2d}/30] {asset}", flush=True)

    print(f"\nSignals done in {time.time() - t0:.0f}s")

    print(f"\n{'=' * 80}")
    print("A vs B: DRAWDOWN COMPARISON")
    print(f"{'=' * 80}")

    for variant, label in [("A", "A: Baseline (all bars)"), ("B", "B: CUSUM sampling")]:
        sigs = []
        for ar in all_results.values():
            for ms, sd in ar.get(variant, {}).get(8, []):
                sigs.append(pd.DataFrame(sd))
        if not sigs:
            print(f"\n  {label}: NO SIGNALS")
            continue
        combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
        combined = orig.apply_nco_wf(combined)
        pcfg = orig.get_pcfg()
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 10:
            print(f"\n  {label}: SIM FAILED")
            continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 5:
            print(f"\n  {label}: NO DPNL")
            continue
        stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                                   sim["n_trades"] / len(dpnl))
        dd_report(label, stats, sim, dpnl)

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
