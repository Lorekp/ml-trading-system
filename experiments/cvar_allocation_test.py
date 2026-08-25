"""CVaR vs Variance portfolio allocation comparison.

Compares NCO (inverse-variance), NCO-CVaR (expected shortfall),
and equal-weight allocation through the full walk-forward pipeline.

Tests at multiple leverage levels for prop firm feasibility.
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.allocation import (
    compute_nco_weights,
    compute_nco_cvar_weights,
    compute_kelly_weights,
)
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
TOP_N = 30


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def apply_allocation_wf(combined, method="nco"):
    """Walk-forward allocation with selectable method."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []

    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]

        if len(past_sigs) < 50:
            parts.append(km_sigs)
            continue

        assets_in_month = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for a in assets_in_month:
            a_past = past_sigs[past_sigs["asset"] == a]
            if len(a_past) < 10:
                continue
            ts_a = pd.to_datetime(a_past["timestamp"])
            rets = pd.Series(a_past["direction"].values * a_past["fwd_return"].values, index=ts_a)
            daily_rets[a] = rets.resample("D").sum()

        if len(daily_rets) < 3:
            # Fall back to Kelly
            kw = compute_kelly_weights(past_sigs)
            kmc = km_sigs.copy()
            kmc["signal_strength"] = kmc.apply(
                lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
            parts.append(kmc)
            continue

        ret_df = pd.DataFrame(daily_rets).fillna(0)
        names = list(ret_df.columns)
        exp_r = np.array([
            float((past_sigs[past_sigs["asset"] == a]["direction"]
                   * past_sigs[past_sigs["asset"] == a]["fwd_return"]).mean())
            if a in past_sigs["asset"].values else 0.0
            for a in names
        ])

        if method == "nco":
            corr = ret_df.corr().values
            weights = compute_nco_weights(corr, exp_r, names, n_clusters=8)
        elif method == "nco_cvar":
            weights = compute_nco_cvar_weights(ret_df, exp_r, names, n_clusters=8, alpha=0.05)
        elif method == "equal":
            weights = {a: 1.0 for a in names}
        else:
            weights = {a: 1.0 for a in names}

        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)

    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_variant(signals_df, method, leverage):
    """Run one allocation + leverage variant."""
    combined = apply_allocation_wf(signals_df, method=method)

    pcfg = PortfolioConfig(
        starting_capital=10000.0,
        max_dd=0.20,
        delever_start=0.10,
        slippage=0.0001,
        max_exposure=0.90,
        max_per_asset=0.05,
        base_sizing_pct=0.04 * leverage,
        max_signal_mult=3.0,
    )

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
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    # Days to 10%
    cumulative = dpnl.cumsum()
    target = pcfg.starting_capital * 0.10
    days_to_target = None
    above = cumulative[cumulative >= target]
    if len(above) > 0:
        days_to_target = (above.index[0] - cumulative.index[0]).days

    return {
        "sharpe": stats["sharpe"],
        "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0),
        "ret_pct": stats["total_return_pct"],
        "max_dd": dd.min() * 100,
        "worst_day": (dpnl / pcfg.starting_capital * 100).min(),
        "pnl_day": pnl_day,
        "win_daily": (dpnl > 0).mean() * 100,
        "n_trades": sim["n_trades"],
        "final_eq": stats["final_equity"],
        "days_to_10pct": days_to_target,
    }


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 90)
    print("CVaR vs VARIANCE ALLOCATION COMPARISON")
    print(f"  Dynamic top-{TOP_N}, decay_T3, OOS {TEST_START}+")
    print("=" * 90)

    # ==================== PHASE 1: Generate signals ====================
    print("\nPhase 1: Loading data + generating signals...", flush=True)

    asset_data = {}
    asset_volumes = {}

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000:
                continue
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
                continue
            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
            close_arr = ohlcv_15m["close"].values.astype(np.float64)
            timestamps = ohlcv_15m.index

            asset_data[symbol] = {
                "close": close, "close_arr": close_arr, "fc": fc,
                "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "timestamps": timestamps,
                "start_date": ohlcv_15m.index.min(),
                "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception:
            pass

    print(f"  Loaded {len(asset_data)} assets in {time.time() - t0:.0f}s")

    # Walk-forward signal generation with decay_T3
    sample_idx = next(iter(asset_volumes.values())).index
    test_start_ts = pd.Timestamp(TEST_START + "-01")
    if sample_idx.tz is not None:
        test_start_ts = test_start_ts.tz_localize(sample_idx.tz)
    end_date = max(d["end_date"] for d in asset_data.values())
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    all_signals = []
    print("\nGenerating signals (decay_T3)...", flush=True)

    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)
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

        month_count = 0
        for symbol in selected:
            if symbol not in asset_data: continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]
            n = len(close_arr)
            fc = ad["fc"]; y_all = ad["y_all"]; valid = ad["valid"]
            vol = ad["vol"]; timestamps_idx = ad["timestamps"]

            ms_local = ms
            me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None)
                me_local = me_local.tz_localize(None)

            te = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vt = vol_train.quantile(0.50)
            test_vol = vol.reindex(te)
            hv = (test_vol > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5: continue

            train_start = ms_local - pd.DateOffset(months=LB)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200: continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg()
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception: continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            mask = conf > CONF_THR
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te)):
                if not mask[i]: continue
                bar_idx = timestamps_idx.get_loc(te[i])
                if bar_idx + 4 >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]
                rets = [d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, 5)]
                exit_bar = 3 if rets[2] < 0 else 4

                all_signals.append({
                    "timestamp": te[i], "direction": d, "confidence": conf[i],
                    "probability": yp[i], "signal_strength": float(ss[i]),
                    "asset": asset_name, "fwd_return": rets[exit_bar - 1],
                })
                month_count += 1

        print(f"  {str(month)}: {len(selected)} assets, {month_count} signals", flush=True)

    signals_df = pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True)
    print(f"\n  Total: {len(signals_df)} signals, {signals_df['asset'].nunique()} assets")

    # ==================== PHASE 2: Compare allocations ====================
    print(f"\n{'='*90}")
    print("RESULTS: CVaR vs VARIANCE ALLOCATION")
    print(f"{'='*90}")

    methods = ["equal", "nco", "nco_cvar"]
    leverages = [1.0, 1.7, 2.0]

    print(f"\n  {'Method':<12s} {'Lev':>4s} {'Sharpe':>7s} {'Sortino':>8s} {'Calmar':>7s} "
          f"{'MaxDD%':>7s} {'WrstDay%':>8s} {'PnL/d':>7s} {'Days10%':>7s} {'Trades':>6s}")
    print("  " + "-" * 85)

    results = []
    for lev in leverages:
        for method in methods:
            t1 = time.time()
            r = run_variant(signals_df, method, lev)
            dt = time.time() - t1
            if r is None:
                print(f"  {method:<12s} {lev:>4.1f} -- NO DATA --")
                continue

            results.append({"method": method, "leverage": lev, **r})
            d10 = f"{r['days_to_10pct']}" if r['days_to_10pct'] else "N/A"
            print(f"  {method:<12s} {lev:>4.1f} {r['sharpe']:>7.3f} {r['sortino']:>8.1f} "
                  f"{r['calmar']:>7.1f} {r['max_dd']:>7.2f} {r['worst_day']:>8.2f} "
                  f"${r['pnl_day']:>6.2f} {d10:>7s} {r['n_trades']:>6d}  ({dt:.0f}s)")

    # Summary
    print(f"\n{'='*90}")
    print("COMPARISON AT 2.0x LEVERAGE (prop firm target)")
    print(f"{'='*90}")
    for method in methods:
        r = next((x for x in results if x["method"] == method and x["leverage"] == 2.0), None)
        if r:
            d10 = f"{r['days_to_10pct']} days" if r['days_to_10pct'] else "N/A"
            print(f"\n  {method.upper()}")
            print(f"    Sharpe:     {r['sharpe']:.3f}")
            print(f"    Sortino:    {r['sortino']:.1f}")
            print(f"    Calmar:     {r['calmar']:.1f}")
            print(f"    Max DD:     {r['max_dd']:.2f}%")
            print(f"    Worst Day:  {r['worst_day']:.2f}%")
            print(f"    PnL/day:    ${r['pnl_day']:.2f}")
            print(f"    Days to 10%: {d10}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
