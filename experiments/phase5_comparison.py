"""Phase 5: Full comparison — baseline vs optimized config.

Baseline: 18f_C0.01, 14mo, thr=0.25, uniform sizing
Optimized: 18f_C0.01, 14mo, thr=0.25, skip high-vol regime

Also tests expanding lookback for both.
Full stats: Sharpe, Sortino, Calmar, DD, daily/monthly/yearly PnL, streaks, VaR.
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
TEST_START = "2024-01"
CONF_THRESHOLDS = [0.10, 0.12, 0.14]  # test multiple


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


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
    vol_20d = close.pct_change().rolling(20 * 24 * 4).std() * np.sqrt(365.25 * 24 * 4)

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)
    test_months = pd.period_range(
        start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    # key: (lookback_label) -> signals with vol_regime
    results = {"14mo": [], "exp": []}

    for month in test_months:
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

        for lb_key, lb_months in [("14mo", 14), ("exp", -1)]:
            if lb_months == -1:
                train_start = start_date
            else:
                train_start = ms - pd.DateOffset(months=lb_months)
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

            vol_20d_te = vol_20d.reindex(te).values

            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
                "vol_regime": vol_20d_te[finite],
            })
            signals["asset"] = asset
                        # Don't filter here — we'll filter at analysis time
            pass  # keep all signals, filter by threshold later
            if len(signals) > 0:
                results[lb_key].append(signals)

    return asset, results


def full_stats(combined, pcfg, label):
    if len(combined) < 50:
        print(f"\n  {label}: INSUFFICIENT DATA ({len(combined)} signals)")
        return

    combined_nco = orig.apply_nco_wf(combined)
    sim = simulate_portfolio(combined_nco, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"\n  {label}: SIM FAILED")
        return
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        print(f"\n  {label}: NO DPNL")
        return
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))

    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    peak = equity.expanding().max()
    dd = (equity - peak) / peak

    # Drawdown periods
    dd_periods = []
    in_dd = False
    dd_start = dd_trough = None
    for ts_i, dd_val in dd.items():
        if dd_val < -0.001 and not in_dd:
            in_dd = True; dd_start = ts_i; dd_trough = dd_val
        elif dd_val < -0.001 and in_dd:
            if dd_val < dd_trough: dd_trough = dd_val
        elif dd_val >= -0.001 and in_dd:
            in_dd = False
            dd_periods.append({"trough": dd_trough * 100, "dur": (ts_i - dd_start).days})
    if in_dd and dd_start:
        dd_periods.append({"trough": dd_trough * 100, "dur": (dd.index[-1] - dd_start).days})
    dd_periods.sort(key=lambda x: x["trough"])

    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    # Streaks
    daily_win = (dpnl > 0).astype(int)
    streaks = []
    cur = 1
    for i in range(1, len(daily_win)):
        if daily_win.iloc[i] == daily_win.iloc[i-1]:
            cur += 1
        else:
            streaks.append((daily_win.iloc[i-1], cur))
            cur = 1
    streaks.append((daily_win.iloc[-1], cur))
    win_s = [s for v, s in streaks if v == 1]
    loss_s = [s for v, s in streaks if v == 0]

    # Flat bet stats
    pnls_flat = combined["direction"].values * combined["fwd_return"].values - 0.0001
    fb_wr = (pnls_flat > 0).mean() * 100
    fb_bps = pnls_flat.mean() * 10000

    print(f"\n  {'='*65}")
    print(f"  {label}")
    print(f"  {'='*65}")
    print(f"  Sharpe:           {stats['sharpe']:.3f}")
    print(f"  Sortino:          {stats.get('sortino', 0):.3f}")
    print(f"  Calmar:           {stats.get('calmar', 0):.3f}")
    print(f"  Total Return:     {stats['total_return_pct']:.1f}%")
    print(f"  Final Equity:     ${stats['final_equity']:,.2f}")
    print(f"  PnL/day:          ${pnl_day:.2f} ({pnl_day/pcfg.starting_capital*100:.3f}%)")
    print(f"  Max Drawdown:     {dd.min()*100:.2f}%")
    print(f"  Worst Day:        {(dpnl/pcfg.starting_capital*100).min():.2f}%")
    print(f"  VaR 95%:          ${np.percentile(dpnl, 5):.2f}")
    print(f"  CVaR 95%:         ${dpnl[dpnl <= np.percentile(dpnl, 5)].mean():.2f}")
    print(f"  ---")
    print(f"  Trades (sim):     {sim['n_trades']}")
    print(f"  Trades (signals): {len(combined)}")
    print(f"  Days:             {stats['n_days']}")
    print(f"  Daily Win%:       {(dpnl>0).mean()*100:.1f}%")
    print(f"  Flat-bet Win%:    {fb_wr:.1f}%")
    print(f"  Flat-bet bps:     {fb_bps:.1f}")
    print(f"  ---")
    print(f"  Pos Months:       {pos_m}/{len(monthly)} ({pos_m/max(len(monthly),1)*100:.0f}%)")
    if monthly.max() > 0:
        print(f"  Best Month:       ${monthly.max():.2f} ({monthly.idxmax().strftime('%Y-%m')})")
    if monthly.min() < 0:
        print(f"  Worst Month:      ${monthly.min():.2f} ({monthly.idxmin().strftime('%Y-%m')})")
    print(f"  Max Win Streak:   {max(win_s) if win_s else 0}d")
    print(f"  Max Loss Streak:  {max(loss_s) if loss_s else 0}d")
    if dd_periods:
        print(f"  Top 3 DDs:        {dd_periods[0]['trough']:.2f}%"
              + (f", {dd_periods[1]['trough']:.2f}%" if len(dd_periods) > 1 else "")
              + (f", {dd_periods[2]['trough']:.2f}%" if len(dd_periods) > 2 else ""))
        durations = [p["dur"] for p in dd_periods]
        print(f"  Avg Underwater:   {np.mean(durations):.1f}d")
        print(f"  Max Underwater:   {max(durations)}d")
    # Yearly
    yearly = dpnl.resample("YE").sum()
    for dt, pnl in yearly.items():
        print(f"  {dt.year}: ${pnl:>10.2f}")


def main():
    t0 = time.time()
    print("=" * 80)
    print("FULL COMPARISON: BASELINE vs OPTIMIZED")
    print(f"  18f_C0.01, thr=0.25, 14mo + expanding")
    print(f"  Baseline = uniform sizing")
    print(f"  Optimized = skip high-vol regime")
    print("=" * 80, flush=True)

    all_signals = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, res = f.result()
            done += 1
            all_signals[asset] = res
            n14 = sum(len(s) for s in res.get("14mo", []))
            nex = sum(len(s) for s in res.get("exp", []))
            print(f"  [{done:>2d}/30] {asset}: 14mo={n14} exp={nex}", flush=True)

    print(f"\n  Signals done in {time.time() - t0:.0f}s")

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.21, max_signal_mult=2.0,  # ~5.2x
    )

    for lb_key, lb_label in [("14mo", "14-MONTH ROLLING"), ("exp", "EXPANDING")]:
        parts = []
        for sigs_list in all_signals.values():
            for s in sigs_list.get(lb_key, []):
                parts.append(s)
        if not parts:
            print(f"\n  {lb_label}: NO DATA")
            continue
        combined = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        print(f"\n  {lb_label}: {len(combined)} total signals, {combined['asset'].nunique()} assets")

        # Compute vol regime terciles
        vol_vals = combined["vol_regime"].dropna()
        if len(vol_vals) > 30:
            try:
                terciles = pd.qcut(vol_vals, 3, labels=["Low", "Med", "High"], duplicates="drop")
            except ValueError:
                terciles = pd.Series("Med", index=vol_vals.index)
        else:
            terciles = pd.Series("Med", index=combined.index)

        # A: Baseline (uniform)
        combined_a = combined.drop(columns=["vol_regime"]).copy()
        full_stats(combined_a, pcfg, f"{lb_label} — A: BASELINE (uniform sizing)")

        # B: Skip high-vol
        combined_b = combined[terciles.values != "High"].drop(columns=["vol_regime"]).copy()
        full_stats(combined_b, pcfg, f"{lb_label} — B: SKIP HIGH-VOL")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
