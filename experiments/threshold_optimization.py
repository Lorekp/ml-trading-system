"""Confidence threshold optimization for PnL (not just Sharpe).

At higher thresholds: fewer trades, higher bps/trade, higher win rate.
But PnL/day depends on BOTH bps/trade AND trade count.
With leverage (Kelly sizing), higher Sharpe → more leverage → more PnL.

Tests lb=14mo and expanding, fine-grained thresholds, NCO portfolio sim.
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


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    """Generate RAW (unfiltered) signals for lb=14 and expanding."""
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
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)
    test_months = pd.period_range(
        start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    results = {"14": [], "exp": []}

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

        month_str = str(month)
        train_cutoff = ms - embargo_td

        for lb_key, train_start in [
            ("14", ms - pd.DateOffset(months=14)),
            ("exp", start_date),
        ]:
            if train_start < start_date:
                train_start = start_date
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
            results[lb_key].append((month_str, signals.to_dict("list")))

    return asset, results


def analyze_threshold(all_results, lb_key, threshold, pcfg):
    """Apply threshold, run NCO portfolio, return stats."""
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get(lb_key, []):
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
    dd = ((equity - equity.expanding().max()) / equity.expanding().max()).min() * 100
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    # Flat-bet stats too
    pnls_flat = combined["direction"].values * combined["fwd_return"].values - 0.0001
    ts_flat = pd.to_datetime(combined["timestamp"], utc=True)
    daily_flat = pd.Series(5.0 * pnls_flat, index=ts_flat.values).resample("1D").sum()
    daily_flat = daily_flat[daily_flat != 0]
    sr_flat = daily_flat.mean() / daily_flat.std() * np.sqrt(365.25) if len(daily_flat) > 5 and daily_flat.std() > 0 else 0

    # Kelly optimal leverage
    win_rate = (pnls_flat > 0).mean()
    avg_win = pnls_flat[pnls_flat > 0].mean() if (pnls_flat > 0).any() else 0
    avg_loss = abs(pnls_flat[pnls_flat < 0].mean()) if (pnls_flat < 0).any() else 1
    b = avg_win / avg_loss if avg_loss > 0 else 1
    kelly_f = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
    half_kelly = max(0, kelly_f / 2)

    return {
        "sharpe_nco": stats["sharpe"],
        "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0),
        "ret_pct": stats["total_return_pct"],
        "max_dd": dd,
        "pnl_day": pnl_day,
        "final_eq": stats["final_equity"],
        "win_rate_daily": (dpnl > 0).mean() * 100,
        "n_trades": sim["n_trades"],
        "n_days": stats["n_days"],
        "sharpe_flat": sr_flat,
        "win_rate_flat": win_rate * 100,
        "bps_flat": pnls_flat.mean() * 10000,
        "n_trades_flat": len(combined),
        "kelly_f": kelly_f,
        "half_kelly": half_kelly,
        # PnL at various leverage levels (on flat-bet daily)
        "pnl_1x": daily_flat.sum(),
        "pnl_2x": (daily_flat * 2).sum(),
        "pnl_hk": (daily_flat * max(1, half_kelly * 10)).sum(),  # half-kelly scaled
    }


def main():
    t0 = time.time()
    print("=" * 90)
    print("THRESHOLD OPTIMIZATION FOR PnL (lb=14mo + expanding)")
    print(f"  Strict OOS: test {TEST_START}+, {W} workers")
    print("=" * 90, flush=True)

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

    thresholds = [0.00, 0.03, 0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.20, 0.25]

    for lb_key, lb_label in [("14", "14-month rolling"), ("exp", "Expanding")]:
        print(f"\n{'='*90}")
        print(f"  {lb_label.upper()}")
        print(f"{'='*90}")

        # Header
        print(f"  {'Thr':>5s} {'SR_nco':>7s} {'Sort':>6s} {'Calm':>6s} {'Ret%':>7s} "
              f"{'MaxDD':>7s} {'PnL/d':>8s} {'Win%d':>6s} {'SR_fb':>7s} {'Win%':>5s} "
              f"{'bps':>6s} {'Trades':>7s} {'Kelly':>6s} {'HK_PnL':>8s}")
        print("  " + "-" * 105)

        # Run all thresholds with standard portfolio config
        pcfg = orig.get_pcfg()
        best_pnl = -999
        best_thr = 0
        best_sharpe_thr = 0
        best_sharpe = -999

        for thr in thresholds:
            r = analyze_threshold(all_results, lb_key, thr, pcfg)
            if r is None:
                print(f"  {thr:>5.2f}  NO DATA")
                continue

            marker = ""
            if r["pnl_day"] > best_pnl:
                best_pnl = r["pnl_day"]
                best_thr = thr
            if r["sharpe_nco"] > best_sharpe:
                best_sharpe = r["sharpe_nco"]
                best_sharpe_thr = thr

            print(f"  {thr:>5.2f} {r['sharpe_nco']:>7.3f} {r['sortino']:>6.3f} {r['calmar']:>6.3f} "
                  f"{r['ret_pct']:>6.1f}% {r['max_dd']:>6.2f}% ${r['pnl_day']:>7.2f} "
                  f"{r['win_rate_daily']:>5.1f}% {r['sharpe_flat']:>7.3f} {r['win_rate_flat']:>4.1f}% "
                  f"{r['bps_flat']:>5.1f} {r['n_trades_flat']:>7d} {r['half_kelly']:>5.3f} "
                  f"${r['pnl_hk']:>7.0f}")

        print(f"\n  Best PnL/day: threshold={best_thr:.2f} (${best_pnl:.2f}/day)")
        print(f"  Best Sharpe:  threshold={best_sharpe_thr:.2f} (SR={best_sharpe:.3f})")

        # Now test with LEVERAGED portfolio configs at the best thresholds
        print(f"\n  --- Leverage Analysis at best PnL threshold ({best_thr:.2f}) ---")
        for leverage_label, sizing_pct in [
            ("0.5x (conservative)", 0.02),
            ("1x (baseline)", 0.04),
            ("1.5x", 0.06),
            ("2x", 0.08),
            ("3x", 0.12),
        ]:
            pcfg_lev = PortfolioConfig(
                starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
                slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
                base_sizing_pct=sizing_pct, max_signal_mult=2.0,
            )
            r = analyze_threshold(all_results, lb_key, best_thr, pcfg_lev)
            if r:
                print(f"  {leverage_label:<22s} SR={r['sharpe_nco']:>6.3f}  Ret={r['ret_pct']:>7.1f}%  "
                      f"MaxDD={r['max_dd']:>6.2f}%  PnL/d=${r['pnl_day']:>8.2f}  "
                      f"Final=${r['final_eq']:>10,.0f}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
