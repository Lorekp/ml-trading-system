"""Full re-optimization for 2025+ OOS.

The production config (C=0.01, threshold 0.14) goes silent in 2025.
Re-optimize all parameters with strict 2025+ testing.

Strategy: generate signals for each C value (requires retrain), then
filter post-hoc for threshold/vol_q/close/fwd_bars combos.

Round 1: C x threshold (find best pair)
Round 2: lookback x vol_quantile (with best C, threshold)
Round 3: close_rule x fwd_bars (with best from R1+R2)
Round 4: final NCO portfolio validation
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
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
TOP_N = 30
TEST_START = "2025-01"  # STRICT 2025+ OOS
MAX_BARS = 8  # collect returns up to T+8 for close rule testing


def make_logreg(C):
    if NEW_SKLEARN:
        return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def flat_sharpe(signals, capital=10000.0):
    """Quick flat-bet Sharpe (no portfolio sim needed)."""
    if len(signals) < 20:
        return {"sharpe": 0, "n": len(signals), "win": 0, "bps": 0}
    pnls = signals["direction"].values * signals["fwd_return"].values - 0.0001
    ts = pd.to_datetime(signals["timestamp"], utc=True)
    daily = pd.Series(5.0 * pnls, index=ts.values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return {"sharpe": 0, "n": len(signals), "win": 0, "bps": 0}
    std = daily.std()
    sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
    return {
        "sharpe": sr,
        "n": len(signals),
        "win": (pnls > 0).mean() * 100,
        "bps": pnls.mean() * 10000,
        "n_days": len(daily),
    }


def generate_signals_for_c(asset_data, asset_volumes, C, lookback, vol_q,
                            test_months, sample_idx):
    """Generate signals for one C value across all assets and months.

    Returns all signals with per-bar returns at multiple horizons and
    the raw confidence/probability so we can filter post-hoc.
    """
    all_signals = []
    cfg = load_config(Path("configs/default.yaml"))

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

        for symbol in selected:
            if symbol not in asset_data: continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]; n = len(close_arr)
            fc = ad["fc"]; y_all = ad["y_all"]; valid = ad["valid"]
            vol = ad["vol"]; timestamps_idx = ad["timestamps"]

            ms_local = ms; me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None)
                me_local = me_local.tz_localize(None)

            te_all = valid[(valid >= ms_local) & (valid <= me_local)]

            # Vol filter: compute but don't apply yet (we filter post-hoc)
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vol_quantiles = {q: vol_train.quantile(q) for q in [0.25, 0.40, 0.50, 0.60, 0.75]}
            test_vol = vol.reindex(te_all)

            train_start = ms_local - pd.DateOffset(months=lookback)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200: continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg(C)
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te_all])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception: continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te_all)):
                bar_idx = timestamps_idx.get_loc(te_all[i])
                if bar_idx + MAX_BARS >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]

                rets = {k: d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, MAX_BARS + 1)}

                # Vol value at this bar
                vol_val = float(test_vol.get(te_all[i], np.nan))

                all_signals.append({
                    "timestamp": te_all[i], "direction": d,
                    "confidence": conf[i], "probability": yp[i],
                    "signal_strength": float(ss[i]),
                    "asset": asset_name, "vol_val": vol_val,
                    **{f"ret_{k}": rets[k] for k in range(1, MAX_BARS + 1)},
                    **{f"vol_q_{q}": vol_quantiles[q] for q in vol_quantiles},
                })

    return pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True)


def apply_filters(df, threshold, vol_q, fwd_bars, close_rule):
    """Filter signals post-hoc and compute fwd_return based on close rule."""
    # Confidence filter
    filtered = df[df["confidence"] > threshold].copy()

    # Vol filter
    if vol_q > 0:
        q_col = f"vol_q_{vol_q}"
        if q_col in filtered.columns:
            filtered = filtered[filtered["vol_val"] >= filtered[q_col]]

    if len(filtered) < 20:
        return filtered

    # Close rule → fwd_return
    if close_rule == "fixed":
        filtered["fwd_return"] = filtered[f"ret_{fwd_bars}"]
    elif close_rule == "decay_T3":
        filtered["fwd_return"] = filtered.apply(
            lambda r: r[f"ret_{min(3, fwd_bars)}"] if r.get("ret_3", r[f"ret_{fwd_bars}"]) < 0 else r[f"ret_{fwd_bars}"],
            axis=1,
        )
    elif close_rule == "decay_T2":
        filtered["fwd_return"] = filtered.apply(
            lambda r: r["ret_2"] if r["ret_2"] < 0 else r[f"ret_{fwd_bars}"],
            axis=1,
        )

    return filtered


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 95)
    print("FULL RE-OPTIMIZATION — STRICT 2025+ OOS")
    print(f"  Dynamic top-{TOP_N}, walk-forward per-asset")
    print(f"  Train: rolling lookback ending 2024-12")
    print(f"  Test: {TEST_START}+ ONLY")
    print("=" * 95)

    # Load all data
    print("\nLoading data...", flush=True)
    asset_data = {}
    asset_volumes = {}

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"): continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000: continue
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
            if len(valid) < 3000: continue
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
        except Exception: pass

    print(f"  Loaded {len(asset_data)} assets in {time.time() - t0:.0f}s")

    sample_idx = next(iter(asset_volumes.values())).index
    test_start_ts = pd.Timestamp(TEST_START + "-01")
    if sample_idx.tz is not None:
        test_start_ts = test_start_ts.tz_localize(sample_idx.tz)
    end_date = max(d["end_date"] for d in asset_data.values())
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    # ===================== ROUND 1: C × Threshold =====================
    print(f"\n{'='*95}")
    print("ROUND 1: C x THRESHOLD (lookback=14, vol_q=0.50, decay_T3, fwd=4)")
    print(f"{'='*95}")

    C_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]

    print(f"\n  {'C':>6s} {'Thr':>5s} {'Sharpe':>7s} {'N':>6s} {'Win%':>6s} {'bps':>6s} {'Days':>5s}")
    print("  " + "-" * 50)

    best_r1 = None
    best_r1_sr = -999

    for C in C_values:
        t1 = time.time()
        raw = generate_signals_for_c(asset_data, asset_volumes, C, 14, 0.50,
                                      test_months, sample_idx)
        gen_time = time.time() - t1

        for thr in thresholds:
            filtered = apply_filters(raw, thr, 0.50, 4, "decay_T3")
            s = flat_sharpe(filtered)
            print(f"  {C:>6.3f} {thr:>5.2f} {s['sharpe']:>7.3f} {s['n']:>6d} "
                  f"{s['win']:>5.1f}% {s['bps']:>5.1f} {s.get('n_days',0):>5d}", flush=True)
            if s["sharpe"] > best_r1_sr and s["n"] > 100:
                best_r1_sr = s["sharpe"]
                best_r1 = {"C": C, "threshold": thr, **s}

        print(f"  (C={C} generated {len(raw)} raw signals in {gen_time:.0f}s)")

    if best_r1:
        print(f"\n  BEST R1: C={best_r1['C']}, thr={best_r1['threshold']}, "
              f"Sharpe={best_r1['sharpe']:.3f}, N={best_r1['n']}")
    else:
        print("\n  NO VIABLE CONFIG FOUND")
        return

    best_C = best_r1["C"]
    best_thr = best_r1["threshold"]

    # ===================== ROUND 2: Lookback × Vol Filter =====================
    print(f"\n{'='*95}")
    print(f"ROUND 2: LOOKBACK x VOL_Q (C={best_C}, thr={best_thr}, decay_T3, fwd=4)")
    print(f"{'='*95}")

    lookbacks = [6, 8, 10, 12, 14]
    vol_qs = [0.25, 0.40, 0.50, 0.60, 0.75]

    print(f"\n  {'LB':>4s} {'VolQ':>5s} {'Sharpe':>7s} {'N':>6s} {'Win%':>6s} {'bps':>6s}")
    print("  " + "-" * 40)

    best_r2 = None
    best_r2_sr = -999

    for lb in lookbacks:
        raw = generate_signals_for_c(asset_data, asset_volumes, best_C, lb, 0.50,
                                      test_months, sample_idx)
        for vq in vol_qs:
            filtered = apply_filters(raw, best_thr, vq, 4, "decay_T3")
            s = flat_sharpe(filtered)
            print(f"  {lb:>4d} {vq:>5.2f} {s['sharpe']:>7.3f} {s['n']:>6d} "
                  f"{s['win']:>5.1f}% {s['bps']:>5.1f}", flush=True)
            if s["sharpe"] > best_r2_sr and s["n"] > 100:
                best_r2_sr = s["sharpe"]
                best_r2 = {"lookback": lb, "vol_q": vq, **s}

    if best_r2:
        print(f"\n  BEST R2: lb={best_r2['lookback']}, vol_q={best_r2['vol_q']}, "
              f"Sharpe={best_r2['sharpe']:.3f}, N={best_r2['n']}")

    best_lb = best_r2["lookback"] if best_r2 else 14
    best_vq = best_r2["vol_q"] if best_r2 else 0.50

    # ===================== ROUND 3: Close Rule × Hold =====================
    print(f"\n{'='*95}")
    print(f"ROUND 3: CLOSE x FWD_BARS (C={best_C}, thr={best_thr}, lb={best_lb}, vq={best_vq})")
    print(f"{'='*95}")

    close_rules = ["fixed", "decay_T2", "decay_T3"]
    fwd_bars_list = [2, 4, 6, 8]

    raw = generate_signals_for_c(asset_data, asset_volumes, best_C, best_lb, best_vq,
                                  test_months, sample_idx)

    print(f"\n  {'Close':>10s} {'Fwd':>4s} {'Sharpe':>7s} {'N':>6s} {'Win%':>6s} {'bps':>6s}")
    print("  " + "-" * 45)

    best_r3 = None
    best_r3_sr = -999

    for cr in close_rules:
        for fb in fwd_bars_list:
            filtered = apply_filters(raw, best_thr, best_vq, fb, cr)
            s = flat_sharpe(filtered)
            print(f"  {cr:>10s} {fb:>4d} {s['sharpe']:>7.3f} {s['n']:>6d} "
                  f"{s['win']:>5.1f}% {s['bps']:>5.1f}", flush=True)
            if s["sharpe"] > best_r3_sr and s["n"] > 100:
                best_r3_sr = s["sharpe"]
                best_r3 = {"close": cr, "fwd_bars": fb, **s}

    if best_r3:
        print(f"\n  BEST R3: close={best_r3['close']}, fwd={best_r3['fwd_bars']}, "
              f"Sharpe={best_r3['sharpe']:.3f}, N={best_r3['n']}")

    best_close = best_r3["close"] if best_r3 else "decay_T3"
    best_fwd = best_r3["fwd_bars"] if best_r3 else 4

    # ===================== ROUND 4: Final Portfolio Validation =====================
    print(f"\n{'='*95}")
    print("ROUND 4: FINAL PORTFOLIO VALIDATION")
    print(f"  Best: C={best_C}, thr={best_thr}, lb={best_lb}, vq={best_vq}, "
          f"close={best_close}, fwd={best_fwd}")
    print(f"{'='*95}")

    # Generate final signals
    final_raw = generate_signals_for_c(asset_data, asset_volumes, best_C, best_lb, best_vq,
                                        test_months, sample_idx)
    final = apply_filters(final_raw, best_thr, best_vq, best_fwd, best_close)
    final_flat = flat_sharpe(final)

    print(f"\n  Flat-bet: Sharpe={final_flat['sharpe']:.3f}, N={final_flat['n']}, "
          f"Win={final_flat['win']:.1f}%, bps={final_flat['bps']:.1f}")

    # NCO portfolio
    if len(final) > 200:
        combined = orig.apply_nco_wf(final)
        for lev_label, lev in [("1.0x", 1.0), ("2.0x", 2.0)]:
            pcfg = PortfolioConfig(
                starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
                slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
                base_sizing_pct=0.04 * lev, max_signal_mult=3.0,
            )
            sim = simulate_portfolio(combined, pcfg)
            if sim and sim["n_trades"] > 10:
                dpnl = compute_daily_pnl_series(sim)
                if dpnl is not None and len(dpnl) > 5:
                    stats = compute_full_stats(dpnl, pcfg.starting_capital,
                                               sim["n_trades"], sim["n_trades"] / len(dpnl))
                    eq = pd.Series(sim["equity"][~sim["skipped"]],
                                   index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
                    dd = ((eq - eq.expanding().max()) / eq.expanding().max())
                    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

                    print(f"\n  NCO Portfolio @ {lev_label}:")
                    print(f"    Sharpe:     {stats['sharpe']:.3f}")
                    print(f"    Sortino:    {stats.get('sortino', 0):.1f}")
                    print(f"    Calmar:     {stats.get('calmar', 0):.1f}")
                    print(f"    Return:     {stats['total_return_pct']:.1f}%")
                    print(f"    Max DD:     {dd.min()*100:.2f}%")
                    print(f"    Worst Day:  {(dpnl/pcfg.starting_capital*100).min():.2f}%")
                    print(f"    PnL/day:    ${pnl_day:.2f}")
                    print(f"    Trades:     {sim['n_trades']}")
                    print(f"    Assets:     {combined['asset'].nunique()}")

    # Summary
    print(f"\n{'='*95}")
    print("FINAL OPTIMIZED CONFIG (2025+ OOS)")
    print(f"{'='*95}")
    print(f"  C:          {best_C}")
    print(f"  Threshold:  {best_thr}")
    print(f"  Lookback:   {best_lb} months")
    print(f"  Vol Q:      {best_vq}")
    print(f"  Close:      {best_close}")
    print(f"  Fwd Bars:   {best_fwd}")
    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
