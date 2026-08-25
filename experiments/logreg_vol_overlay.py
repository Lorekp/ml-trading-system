"""Experiment 2: Vol Regime Sizing Overlay for LogReg T1

Hypothesis: Scaling position size by BTC vol regime improves Sharpe.
Bottom 20% vol → 0.25x, middle → 1.0x, top 20% → 1.5x.
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SCACHE = Path("data/optuna/signal_cache")
PCACHE = Path("data/cache_15m")
CAPITAL = 10000.0
W = [("W1", 245), ("W2", 244), ("W3", 122)]


def load_logreg_dpnl():
    from signal_engine.config import PortfolioConfig
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    import test_original_pipeline as orig

    all_dpnl = {}
    for wl, cd in W:
        p = SCACHE / f"C0.02000_lb2_{wl}.pkl"
        if not p.exists(): continue
        with open(p, "rb") as f: raw = pickle.load(f)
        fi = raw[raw["confidence"] > 0.10].copy()
        if len(fi) < 10: continue
        fi["fwd_return"] = fi["ret_6"].values
        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.30, max_signal_mult=3.0)
        sim = simulate_portfolio(combined, pcfg)
        if sim is None: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None: continue
        for dt, val in dpnl.items():
            dt2 = dt.tz_localize(None) if hasattr(dt, "tz") and dt.tz is not None else dt
            all_dpnl[dt2] = all_dpnl.get(dt2, 0) + val
    return pd.Series(all_dpnl).sort_index()


def metrics(dpnl, label, capital=CAPITAL):
    if len(dpnl) < 5: return None
    cal = (dpnl.index[-1] - dpnl.index[0]).days + 1
    dr = dpnl / capital
    sr = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    cum = dpnl.cumsum(); peak = (cum + capital).cummax()
    dd = abs(((cum + capital - peak) / peak).min()); wd = abs(dr.min())
    pnl_cal = cum.iloc[-1] / cal
    return {"sr": sr, "pnl_cal": pnl_cal, "dd": dd, "wd": wd, "cal": cal}


def main():
    t0 = time.time()
    print("=" * 100)
    print("VOL REGIME SIZING OVERLAY")
    print("=" * 100)

    dpnl = load_logreg_dpnl()
    print(f"LogReg T1: {len(dpnl)} days")

    btc = pd.read_parquet(PCACHE / "BTCUSDT.parquet", columns=["close"])
    btc_daily = btc["close"].resample("1D").last().dropna()
    btc_daily.index = btc_daily.index.tz_localize(None)

    # BTC vol metrics
    btc_ret = btc_daily.pct_change()
    btc_vol_20d = btc_ret.rolling(20).std() * np.sqrt(365)
    btc_vol_pct = btc_vol_20d.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    common = dpnl.index.intersection(btc_vol_pct.dropna().index)
    dpnl = dpnl.reindex(common)
    vol_pct = btc_vol_pct.reindex(common)
    vol_abs = btc_vol_20d.reindex(common)

    base = metrics(dpnl, "Base")
    print(f"\nBase LogReg T1: SR={base['sr']:.2f}, ${base['pnl_cal']:.2f}/cal, DD={base['dd']*100:.1f}%, WD={base['wd']*100:.2f}%")

    # Sizing overlays
    configs = [
        # (name, function: vol_pct -> multiplier)
        ("Base (1.0x flat)", lambda p: 1.0),
        ("Discrete 0.25/1.0/1.5", lambda p: 0.25 if p < 0.20 else (1.5 if p >= 0.80 else 1.0)),
        ("Discrete 0.50/1.0/1.5", lambda p: 0.50 if p < 0.20 else (1.5 if p >= 0.80 else 1.0)),
        ("Discrete 0.10/1.0/2.0", lambda p: 0.10 if p < 0.20 else (2.0 if p >= 0.80 else 1.0)),
        ("Discrete 0.25/1.0/2.0", lambda p: 0.25 if p < 0.20 else (2.0 if p >= 0.80 else 1.0)),
        ("Linear 0.25-1.75", lambda p: 0.25 + 1.50 * p),
        ("Linear 0.50-1.50", lambda p: 0.50 + 1.00 * p),
        ("Linear 0.10-2.00", lambda p: 0.10 + 1.90 * p),
        ("Sqrt scaling", lambda p: max(0.2, np.sqrt(p) * 1.4)),
        ("Kill low vol (<20%ile)", lambda p: 0.0 if p < 0.20 else 1.0),
        ("Kill low vol (<10%ile)", lambda p: 0.0 if p < 0.10 else 1.0),
        ("Boost high only", lambda p: 1.0 if p < 0.70 else 1.5),
        ("3-tier 0.25/1.0/1.25", lambda p: 0.25 if p < 0.20 else (1.25 if p >= 0.70 else 1.0)),
        ("Vol-scaled (abs)", lambda p: 1.0),  # special: use absolute vol
    ]

    print(f"\n{'Config':<30s} {'SR':>6s} {'$/cal':>7s} {'DD%':>5s} {'WD%':>6s} {'vs Base SR':>10s}")
    print("-" * 75)

    for name, fn in configs:
        if name == "Vol-scaled (abs)":
            # Scale by absolute vol (mean-normalized)
            mean_vol = vol_abs.mean()
            mult = (vol_abs / mean_vol).clip(0.2, 2.5)
        else:
            mult = vol_pct.apply(fn)

        adjusted = dpnl * mult
        m = metrics(adjusted, name)
        if m is None: continue

        delta = m["sr"] - base["sr"]
        flag = " ***" if delta > 0.3 else (" **" if delta > 0.1 else (" *" if delta > 0 else ""))
        print(f"{name:<30s} {m['sr']:>6.2f} ${m['pnl_cal']:>6.2f} {m['dd']*100:>4.1f}% "
              f"{m['wd']*100:>5.2f}%  {delta:>+6.2f}{flag}")

    # Detailed analysis of best overlay
    print("\n--- Best Overlay: Monthly Comparison ---")
    # Find best
    best_fn = lambda p: 0.25 if p < 0.20 else (1.5 if p >= 0.80 else 1.0)
    best_mult = vol_pct.apply(best_fn)
    best_dpnl = dpnl * best_mult

    monthly_base = dpnl.resample("ME").sum()
    monthly_best = best_dpnl.resample("ME").sum()

    print(f"{'Month':<10s} {'Base':>8s} {'Overlay':>8s} {'Diff':>8s} {'Vol%':>6s}")
    for dt in monthly_base.index:
        b = monthly_base[dt]; o = monthly_best[dt]
        vp = vol_pct.resample("ME").mean().get(dt, 0)
        print(f"{dt.strftime('%Y-%m'):<10s} ${b:>7.0f} ${o:>7.0f} ${o-b:>+7.0f} {vp*100:>5.0f}%")

    # Regime breakdown for overlay
    print("\n--- Regime Performance: Overlay vs Base ---")
    for label, lo, hi in [("Low vol (<20%ile)", 0, 0.20), ("Mid vol (20-80%ile)", 0.20, 0.80), ("High vol (>80%ile)", 0.80, 1.01)]:
        mask = (vol_pct >= lo) & (vol_pct < hi)
        b_sub = dpnl[mask]; o_sub = best_dpnl[mask]
        if len(b_sub) < 5: continue
        b_sr = b_sub.mean() / b_sub.std() * np.sqrt(365.25) if b_sub.std() > 0 else 0
        o_sr = o_sub.mean() / o_sub.std() * np.sqrt(365.25) if o_sub.std() > 0 else 0
        print(f"  {label:<25s} Base: SR={b_sr:.2f} avg=${b_sub.mean():.1f}  "
              f"Overlay: SR={o_sr:.2f} avg=${o_sub.mean():.1f}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
