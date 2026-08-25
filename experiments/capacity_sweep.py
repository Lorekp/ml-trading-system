#!/usr/bin/env python
"""Capacity investigation — sweep capital from $10K to $5M with realistic slippage.

Usage:
    PYTHONPATH="src;." python experiments/capacity_sweep.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig
from signal_engine.execution import get_vip_tier
from signal_engine.slippage import load_per_symbol_calibration

CAPITAL_LEVELS = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 5_000_000]


def run_year(yr_signals, cap, port, breaker, improvements, slippage_model, asset_volumes):
    sim = simulate_with_improvements(
        yr_signals, port, breaker, improvements,
        slippage_model=slippage_model, asset_volumes=asset_volumes)
    if sim is None or sim["n_trades"] < 5:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    pnl = float(dpnl.sum())
    running_eq = cap
    daily_rets = []
    eq_series = []
    for dp in dpnl.values:
        r = dp / running_eq if running_eq > 0 else 0.0
        daily_rets.append(r)
        running_eq = max(running_eq + dp, 0.0)
        eq_series.append(running_eq)
    daily_rets = np.array(daily_rets)
    eq_arr = np.array(eq_series)

    sr = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365.25)) if daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd_arr = np.where(peak > 0, (peak - eq_arr) / peak, 0)
    max_dd = float(dd_arr.max() * 100)

    return {
        "days": len(dpnl), "sr": sr, "pnl": pnl,
        "ret_pct": pnl / cap * 100, "max_dd": max_dd,
        "trades": sim["n_trades"],
    }


def main():
    T0 = time.time()

    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    months = pd.period_range("2020-07", "2026-02", freq="M")

    def _regime_sel(ms):
        btc_vol = vol_series.get("BTC")
        n_assets = 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                if current > avail.quantile(0.80): n_assets = 15
                elif current < avail.quantile(0.20): n_assets = 40
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in hold_sigs:
            confs = []
            for tf in hold_sigs[name]:
                s = hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                w = s[((ts >= cutoff) & (ts < ms)).values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            if not confs: continue
            vol = asset_volumes.get(name)
            if vol is None: continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0: continue
            scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:n_assets]]

    def _s8_sel(ms):
        active_core = [a for a in OBVIOUS_CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in OBVIOUS_CORE]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    prepared_all = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared_all.columns: prepared_all[c] = 1.0
        prepared_all[c] = prepared_all[c].fillna(1.0)
    prepared_all["probability"] = prepared_all["probability"].astype(float)
    prepared_all = apply_vol(prepared_all.sort_values("timestamp").reset_index(drop=True), vol_series)

    ts_all = pd.to_datetime(prepared_all["timestamp"])
    if ts_all.dt.tz is not None:
        ts_all = ts_all.dt.tz_localize(None)
    prepared_all["_ts"] = ts_all
    years = sorted(ts_all.dt.year.unique())

    # Load per-symbol calibration
    per_symbol_cal = load_per_symbol_calibration()

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}
    improvements = {"max_signals_per_bar": 20}

    print("=" * 100)
    print("  CAPACITY SWEEP: Config C with realistic slippage (spread + sqrt impact + fees)")
    print(f"  Capital levels: {', '.join(f'${c/1000:.0f}K' if c < 1e6 else f'${c/1e6:.0f}M' for c in CAPITAL_LEVELS)}")
    print(f"  Period: 2020-07 to 2026-02 | Per-year independent sims")
    print("=" * 100)

    # Also run flat slippage baseline at $200K for comparison
    all_results = {}

    for cap in CAPITAL_LEVELS:
        cap_label = f"${cap/1000:.0f}K" if cap < 1_000_000 else f"${cap/1e6:.0f}M"
        vip_name, maker_bps, taker_bps = get_vip_tier(cap)

        port = PortfolioConfig(
            starting_capital=cap, max_dd=0.30, delever_start=0.20, slippage=0.0001,
            max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

        slippage_model = {
            "calibration": per_symbol_cal,
            "vip_tier": (vip_name, maker_bps, taker_bps),
            "max_participation_rate": 0.02,
        }

        year_results = {}
        for yr in years:
            yr_signals = prepared_all[prepared_all["_ts"].dt.year == yr].drop(columns=["_ts"]).reset_index(drop=True)
            if len(yr_signals) < 10:
                continue
            r = run_year(yr_signals, cap, port, breaker, improvements, slippage_model, asset_volumes)
            if r:
                year_results[yr] = r

        if year_results:
            srs = [r["sr"] for r in year_results.values()]
            rets = [r["ret_pct"] for r in year_results.values()]
            dds = [r["max_dd"] for r in year_results.values()]
            trades = [r["trades"] for r in year_results.values()]
            all_results[cap] = {
                "vip": vip_name, "year_results": year_results,
                "median_sr": float(np.median(srs)),
                "median_ret": float(np.median(rets)),
                "median_dd": float(np.median(dds)),
                "mean_sr": float(np.mean(srs)),
                "mean_ret": float(np.mean(rets)),
                "median_trades": int(np.median(trades)),
            }
            t = time.time() - T0
            print(f"  {cap_label:>6} ({vip_name}): median SR={np.median(srs):.2f}  "
                  f"ret={np.median(rets):.0f}%  DD={np.median(dds):.1f}%  "
                  f"trades={int(np.median(trades)):,}  ({t:.0f}s)")

    # Also run flat slippage at $200K for reference
    print("\n  Running flat-slippage baseline at $200K for comparison...")
    flat_port = PortfolioConfig(
        starting_capital=200_000, max_dd=0.30, delever_start=0.20, slippage=0.0001,
        max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    flat_results = {}
    for yr in years:
        yr_signals = prepared_all[prepared_all["_ts"].dt.year == yr].drop(columns=["_ts"]).reset_index(drop=True)
        if len(yr_signals) < 10:
            continue
        r = run_year(yr_signals, 200_000, flat_port, breaker, improvements, None, None)
        if r:
            flat_results[yr] = r

    # ── TABLE 1: Main sweep ──
    print(f"\n{'=' * 100}")
    print("  TABLE 1: CAPACITY SWEEP SUMMARY")
    print(f"{'=' * 100}")
    print(f"  {'Capital':>8} {'VIP':>5} {'Med SR':>7} {'Med Ret%':>9} {'Med DD%':>8} {'Mean SR':>8} {'Mean Ret%':>10} {'Trades':>8}")
    print("  " + "-" * 75)
    for cap in CAPITAL_LEVELS:
        if cap not in all_results:
            continue
        r = all_results[cap]
        cap_label = f"${cap/1000:.0f}K" if cap < 1_000_000 else f"${cap/1e6:.0f}M"
        print(f"  {cap_label:>8} {r['vip']:>5} {r['median_sr']:>7.2f} {r['median_ret']:>8.1f}% "
              f"{r['median_dd']:>7.1f}% {r['mean_sr']:>8.2f} {r['mean_ret']:>9.1f}% {r['median_trades']:>8,}")

    # Flat baseline
    if flat_results:
        flat_srs = [r["sr"] for r in flat_results.values()]
        flat_rets = [r["ret_pct"] for r in flat_results.values()]
        flat_dds = [r["max_dd"] for r in flat_results.values()]
        print("  " + "-" * 75)
        print(f"  {'$200K':>8} {'flat':>5} {np.median(flat_srs):>7.2f} {np.median(flat_rets):>8.1f}% "
              f"{np.median(flat_dds):>7.1f}%  (0.01bps flat slippage — previous model)")

    # ── TABLE 2: Per-year at key levels ──
    key_caps = [c for c in [200_000, 1_000_000, 5_000_000] if c in all_results]
    if key_caps:
        print(f"\n{'=' * 100}")
        print("  TABLE 2: PER-YEAR BREAKDOWN AT KEY CAPITAL LEVELS")
        print(f"{'=' * 100}")
        header = f"  {'Year':<6}"
        for cap in key_caps:
            cap_label = f"${cap/1000:.0f}K" if cap < 1_000_000 else f"${cap/1e6:.0f}M"
            header += f" {'SR':>6} {'Ret%':>7} |"
        print(header)
        cap_labels = []
        for cap in key_caps:
            cap_label = f"${cap/1000:.0f}K" if cap < 1_000_000 else f"${cap/1e6:.0f}M"
            cap_labels.append(cap_label)
        print(f"  {'':>6} " + " ".join(f"{'-- ' + l + ' --':>16}" for l in cap_labels))
        print("  " + "-" * (6 + 17 * len(key_caps)))
        for yr in years:
            row = f"  {yr:<6}"
            for cap in key_caps:
                yr_data = all_results[cap]["year_results"].get(yr)
                if yr_data:
                    row += f" {yr_data['sr']:>6.2f} {yr_data['ret_pct']:>6.1f}% |"
                else:
                    row += f" {'--':>6} {'--':>7} |"
            print(row)

    # ── TABLE 3: Capacity ceiling ──
    print(f"\n{'=' * 100}")
    print("  TABLE 3: CAPACITY CEILING")
    print(f"{'=' * 100}")

    thresholds = [
        ("Median SR drops below 4.0", lambda r: r["median_sr"] < 4.0),
        ("Median SR drops below 3.0", lambda r: r["median_sr"] < 3.0),
        ("Median SR drops below 2.0", lambda r: r["median_sr"] < 2.0),
        ("Median return below 50%", lambda r: r["median_ret"] < 50),
        ("Median return below 20%", lambda r: r["median_ret"] < 20),
        ("Median DD exceeds 10%", lambda r: r["median_dd"] > 10),
    ]

    for label, test_fn in thresholds:
        ceiling = "Never (within tested range)"
        for cap in CAPITAL_LEVELS:
            if cap in all_results and test_fn(all_results[cap]):
                cap_label = f"${cap/1000:.0f}K" if cap < 1_000_000 else f"${cap/1e6:.0f}M"
                ceiling = cap_label
                break
        print(f"  {label:<35} {ceiling}")

    # ── Slippage cost comparison ──
    if 200_000 in all_results and flat_results:
        real_sr = all_results[200_000]["median_sr"]
        flat_sr = np.median(flat_srs)
        real_ret = all_results[200_000]["median_ret"]
        flat_ret = np.median(flat_rets)
        print(f"\n  SLIPPAGE IMPACT AT $200K:")
        print(f"    Flat (0.01bps):    SR {flat_sr:.2f}  Return {flat_ret:.1f}%")
        print(f"    Realistic model:   SR {real_sr:.2f}  Return {real_ret:.1f}%")
        print(f"    Cost of reality:   SR {real_sr - flat_sr:+.2f}  Return {real_ret - flat_ret:+.1f}%")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
