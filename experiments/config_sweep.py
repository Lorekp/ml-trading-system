#!/usr/bin/env python
"""Extensive config sweep using the EXACT yolo_compound.py pipeline.

Same realistic slippage, compounding, bar-level SL, participation caps.
Sweeps sizing, exposure, deleveraging, correlation throttle, and max_dd.

Usage:
    PYTHONPATH="src;." python experiments/config_sweep.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from itertools import product

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import (
    simulate_with_improvements, apply_bar_level_sl, CACHE_DIR, TF_RESAMPLE,
)
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig
from signal_engine.slippage import load_per_symbol_calibration
from signal_engine.execution import get_vip_tier

# Import run_full from yolo_compound — the exact production pipeline
from experiments.yolo_compound import run_full

CAP = 200_000.0


def main():
    T0 = time.time()
    print("=" * 120)
    print("  EXTENSIVE CONFIG SWEEP — $200K compounding, realistic slippage + bar-level SL")
    print("=" * 120)

    # ── Load data (same as yolo_compound.py) ──
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
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)
    print(f"  {len(prepared):,} signals loaded ({time.time()-T0:.1f}s)")

    # ── Load close cache for bar-level SL ──
    close_cache_path = CACHE_DIR / "close_cache.pkl"
    if close_cache_path.exists():
        with open(close_cache_path, "rb") as f:
            close_cache = pickle.load(f)
        print(f"  Close cache: {len(close_cache)} pairs")
    else:
        print("  ERROR: close_cache.pkl not found. Run yolo_compound.py first to build it.")
        return

    # ── Slippage model (same as yolo_compound) ──
    per_symbol_cal = load_per_symbol_calibration()
    vip_name, maker_bps, taker_bps = get_vip_tier(CAP)
    slippage_model = {
        "calibration": per_symbol_cal,
        "vip_tier": (vip_name, maker_bps, taker_bps),
        "max_participation_rate": 0.02,
        "guard_pct": 0.50,
    }

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    # ══════════════════════════════════════════════════════════
    #  PARAMETER GRID — extensive range
    # ══════════════════════════════════════════════════════════

    sizing_pcts      = [0.03, 0.06, 0.10, 0.15, 0.20, 0.30]
    max_per_assets   = [0.06, 0.15, 0.25]
    max_exposures    = [0.90, 0.95]
    max_dds          = [0.10, 0.20, 0.30, 0.50]
    delever_starts   = [0.05, 0.10, 0.20]
    corr_throttles   = [None, 0.90]  # None = no throttle
    max_sig_mults    = [2.0]

    # Smart pruning: only test combos where delever_start < max_dd
    # and max_per_asset <= max_exposure
    configs = []

    for sz, mpa, mex, mdd, ds, ct, msm in product(
        sizing_pcts, max_per_assets, max_exposures, max_dds,
        delever_starts, corr_throttles, max_sig_mults,
    ):
        # Prune invalid combos
        if ds >= mdd: continue           # delever must start before max_dd
        if mpa > mex: continue           # per-asset can't exceed total
        if sz > mpa * 2: continue        # sizing shouldn't be >> per-asset
        if mdd < 0.10 and sz > 0.15: continue  # tight DD + huge sizing = halts instantly
        if mex < 0.90 and sz > 0.20: continue  # low exposure + high sizing = contradictory

        label = f"sz{sz:.0%}_mpa{mpa:.0%}_exp{mex:.0%}_dd{mdd:.0%}_dl{ds:.0%}"
        if ct: label += f"_ct{ct}"
        if msm != 2.0: label += f"_m{msm:.0f}x"

        imp = {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75}
        if ct:
            imp.update({"correlation_throttle": ct, "correlation_direction": "both",
                        "correlation_scale": 0.5, "max_signals_per_bar": 15})

        port = PortfolioConfig(
            starting_capital=CAP, max_dd=mdd, delever_start=ds, slippage=0.0001,
            max_exposure=mex, max_per_asset=mpa, max_per_strategy=0.50,
            base_sizing_pct=sz, max_signal_mult=msm, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
        )
        configs.append((label, port, imp))

    print(f"  Grid: {len(configs)} configs after pruning")

    # ── Pre-compute SL-adjusted signals per unique sizing value ──
    # SL only depends on base_sizing_pct (max_risk_per_trade and sl_cap are constant).
    # Pre-compute once per sizing → reuse across all configs with same sizing.
    unique_sizings = sorted(set(port.base_sizing_pct for _, port, _ in configs))
    sl_cache = {}
    print(f"  Pre-computing bar-level SL for {len(unique_sizings)} sizing values...")
    for sz in unique_sizings:
        t1 = time.time()
        sl_prepared, sl_count = apply_bar_level_sl(
            prepared.copy(), close_cache,
            max_risk_per_trade=0.03, sl_cap=0.75, base_sizing_pct=sz,
        )
        sl_cache[sz] = sl_prepared
        print(f"    sz={sz:.0%}: {sl_count} SL triggers ({time.time()-t1:.1f}s)")
    print()

    # ── Run sweep (simulation only, SL already applied) ──
    results = []
    for idx, (label, port, imp) in enumerate(configs):
        t1 = time.time()
        # Use pre-computed SL-adjusted signals for this sizing
        prep = sl_cache[port.base_sizing_pct].copy()
        r = run_full(prep, label, port, breaker, imp,
                     slippage_model, asset_volumes, close_cache=None)  # None = skip SL (already done)
        dt = time.time() - t1
        if r:
            r["config"] = {
                "sizing": port.base_sizing_pct, "max_per_asset": port.max_per_asset,
                "max_exposure": port.max_exposure, "max_dd": port.max_dd,
                "delever_start": port.delever_start, "max_signal_mult": port.max_signal_mult,
                "corr_throttle": imp.get("correlation_throttle"),
            }
            results.append(r)
        if (idx + 1) % 25 == 0 or dt > 5:
            sr_str = f"SR={r['sr']:.2f}" if r else "FAIL"
            print(f"  [{idx+1}/{len(configs)}] {label}: {sr_str} ({dt:.1f}s)")

    # ── Dump ALL results to CSV for analysis ──
    csv_rows = []
    for r in results:
        csv_rows.append({
            "label": r["label"],
            "sizing": r["config"]["sizing"],
            "max_per_asset": r["config"]["max_per_asset"],
            "max_exposure": r["config"]["max_exposure"],
            "max_dd": r["config"]["max_dd"],
            "delever_start": r["config"]["delever_start"],
            "corr_throttle": r["config"]["corr_throttle"] or "none",
            "sharpe": r["sr"],
            "cagr": r["cagr"],
            "max_drawdown": r["max_dd"],
            "final_eq": r["final_eq"],
            "trades": r["trades"],
            "win_rate": r["win_rate"],
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = CACHE_DIR / "sweep_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  Saved {len(csv_df)} results to {csv_path}")

    # ── Results table sorted by Sharpe ──
    results.sort(key=lambda x: -x["sr"])

    print(f"\n{'=' * 140}")
    print(f"  TOP 30 BY SHARPE RATIO (of {len(results)} successful configs)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} {'Config':<55} {'SR':>5} {'CAGR':>6} {'MaxDD':>6} {'Final':>14} "
          f"{'Trades':>7} {'WR':>5}")
    print(f"  {'-' * 130}")
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} {r['label']:<55} {r['sr']:>5.2f} {r['cagr']:>5.0f}% "
              f"{r['max_dd']:>5.1f}% ${r['final_eq']:>12,.0f} {r['trades']:>7,} "
              f"{r['win_rate']:>4.0f}%")

    # ── Top 30 by CAGR ──
    by_cagr = sorted(results, key=lambda x: -x["cagr"])
    print(f"\n{'=' * 140}")
    print(f"  TOP 30 BY CAGR")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} {'Config':<55} {'SR':>5} {'CAGR':>6} {'MaxDD':>6} {'Final':>14} "
          f"{'Trades':>7} {'WR':>5}")
    print(f"  {'-' * 130}")
    for i, r in enumerate(by_cagr[:30]):
        print(f"  {i+1:>3} {r['label']:<55} {r['sr']:>5.2f} {r['cagr']:>5.0f}% "
              f"{r['max_dd']:>5.1f}% ${r['final_eq']:>12,.0f} {r['trades']:>7,} "
              f"{r['win_rate']:>4.0f}%")

    # ── Top 30 by Sharpe * sqrt(CAGR) (risk-adjusted growth) ──
    by_bang = sorted(results, key=lambda x: -x["sr"] * np.sqrt(max(x["cagr"], 0.01)))
    print(f"\n{'=' * 140}")
    print(f"  TOP 30 BY SHARPE × sqrt(CAGR)  (risk-adjusted growth)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} {'Config':<55} {'SR':>5} {'CAGR':>6} {'MaxDD':>6} {'Final':>14} "
          f"{'Score':>7}")
    print(f"  {'-' * 130}")
    for i, r in enumerate(by_bang[:30]):
        score = r["sr"] * np.sqrt(max(r["cagr"], 0.01))
        print(f"  {i+1:>3} {r['label']:<55} {r['sr']:>5.2f} {r['cagr']:>5.0f}% "
              f"{r['max_dd']:>5.1f}% ${r['final_eq']:>12,.0f} {score:>7.1f}")

    # ── Pareto frontier: max CAGR for each DD bucket ──
    dd_buckets = [(0, 3), (3, 5), (5, 10), (10, 15), (15, 25), (25, 50)]
    print(f"\n{'=' * 140}")
    print(f"  PARETO FRONTIER: Best CAGR per drawdown bucket")
    print(f"{'=' * 140}")
    for lo, hi in dd_buckets:
        bucket = [r for r in results if lo <= r["max_dd"] < hi]
        if not bucket:
            print(f"  DD {lo:>2}-{hi:>2}%: no configs")
            continue
        best = max(bucket, key=lambda x: x["cagr"])
        print(f"  DD {lo:>2}-{hi:>2}%: {best['label']:<55} SR={best['sr']:.2f} "
              f"CAGR={best['cagr']:.0f}% DD={best['max_dd']:.1f}% ${best['final_eq']:>12,.0f}")

    print(f"\n  Total: {len(results)}/{len(configs)} configs succeeded | Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
