#!/usr/bin/env python
"""Multi-dimensional analysis of config sweep results.

Runs the exact same grid as config_sweep.py but outputs pivot tables
and interaction matrices instead of ranked lists.

Usage:
    PYTHONPATH="src;." python -u experiments/analyze_sweep.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
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
from experiments.yolo_compound import run_full

CAP = 200_000.0


def main():
    T0 = time.time()

    # ── Load data (exact yolo_compound pipeline) ──
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
    print(f"  Signals: {len(prepared):,} ({time.time()-T0:.0f}s)")

    with open(CACHE_DIR / "close_cache.pkl", "rb") as f:
        close_cache = pickle.load(f)
    print(f"  Close cache: {len(close_cache)} pairs")

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

    # ── Grid (same as config_sweep.py) ──
    sizing_pcts      = [0.03, 0.06, 0.10, 0.15, 0.20, 0.30]
    max_per_assets   = [0.06, 0.15, 0.25]
    max_exposures    = [0.90, 0.95]
    max_dds          = [0.10, 0.20, 0.30, 0.50]
    delever_starts   = [0.05, 0.10, 0.20]
    corr_throttles   = [None, 0.90]

    rows = []
    count = 0

    for sz, mpa, mex, mdd, ds, ct in product(
        sizing_pcts, max_per_assets, max_exposures, max_dds,
        delever_starts, corr_throttles,
    ):
        if ds >= mdd: continue
        if mpa > mex: continue
        if sz > mpa * 2: continue
        if mdd < 0.10 and sz > 0.15: continue
        if mex < 0.90 and sz > 0.20: continue

        label = f"sz{sz:.0%}_mpa{mpa:.0%}_exp{mex:.0%}_dd{mdd:.0%}_dl{ds:.0%}"
        if ct: label += f"_ct{ct}"

        imp = {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75}
        if ct:
            imp.update({"correlation_throttle": ct, "correlation_direction": "both",
                        "correlation_scale": 0.5, "max_signals_per_bar": 15})

        port = PortfolioConfig(
            starting_capital=CAP, max_dd=mdd, delever_start=ds, slippage=0.0001,
            max_exposure=mex, max_per_asset=mpa, max_per_strategy=0.50,
            base_sizing_pct=sz, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
        )

        r = run_full(prepared.copy(), label, port, breaker, imp,
                     slippage_model, asset_volumes, close_cache=close_cache)
        count += 1
        if r:
            rows.append({
                "sizing": sz, "max_per_asset": mpa, "max_exposure": mex,
                "max_dd": mdd, "delever_start": ds,
                "corr_throttle": "CT" if ct else "none",
                "sharpe": r["sr"], "cagr": r["cagr"], "max_drawdown": r["max_dd"],
                "final_eq": r["final_eq"], "trades": r["trades"], "win_rate": r["win_rate"],
            })
        if count % 100 == 0:
            print(f"  [{count}/540]")

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} configs completed in {time.time()-T0:.0f}s\n")

    # ═══════════════════════════════════════════════════
    W = 100

    def header(title):
        print(f"\n{'=' * W}")
        print(f"  {title}")
        print(f"{'=' * W}")

    # ── DIM 1: SIZING ──
    header("DIM 1: SIZING (the biggest lever)")
    g = df.groupby("sizing").agg(
        SR_mean=("sharpe", "mean"), SR_std=("sharpe", "std"),
        CAGR_mean=("cagr", "mean"), CAGR_max=("cagr", "max"),
        DD_mean=("max_drawdown", "mean"), DD_max=("max_drawdown", "max"),
        n=("sharpe", "count"),
    )
    print(f"  {'Sizing':>7} {'SR avg':>7} {'SR std':>7} {'CAGR avg':>9} {'CAGR max':>9} "
          f"{'DD avg':>7} {'DD max':>7} {'n':>4}")
    for sz, row in g.iterrows():
        print(f"  {sz:>7.0%} {row.SR_mean:>7.2f} {row.SR_std:>7.2f} "
              f"{row.CAGR_mean:>8.0f}% {row.CAGR_max:>8.0f}% "
              f"{row.DD_mean:>6.1f}% {row.DD_max:>6.1f}% {int(row.n):>4}")

    # ── DIM 2: CORR THROTTLE ──
    header("DIM 2: CORRELATION THROTTLE")
    g = df.groupby("corr_throttle").agg(
        SR=("sharpe", "mean"), CAGR=("cagr", "mean"), DD=("max_drawdown", "mean"),
    ).round(2)
    for ct, row in g.iterrows():
        print(f"  {ct:>6}: SR={row.SR:.2f}  CAGR={row.CAGR:.0f}%  DD={row.DD:.1f}%")

    # ── DIM 3: PER-ASSET CAP ──
    header("DIM 3: PER-ASSET CONCENTRATION CAP")
    g = df.groupby("max_per_asset").agg(
        SR=("sharpe", "mean"), CAGR=("cagr", "mean"), CAGR_max=("cagr", "max"),
        DD=("max_drawdown", "mean"),
    ).round(2)
    for mpa, row in g.iterrows():
        print(f"  {mpa:>5.0%}: SR={row.SR:.2f}  CAGR avg={row.CAGR:.0f}%  max={row.CAGR_max:.0f}%  DD={row.DD:.1f}%")

    # ── DIM 4: TOTAL EXPOSURE ──
    header("DIM 4: TOTAL EXPOSURE CAP")
    g = df.groupby("max_exposure").agg(
        SR=("sharpe", "mean"), CAGR=("cagr", "mean"), DD=("max_drawdown", "mean"),
    ).round(2)
    for mex, row in g.iterrows():
        print(f"  {mex:>5.0%}: SR={row.SR:.2f}  CAGR={row.CAGR:.0f}%  DD={row.DD:.1f}%")

    # ── DIM 5: MAX DD TOLERANCE ──
    header("DIM 5: MAX DRAWDOWN TOLERANCE")
    g = df.groupby("max_dd").agg(
        SR=("sharpe", "mean"), CAGR=("cagr", "mean"), CAGR_max=("cagr", "max"),
        DD=("max_drawdown", "mean"), DD_max=("max_drawdown", "max"),
    ).round(2)
    for mdd, row in g.iterrows():
        print(f"  {mdd:>5.0%}: SR={row.SR:.2f}  CAGR avg={row.CAGR:.0f}%  "
              f"max={row.CAGR_max:.0f}%  DD avg={row.DD:.1f}%  max={row.DD_max:.1f}%")

    # ── DIM 6: DELEVER START ──
    header("DIM 6: DELEVERAGING START")
    g = df.groupby("delever_start").agg(
        SR=("sharpe", "mean"), CAGR=("cagr", "mean"), DD=("max_drawdown", "mean"),
    ).round(2)
    for ds, row in g.iterrows():
        print(f"  {ds:>5.0%}: SR={row.SR:.2f}  CAGR={row.CAGR:.0f}%  DD={row.DD:.1f}%")

    # ═══════════════════════════════════════════════════
    #  2-WAY INTERACTIONS (pivot tables)
    # ═══════════════════════════════════════════════════

    header("INTERACTION: SIZING x CORR THROTTLE (mean Sharpe)")
    print(df.pivot_table("sharpe", "sizing", "corr_throttle", "mean").round(2).to_string())

    header("INTERACTION: SIZING x CORR THROTTLE (mean CAGR)")
    print(df.pivot_table("cagr", "sizing", "corr_throttle", "mean").round(0).to_string())

    header("INTERACTION: SIZING x PER-ASSET CAP (mean CAGR)")
    print(df.pivot_table("cagr", "sizing", "max_per_asset", "mean").round(0).to_string())

    header("INTERACTION: SIZING x PER-ASSET CAP (mean Sharpe)")
    print(df.pivot_table("sharpe", "sizing", "max_per_asset", "mean").round(2).to_string())

    header("INTERACTION: SIZING x PER-ASSET CAP (mean MaxDD)")
    print(df.pivot_table("max_drawdown", "sizing", "max_per_asset", "mean").round(1).to_string())

    header("INTERACTION: SIZING x MAX_DD TOLERANCE (mean Sharpe)")
    print(df.pivot_table("sharpe", "sizing", "max_dd", "mean").round(2).to_string())

    header("INTERACTION: SIZING x MAX_DD TOLERANCE (mean CAGR)")
    print(df.pivot_table("cagr", "sizing", "max_dd", "mean").round(0).to_string())

    header("INTERACTION: SIZING x DELEVER START (mean Sharpe)")
    print(df.pivot_table("sharpe", "sizing", "delever_start", "mean").round(2).to_string())

    # ═══════════════════════════════════════════════════
    #  3-WAY: SIZING x PER-ASSET x CT
    # ═══════════════════════════════════════════════════

    header("3-WAY: SIZING x PER-ASSET x CT (mean Sharpe)")
    g = df.groupby(["sizing", "max_per_asset", "corr_throttle"])["sharpe"].mean().round(2)
    print(g.unstack(["max_per_asset", "corr_throttle"]).to_string())

    header("3-WAY: SIZING x PER-ASSET x CT (mean CAGR)")
    g = df.groupby(["sizing", "max_per_asset", "corr_throttle"])["cagr"].mean().round(0)
    print(g.unstack(["max_per_asset", "corr_throttle"]).to_string())

    # ═══════════════════════════════════════════════════
    #  EFFICIENT FRONTIER
    # ═══════════════════════════════════════════════════

    header("EFFICIENT FRONTIER: Best Sharpe per CAGR bucket")
    cagr_buckets = [(10, 20), (20, 40), (40, 70), (70, 120), (120, 200), (200, 300), (300, 400)]
    for lo, hi in cagr_buckets:
        bucket = df[(df["cagr"] >= lo) & (df["cagr"] < hi)]
        if len(bucket) == 0:
            print(f"  CAGR {lo:>3}-{hi:>3}%: ---")
            continue
        best = bucket.loc[bucket["sharpe"].idxmax()]
        print(f"  CAGR {lo:>3}-{hi:>3}%: sz={best.sizing:.0%} mpa={best.max_per_asset:.0%} "
              f"exp={best.max_exposure:.0%} dd={best.max_dd:.0%} dl={best.delever_start:.0%} "
              f"ct={best.corr_throttle}  =>  SR={best.sharpe:.2f} CAGR={best.cagr:.0f}% "
              f"DD={best.max_drawdown:.1f}%")

    header("EFFICIENT FRONTIER: Best Sharpe per MaxDD bucket")
    dd_buckets = [(0, 2), (2, 5), (5, 8), (8, 12), (12, 15)]
    for lo, hi in dd_buckets:
        bucket = df[(df["max_drawdown"] >= lo) & (df["max_drawdown"] < hi)]
        if len(bucket) == 0:
            print(f"  DD {lo:>2}-{hi:>2}%: ---")
            continue
        best = bucket.loc[bucket["cagr"].idxmax()]
        print(f"  DD {lo:>2}-{hi:>2}%: sz={best.sizing:.0%} mpa={best.max_per_asset:.0%} "
              f"exp={best.max_exposure:.0%} dd={best.max_dd:.0%} dl={best.delever_start:.0%} "
              f"ct={best.corr_throttle}  =>  SR={best.sharpe:.2f} CAGR={best.cagr:.0f}% "
              f"DD={best.max_drawdown:.1f}%")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
