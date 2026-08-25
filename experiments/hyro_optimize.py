#!/usr/bin/env python
"""Find the most aggressive config that still has >99.7% (3-sigma) survival.

Sweeps sizing, exposure, per-asset caps, signal limits, CT threshold
to maximize returns while maintaining 100% survival at 3-sigma over 1 year.

Usage:
    PYTHONPATH="src;." python experiments/hyro_optimize.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig
from signal_engine.slippage import load_per_symbol_calibration
from signal_engine.execution import get_vip_tier

CAP = 200_000.0
N_SIMS = 5_000       # per config (faster sweep, rerun winner at 10K)
BLOCK_SIZE = 5
HORIZON = 365

DAILY_DD_LIMIT = 0.05
TRAILING_DD_LIMIT = 0.10


def prepare_data():
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

    _ts = pd.to_datetime(prepared["timestamp"])
    if _ts.dt.tz is not None: _ts = _ts.dt.tz_localize(None)
    prepared = prepared[~_ts.dt.year.isin({2021})].reset_index(drop=True)

    return prepared, asset_volumes


def get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes):
    sim = simulate_with_improvements(prepared, port, breaker, imp,
        slippage_model=slippage_model, asset_volumes=asset_volumes)
    dpnl = compute_daily_pnl_series(sim)
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)
    eq = CAP
    rets = []
    for dp in dpnl.values:
        r = dp / eq if eq > 0 else 0
        rets.append(r)
        eq = max(eq + dp, 1.0)
    return np.array(rets)


def mc_survival(daily_rets, n_sims, horizon, block_size):
    n_days = len(daily_rets)
    n_blocks = (horizon + block_size - 1) // block_size

    busts = 0
    worst_daily_all = np.zeros(n_sims)
    worst_trail_all = np.zeros(n_sims)
    final_eq = np.zeros(n_sims)

    for i in range(n_sims):
        block_starts = np.random.randint(0, n_days - block_size, size=n_blocks)
        sim_rets = np.concatenate([daily_rets[s:s+block_size] for s in block_starts])[:horizon]

        eq = 1.0
        peak = 1.0
        busted = False
        max_daily = 0.0
        max_trail = 0.0

        for r in sim_rets:
            day_loss = -r if r < 0 else 0
            max_daily = max(max_daily, day_loss)
            if day_loss >= DAILY_DD_LIMIT:
                busted = True

            eq *= (1 + r)
            if eq > peak:
                peak = eq
            trail = (peak - eq) / peak if peak > 0 else 0
            max_trail = max(max_trail, trail)
            if trail >= TRAILING_DD_LIMIT:
                busted = True

        if busted:
            busts += 1
        worst_daily_all[i] = max_daily
        worst_trail_all[i] = max_trail
        final_eq[i] = eq

    return {
        "survive_pct": (1 - busts / n_sims) * 100,
        "daily_p95": np.percentile(worst_daily_all, 95) * 100,
        "daily_p997": np.percentile(worst_daily_all, 99.7) * 100,
        "trail_p95": np.percentile(worst_trail_all, 95) * 100,
        "trail_p997": np.percentile(worst_trail_all, 99.7) * 100,
        "med_return": (np.median(final_eq) - 1) * 100,
        "p5_return": (np.percentile(final_eq, 5) - 1) * 100,
    }


def main():
    T0 = time.time()
    print("=" * 110)
    print("  HYROTRADER CONFIG OPTIMIZER — Find Max Returns at 3-sigma Survival")
    print(f"  {N_SIMS:,} sims | {HORIZON}-day horizon | 5% daily / 10% trailing DD limits")
    print("=" * 110)

    prepared, asset_volumes = prepare_data()

    per_symbol_cal = load_per_symbol_calibration()
    vip_name, maker_bps, taker_bps = get_vip_tier(CAP)
    slippage_model = {"calibration": per_symbol_cal, "vip_tier": (vip_name, maker_bps, taker_bps),
                      "max_participation_rate": 0.02, "guard_pct": 0.50}
    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    # ── Phase 1: Sizing sweep (main lever) ──
    print("\n  PHASE 1: Base sizing sweep (other params at PROD baseline)")
    print(f"  {'Sizing':>7s} | {'Survive':>8s} | {'Daily p997':>10s} | {'Trail p997':>10s} | {'Med Ret':>8s} | {'p5 Ret':>8s}")
    print(f"  {'-'*65}")

    sizing_results = {}
    for sizing in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25]:
        port = PortfolioConfig(
            starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.10, max_per_strategy=0.50,
            base_sizing_pct=sizing, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        imp = {"max_signals_per_bar": 15, "correlation_throttle": 0.90,
               "correlation_direction": "both", "correlation_scale": 0.5}

        rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
        r = mc_survival(rets, N_SIMS, HORIZON, BLOCK_SIZE)
        sizing_results[sizing] = r

        safe = "OK" if r["survive_pct"] >= 99.7 else "RISKY"
        print(f"  {sizing*100:>6.0f}% | {r['survive_pct']:>7.1f}% | {r['daily_p997']:>9.2f}% | "
              f"{r['trail_p997']:>9.2f}% | {r['med_return']:>+7.1f}% | {r['p5_return']:>+7.1f}% {safe}")

    # Find max sizing with 3-sigma survival
    max_safe_sizing = max((s for s, r in sizing_results.items() if r["survive_pct"] >= 99.7), default=0.06)
    print(f"\n  >> Max safe sizing: {max_safe_sizing*100:.0f}%")

    # ── Phase 2: Tune other params around best sizing ──
    print(f"\n  PHASE 2: Tune around {max_safe_sizing*100:.0f}% sizing")
    print(f"  {'Config':>30s} | {'Survive':>8s} | {'Daily p997':>10s} | {'Trail p997':>10s} | {'Med Ret':>8s} | {'p5 Ret':>8s}")
    print(f"  {'-'*85}")

    tweaks = []

    # Exposure caps
    for max_exp in [0.85, 0.90, 0.95]:
        for max_pa in [0.06, 0.10, 0.15, 0.20]:
            label = f"exp={max_exp:.0%} pa={max_pa:.0%}"
            port = PortfolioConfig(
                starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
                max_exposure=max_exp, max_per_asset=max_pa, max_per_strategy=0.50,
                base_sizing_pct=max_safe_sizing, max_signal_mult=2.0, sizing_mode="probability_edge",
                use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
            imp = {"max_signals_per_bar": 15, "correlation_throttle": 0.90,
                   "correlation_direction": "both", "correlation_scale": 0.5}
            rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
            r = mc_survival(rets, N_SIMS, HORIZON, BLOCK_SIZE)
            tweaks.append((label, r, port, imp))
            safe = "OK" if r["survive_pct"] >= 99.7 else "RISKY"
            print(f"  {label:>30s} | {r['survive_pct']:>7.1f}% | {r['daily_p997']:>9.2f}% | "
                  f"{r['trail_p997']:>9.2f}% | {r['med_return']:>+7.1f}% | {r['p5_return']:>+7.1f}% {safe}")

    # Max signals per bar
    for msb in [10, 15, 20, 25]:
        label = f"maxSig={msb}"
        port = PortfolioConfig(
            starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.10, max_per_strategy=0.50,
            base_sizing_pct=max_safe_sizing, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        imp = {"max_signals_per_bar": msb, "correlation_throttle": 0.90,
               "correlation_direction": "both", "correlation_scale": 0.5}
        rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
        r = mc_survival(rets, N_SIMS, HORIZON, BLOCK_SIZE)
        tweaks.append((label, r, port, imp))
        safe = "OK" if r["survive_pct"] >= 99.7 else "RISKY"
        print(f"  {label:>30s} | {r['survive_pct']:>7.1f}% | {r['daily_p997']:>9.2f}% | "
              f"{r['trail_p997']:>9.2f}% | {r['med_return']:>+7.1f}% | {r['p5_return']:>+7.1f}% {safe}")

    # CT threshold
    for ct in [0.80, 0.90, None]:
        label = f"CT={'off' if ct is None else f'{ct:.0%}'}"
        port = PortfolioConfig(
            starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.10, max_per_strategy=0.50,
            base_sizing_pct=max_safe_sizing, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        imp = {"max_signals_per_bar": 15}
        if ct is not None:
            imp["correlation_throttle"] = ct
            imp["correlation_direction"] = "both"
            imp["correlation_scale"] = 0.5
        rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
        r = mc_survival(rets, N_SIMS, HORIZON, BLOCK_SIZE)
        tweaks.append((label, r, port, imp))
        safe = "OK" if r["survive_pct"] >= 99.7 else "RISKY"
        print(f"  {label:>30s} | {r['survive_pct']:>7.1f}% | {r['daily_p997']:>9.2f}% | "
              f"{r['trail_p997']:>9.2f}% | {r['med_return']:>+7.1f}% | {r['p5_return']:>+7.1f}% {safe}")

    # Signal mult
    for sm in [1.5, 2.0, 2.5, 3.0]:
        label = f"sigMult={sm:.1f}"
        port = PortfolioConfig(
            starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.10, max_per_strategy=0.50,
            base_sizing_pct=max_safe_sizing, max_signal_mult=sm, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        imp = {"max_signals_per_bar": 15, "correlation_throttle": 0.90,
               "correlation_direction": "both", "correlation_scale": 0.5}
        rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
        r = mc_survival(rets, N_SIMS, HORIZON, BLOCK_SIZE)
        tweaks.append((label, r, port, imp))
        safe = "OK" if r["survive_pct"] >= 99.7 else "RISKY"
        print(f"  {label:>30s} | {r['survive_pct']:>7.1f}% | {r['daily_p997']:>9.2f}% | "
              f"{r['trail_p997']:>9.2f}% | {r['med_return']:>+7.1f}% | {r['p5_return']:>+7.1f}% {safe}")

    # ── Phase 3: Best combo ──
    safe_tweaks = [(l, r) for l, r, _, _ in tweaks if r["survive_pct"] >= 99.7]
    if safe_tweaks:
        best_label, best_r = max(safe_tweaks, key=lambda x: x[1]["med_return"])
        print(f"\n  >> Best safe tweak: {best_label} -> median return {best_r['med_return']:+.1f}%")

    # ── Phase 4: Try pushing sizing +1-2% with best tweak params ──
    print(f"\n  PHASE 3: Push sizing above {max_safe_sizing*100:.0f}% with best params")
    for sizing in [max_safe_sizing + 0.01, max_safe_sizing + 0.02, max_safe_sizing + 0.03, max_safe_sizing + 0.05]:
        if sizing > 0.30:
            break
        port = PortfolioConfig(
            starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.10, max_per_strategy=0.50,
            base_sizing_pct=sizing, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        imp = {"max_signals_per_bar": 15, "correlation_throttle": 0.90,
               "correlation_direction": "both", "correlation_scale": 0.5}
        rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
        # Run at 10K for precision on the edge
        r = mc_survival(rets, 10_000, HORIZON, BLOCK_SIZE)
        safe = "OK" if r["survive_pct"] >= 99.7 else "RISKY"
        print(f"  {sizing*100:>6.0f}% | {r['survive_pct']:>7.2f}% | {r['daily_p997']:>9.2f}% | "
              f"{r['trail_p997']:>9.2f}% | {r['med_return']:>+7.1f}% | {r['p5_return']:>+7.1f}% {safe}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
