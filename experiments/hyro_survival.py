#!/usr/bin/env python
"""HyroTrader survival probability — Monte Carlo simulation.

Uses block bootstrap on actual daily PnL to estimate the probability of
breaching the 5% daily drawdown rule (swing) and 10% trailing DD rule
across different configs and time horizons.

Reports 2-sigma (95%) and 3-sigma (99.7%) confidence bounds.

Usage:
    PYTHONPATH="src;." python experiments/hyro_survival.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

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
N_SIMS = 10_000
BLOCK_SIZE = 5          # 5-day blocks preserve autocorrelation
HORIZONS = [30, 90, 180, 365]  # days

# HyroTrader rules (swing upgrade)
DAILY_DD_LIMIT = 0.05   # 5% daily drawdown from start-of-day
TRAILING_DD_LIMIT = 0.10  # 10% trailing from peak


def prepare_data():
    """Load and prepare signal data (same as yolo_compound.py)."""
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

    # Exclude 2021
    _ts = pd.to_datetime(prepared["timestamp"])
    if _ts.dt.tz is not None: _ts = _ts.dt.tz_localize(None)
    prepared = prepared[~_ts.dt.year.isin({2021})].reset_index(drop=True)

    return prepared, asset_volumes


def get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes):
    """Run backtest and return daily return series (relative to running equity)."""
    sim = simulate_with_improvements(prepared, port, breaker, imp,
        slippage_model=slippage_model, asset_volumes=asset_volumes)
    dpnl = compute_daily_pnl_series(sim)
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    # Convert to daily returns relative to running equity
    eq = CAP
    rets = []
    for dp in dpnl.values:
        r = dp / eq if eq > 0 else 0
        rets.append(r)
        eq = max(eq + dp, 1.0)
    return np.array(rets)


def simulate_survival(daily_rets, n_sims, horizon_days, block_size):
    """Block bootstrap Monte Carlo for survival probability.

    Returns dict with bust probabilities for daily DD and trailing DD rules.
    """
    n_days = len(daily_rets)
    n_blocks = (horizon_days + block_size - 1) // block_size

    daily_bust = 0
    trail_bust = 0
    worst_daily = np.zeros(n_sims)
    worst_trail = np.zeros(n_sims)
    final_equity = np.zeros(n_sims)

    for i in range(n_sims):
        # Sample blocks with replacement
        block_starts = np.random.randint(0, n_days - block_size, size=n_blocks)
        sim_rets = np.concatenate([daily_rets[s:s+block_size] for s in block_starts])[:horizon_days]

        # Simulate equity path
        eq = 1.0  # normalized
        peak = 1.0
        busted_daily = False
        busted_trail = False
        max_daily_dd = 0.0
        max_trail_dd = 0.0

        for r in sim_rets:
            # Daily DD check (swing: loss from start-of-day)
            day_loss = -r if r < 0 else 0
            max_daily_dd = max(max_daily_dd, day_loss)
            if day_loss >= DAILY_DD_LIMIT:
                busted_daily = True

            eq *= (1 + r)
            if eq > peak:
                peak = eq

            # Trailing DD check
            trail_dd = (peak - eq) / peak if peak > 0 else 0
            max_trail_dd = max(max_trail_dd, trail_dd)
            if trail_dd >= TRAILING_DD_LIMIT:
                busted_trail = True

        if busted_daily:
            daily_bust += 1
        if busted_trail:
            trail_bust += 1
        worst_daily[i] = max_daily_dd
        worst_trail[i] = max_trail_dd
        final_equity[i] = eq

    return {
        "daily_bust_pct": daily_bust / n_sims * 100,
        "trail_bust_pct": trail_bust / n_sims * 100,
        "either_bust_pct": sum(1 for i in range(n_sims)
            if worst_daily[i] >= DAILY_DD_LIMIT or worst_trail[i] >= TRAILING_DD_LIMIT) / n_sims * 100,
        "worst_daily_p50": np.percentile(worst_daily, 50) * 100,
        "worst_daily_p95": np.percentile(worst_daily, 95) * 100,
        "worst_daily_p99": np.percentile(worst_daily, 99) * 100,
        "worst_daily_p997": np.percentile(worst_daily, 99.7) * 100,
        "worst_trail_p50": np.percentile(worst_trail, 50) * 100,
        "worst_trail_p95": np.percentile(worst_trail, 95) * 100,
        "worst_trail_p99": np.percentile(worst_trail, 99) * 100,
        "worst_trail_p997": np.percentile(worst_trail, 99.7) * 100,
        "median_return": (np.median(final_equity) - 1) * 100,
        "p5_return": (np.percentile(final_equity, 5) - 1) * 100,
        "p95_return": (np.percentile(final_equity, 95) - 1) * 100,
    }


def main():
    T0 = time.time()

    print("=" * 100)
    print("  HYROTRADER SURVIVAL PROBABILITY — Monte Carlo Block Bootstrap")
    print(f"  {N_SIMS:,} simulations | {BLOCK_SIZE}-day blocks | HyroTrader swing rules")
    print(f"  Daily DD limit: {DAILY_DD_LIMIT*100:.0f}% | Trailing DD limit: {TRAILING_DD_LIMIT*100:.0f}%")
    print("=" * 100)

    prepared, asset_volumes = prepare_data()
    print(f"  Signals: {len(prepared):,}")

    per_symbol_cal = load_per_symbol_calibration()
    vip_name, maker_bps, taker_bps = get_vip_tier(CAP)
    slippage_model = {"calibration": per_symbol_cal, "vip_tier": (vip_name, maker_bps, taker_bps),
                      "max_participation_rate": 0.02, "guard_pct": 0.50}
    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    configs = OrderedDict()
    configs["PROD"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
            base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15, "correlation_throttle": 0.90,
         "correlation_direction": "both", "correlation_scale": 0.5})

    configs["YOLO"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.30, delever_start=0.20, slippage=0.0001,
            max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20})

    # Also test intermediate configs
    for sizing in [0.10, 0.15]:
        label = f"SIZE_{int(sizing*100)}PCT"
        configs[label] = (
            PortfolioConfig(
                starting_capital=CAP, max_dd=0.15, delever_start=0.08, slippage=0.0001,
                max_exposure=0.90, max_per_asset=0.10, max_per_strategy=0.50,
                base_sizing_pct=sizing, max_signal_mult=2.0, sizing_mode="probability_edge",
                use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
            {"max_signals_per_bar": 15, "correlation_throttle": 0.90,
             "correlation_direction": "both", "correlation_scale": 0.5})

    for cfg_name, (port, imp) in configs.items():
        t1 = time.time()
        daily_rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)

        print(f"\n{'='*100}")
        print(f"  {cfg_name}  (base_sizing={port.base_sizing_pct*100:.0f}%, "
              f"daily_ret: mean={daily_rets.mean()*100:.3f}%, std={daily_rets.std()*100:.3f}%)")
        print(f"{'='*100}")

        # Historical stats
        worst_day = daily_rets.min()
        print(f"  Historical worst day: {worst_day*100:.2f}%")
        print(f"  Historical daily vol: {daily_rets.std()*100:.3f}%")
        print(f"  Historical Sharpe:    {daily_rets.mean()/daily_rets.std()*np.sqrt(365.25):.2f}")

        for horizon in HORIZONS:
            result = simulate_survival(daily_rets, N_SIMS, horizon, BLOCK_SIZE)

            print(f"\n  --- {horizon}-day horizon ({horizon/30:.0f} months) ---")
            print(f"  Survival rate:")
            print(f"    No daily DD bust:    {100-result['daily_bust_pct']:>6.1f}%")
            print(f"    No trailing DD bust: {100-result['trail_bust_pct']:>6.1f}%")
            print(f"    Survive BOTH rules:  {100-result['either_bust_pct']:>6.1f}%")
            print(f"  Worst daily DD distribution:")
            print(f"    Median (p50):        {result['worst_daily_p50']:>6.2f}%")
            print(f"    2-sigma (p95):       {result['worst_daily_p95']:>6.2f}%  {'<-- BUSTS' if result['worst_daily_p95'] >= DAILY_DD_LIMIT*100 else ''}")
            print(f"    3-sigma (p99.7):     {result['worst_daily_p997']:>6.2f}%  {'<-- BUSTS' if result['worst_daily_p997'] >= DAILY_DD_LIMIT*100 else ''}")
            print(f"  Worst trailing DD distribution:")
            print(f"    Median (p50):        {result['worst_trail_p50']:>6.2f}%")
            print(f"    2-sigma (p95):       {result['worst_trail_p95']:>6.2f}%  {'<-- BUSTS' if result['worst_trail_p95'] >= TRAILING_DD_LIMIT*100 else ''}")
            print(f"    3-sigma (p99.7):     {result['worst_trail_p997']:>6.2f}%  {'<-- BUSTS' if result['worst_trail_p997'] >= TRAILING_DD_LIMIT*100 else ''}")
            print(f"  Return: median={result['median_return']:+.1f}%  "
                  f"p5={result['p5_return']:+.1f}%  p95={result['p95_return']:+.1f}%")

        print(f"\n  ({time.time()-t1:.1f}s)")

    # Summary table
    print(f"\n{'='*100}")
    print(f"  SUMMARY: 1-YEAR SURVIVAL PROBABILITY")
    print(f"{'='*100}")
    print(f"  {'Config':>12s} | {'Sizing':>7s} | {'Survive':>8s} | {'Daily p95':>10s} | {'Trail p95':>10s} | {'Med Return':>11s}")
    print(f"  {'-'*75}")

    for cfg_name, (port, imp) in configs.items():
        daily_rets = get_daily_returns(prepared, port, breaker, imp, slippage_model, asset_volumes)
        result = simulate_survival(daily_rets, N_SIMS, 365, BLOCK_SIZE)
        survive = 100 - result["either_bust_pct"]
        print(f"  {cfg_name:>12s} | {port.base_sizing_pct*100:>6.0f}% | {survive:>7.1f}% | "
              f"{result['worst_daily_p95']:>9.2f}% | {result['worst_trail_p95']:>9.2f}% | "
              f"{result['median_return']:>+10.1f}%")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
