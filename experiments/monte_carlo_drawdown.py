#!/usr/bin/env python
"""Monte Carlo drawdown analysis — 1000 simulations of S8+Sym1.5.

Bootstraps daily PnL series to generate 1000 equity paths.
Checks: does the strategy ever breach MaxDD >10% or DailyDD >5%?

Fully vectorized — all 1000 paths computed simultaneously via numpy.

Usage:
    PYTHONPATH="src;." python experiments/monte_carlo_drawdown.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.dynamic_asset_v2_experiment import (
    run_sim, assemble_signals, OBVIOUS_CORE,
)
from experiments.circuit_breaker_v2_experiment import (
    simulate_with_breaker, apply_vol, get_vip_tier, SPREAD_BPS, CAP,
)
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

CACHE_DIR = Path("experiments/.cache")
N_SIMS = 1000
SEED = 42


def main():
    T0 = time.time()
    print("=" * 80)
    print(f"  MONTE CARLO DRAWDOWN ANALYSIS — {N_SIMS} simulations")
    print("  S8+Sym1.5, no stop losses, bootstrap daily PnL")
    print("=" * 80)

    # ── Load cached data ──
    print("\n  Loading cached data...", flush=True)
    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    asset_volumes = cached["asset_volumes"]
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]

    # ── Build S8+Sym1.5 signals ──
    print("  Building S8+Sym1.5 signals...", flush=True)
    months = pd.period_range("2020-07", "2026-02", freq="M")
    CORE = OBVIOUS_CORE

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
        active_core = [a for a in CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in CORE]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    sym_cfg = {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}

    # ── Run baseline sim to get daily PnL series ──
    print("  Running baseline simulation...", flush=True)
    r = run_sim(merged, vol_series, base_sizing=0.045, symmetric=sym_cfg)
    if r is None:
        print("  ERROR: baseline sim failed")
        return

    # Get the actual daily PnL from the sim
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=0.045, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    sim = simulate_with_breaker(
        apply_vol(prepared, vol_series), port,
        "pnl_ratio_symmetric",
        {"window": 10, "threshold": -0.01, "floor": 0.10,
         "up_threshold": 0.01, "cap": 1.5})

    dpnl = compute_daily_pnl_series(sim)
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    daily_returns = (dpnl / CAP).values  # daily returns as fraction of starting capital
    n_days = len(daily_returns)

    print(f"  Baseline: {n_days} trading days, Sharpe {r['sr']:.2f}")
    print(f"  Daily return: mean={daily_returns.mean()*100:.3f}%, "
          f"std={daily_returns.std()*100:.3f}%")
    print(f"  Baseline MaxDD: {r['max_dd']:.1f}%, DailyDD: {r['max_daily_dd']:.2f}%")

    # ── Monte Carlo: bootstrap daily returns ──
    # Method: resample daily returns WITH replacement to create N_SIMS paths
    # Each path has the same length as the original
    # This preserves the return distribution but shuffles the order
    # (tests whether the SEQUENCE of returns matters for drawdown)

    print(f"\n  Running {N_SIMS} Monte Carlo simulations (vectorized)...", flush=True)
    t1 = time.time()

    rng = np.random.default_rng(SEED)

    # Generate all bootstrap indices at once: (N_SIMS, n_days)
    boot_indices = rng.integers(0, n_days, size=(N_SIMS, n_days))

    # Resample daily returns: (N_SIMS, n_days)
    boot_returns = daily_returns[boot_indices]

    # Compute cumulative equity curves: (N_SIMS, n_days)
    # equity[t] = (1 + r[0]) * (1 + r[1]) * ... * (1 + r[t])
    cum_equity = np.cumprod(1.0 + boot_returns, axis=1)

    # Running max (for drawdown): (N_SIMS, n_days)
    running_max = np.maximum.accumulate(cum_equity, axis=1)

    # Drawdown at each point: (N_SIMS, n_days)
    drawdowns = (running_max - cum_equity) / running_max

    # Max drawdown per simulation: (N_SIMS,)
    max_dds = drawdowns.max(axis=1) * 100  # in percent

    # Max daily loss per simulation: (N_SIMS,)
    max_daily_loss = boot_returns.min(axis=1) * 100  # in percent (negative)

    # Final equity per simulation
    final_equity = cum_equity[:, -1] * CAP

    # Sharpe per simulation
    daily_means = boot_returns.mean(axis=1)
    daily_stds = boot_returns.std(axis=1)
    sharpes = daily_means / np.where(daily_stds > 0, daily_stds, 1e-10) * np.sqrt(365.25)

    elapsed = time.time() - t1
    print(f"  Done ({elapsed:.1f}s)")

    # ── Results ──
    print(f"\n{'='*80}")
    print(f"  RESULTS — {N_SIMS} Monte Carlo Paths")
    print(f"{'='*80}")

    print(f"\n  MAX DRAWDOWN DISTRIBUTION:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(max_dds, p)
        print(f"    p{p:>2}: {val:>6.2f}%")

    breached_10 = (max_dds > 10).sum()
    breached_8 = (max_dds > 8).sum()
    breached_15 = (max_dds > 15).sum()

    print(f"\n  MaxDD BREACH RATES:")
    print(f"    MaxDD > 8%:  {breached_8:>4}/{N_SIMS} ({breached_8/N_SIMS*100:.1f}%)")
    print(f"    MaxDD > 10%: {breached_10:>4}/{N_SIMS} ({breached_10/N_SIMS*100:.1f}%)")
    print(f"    MaxDD > 15%: {breached_15:>4}/{N_SIMS} ({breached_15/N_SIMS*100:.1f}%)")

    print(f"\n  WORST DAILY LOSS DISTRIBUTION:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(max_daily_loss, p)
        print(f"    p{p:>2}: {val:>+7.3f}%")

    breached_daily_5 = (max_daily_loss < -5).sum()
    breached_daily_3 = (max_daily_loss < -3).sum()
    breached_daily_2 = (max_daily_loss < -2).sum()

    print(f"\n  Daily DD BREACH RATES:")
    print(f"    Worst day > -2%: {breached_daily_2:>4}/{N_SIMS} ({breached_daily_2/N_SIMS*100:.1f}%)")
    print(f"    Worst day > -3%: {breached_daily_3:>4}/{N_SIMS} ({breached_daily_3/N_SIMS*100:.1f}%)")
    print(f"    Worst day > -5%: {breached_daily_5:>4}/{N_SIMS} ({breached_daily_5/N_SIMS*100:.1f}%)")

    print(f"\n  SHARPE DISTRIBUTION:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(sharpes, p)
        print(f"    p{p:>2}: {val:>6.2f}")

    print(f"\n  FINAL EQUITY (starting ${CAP:,}):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(final_equity, p)
        print(f"    p{p:>2}: ${val:>12,.0f}")

    # Also do block bootstrap (preserving some autocorrelation)
    print(f"\n{'='*80}")
    print(f"  BLOCK BOOTSTRAP (5-day blocks, preserves short-term autocorrelation)")
    print(f"{'='*80}")

    block_size = 5
    n_blocks = n_days // block_size

    # Generate block starts
    block_starts = rng.integers(0, n_days - block_size, size=(N_SIMS, n_blocks))

    # Build returns from blocks
    block_returns = np.zeros((N_SIMS, n_blocks * block_size))
    for b in range(n_blocks):
        for d in range(block_size):
            block_returns[:, b * block_size + d] = daily_returns[block_starts[:, b] + d]

    # Same analysis
    block_cum = np.cumprod(1.0 + block_returns, axis=1)
    block_max = np.maximum.accumulate(block_cum, axis=1)
    block_dd = (block_max - block_cum) / block_max
    block_max_dds = block_dd.max(axis=1) * 100
    block_daily_loss = block_returns.min(axis=1) * 100

    breached_10_b = (block_max_dds > 10).sum()
    breached_daily_5_b = (block_daily_loss < -5).sum()

    print(f"\n  MaxDD > 10%: {breached_10_b:>4}/{N_SIMS} ({breached_10_b/N_SIMS*100:.1f}%)")
    print(f"  Worst day > -5%: {breached_daily_5_b:>4}/{N_SIMS} ({breached_daily_5_b/N_SIMS*100:.1f}%)")
    print(f"  MaxDD p50: {np.percentile(block_max_dds, 50):.2f}%, "
          f"p95: {np.percentile(block_max_dds, 95):.2f}%, "
          f"p99: {np.percentile(block_max_dds, 99):.2f}%")
    print(f"  Daily loss p1: {np.percentile(block_daily_loss, 1):.3f}%, "
          f"p5: {np.percentile(block_daily_loss, 5):.3f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
