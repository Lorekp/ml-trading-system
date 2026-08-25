#!/usr/bin/env python
"""Trailing / Moving Stop Loss Experiment.

Tests wide trailing stops and hybrid approaches (initial SL + trail after breakeven).
Builds on the finding that fixed 25-40% SLs are in a stable cluster.

Usage:
    PYTHONPATH="src;." python experiments/trailing_sl_experiment.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.stop_loss_experiment import _load_15m_data, CACHE_DIR, TF_MINUTES
from experiments.dynamic_asset_v2_experiment import (
    run_sim, assemble_signals, OBVIOUS_CORE,
)

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]


def apply_moving_sl(signals_df, sl_config, close_cache, high_cache, low_cache):
    """Apply moving/trailing/hybrid stop loss.

    Config keys:
      initial_sl: initial SL distance from entry (e.g. 0.25 = 25%)
      trail_pct: trail distance from peak unrealized (None = no trail)
      breakeven_at: move SL to breakeven after this profit (None = disabled)
      tighten_to: tighten trail to this after breakeven (None = keep trail_pct)
    """
    out = signals_df.copy()
    sig_ts = pd.to_datetime(out["timestamp"])
    if sig_ts.dt.tz is not None:
        sig_ts = sig_ts.dt.tz_localize(None)

    new_fwd = out["fwd_return"].values.copy()
    new_hold = out["hold_bars"].values.copy()
    n_stopped = 0
    n_profit_lock = 0

    initial_sl = sl_config.get("initial_sl", 0.25)
    trail_pct = sl_config.get("trail_pct", None)
    breakeven_at = sl_config.get("breakeven_at", None)
    tighten_to = sl_config.get("tighten_to", None)

    for i in range(len(out)):
        asset = out.iloc[i]["asset"]
        direction = out.iloc[i]["direction"]
        tf_label = out.iloc[i].get("strategy", "15m")
        hold_bars = int(out.iloc[i]["hold_bars"])

        if asset not in close_cache or close_cache[asset] is None:
            continue
        close = close_cache[asset]
        high = high_cache[asset]
        low = low_cache[asset]

        entry_ts = sig_ts.iloc[i]
        entry_idx = close.index.searchsorted(entry_ts)
        if entry_idx >= len(close):
            continue
        entry_price = close.values[entry_idx]
        if entry_price <= 0:
            continue

        tf_min = TF_MINUTES.get(tf_label, 15)
        n_check = hold_bars * (tf_min // 15)

        # State
        peak_profit = 0.0
        hit_breakeven = False
        # sl_level = directional return at which SL triggers (starts negative)
        sl_level = -initial_sl

        for b in range(1, n_check + 1):
            check_idx = entry_idx + b
            if check_idx >= len(close):
                break

            bar_high = high.values[check_idx]
            bar_low = low.values[check_idx]

            if direction > 0:
                worst_dir = (bar_low - entry_price) / entry_price
                best_dir = (bar_high - entry_price) / entry_price
            else:
                worst_dir = -(bar_high - entry_price) / entry_price
                best_dir = -(bar_low - entry_price) / entry_price

            if best_dir > peak_profit:
                peak_profit = best_dir

            # Move SL: breakeven trigger
            if breakeven_at and peak_profit >= breakeven_at and not hit_breakeven:
                hit_breakeven = True
                sl_level = max(sl_level, 0.0)  # move to breakeven

            # Move SL: trail from peak
            if trail_pct is not None and peak_profit > 0:
                active_trail = tighten_to if (hit_breakeven and tighten_to) else trail_pct
                trail_level = peak_profit - active_trail
                if trail_level > sl_level:
                    sl_level = trail_level

            # Check if SL hit
            if worst_dir < sl_level:
                exit_dir = sl_level
                if direction > 0:
                    new_fwd[i] = exit_dir
                else:
                    new_fwd[i] = -exit_dir
                new_hold[i] = max(1, b * 15 // tf_min)
                n_stopped += 1
                if exit_dir > 0:
                    n_profit_lock += 1
                break

    out["fwd_return"] = new_fwd
    out["hold_bars"] = new_hold
    return out, n_stopped, n_profit_lock


def main():
    T0 = time.time()
    print("=" * 100)
    print("  TRAILING / MOVING STOP LOSS EXPERIMENT")
    print("=" * 100)

    # Load cached data
    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    asset_volumes = cached["asset_volumes"]
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]

    # Load 15m data
    all_assets = {n: f"{n}USDT" for n in hold_sigs}
    close_cache, high_cache, low_cache, atr_cache = {}, {}, {}, {}
    print("\n  Loading 15m data...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_load_15m_data, (n, s)): n for n, s in all_assets.items()}
        for f in as_completed(futs):
            name, close, high, low, atr = f.result()
            if close is not None:
                close_cache[name] = close
                high_cache[name] = high
                low_cache[name] = low
    print(f"    {len(close_cache)} assets loaded ({time.time()-T0:.0f}s)")

    # Build S8 signals
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

    merged_base = assemble_signals(hold_sigs, _s8_sel, months)
    print(f"    {len(merged_base):,} signals")

    sym_cfg = {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}

    # ── Variants ──
    variants = [
        ("No SL",               None),
        # Reference: best fixed from earlier
        ("Fixed 25%",           {"initial_sl": 0.25}),
        # Pure wide trailing (50% initial, trail from peak)
        ("50%i + trail 10%",    {"initial_sl": 0.50, "trail_pct": 0.10}),
        ("50%i + trail 15%",    {"initial_sl": 0.50, "trail_pct": 0.15}),
        ("50%i + trail 20%",    {"initial_sl": 0.50, "trail_pct": 0.20}),
        ("50%i + trail 25%",    {"initial_sl": 0.50, "trail_pct": 0.25}),
        # 25% initial + trail after breakeven
        ("25% + BE@3% tr15%",   {"initial_sl": 0.25, "breakeven_at": 0.03, "trail_pct": 0.15, "tighten_to": 0.15}),
        ("25% + BE@5% tr15%",   {"initial_sl": 0.25, "breakeven_at": 0.05, "trail_pct": 0.15, "tighten_to": 0.15}),
        ("25% + BE@5% tr10%",   {"initial_sl": 0.25, "breakeven_at": 0.05, "trail_pct": 0.10, "tighten_to": 0.10}),
        ("25% + BE@5% tr20%",   {"initial_sl": 0.25, "breakeven_at": 0.05, "trail_pct": 0.20, "tighten_to": 0.20}),
        ("25% + BE@10% tr15%",  {"initial_sl": 0.25, "breakeven_at": 0.10, "trail_pct": 0.15, "tighten_to": 0.15}),
        # 30% initial + trail
        ("30% + trail 15%",     {"initial_sl": 0.30, "trail_pct": 0.15}),
        ("30% + trail 20%",     {"initial_sl": 0.30, "trail_pct": 0.20}),
        # 25% initial + trail (no breakeven gate)
        ("25% + trail 15%",     {"initial_sl": 0.25, "trail_pct": 0.15}),
        ("25% + trail 20%",     {"initial_sl": 0.25, "trail_pct": 0.20}),
    ]

    print(f"\n  {'Variant':<22} {'Sharpe':>7} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'Stopped':>7} {'PrLock':>7} {'$/mo':>8}")
    print("  " + "-" * 90)

    for var_name, cfg in variants:
        if cfg is None:
            modified = merged_base.copy()
            n_stopped, n_profit = 0, 0
        else:
            modified, n_stopped, n_profit = apply_moving_sl(
                merged_base, cfg, close_cache, high_cache, low_cache)

        r = run_sim(modified, vol_series, base_sizing=0.045, symmetric=sym_cfg)
        if r:
            print(f"  {var_name:<22} {r['sr']:>7.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
                  f" {r['max_daily_dd']:>6.2f}% {n_stopped:>7,} {n_profit:>7,}"
                  f" ${r['pnl_mo']:>7,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
