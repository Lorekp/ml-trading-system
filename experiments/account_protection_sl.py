#!/usr/bin/env python
"""Account-protection-only stop losses.

SLs that ONLY trigger to prevent catastrophic account damage:
- Shorts: hard cap (unlimited risk from pumps)
- Longs: only if trade would wipe >X% of account
- Dynamic SL distance based on actual position size

Usage:
    PYTHONPATH="src;." python experiments/account_protection_sl.py
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
from experiments.dynamic_asset_v2_experiment import run_sim, assemble_signals, OBVIOUS_CORE


def apply_account_protection_sl(signals_df, config, close_cache, high_cache, low_cache):
    """SLs that only prevent catastrophic loss, not normal volatility.

    For each trade, SL distance = max_acct_loss / estimated_position_fraction.
    Shorts get a hard cap (unlimited risk). Longs get a very wide cap.
    """
    out = signals_df.copy()
    sig_ts = pd.to_datetime(out["timestamp"])
    if sig_ts.dt.tz is not None:
        sig_ts = sig_ts.dt.tz_localize(None)

    new_fwd = out["fwd_return"].values.copy()
    new_hold = out["hold_bars"].values.copy()
    n_stopped = 0
    n_short_stopped = 0
    n_long_stopped = 0

    max_acct_loss = config.get("max_acct_loss_pct", 0.03)
    short_cap = config.get("short_hard_cap", 0.50)
    long_cap = config.get("long_hard_cap", 0.80)

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

        # Estimate position fraction from signal strength
        prob = out.iloc[i].get("probability", 0.55)
        edge = abs(2.0 * prob - 1.0)
        est_size_frac = min(0.045 * edge, 0.045 * 2.0)
        est_size_frac = min(est_size_frac, 0.06)

        # Dynamic SL: caps account loss at max_acct_loss_pct
        if est_size_frac > 0:
            dynamic_sl = max_acct_loss / est_size_frac
        else:
            dynamic_sl = 1.0

        # Direction-specific caps
        if direction > 0:
            sl_pct = min(dynamic_sl, long_cap)
        else:
            sl_pct = min(dynamic_sl, short_cap)

        # Walk price path
        tf_min = TF_MINUTES.get(tf_label, 15)
        n_check = hold_bars * (tf_min // 15)

        for b in range(1, n_check + 1):
            check_idx = entry_idx + b
            if check_idx >= len(close):
                break

            if direction > 0:
                worst_dir = (low.values[check_idx] - entry_price) / entry_price
            else:
                worst_dir = -(high.values[check_idx] - entry_price) / entry_price

            if worst_dir < -sl_pct:
                if direction > 0:
                    new_fwd[i] = -sl_pct
                else:
                    new_fwd[i] = sl_pct
                new_hold[i] = max(1, b * 15 // tf_min)
                n_stopped += 1
                if direction > 0:
                    n_long_stopped += 1
                else:
                    n_short_stopped += 1
                break

    out["fwd_return"] = new_fwd
    out["hold_bars"] = new_hold
    return out, n_stopped, n_long_stopped, n_short_stopped


def main():
    T0 = time.time()
    print("=" * 100)
    print("  ACCOUNT-PROTECTION STOP LOSS EXPERIMENT")
    print("  SLs that only prevent catastrophic damage, not normal volatility")
    print("=" * 100)

    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    asset_volumes = cached["asset_volumes"]
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]

    all_assets = {n: f"{n}USDT" for n in hold_sigs}
    close_cache, high_cache, low_cache = {}, {}, {}
    print("\n  Loading 15m data...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_load_15m_data, (n, s)): n for n, s in all_assets.items()}
        for f in as_completed(futs):
            name, close, high, low, atr = f.result()
            if close is not None:
                close_cache[name] = close
                high_cache[name] = high
                low_cache[name] = low
    print(f"    {len(close_cache)} assets ({time.time()-T0:.0f}s)")

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

    # Count longs vs shorts
    n_long = (merged_base["direction"] > 0).sum()
    n_short = (merged_base["direction"] < 0).sum()
    print(f"    Longs: {n_long:,} ({n_long/len(merged_base)*100:.0f}%), "
          f"Shorts: {n_short:,} ({n_short/len(merged_base)*100:.0f}%)")

    sym_cfg = {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}

    variants = [
        ("No SL",                None),
        # Account protection: different thresholds
        ("3% acct S50% L80%",    {"max_acct_loss_pct": 0.03, "short_hard_cap": 0.50, "long_hard_cap": 0.80}),
        ("3% acct S50% L95%",    {"max_acct_loss_pct": 0.03, "short_hard_cap": 0.50, "long_hard_cap": 0.95}),
        ("5% acct S50% L95%",    {"max_acct_loss_pct": 0.05, "short_hard_cap": 0.50, "long_hard_cap": 0.95}),
        ("5% acct S60% L95%",    {"max_acct_loss_pct": 0.05, "short_hard_cap": 0.60, "long_hard_cap": 0.95}),
        ("5% acct S70% L95%",    {"max_acct_loss_pct": 0.05, "short_hard_cap": 0.70, "long_hard_cap": 0.95}),
        # Shorts-only protection (longs effectively no SL)
        ("Shorts S30% only",     {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.30, "long_hard_cap": 10.0}),
        ("Shorts S40% only",     {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.40, "long_hard_cap": 10.0}),
        ("Shorts S50% only",     {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.50, "long_hard_cap": 10.0}),
        ("Shorts S60% only",     {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.60, "long_hard_cap": 10.0}),
        ("Shorts S70% only",     {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.70, "long_hard_cap": 10.0}),
        ("Shorts S80% only",     {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.80, "long_hard_cap": 10.0}),
        ("Shorts S100% only",    {"max_acct_loss_pct": 10.0, "short_hard_cap": 1.00, "long_hard_cap": 10.0}),
        # Asymmetric: tight on shorts, wide on longs
        ("S40% L80%",            {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.40, "long_hard_cap": 0.80}),
        ("S50% L80%",            {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.50, "long_hard_cap": 0.80}),
        ("S50% L95%",            {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.50, "long_hard_cap": 0.95}),
        ("S60% L95%",            {"max_acct_loss_pct": 10.0, "short_hard_cap": 0.60, "long_hard_cap": 0.95}),
    ]

    print(f"\n  {'Variant':<24} {'Sharpe':>7} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'ShStop':>7} {'LgStop':>7} {'Total':>6} {'$/mo':>8}")
    print("  " + "-" * 95)

    for name, cfg in variants:
        if cfg is None:
            modified = merged_base.copy()
            n_total, n_long_s, n_short_s = 0, 0, 0
        else:
            modified, n_total, n_long_s, n_short_s = apply_account_protection_sl(
                merged_base, cfg, close_cache, high_cache, low_cache)

        r = run_sim(modified, vol_series, base_sizing=0.045, symmetric=sym_cfg)
        if r:
            print(f"  {name:<24} {r['sr']:>7.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
                  f" {r['max_daily_dd']:>6.2f}% {n_short_s:>7,} {n_long_s:>7,}"
                  f" {n_total:>6,} ${r['pnl_mo']:>7,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
