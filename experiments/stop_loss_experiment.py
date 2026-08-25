#!/usr/bin/env python
"""Stop Loss Experiment — Path-dependent SL on S8+Sym1.5 production config.

Tests 12 stop loss approaches using actual 15m price paths to determine
which SL hurts profits the least (or helps) while being HyroTrader-compliant.

HyroTrader rules: mandatory SL within 5 min, max 3% risk per trade.

Usage:
    PYTHONPATH="src;." python experiments/stop_loss_experiment.py [--smoke]
"""

import sys, os, time, warnings, pickle, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.config import *
from experiments.circuit_breaker_v2_experiment import (
    NEEDED_TFS, FULL5, CAP, VIP_TIERS, SPREAD_BPS, BASELINE_WR,
    simulate_with_breaker, recompute_returns, apply_vol, _compute_vol_one,
    get_vip_tier,
)
from signal_engine.portfolio import compute_daily_pnl_series
from experiments.dynamic_asset_v2_experiment import (
    run_sim, assemble_signals, OBVIOUS_CORE, FIXED_30_OVERFIT,
)
from signal_engine.asset_selection import select_top_n_for_month

SMOKE = "--smoke" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
CACHE_DIR = Path("experiments/.cache")
FULL5_HOLDS = {"15m": 2, "1h": 4, "3h": 16, "4h": 6, "24h": 12}

# TF label → minutes per bar (for computing 15m checkpoints)
TF_MINUTES = {"15m": 15, "1h": 60, "3h": 180, "4h": 240, "24h": 1440}


def _load_15m_data(args):
    """Load 15m close + high + low for one asset (for SL checking)."""
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        if ohlcv_15m.index.tz is not None:
            ohlcv_15m.index = ohlcv_15m.index.tz_localize(None)
        close = ohlcv_15m["close"]
        high = ohlcv_15m["high"]
        low = ohlcv_15m["low"]
        # ATR(14) on 15m bars
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        return name, close, high, low, atr
    except:
        return name, None, None, None, None


def apply_stop_loss(signals_df, sl_type, sl_params, close_cache, high_cache,
                    low_cache, atr_cache):
    """Apply path-dependent stop loss to all signals.

    Walks through 15m bars during each signal's hold period.
    Uses HIGH/LOW (not just close) for more accurate SL trigger detection.

    Returns: (modified_signals_df, n_stopped, n_tp_hit)
    """
    out = signals_df.copy()
    sig_ts = pd.to_datetime(out["timestamp"])
    if sig_ts.dt.tz is not None:
        sig_ts = sig_ts.dt.tz_localize(None)

    new_fwd = out["fwd_return"].values.copy()
    new_hold = out["hold_bars"].values.copy()
    n_stopped = 0
    n_tp = 0

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
        atr = atr_cache[asset]

        entry_ts = sig_ts.iloc[i]
        entry_idx = close.index.searchsorted(entry_ts)
        if entry_idx >= len(close):
            continue
        entry_price = close.values[entry_idx]
        if entry_price <= 0:
            continue

        # ATR at entry
        entry_atr = atr.values[entry_idx] if entry_idx < len(atr) and np.isfinite(atr.values[entry_idx]) else entry_price * 0.01
        entry_atr_pct = entry_atr / entry_price

        # Compute SL/TP levels
        sl_pct = None  # directional loss threshold (positive number)
        tp_pct = None  # directional profit threshold (positive number)

        if sl_type == "fixed":
            sl_pct = sl_params["sl_pct"]
        elif sl_type == "atr":
            sl_pct = entry_atr_pct * sl_params["sl_mult"]
            if "tp_mult" in sl_params:
                tp_pct = entry_atr_pct * sl_params["tp_mult"]
        elif sl_type == "trailing":
            sl_pct = sl_params.get("trail_pct", entry_atr_pct * sl_params.get("trail_mult", 2.0))

        if sl_pct is None:
            continue

        # Number of 15m bars to check
        tf_min = TF_MINUTES.get(tf_label, 15)
        n_check_bars = hold_bars * (tf_min // 15)

        # Walk through 15m bars
        peak_directional = 0.0
        exited = False
        for b in range(1, n_check_bars + 1):
            check_idx = entry_idx + b
            if check_idx >= len(close):
                break

            # Use high/low for better SL detection
            bar_high = high.values[check_idx]
            bar_low = low.values[check_idx]
            bar_close = close.values[check_idx]

            # Worst case within bar (for SL check)
            if direction > 0:
                worst_price = bar_low   # long: worst is low
                best_price = bar_high   # long: best is high
            else:
                worst_price = bar_high  # short: worst is high
                best_price = bar_low    # short: best is low

            worst_ret = direction * (worst_price - entry_price) / entry_price
            best_ret = direction * (best_price - entry_price) / entry_price
            close_ret = (bar_close - entry_price) / entry_price

            # Trailing stop: update peak
            if sl_type == "trailing":
                close_dir_ret = direction * close_ret
                if close_dir_ret > peak_directional:
                    peak_directional = close_dir_ret

                # Trail from peak
                trail_loss = peak_directional - (-worst_ret if direction > 0 else worst_ret)
                # Actually: check if drop from peak exceeds trail_pct
                drop_from_peak = peak_directional - (direction * (bar_low if direction > 0 else bar_high) - entry_price) / entry_price * direction
                # Simpler: directional return at worst point
                if peak_directional - worst_ret > sl_pct and peak_directional > 0:
                    # Stopped by trailing stop
                    exit_ret = direction * (peak_directional - sl_pct)
                    new_fwd[i] = exit_ret / direction if direction != 0 else 0
                    # Actually simpler: exit at SL level
                    sl_price = entry_price * (1 + direction * (peak_directional - sl_pct))
                    new_fwd[i] = (sl_price - entry_price) / entry_price
                    new_hold[i] = max(1, b * 15 // tf_min)
                    n_stopped += 1
                    exited = True
                    break
            else:
                # Fixed / ATR stop: check if worst return hits SL
                if worst_ret < -sl_pct:
                    # Stopped out — exit at SL price
                    if direction > 0:
                        sl_price = entry_price * (1 - sl_pct)
                    else:
                        sl_price = entry_price * (1 + sl_pct)
                    new_fwd[i] = (sl_price - entry_price) / entry_price
                    new_hold[i] = max(1, b * 15 // tf_min)
                    n_stopped += 1
                    exited = True
                    break

                # Take profit check (for barrier configs)
                if tp_pct is not None and best_ret > tp_pct:
                    if direction > 0:
                        tp_price = entry_price * (1 + tp_pct)
                    else:
                        tp_price = entry_price * (1 - tp_pct)
                    new_fwd[i] = (tp_price - entry_price) / entry_price
                    new_hold[i] = max(1, b * 15 // tf_min)
                    n_tp += 1
                    exited = True
                    break

    out["fwd_return"] = new_fwd
    out["hold_bars"] = new_hold
    return out, n_stopped, n_tp


def main():
    T0 = time.time()
    print("=" * 110)
    print("  STOP LOSS EXPERIMENT — S8+Sym1.5 Production Config")
    print("  Path-dependent SL using 15m high/low price paths")
    print("=" * 110)

    # ── Load cached prep data ──
    print("\n  Loading cached data...", flush=True)
    prep_cache = CACHE_DIR / "v2_prep_cache.pkl"
    with open(prep_cache, "rb") as f:
        cached = pickle.load(f)
    asset_volumes = cached["asset_volumes"]
    asset_first_dates = cached["asset_first_dates"]
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    print(f"    {len(hold_sigs)} assets with signals")

    # ── Load 15m price data for SL checking ──
    print("\n  Loading 15m close/high/low + ATR per asset...", flush=True)
    t1 = time.time()

    # Get all assets that appear in signals
    all_assets = {}
    for name in hold_sigs:
        all_assets[name] = f"{name}USDT"
    if SMOKE:
        all_assets = {k: v for k, v in list(all_assets.items())[:15]}

    close_cache, high_cache, low_cache, atr_cache = {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_load_15m_data, (n, s)): n for n, s in all_assets.items()}
        for f in as_completed(futs):
            name, close, high, low, atr = f.result()
            if close is not None:
                close_cache[name] = close
                high_cache[name] = high
                low_cache[name] = low
                atr_cache[name] = atr
    print(f"    {len(close_cache)} assets loaded ({time.time()-t1:.0f}s)")

    # ── Build S8+Sym1.5 merged signals (baseline) ──
    print("\n  Building S8+Sym1.5 baseline signals...", flush=True)
    months = pd.period_range("2020-07", "2026-02", freq="M")
    CORE = OBVIOUS_CORE
    sym_cfg = {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}

    # Inline S8 selector (same as dynamic_asset_v2:784-793)
    def _regime_sel(ms):
        """S8: Core9 + regime-adaptive expansion by confidence × volume."""
        btc_vol = vol_series.get("BTC")
        n_assets = 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                p80 = avail.quantile(0.80)
                p20 = avail.quantile(0.20)
                if current > p80: n_assets = 15
                elif current < p20: n_assets = 40
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in hold_sigs:
            confs = []
            for tf in hold_sigs[name]:
                s = hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                mask = (ts >= cutoff) & (ts < ms)
                w = s[mask.values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            if not confs: continue
            avg_conf = float(np.mean(confs))
            vol = asset_volumes.get(name)
            if vol is None: continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0: continue
            scored.append((name, avg_conf * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])
        full = [a for a, _ in scored[:n_assets]]
        return full

    def _s8_sel(ms):
        active_core = [a for a in CORE if a in hold_sigs]
        full = _regime_sel(ms)
        dynamic = [a for a in full if a not in CORE]
        return active_core + dynamic

    merged_base = assemble_signals(hold_sigs, _s8_sel, months)
    if merged_base is None:
        print("  ERROR: no signals assembled")
        return
    print(f"    {len(merged_base):,} signals in baseline")

    # ── Define SL variants ──
    sl_variants = [
        ("No SL (baseline)", None, {}),
        ("Fixed 1%",    "fixed", {"sl_pct": 0.01}),
        ("Fixed 2%",    "fixed", {"sl_pct": 0.02}),
        ("Fixed 3%",    "fixed", {"sl_pct": 0.03}),
        ("Fixed 5%",    "fixed", {"sl_pct": 0.05}),
        ("1x ATR",      "atr",   {"sl_mult": 1.0}),
        ("2x ATR",      "atr",   {"sl_mult": 2.0}),
        ("3x ATR",      "atr",   {"sl_mult": 3.0}),
        ("2x ATR + 3x TP", "atr", {"sl_mult": 2.0, "tp_mult": 3.0}),
        ("3x ATR + 5x TP", "atr", {"sl_mult": 3.0, "tp_mult": 5.0}),
        ("Trail 2%",    "trailing", {"trail_pct": 0.02}),
        ("Trail 2x ATR", "trailing", {"trail_mult": 2.0}),
    ]

    if SMOKE:
        sl_variants = [sl_variants[0], sl_variants[2], sl_variants[5], sl_variants[8]]

    # ── Run variants ──
    print(f"\n  Running {len(sl_variants)} variants...", flush=True)
    all_results = {}

    for var_name, sl_type, sl_params in sl_variants:
        t1 = time.time()

        if sl_type is None:
            # Baseline — no SL
            modified = merged_base.copy()
            n_stopped, n_tp = 0, 0
        else:
            modified, n_stopped, n_tp = apply_stop_loss(
                merged_base, sl_type, sl_params,
                close_cache, high_cache, low_cache, atr_cache)

        # Run S8+Sym1.5 sim
        r = run_sim(modified, vol_series, base_sizing=0.045, symmetric=sym_cfg)

        if r is not None:
            r["n_stopped"] = n_stopped
            r["n_tp"] = n_tp
            r["pct_stopped"] = n_stopped / len(modified) * 100 if len(modified) > 0 else 0
            all_results[var_name] = r
            elapsed = time.time() - t1
            print(f"    {var_name:<20} Sharpe={r['sr']:.3f} PnL=${r['pnl']:>10,.0f} "
                  f"MaxDD={r['max_dd']:.1f}% stopped={n_stopped:,} tp={n_tp:,} ({elapsed:.0f}s)",
                  flush=True)
        else:
            print(f"    {var_name:<20} FAILED", flush=True)

    # ── Results table ──
    print(f"\n{'='*120}")
    print("  RESULTS — S8+Sym1.5 with Stop Loss Variants")
    print(f"{'='*120}")
    print(f"\n  {'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'Calmar':>7} {'PF':>6} {'Trades':>7} {'Stopped':>8} {'TP':>6} {'$/mo':>8}")
    print("  " + "-" * 115)

    base_sr = all_results.get("No SL (baseline)", {}).get("sr", 0)
    for var_name, *_ in sl_variants:
        r = all_results.get(var_name)
        if r is None:
            print(f"  {var_name:<22} FAILED")
            continue
        ds = r["sr"] - base_sr
        print(f"  {var_name:<22} {r['sr']:>7.3f} {r['sortino']:>8.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% {r['calmar']:>7.2f} {r['pf']:>5.2f}"
              f" {r['n_trades']:>7,} {r.get('n_stopped',0):>8,} {r.get('n_tp',0):>6,}"
              f" ${r['pnl_mo']:>7,.0f}")

    # ── Ranking ──
    print(f"\n{'='*120}")
    print("  RANKING BY PNL (HyroTrader-compliant: MaxDD < 8%, DailyDD < 3%)")
    print(f"{'='*120}")
    ranked = sorted(all_results.items(), key=lambda x: -x[1]["pnl"])
    print(f"\n  {'#':>3} {'Variant':<22} {'Sharpe':>7} {'dSh':>6} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7} {'$/mo':>8} {'Compliant':>10}")
    print("  " + "-" * 95)
    for i, (name, r) in enumerate(ranked, 1):
        ds = r["sr"] - base_sr
        compliant = r["max_dd"] < 8.0 and r["max_daily_dd"] > -3.0
        tag = "YES" if compliant else "NO"
        print(f"  {i:>3} {name:<22} {r['sr']:>7.3f} {ds:>+5.2f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% ${r['pnl_mo']:>7,.0f} {tag:>10}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
