#!/usr/bin/env python
"""Realistic compliance SL modeling using 1-minute price paths.

Instead of assuming perfect fills at SL price, checks actual 1m candles
during each trade's hold period and models execution as:
  - Trigger: when 1m candle's low (long) or high (short) crosses SL
  - Fill: WORST price of the trigger candle (conservative)
  - Fee: taker fee (5.5bps VIP0)

Compares ideal vs realistic SL at multiple levels.

Usage:
    PYTHONPATH="src;." python experiments/compliance_sl_realistic.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m
from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol, CAP
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
TAKER_FEE_BPS = 5.5  # VIP0 Bybit taker fee

PORT = PortfolioConfig(
    starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
    base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

BREAKER = {"window": 10, "threshold": -0.01, "floor": 0.10,
           "up_threshold": 0.01, "cap": 1.5}

BASE_IMP = {
    "correlation_throttle": 0.90,
    "correlation_direction": "both",
    "correlation_scale": 0.5,
    "max_signals_per_bar": 15,
}

# TF label -> minutes per bar
TF_MINUTES = {"15m": 15, "1h": 60, "3h": 180, "4h": 240, "24h": 1440}


def load_1m_cached(asset):
    """Load 1m OHLCV for an asset, return (high, low, close) Series."""
    try:
        sym = f"{asset}USDT"
        df = load_binance_1m(sym, DATA_DIRS)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return asset, df["high"], df["low"], df["close"]
    except Exception:
        return asset, None, None, None


def check_sl_realistic(entry_ts, entry_price, direction, hold_bars, tf_label,
                       high_1m, low_1m, close_1m, sl_pct):
    """Check if SL triggers using 1m price path. Returns (triggered, fill_return).

    fill_return is the realized return AFTER realistic execution:
    - Fill at WORST price of the trigger candle (conservative)
    - Plus taker fee
    """
    tf_min = TF_MINUTES.get(tf_label, 15)
    n_minutes = hold_bars * tf_min

    # Find entry index in 1m data
    idx = close_1m.index.searchsorted(entry_ts)
    if idx >= len(close_1m):
        return False, None, 0

    # Walk through each 1m candle during hold period
    for m in range(1, n_minutes + 1):
        check_idx = idx + m
        if check_idx >= len(close_1m):
            break

        if direction > 0:  # long: SL below entry
            sl_price = entry_price * (1 - sl_pct)
            worst = low_1m.values[check_idx]
            if worst <= sl_price:
                # Triggered! Fill at worst price of this candle + taker fee
                fill_price = worst
                fill_return = (fill_price - entry_price) / entry_price
                fill_return -= TAKER_FEE_BPS / 10000  # taker fee
                return True, fill_return, m
        else:  # short: SL above entry
            sl_price = entry_price * (1 + sl_pct)
            worst = high_1m.values[check_idx]
            if worst >= sl_price:
                fill_price = worst
                fill_return = -(fill_price - entry_price) / entry_price
                fill_return -= TAKER_FEE_BPS / 10000
                return True, fill_return, m

    return False, None, 0


def main():
    T0 = time.time()
    print("=" * 110)
    print("  REALISTIC COMPLIANCE SL — 1m price paths + taker execution")
    print(f"  Taker fee: {TAKER_FEE_BPS}bps | Fill: worst of trigger candle")
    print("=" * 110)

    # ── Build signals ──
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
    print(f"  {len(prepared):,} signals assembled ({time.time()-T0:.0f}s)")

    # ── Run baseline simulation (no SL) to get executed trades ──
    sim = simulate_with_improvements(prepared, PORT, BREAKER, BASE_IMP)
    dpnl_base = compute_daily_pnl_series(sim)
    if dpnl_base.index.tz is not None:
        dpnl_base.index = dpnl_base.index.tz_localize(None)
    base_pnl = float(dpnl_base.sum())
    base_dr = dpnl_base / CAP
    base_sr = float(base_dr.mean() / base_dr.std() * np.sqrt(365.25))
    n_trades = sim["n_trades"]
    print(f"  Baseline: {n_trades:,} trades, SR={base_sr:.2f}, PnL=${base_pnl:,.0f}")

    # ── Extract executed trades ──
    executed_mask = ~sim["skipped"]
    exec_idx = np.where(executed_mask)[0]
    trades = []
    for i in exec_idx:
        ts = pd.Timestamp(sim["timestamps"][i])
        if hasattr(ts, 'tz') and ts.tz is not None:
            ts = ts.tz_localize(None)
        trades.append({
            "idx": i,
            "timestamp": ts,
            "asset": prepared.iloc[i]["asset"],
            "direction": prepared.iloc[i]["direction"],
            "fwd_return": prepared.iloc[i]["fwd_return"],
            "hold_bars": int(prepared.iloc[i].get("hold_bars", 4)),
            "strategy": prepared.iloc[i].get("strategy", "15m"),
            "pnl": sim["pnl"][i],
            "size": sim["sizes"][i],
        })
    print(f"  {len(trades):,} executed trades extracted")

    # ── Load 1m data for all traded assets ──
    unique_assets = list(set(t["asset"] for t in trades))
    print(f"  Loading 1m data for {len(unique_assets)} assets...", flush=True)
    t1 = time.time()

    high_1m = {}
    low_1m = {}
    close_1m = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(load_1m_cached, a): a for a in unique_assets}
        loaded = 0
        for f in as_completed(futs):
            asset, h, l, c = f.result()
            if h is not None:
                high_1m[asset] = h
                low_1m[asset] = l
                close_1m[asset] = c
                loaded += 1
    print(f"  Loaded {loaded}/{len(unique_assets)} assets ({time.time()-t1:.0f}s)")

    # ── Check SL at each level ──
    sl_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]

    print(f"\n  {'SL%':>5} {'Triggers':>9} {'Ideal PnL':>12} {'Real PnL':>12}"
          f" {'Slippage':>9} {'Worst Slip':>11} {'Avg Min':>8}")
    print("  " + "-" * 80)

    for sl_pct in sl_levels:
        t2 = time.time()
        n_triggers = 0
        ideal_total_pnl = 0.0  # PnL with perfect SL fills
        real_total_pnl = 0.0   # PnL with realistic fills
        base_total_pnl = 0.0   # PnL without any SL
        slippages = []
        trigger_minutes = []

        for trade in trades:
            asset = trade["asset"]
            if asset not in close_1m:
                # No 1m data — use original PnL
                ideal_total_pnl += trade["pnl"]
                real_total_pnl += trade["pnl"]
                base_total_pnl += trade["pnl"]
                continue

            base_total_pnl += trade["pnl"]

            # Extract TF from strategy field
            tf_label = trade["strategy"]
            if "/" in tf_label:
                tf_label = tf_label.split("/")[-1]
            if tf_label not in TF_MINUTES:
                tf_label = "15m"

            entry_ts = trade["timestamp"]
            entry_idx = close_1m[asset].index.searchsorted(entry_ts)
            if entry_idx >= len(close_1m[asset]):
                ideal_total_pnl += trade["pnl"]
                real_total_pnl += trade["pnl"]
                continue
            entry_price = close_1m[asset].values[entry_idx]
            if entry_price <= 0:
                ideal_total_pnl += trade["pnl"]
                real_total_pnl += trade["pnl"]
                continue

            triggered, fill_return, trigger_min = check_sl_realistic(
                entry_ts, entry_price, trade["direction"],
                trade["hold_bars"], tf_label,
                high_1m[asset], low_1m[asset], close_1m[asset],
                sl_pct)

            if triggered:
                n_triggers += 1
                trigger_minutes.append(trigger_min)

                # Ideal: perfect fill at SL price
                if trade["direction"] > 0:
                    ideal_return = -sl_pct - 0.0001  # SL price + slippage
                else:
                    ideal_return = -sl_pct - 0.0001
                ideal_pnl = trade["size"] * ideal_return
                ideal_total_pnl += ideal_pnl

                # Realistic: fill at worst of trigger candle + taker fee
                real_pnl = trade["size"] * fill_return
                real_total_pnl += real_pnl

                # Slippage = difference between ideal and realistic
                slip_bps = (abs(fill_return) - sl_pct) * 10000
                slippages.append(slip_bps)
            else:
                # SL not triggered — original PnL
                ideal_total_pnl += trade["pnl"]
                real_total_pnl += trade["pnl"]

        # Delta from baseline
        ideal_delta = ideal_total_pnl - base_total_pnl
        real_delta = real_total_pnl - base_total_pnl
        slip_delta = ideal_total_pnl - real_total_pnl

        avg_slip = np.mean(slippages) if slippages else 0
        max_slip = max(slippages) if slippages else 0
        avg_min = np.mean(trigger_minutes) if trigger_minutes else 0

        print(f"  {sl_pct*100:>4.0f}% {n_triggers:>9}"
              f" ${ideal_delta:>+10,.0f} ${real_delta:>+10,.0f}"
              f" ${slip_delta:>+8,.0f} {max_slip:>10.1f}bp {avg_min:>7.0f}m"
              f"  ({time.time()-t2:.0f}s)")

    # ── Summary ──
    print(f"\n{'=' * 110}")
    print("  LEGEND")
    print(f"{'=' * 110}")
    print("  Triggers:   trades stopped out (checked via 1m high/low paths)")
    print("  Ideal PnL:  delta vs no-SL if fills were perfect at SL price")
    print("  Real PnL:   delta vs no-SL with worst-of-candle fill + taker fee")
    print("  Slippage:   cost of realistic vs ideal execution")
    print("  Worst Slip: worst single-trade execution slippage (bps)")
    print("  Avg Min:    average minutes into hold before SL triggers")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
