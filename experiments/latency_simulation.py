#!/usr/bin/env python
"""Latency simulation — model 10s execution delay using tick data.

Memory-efficient: processes one asset at a time, never loads all tick data.
Uses bookTicker (best bid/ask) where available, 1m candle fallback elsewhere.

Usage:
    PYTHONPATH="src;." python experiments/latency_simulation.py
"""

import sys, os, time, warnings, pickle, gc
import numpy as np
import pandas as pd
from pathlib import Path

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
TICK_DIR = Path("data/bookTicker_parquet")
LATENCY_S = 10
TAKER_FEE_BPS = 5.5

PORT = PortfolioConfig(
    starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
    base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
BREAKER = {"window": 10, "threshold": -0.01, "floor": 0.10,
           "up_threshold": 0.01, "cap": 1.5}
BASE_IMP = {"correlation_throttle": 0.90, "correlation_direction": "both",
            "correlation_scale": 0.5, "max_signals_per_bar": 15}


def load_tick_for_date(asset, date):
    """Load ONE day of bookTicker for an asset. Returns small DataFrame or None."""
    sym = f"{asset}USDT"
    path = TICK_DIR / sym / f"{date}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["best_bid_price", "best_ask_price", "transaction_time"])
        df.index = pd.to_datetime(df["transaction_time"], unit="ms")
        return df[["best_bid_price", "best_ask_price"]]
    except Exception:
        return None


def load_1m_for_asset(asset):
    """Load 1m candles (lightweight, just open/close)."""
    try:
        sym = f"{asset}USDT"
        df = load_binance_1m(sym, DATA_DIRS)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["open", "close"]]
    except Exception:
        return None


def process_asset_trades(asset, asset_trades, candle_1m):
    """Process all trades for one asset. Load tick data day-by-day."""
    results = []  # (original_pnl, adjusted_pnl, slippage_bps, source)

    # Group trades by date for efficient tick loading
    by_date = {}
    for t in asset_trades:
        d = t["timestamp"].strftime("%Y-%m-%d")
        by_date.setdefault(d, []).append(t)

    # Check which dates have tick data
    sym = f"{asset}USDT"
    tick_path = TICK_DIR / sym
    has_tick_dir = tick_path.exists()

    for date_str, day_trades in by_date.items():
        tick_df = None
        if has_tick_dir:
            tick_df = load_tick_for_date(asset, date_str)

        for trade in day_trades:
            ts = trade["timestamp"]
            direction = trade["direction"]
            target_ts = ts + pd.Timedelta(seconds=LATENCY_S)
            fill_price = None
            source = "none"

            # Try tick data
            if tick_df is not None and len(tick_df) > 0:
                idx = tick_df.index.searchsorted(target_ts)
                if 0 <= idx < len(tick_df):
                    row = tick_df.iloc[idx]
                    if direction > 0:
                        fill_price = float(row["best_ask_price"])
                    else:
                        fill_price = float(row["best_bid_price"])
                    source = "tick"

            # Fallback to 1m candle
            if fill_price is None and candle_1m is not None:
                idx = candle_1m.index.searchsorted(target_ts)
                if 0 <= idx < len(candle_1m):
                    fill_price = float(candle_1m.iloc[idx]["open"])
                    source = "1m"

            if fill_price is None or fill_price <= 0:
                results.append((trade["pnl"], trade["pnl"], 0.0, "none"))
                continue

            # Get entry price (close of signal bar)
            entry_idx = candle_1m.index.searchsorted(ts) if candle_1m is not None else -1
            if candle_1m is not None and 0 < entry_idx <= len(candle_1m):
                entry_price = float(candle_1m.iloc[entry_idx - 1]["close"])
            else:
                results.append((trade["pnl"], trade["pnl"], 0.0, "none"))
                continue

            if entry_price <= 0:
                results.append((trade["pnl"], trade["pnl"], 0.0, "none"))
                continue

            # Slippage
            if direction > 0:
                slip_bps = (fill_price - entry_price) / entry_price * 10000
            else:
                slip_bps = (entry_price - fill_price) / entry_price * 10000

            # Adjusted PnL
            exit_price = entry_price * (1 + trade["fwd_return"])
            if direction > 0:
                real_return = (exit_price - fill_price) / fill_price
            else:
                real_return = (fill_price - exit_price) / fill_price
            real_return -= TAKER_FEE_BPS / 10000  # taker fee on entry

            real_pnl = trade["size"] * real_return
            results.append((trade["pnl"], real_pnl, slip_bps, source))

        # Free tick data after each date
        del tick_df

    return results


def main():
    T0 = time.time()
    print("=" * 100)
    print(f"  LATENCY SIMULATION — {LATENCY_S}s delay, tick-level fills")
    print(f"  Taker fee: {TAKER_FEE_BPS}bps | Memory-efficient (per-asset, per-day)")
    print("=" * 100)

    # Build signals
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

    # Run baseline
    sim = simulate_with_improvements(prepared, PORT, BREAKER, BASE_IMP)
    dpnl_base = compute_daily_pnl_series(sim)
    if dpnl_base.index.tz is not None:
        dpnl_base.index = dpnl_base.index.tz_localize(None)
    base_pnl = float(dpnl_base.sum())
    base_dr = dpnl_base / CAP
    base_sr = float(base_dr.mean() / base_dr.std() * np.sqrt(365.25))
    n_trades = sim["n_trades"]
    print(f"  Baseline: SR={base_sr:.2f}, PnL=${base_pnl:,.0f}, {n_trades:,} trades")

    # Extract executed trades grouped by asset
    executed = ~sim["skipped"]
    exec_idx = np.where(executed)[0]
    trades_by_asset = {}
    for i in exec_idx:
        ts = pd.Timestamp(sim["timestamps"][i])
        if hasattr(ts, 'tz') and ts.tz is not None:
            ts = ts.tz_localize(None)
        asset = prepared.iloc[i]["asset"]
        trades_by_asset.setdefault(asset, []).append({
            "idx": i, "timestamp": ts,
            "direction": float(prepared.iloc[i]["direction"]),
            "fwd_return": float(prepared.iloc[i]["fwd_return"]),
            "pnl": sim["pnl"][i],
            "size": sim["sizes"][i],
        })

    unique_assets = sorted(trades_by_asset.keys())
    print(f"  {len(exec_idx):,} trades across {len(unique_assets)} assets\n")

    # Process each asset sequentially (memory-efficient)
    all_slips = []
    total_base_pnl = 0.0
    total_adj_pnl = 0.0
    tick_fills = 0
    candle_fills = 0
    no_data = 0

    for ai, asset in enumerate(unique_assets):
        candle_1m = load_1m_for_asset(asset)
        results = process_asset_trades(asset, trades_by_asset[asset], candle_1m)

        for orig_pnl, adj_pnl, slip, source in results:
            total_base_pnl += orig_pnl
            total_adj_pnl += adj_pnl
            if source == "tick":
                tick_fills += 1
                all_slips.append(slip)
            elif source == "1m":
                candle_fills += 1
                all_slips.append(slip)
            else:
                no_data += 1

        del candle_1m
        gc.collect()

        if (ai + 1) % 20 == 0:
            print(f"  Processed {ai+1}/{len(unique_assets)} assets...", flush=True)

    print(f"  Done: {tick_fills:,} tick | {candle_fills:,} 1m | {no_data:,} no data")

    # Results
    slips = np.array(all_slips)
    delta = total_adj_pnl - total_base_pnl

    print(f"\n{'=' * 100}")
    print(f"  RESULTS — {LATENCY_S}s LATENCY + {TAKER_FEE_BPS}bps TAKER FEE")
    print(f"{'=' * 100}")
    print(f"  Baseline PnL (0s, 1bps slip):   ${total_base_pnl:>12,.0f}")
    print(f"  Adjusted PnL ({LATENCY_S}s, taker):     ${total_adj_pnl:>12,.0f}")
    print(f"  Delta:                           ${delta:>12,.0f} ({delta/total_base_pnl*100:+.1f}%)")

    if len(slips) > 0:
        print(f"\n  Slippage from {LATENCY_S}s delay ({len(slips):,} trades):")
        print(f"    Mean:   {slips.mean():>+.2f} bps")
        print(f"    Median: {np.median(slips):>+.2f} bps")
        print(f"    Std:    {slips.std():>.2f} bps")
        print(f"    p5/p95: {np.percentile(slips, 5):>+.1f} / {np.percentile(slips, 95):>+.1f} bps")
        worse = (slips > 0).sum()
        better = (slips < 0).sum()
        print(f"    Worse fill: {worse:,} ({worse/len(slips)*100:.0f}%)")
        print(f"    Better fill: {better:,} ({better/len(slips)*100:.0f}%)")

    years = 5.5
    annual_drag = abs(delta) / years
    print(f"\n  Annual drag: ${annual_drag:,.0f}/yr")
    print(f"  Daily drag:  ${annual_drag/365:,.0f}/day")
    print(f"  % of PnL:    {abs(delta)/total_base_pnl*100:.1f}%")
    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
