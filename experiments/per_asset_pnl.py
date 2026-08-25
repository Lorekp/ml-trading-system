#!/usr/bin/env python
"""Per-asset PnL breakdown for yolo_compound.py PROD config.

Loads v2_prep_cache.pkl, runs simulate_with_improvements() with PROD config,
then groups per-trade PnL by asset and prints ranked performance.

Usage:
    PYTHONPATH="src;." python experiments/per_asset_pnl.py
"""
import sys, os, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, ".."))
sys.path.insert(0, "src")
sys.path.insert(0, ".")
sys.path.insert(0, _HERE)
warnings.filterwarnings("ignore")

from drawdown_improvements import (
    simulate_with_improvements, apply_bar_level_sl, CACHE_DIR, TF_RESAMPLE,
)
from circuit_breaker_v2_experiment import apply_vol
from dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.config import PortfolioConfig
from signal_engine.slippage import load_per_symbol_calibration
from signal_engine.execution import get_vip_tier
from signal_engine.data import load_binance_1m, resample_ohlcv

CAP = 200_000.0

# Meme coin classification (rough — user can adjust)
MEME_TOKENS = {
    "DOGE", "SHIB", "1000SHIB", "PEPE", "1000PEPE", "FLOKI", "1000FLOKI",
    "BONK", "1000BONK", "WIF", "NEIRO", "POPCAT", "TURBO", "BRETT", "MOG",
    "PIPPIN", "GOAT", "PNUT", "ACT", "MOODENG", "CHILLGUY", "FARTCOIN",
    "PENGU", "TRUMP", "MELANIA", "VINE", "ANIME", "MUBARAK", "BANANAS31",
    "BANANA", "WOJAK", "MEW", "MYRO", "TRB",
}

# Major coins
MAJORS = {"BTC", "ETH", "SOL", "BNB"}


def classify(asset: str) -> str:
    a = asset.upper()
    if a in MEME_TOKENS:
        return "meme"
    if a in MAJORS:
        return "major"
    return "alt"


def main():
    print("Loading prep cache...")
    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    print(f"  hold_sigs: {len(hold_sigs)} assets")
    print(f"  vol_series: {len(vol_series)} assets")

    # Build S8 schedule callback — exact port of yolo_compound.py:104-136
    months = pd.period_range("2020-07", "2026-04", freq="M")

    def _regime_sel(ms):
        btc_vol = vol_series.get("BTC")
        n_assets = 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                if current > avail.quantile(0.80):
                    n_assets = 15
                elif current < avail.quantile(0.20):
                    n_assets = 40
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in hold_sigs:
            confs = []
            for tf in hold_sigs[name]:
                s = hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None:
                    ts = ts.dt.tz_localize(None)
                w = s[((ts >= cutoff) & (ts < ms)).values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            if not confs:
                continue
            vol = asset_volumes.get(name)
            if vol is None:
                continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0:
                continue
            scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:n_assets]]

    def _s8_sel(ms):
        active_core = [a for a in OBVIOUS_CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in OBVIOUS_CORE]

    print("Assembling signals...")
    prepared = assemble_signals(hold_sigs, _s8_sel, months)
    if prepared is None or len(prepared) == 0:
        print("No signals!")
        return
    prepared = prepared.copy()
    prepared["probability"] = prepared["probability"].astype(float)
    print(f"  Signals: {len(prepared):,}")

    prepared = apply_vol(
        prepared.sort_values("timestamp").reset_index(drop=True), vol_series)
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns:
            prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)

    # Close cache for bar-level SL
    print("Building close cache...")
    from drawdown_improvements import CACHE_DIR as CD
    close_cache_path = CD / "close_cache.pkl"
    close_cache = None
    if close_cache_path.exists():
        with open(close_cache_path, "rb") as f:
            close_cache = pickle.load(f)
        print(f"  Loaded close cache: {len(close_cache)} pairs")
    else:
        print("  No close cache — skipping bar-level SL")

    # Slippage
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

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.300, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    imp = {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75}

    # Apply bar-level SL if we have close cache
    if close_cache:
        prepared, sl_count = apply_bar_level_sl(
            prepared, close_cache,
            max_risk_per_trade=imp.get("max_risk_per_trade", 0.03),
            sl_cap=imp.get("sl_cap", 0.75),
            base_sizing_pct=port.base_sizing_pct)
        print(f"  Bar-level SL triggered: {sl_count}")

    print("Running simulation...")
    sim = simulate_with_improvements(
        prepared, port, breaker, imp,
        slippage_model=slippage_model, asset_volumes=asset_volumes)

    # Build per-trade dataframe
    n = len(prepared)
    trades = pd.DataFrame({
        "timestamp": prepared["timestamp"].values,
        "asset": prepared["asset"].values,
        "direction": prepared["direction"].values,
        "size": sim["sizes"],
        "pnl": sim["pnl"],
        "skipped": sim["skipped"],
    })
    trades = trades[~trades["skipped"]].copy()
    trades["category"] = trades["asset"].map(classify)
    print(f"  Executed trades: {len(trades):,}")
    print(f"  Total PnL: ${trades['pnl'].sum():,.0f}")
    print(f"  Final equity: ${sim['final_equity']:,.0f}")

    # Per-asset stats
    agg = trades.groupby("asset").agg(
        n_trades=("pnl", "size"),
        total_pnl=("pnl", "sum"),
        total_notional=("size", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_pnl=("pnl", "mean"),
    ).reset_index()
    agg["category"] = agg["asset"].map(classify)
    agg["roi_pct"] = (agg["total_pnl"] / agg["total_notional"] * 100).fillna(0)
    agg = agg.sort_values("total_pnl", ascending=False)

    print()
    print("=" * 100)
    print("  PER-CATEGORY SUMMARY")
    print("=" * 100)
    by_cat = trades.groupby("category").agg(
        n_trades=("pnl", "size"),
        total_pnl=("pnl", "sum"),
        total_notional=("size", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
    ).reset_index()
    by_cat["roi_pct"] = by_cat["total_pnl"] / by_cat["total_notional"] * 100
    by_cat["pct_of_trades"] = by_cat["n_trades"] / len(trades) * 100
    print(f"  {'Category':<10s} {'Trades':>8s} {'% of all':>9s} {'Total PnL':>14s} "
          f"{'Notional':>14s} {'ROI%':>8s} {'Win%':>7s}")
    print("  " + "-" * 80)
    for _, r in by_cat.iterrows():
        print(f"  {r['category']:<10s} {r['n_trades']:>8,d} {r['pct_of_trades']:>8.1f}% "
              f"${r['total_pnl']:>+12,.0f} ${r['total_notional']:>12,.0f} "
              f"{r['roi_pct']:>+7.3f}% {r['win_rate']:>6.1f}%")

    print()
    print("=" * 100)
    print("  TOP 20 BY PnL")
    print("=" * 100)
    print(f"  {'Asset':<10s} {'Cat':<5s} {'Trades':>8s} {'Total PnL':>14s} "
          f"{'ROI%':>8s} {'Win%':>7s}")
    print("  " + "-" * 70)
    for _, r in agg.head(20).iterrows():
        print(f"  {r['asset']:<10s} {r['category']:<5s} {r['n_trades']:>8,d} "
              f"${r['total_pnl']:>+12,.0f} {r['roi_pct']:>+7.3f}% "
              f"{r['win_rate']:>6.1f}%")

    print()
    print("=" * 100)
    print("  BOTTOM 20 BY PnL")
    print("=" * 100)
    print(f"  {'Asset':<10s} {'Cat':<5s} {'Trades':>8s} {'Total PnL':>14s} "
          f"{'ROI%':>8s} {'Win%':>7s}")
    print("  " + "-" * 70)
    for _, r in agg.tail(20).iloc[::-1].iterrows():
        print(f"  {r['asset']:<10s} {r['category']:<5s} {r['n_trades']:>8,d} "
              f"${r['total_pnl']:>+12,.0f} {r['roi_pct']:>+7.3f}% "
              f"{r['win_rate']:>6.1f}%")

    # Meme specifically
    memes = agg[agg["category"] == "meme"].sort_values("total_pnl", ascending=False)
    if len(memes) > 0:
        print()
        print("=" * 100)
        print("  ALL MEME COINS (ranked)")
        print("=" * 100)
        print(f"  {'Asset':<10s} {'Trades':>8s} {'Total PnL':>14s} {'ROI%':>8s} {'Win%':>7s}")
        print("  " + "-" * 60)
        for _, r in memes.iterrows():
            print(f"  {r['asset']:<10s} {r['n_trades']:>8,d} "
                  f"${r['total_pnl']:>+12,.0f} {r['roi_pct']:>+7.3f}% "
                  f"{r['win_rate']:>6.1f}%")


if __name__ == "__main__":
    main()
