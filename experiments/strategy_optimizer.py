#!/usr/bin/env python
"""Tier 1 Strategy Optimizer — Parameter sweeps on cached signals.

Sweeps portfolio-level parameters (sizing, constraints, vol overlay, weights)
on existing cached signals WITHOUT regenerating them. Each config takes <10s.

Usage:
    python experiments/strategy_optimizer.py
"""

import sys, os, time, warnings, itertools
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.data import load_binance_1m, resample_ohlcv
from pathlib import Path

CAPITAL = 100_000.0
BYBIT_RT_BPS = 4.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]


def load_signals():
    """Load cached signals for all 3 strategies."""
    sigs = {}
    for name in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        path = Path(f"results/signals_cache/{name}.parquet")
        if not path.exists():
            print(f"  MISSING: {path}")
            return None
        sigs[name] = pd.read_parquet(path)
    return sigs


def merge_signals(sigs):
    """Merge all strategies into a single DataFrame."""
    parts = []
    for name, df in sigs.items():
        d = df.copy()
        if "strategy" not in d.columns:
            d["strategy"] = name
        if "regime_exposure_mult" not in d.columns:
            d["regime_exposure_mult"] = 1.0
        if "sizing_mult" not in d.columns:
            d["sizing_mult"] = 1.0
        if "hold_bars" not in d.columns:
            d["hold_bars"] = 4
        d["regime_exposure_mult"] = d["regime_exposure_mult"].fillna(1.0)
        d["sizing_mult"] = d["sizing_mult"].fillna(1.0)
        d["probability"] = d["probability"].astype(float)
        parts.append(d)
    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return merged


def bybit_sim(signals, rt_bps=BYBIT_RT_BPS, capital=CAPITAL,
              base_pct=0.02, max_mult=2.0):
    """Quick Bybit execution cost sim with configurable sizing."""
    sub = signals.copy()
    rt = rt_bps / 10000.0
    gross = sub["direction"].values * sub["fwd_return"].values
    if "return_already_signed" in sub.columns:
        signed_mask = sub["return_already_signed"].values.astype(bool)
        gross[signed_mask] = sub["fwd_return"].values[signed_mask]
    net = gross - rt

    prob = sub["probability"].values if "probability" in sub.columns else np.full(len(sub), np.nan)
    sizing_mult = sub["sizing_mult"].values if "sizing_mult" in sub.columns else np.ones(len(sub))

    edge = np.where(np.isfinite(prob), np.abs(2 * prob - 1), 0.1)
    edge = np.clip(edge, 0, 1)

    eq = capital
    pnl = np.zeros(len(sub))
    for i in range(len(sub)):
        sz = min(eq * base_pct * edge[i] * sizing_mult[i], eq * base_pct * max_mult)
        sz = max(sz, 0)
        pnl[i] = sz * net[i]
        eq += pnl[i]

    ts = pd.to_datetime(sub["timestamp"].values)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily != 0]

    if len(daily) > 30 and daily.std() > 0:
        sr = float(daily.mean() / daily.std() * np.sqrt(365.25))
    else:
        sr = 0.0

    cum_eq = capital + daily.cumsum()
    dd = float(((cum_eq.cummax() - cum_eq) / cum_eq.cummax()).max() * 100) if len(cum_eq) > 0 else 0

    return {
        "sr": sr, "pnl": float(pnl.sum()), "eq": eq, "daily": daily,
        "n": len(sub), "mean_net_bps": float(net.mean() * 10000),
        "max_dd_pct": dd,
    }


def portfolio_sim(signals, config, label=""):
    """Run full simulate_portfolio and return SR + stats."""
    sim = simulate_portfolio(signals, config)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)
    stats = compute_full_stats(dpnl, config.starting_capital, sim["n_trades"],
                                sim["n_trades"] / max(len(dpnl), 1))
    return {
        "sr": stats.get("sharpe", 0),
        "sortino": stats.get("sortino", 0),
        "max_dd": stats.get("max_dd_pct", 0),
        "n_trades": sim["n_trades"],
        "n_skipped": int(sim["skipped"].sum()),
        "final_eq": sim["final_equity"],
        "pnl": float(dpnl.sum()),
        "win_rate": stats.get("win_rate_days", 0),
        "profit_factor": stats.get("profit_factor", 0),
    }


def load_btc_daily():
    """Load BTC daily close for vol overlay."""
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        return btc_daily
    except Exception as e:
        print(f"  BTC daily load failed: {e}")
        return None


def apply_vol(signals, btc_daily, vol_cfg):
    """Apply vol overlay, return new signals DataFrame."""
    if btc_daily is None or not vol_cfg.enabled:
        out = signals.copy()
        out["sizing_mult"] = 1.0
        return out
    return apply_vol_overlay(signals, btc_daily, vol_cfg)


# ============================================================
# SWEEP FUNCTIONS
# ============================================================

def sweep_sizing(signals, btc_daily):
    """Sweep base_sizing_pct and max_signal_mult using bybit_sim."""
    print("\n" + "=" * 80)
    print("  SWEEP 1A: Position Sizing (bybit_sim)")
    print("=" * 80)

    results = []
    for base_pct in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        for max_mult in [2.0, 3.0, 4.0]:
            r = bybit_sim(signals, base_pct=base_pct, max_mult=max_mult)
            results.append({
                "base_pct": base_pct, "max_mult": max_mult,
                "sr": r["sr"], "pnl": r["pnl"], "dd": r["max_dd_pct"],
                "net_bps": r["mean_net_bps"],
            })
            print(f"  base={base_pct:.2f} max_mult={max_mult:.1f}: "
                  f"SR={r['sr']:.3f}  DD={r['max_dd_pct']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  net={r['mean_net_bps']:.2f}bps")

    df = pd.DataFrame(results).sort_values("sr", ascending=False)
    best = df.iloc[0]
    print(f"\n  BEST: base={best['base_pct']:.2f} max_mult={best['max_mult']:.1f} "
          f"SR={best['sr']:.3f} DD={best['dd']:.1f}%")
    return df


def sweep_strategy_constraint(signals):
    """Sweep max_per_strategy using simulate_portfolio."""
    print("\n" + "=" * 80)
    print("  SWEEP 1B: max_per_strategy (simulate_portfolio)")
    print("=" * 80)

    results = []
    for max_strat in [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        config = PortfolioConfig(
            starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
            max_per_strategy=max_strat, base_sizing_pct=0.02,
            max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
        )
        r = portfolio_sim(signals, config)
        if r:
            results.append({"max_per_strategy": max_strat, **r})
            print(f"  max_strat={max_strat:.2f}: SR={r['sr']:.3f}  DD={r['max_dd']:.1f}%  "
                  f"trades={r['n_trades']:,}  skipped={r['n_skipped']:,}")

    df = pd.DataFrame(results).sort_values("sr", ascending=False)
    if len(df) > 0:
        best = df.iloc[0]
        print(f"\n  BEST: max_strat={best['max_per_strategy']:.2f} SR={best['sr']:.3f}")
    return df


def sweep_vol_overlay(signals, btc_daily):
    """Sweep vol overlay configurations."""
    print("\n" + "=" * 80)
    print("  SWEEP 1C: Vol Overlay Variants (bybit_sim)")
    print("=" * 80)

    configs = {
        "OFF": VolOverlayConfig(enabled=False),
        "current": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                     low_pct=0.20, high_pct=0.80,
                                     low_mult=0.50, mid_mult=1.00, high_mult=1.25),
        "aggressive": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                        low_pct=0.20, high_pct=0.80,
                                        low_mult=0.25, mid_mult=1.00, high_mult=1.50),
        "inverted": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                      low_pct=0.20, high_pct=0.80,
                                      low_mult=1.25, mid_mult=1.00, high_mult=0.50),
        "flat_high": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                       low_pct=0.20, high_pct=0.80,
                                       low_mult=0.50, mid_mult=1.00, high_mult=1.00),
        "no_penalty": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                        low_pct=0.20, high_pct=0.80,
                                        low_mult=1.00, mid_mult=1.00, high_mult=1.25),
    }

    results = []
    for name, vcfg in configs.items():
        df_v = apply_vol(signals, btc_daily, vcfg)
        r = bybit_sim(df_v)
        results.append({"vol_config": name, "sr": r["sr"], "pnl": r["pnl"],
                         "dd": r["max_dd_pct"], "net_bps": r["mean_net_bps"]})
        print(f"  {name:<15}: SR={r['sr']:.3f}  DD={r['max_dd_pct']:.1f}%  "
              f"PnL=${r['pnl']:,.0f}  net={r['mean_net_bps']:.2f}bps")

    df = pd.DataFrame(results).sort_values("sr", ascending=False)
    if len(df) > 0:
        print(f"\n  BEST: {df.iloc[0]['vol_config']} SR={df.iloc[0]['sr']:.3f}")
    return df


def sweep_regime_mult(signals):
    """Test regime_exposure_multiplier ON vs OFF."""
    print("\n" + "=" * 80)
    print("  SWEEP 1D: Regime Exposure Multiplier (simulate_portfolio)")
    print("=" * 80)

    results = []
    for use_regime in [True, False]:
        config = PortfolioConfig(
            starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
            max_per_strategy=0.40, base_sizing_pct=0.02,
            max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=use_regime, regime_exposure_floor=0.50,
        )
        r = portfolio_sim(signals, config)
        if r:
            results.append({"use_regime_mult": use_regime, **r})
            print(f"  regime_mult={use_regime}: SR={r['sr']:.3f}  DD={r['max_dd']:.1f}%  "
                  f"trades={r['n_trades']:,}")

    return pd.DataFrame(results)


def sweep_combined_best(signals, btc_daily, best_sizing, best_strat, best_vol, best_regime):
    """Test the combined best configuration from individual sweeps."""
    print("\n" + "=" * 80)
    print("  COMBINED BEST CONFIG (bybit_sim + simulate_portfolio)")
    print("=" * 80)

    # Apply best vol overlay
    vol_configs = {
        "OFF": VolOverlayConfig(enabled=False),
        "current": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                     low_pct=0.20, high_pct=0.80,
                                     low_mult=0.50, mid_mult=1.00, high_mult=1.25),
        "aggressive": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                        low_pct=0.20, high_pct=0.80,
                                        low_mult=0.25, mid_mult=1.00, high_mult=1.50),
        "inverted": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                      low_pct=0.20, high_pct=0.80,
                                      low_mult=1.25, mid_mult=1.00, high_mult=0.50),
        "flat_high": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                       low_pct=0.20, high_pct=0.80,
                                       low_mult=0.50, mid_mult=1.00, high_mult=1.00),
        "no_penalty": VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                        low_pct=0.20, high_pct=0.80,
                                        low_mult=1.00, mid_mult=1.00, high_mult=1.25),
    }
    vol_cfg = vol_configs.get(best_vol, vol_configs["current"])
    sigs_v = apply_vol(signals, btc_daily, vol_cfg)

    # bybit_sim with best sizing
    r_bybit = bybit_sim(sigs_v, base_pct=best_sizing[0], max_mult=best_sizing[1])
    print(f"  Bybit sim:  SR={r_bybit['sr']:.3f}  DD={r_bybit['max_dd_pct']:.1f}%  "
          f"PnL=${r_bybit['pnl']:,.0f}")

    # simulate_portfolio with best constraints
    config = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=best_strat, base_sizing_pct=best_sizing[0],
        max_signal_mult=best_sizing[1], sizing_mode="probability_edge",
        use_regime_exposure_multiplier=best_regime, regime_exposure_floor=0.50,
    )
    r_port = portfolio_sim(sigs_v, config, "Combined best")
    if r_port:
        print(f"  Portfolio:  SR={r_port['sr']:.3f}  DD={r_port['max_dd']:.1f}%  "
              f"trades={r_port['n_trades']:,}")

    # Per-strategy breakdown
    print(f"\n  Per-Strategy (Bybit, best sizing):")
    for strat in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        s = sigs_v[sigs_v["strategy"] == strat] if "strategy" in sigs_v.columns else pd.DataFrame()
        if len(s) > 10:
            r = bybit_sim(s.sort_values("timestamp"), base_pct=best_sizing[0],
                          max_mult=best_sizing[1])
            print(f"    {strat:<15}: SR={r['sr']:.3f}  trades={r['n']:,}  "
                  f"net={r['mean_net_bps']:.1f}bps  pnl=${r['pnl']:,.0f}")

    return r_bybit, r_port


def sweep_confidence_filter(signals, btc_daily):
    """Sweep post-hoc confidence filters on cached signals.

    The cached signals already passed a 70th percentile filter during generation.
    This applies an additional absolute confidence threshold to cut marginal trades.
    """
    print("\n" + "=" * 80)
    print("  SWEEP 1F: Post-Hoc Confidence Filter (bybit_sim)")
    print("=" * 80)

    # Apply inverted vol overlay (best from 1C)
    vol_inv = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                low_pct=0.20, high_pct=0.80,
                                low_mult=1.25, mid_mult=1.00, high_mult=0.50)
    sigs_v = apply_vol(signals, btc_daily, vol_inv)

    results = []

    # Check what confidence values look like
    if "confidence" in sigs_v.columns:
        conf = sigs_v["confidence"].values
        print(f"  Confidence stats: mean={np.nanmean(conf):.4f} "
              f"median={np.nanmedian(conf):.4f} "
              f"p75={np.nanpercentile(conf, 75):.4f} "
              f"p90={np.nanpercentile(conf, 90):.4f}")

        # Sweep absolute confidence thresholds
        for min_conf in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
            filtered = sigs_v[sigs_v["confidence"] >= min_conf].copy()
            if len(filtered) < 100:
                continue
            r = bybit_sim(filtered.sort_values("timestamp"))
            results.append({
                "min_confidence": min_conf,
                "n_signals": len(filtered),
                "pct_kept": len(filtered) / len(sigs_v) * 100,
                "sr": r["sr"], "pnl": r["pnl"], "dd": r["max_dd_pct"],
                "net_bps": r["mean_net_bps"],
            })
            print(f"  conf>={min_conf:.2f}: {len(filtered):>7,} signals ({len(filtered)/len(sigs_v)*100:.0f}%)  "
                  f"SR={r['sr']:.3f}  DD={r['max_dd_pct']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  net={r['mean_net_bps']:.2f}bps")

        # Per-strategy confidence sweep for logreg_15m only
        print(f"\n  --- logreg_15m only (highest trade count, lowest per-trade edge) ---")
        for min_conf in [0.00, 0.04, 0.08, 0.10, 0.12, 0.15, 0.20]:
            mask_15m = sigs_v["strategy"] == "logreg_15m"
            mask_other = ~mask_15m
            filtered_15m = sigs_v[mask_15m & (sigs_v["confidence"] >= min_conf)]
            other = sigs_v[mask_other]
            combined = pd.concat([filtered_15m, other]).sort_values("timestamp")
            if len(combined) < 100:
                continue
            r = bybit_sim(combined)
            n_15m = len(filtered_15m)
            print(f"  15m conf>={min_conf:.2f}: {n_15m:>7,} 15m signals  "
                  f"SR={r['sr']:.3f}  DD={r['max_dd_pct']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  net={r['mean_net_bps']:.2f}bps")
    else:
        print("  WARNING: 'confidence' column not found in signals!")

    df = pd.DataFrame(results).sort_values("sr", ascending=False) if results else pd.DataFrame()
    if len(df) > 0:
        best = df.iloc[0]
        print(f"\n  BEST: conf>={best['min_confidence']:.2f} "
              f"({best['pct_kept']:.0f}% kept) SR={best['sr']:.3f}")
    return df


def bootstrap_ci(daily_pnl, capital, n=5000):
    """Bootstrap 95% CI for Sharpe ratio."""
    dr = (daily_pnl / capital).values
    rng = np.random.RandomState(42)
    srs = np.empty(n)
    for i in range(n):
        s = rng.choice(dr, size=len(dr), replace=True)
        srs[i] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    return float(np.percentile(srs, 2.5)), float(np.percentile(srs, 97.5))


# ============================================================
# MAIN
# ============================================================

def main():
    T0 = time.time()
    print("=" * 80)
    print("  TIER 1 STRATEGY OPTIMIZER — Parameter Sweeps on Cached Signals")
    print("=" * 80)

    # Load signals
    print("\nLoading cached signals...")
    sigs = load_signals()
    if sigs is None:
        return
    for name, df in sigs.items():
        print(f"  {name}: {len(df):,} signals")

    merged = merge_signals(sigs)
    print(f"  Total merged: {len(merged):,}")

    # Load BTC daily for vol overlay
    print("\nLoading BTC daily for vol overlay...")
    btc_daily = load_btc_daily()

    # Baseline
    print("\n" + "=" * 80)
    print("  BASELINE (current config)")
    print("=" * 80)

    # Apply current vol overlay
    vol_current = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                    low_pct=0.20, high_pct=0.80,
                                    low_mult=0.50, mid_mult=1.00, high_mult=1.25)
    merged_vol = apply_vol(merged, btc_daily, vol_current)
    r_base = bybit_sim(merged_vol)
    ci = bootstrap_ci(r_base["daily"], CAPITAL) if len(r_base["daily"]) > 30 else (0, 0)
    print(f"  Bybit SR: {r_base['sr']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
          f"DD={r_base['max_dd_pct']:.1f}%  PnL=${r_base['pnl']:,.0f}")

    # Run sweeps
    df_sizing = sweep_sizing(merged_vol, btc_daily)
    df_strat = sweep_strategy_constraint(merged_vol)
    df_vol = sweep_vol_overlay(merged, btc_daily)
    df_regime = sweep_regime_mult(merged_vol)
    df_conf = sweep_confidence_filter(merged, btc_daily)

    # Extract best from each sweep
    best_sizing = (df_sizing.iloc[0]["base_pct"], df_sizing.iloc[0]["max_mult"])
    best_strat = df_strat.iloc[0]["max_per_strategy"] if len(df_strat) > 0 else 0.40
    best_vol = df_vol.iloc[0]["vol_config"] if len(df_vol) > 0 else "current"
    best_regime = df_regime.sort_values("sr", ascending=False).iloc[0]["use_regime_mult"] if len(df_regime) > 0 else True

    print(f"\n{'='*80}")
    print(f"  INDIVIDUAL SWEEP WINNERS:")
    print(f"  Sizing: base={best_sizing[0]:.2f}, max_mult={best_sizing[1]:.1f}")
    print(f"  Strategy cap: {best_strat:.2f}")
    print(f"  Vol overlay: {best_vol}")
    print(f"  Regime mult: {best_regime}")
    print(f"{'='*80}")

    # Combined best (original, without confidence filter)
    r_bybit_best, r_port_best = sweep_combined_best(
        merged, btc_daily, best_sizing, best_strat, best_vol, best_regime
    )

    # ================================================================
    # COMBINED BEST WITH CONFIDENCE FILTER
    # ================================================================
    print(f"\n{'='*80}")
    print("  COMBINED: Inverted Vol + Confidence Filter + Sizing Sweep")
    print(f"{'='*80}")

    vol_inv = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                low_pct=0.20, high_pct=0.80,
                                low_mult=1.25, mid_mult=1.00, high_mult=0.50)
    sigs_inv = apply_vol(merged, btc_daily, vol_inv)

    # For each confidence threshold, also sweep sizing
    for min_conf in [0.06, 0.08, 0.10, 0.12]:
        filtered = sigs_inv[sigs_inv["confidence"] >= min_conf].sort_values("timestamp")
        if len(filtered) < 100:
            continue
        print(f"\n  --- conf >= {min_conf:.2f} ({len(filtered):,} signals, {len(filtered)/len(sigs_inv)*100:.0f}%) ---")
        for base_pct in [0.02, 0.04, 0.06, 0.08]:
            r = bybit_sim(filtered, base_pct=base_pct, max_mult=4.0)
            print(f"    base={base_pct:.2f}: SR={r['sr']:.3f}  DD={r['max_dd_pct']:.1f}%  "
                  f"PnL=${r['pnl']:,.0f}  net={r['mean_net_bps']:.2f}bps")

    # Year-by-year analysis for conf >= 0.08 (the practical sweet spot)
    print(f"\n{'='*80}")
    print("  YEAR-BY-YEAR: conf >= 0.08 + inverted vol")
    print(f"{'='*80}")
    filtered_08 = sigs_inv[sigs_inv["confidence"] >= 0.08].sort_values("timestamp")
    r_08 = bybit_sim(filtered_08)
    ts = pd.to_datetime(filtered_08["timestamp"])
    for year in sorted(ts.dt.year.unique()):
        year_mask = ts.dt.year == year
        year_sigs = filtered_08[year_mask.values]
        if len(year_sigs) > 50:
            r_y = bybit_sim(year_sigs.sort_values("timestamp"))
            print(f"  {year}: SR={r_y['sr']:.3f}  signals={len(year_sigs):,}  "
                  f"PnL=${r_y['pnl']:,.0f}  net={r_y['mean_net_bps']:.1f}bps")

    # Bootstrap CI for the combined best
    print(f"\n{'='*80}")
    print("  FINAL COMPARISON")
    print(f"{'='*80}")
    ci_08 = bootstrap_ci(r_08["daily"], CAPITAL) if len(r_08["daily"]) > 30 else (0, 0)
    print(f"  Baseline (current):        SR={r_base['sr']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
          f"PnL=${r_base['pnl']:,.0f}  DD={r_base['max_dd_pct']:.1f}%")
    print(f"  + Inverted vol only:       SR={r_bybit_best['sr']:.3f}  "
          f"PnL=${r_bybit_best['pnl']:,.0f}  DD={r_bybit_best['max_dd_pct']:.1f}%")
    print(f"  + Inv vol + conf>=0.08:    SR={r_08['sr']:.3f} [{ci_08[0]:.3f}, {ci_08[1]:.3f}]  "
          f"PnL=${r_08['pnl']:,.0f}  DD={r_08['max_dd_pct']:.1f}%  ({len(filtered_08):,} trades)")

    # Final CI
    if r_bybit_best and len(r_bybit_best["daily"]) > 30:
        ci_best = bootstrap_ci(r_bybit_best["daily"], CAPITAL)
        print(f"\n  Combined best Bybit SR: {r_bybit_best['sr']:.3f} "
              f"[{ci_best[0]:.3f}, {ci_best[1]:.3f}] 95% CI")
        print(f"  vs baseline:            {r_base['sr']:.3f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}] 95% CI")
        delta = r_bybit_best['sr'] - r_base['sr']
        print(f"  Delta: {delta:+.3f} SR")

    # Write report
    report_path = Path("experiments/REPORT_optimization.md")
    with open(report_path, "w") as f:
        f.write("# Tier 1 Strategy Optimization Results\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Baseline SR (Bybit):** {r_base['sr']:.3f}\n\n")

        f.write("## Sizing Sweep\n\n")
        f.write("| base_pct | max_mult | SR | DD% | PnL |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_sizing.iterrows():
            f.write(f"| {row['base_pct']:.2f} | {row['max_mult']:.1f} | "
                    f"{row['sr']:.3f} | {row['dd']:.1f} | ${row['pnl']:,.0f} |\n")

        f.write("\n## Strategy Constraint Sweep\n\n")
        f.write("| max_per_strategy | SR | DD% | Trades | Skipped |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_strat.iterrows():
            f.write(f"| {row['max_per_strategy']:.2f} | {row['sr']:.3f} | "
                    f"{row['max_dd']:.1f} | {row['n_trades']:,} | {row['n_skipped']:,} |\n")

        f.write("\n## Vol Overlay Sweep\n\n")
        f.write("| Config | SR | DD% | PnL | Net bps |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_vol.iterrows():
            f.write(f"| {row['vol_config']} | {row['sr']:.3f} | "
                    f"{row['dd']:.1f} | ${row['pnl']:,.0f} | {row['net_bps']:.2f} |\n")

        f.write("\n## Regime Multiplier\n\n")
        f.write("| use_regime_mult | SR | DD% | Trades |\n")
        f.write("|---|---|---|---|\n")
        for _, row in df_regime.iterrows():
            f.write(f"| {row['use_regime_mult']} | {row['sr']:.3f} | "
                    f"{row['max_dd']:.1f} | {row['n_trades']:,} |\n")

        f.write(f"\n## Combined Best\n\n")
        f.write(f"- Sizing: base={best_sizing[0]:.2f}, max_mult={best_sizing[1]:.1f}\n")
        f.write(f"- Strategy cap: {best_strat:.2f}\n")
        f.write(f"- Vol overlay: {best_vol}\n")
        f.write(f"- Regime mult: {best_regime}\n")
        if r_bybit_best:
            f.write(f"- **SR: {r_bybit_best['sr']:.3f}** (vs baseline {r_base['sr']:.3f})\n")
            f.write(f"- DD: {r_bybit_best['max_dd_pct']:.1f}%\n")
            f.write(f"- PnL: ${r_bybit_best['pnl']:,.0f}\n")

    print(f"\n  Report saved to {report_path}")
    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
