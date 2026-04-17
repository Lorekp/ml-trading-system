#!/usr/bin/env python
"""Calibrate execution model from available BBO data.

Analyzes maker fill rates and spreads by volume tier across all available
BBO data (Binance bookTicker + Hyperliquid L2). Saves calibration to
data/execution_calibration.json for use by signal_engine.execution.

Usage:
    PYTHONPATH=src python scripts/calibrate_execution_model.py
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.execution import check_maker_fill, DEFAULT_TICK_SIZES

BBO_DIRS = [Path("data/hl-bbo"), Path("data/bookTicker_parquet")]
OUTPUT = Path("data/execution_calibration.json")


def analyze_symbol(sym, bbo_dir, n_sample_days=30):
    """Analyze maker fill rate and spread for one symbol.

    Samples up to n_sample_days randomly, generates synthetic signals
    every 15 minutes, checks 60s maker fill rate.
    """
    sym_dir = bbo_dir / sym
    if not sym_dir.exists():
        return None

    pqs = sorted(sym_dir.glob("*.parquet"))
    if not pqs:
        return None

    # Sample days
    if len(pqs) > n_sample_days:
        indices = np.linspace(0, len(pqs) - 1, n_sample_days, dtype=int)
        pqs = [pqs[i] for i in indices]

    all_fill_rates = []
    all_spreads = []
    total_volume = 0

    tick = DEFAULT_TICK_SIZES.get(sym, 0.01)

    for pq in pqs:
        try:
            df = pd.read_parquet(pq)
            if len(df) < 100:
                continue

            ts_ns = df["transaction_time"].values.astype(np.int64) * 1_000_000  # ms → ns
            bbid = df["best_bid_price"].values.astype(np.float64)
            bask = df["best_ask_price"].values.astype(np.float64)

            # Filter out zeros
            valid = (bbid > 0) & (bask > 0)
            ts_ns = ts_ns[valid]
            bbid = bbid[valid]
            bask = bask[valid]

            if len(ts_ns) < 50:
                continue

            # Compute spread
            mid = (bbid + bask) / 2
            spread_bps = (bask - bbid) / mid * 10000
            all_spreads.extend(spread_bps.tolist())

            # Approximate volume from BBO qty (rough proxy)
            total_volume += float(df["best_bid_qty"].sum() * df["best_bid_price"].mean())

            # Generate synthetic signals every 15 min and check fill
            sig_interval_ns = 15 * 60 * 1_000_000_000
            start_ns = ts_ns[0] + sig_interval_ns
            end_ns = ts_ns[-1] - 60 * 1_000_000_000

            sig_times = np.arange(start_ns, end_ns, sig_interval_ns)
            if len(sig_times) == 0:
                continue

            # Test with long direction
            dirs = np.ones(len(sig_times))
            filled, _ = check_maker_fill(ts_ns, bbid, bask, sig_times, dirs,
                                         timeout_ns=60_000_000_000, tick=tick)
            fill_rate = float(filled.mean())

            # Test with short direction
            dirs_short = -np.ones(len(sig_times))
            filled_s, _ = check_maker_fill(ts_ns, bbid, bask, sig_times, dirs_short,
                                           timeout_ns=60_000_000_000, tick=tick)
            fill_rate_short = float(filled_s.mean())

            avg_fill = (fill_rate + fill_rate_short) / 2
            all_fill_rates.append(avg_fill)

        except Exception:
            continue

    if not all_fill_rates:
        return None

    return {
        "symbol": sym,
        "fill_rate": float(np.mean(all_fill_rates)),
        "fill_rate_std": float(np.std(all_fill_rates)),
        "spread_median_bps": float(np.median(all_spreads)) if all_spreads else 0,
        "spread_p90_bps": float(np.percentile(all_spreads, 90)) if all_spreads else 0,
        "n_days": len(all_fill_rates),
        "approx_daily_volume": total_volume / max(len(all_fill_rates), 1),
    }


def main():
    T0 = time.time()
    print("=" * 70)
    print("  CALIBRATE EXECUTION MODEL")
    print("=" * 70)

    # Find all symbols with BBO data
    all_symbols = set()
    for bbo_dir in BBO_DIRS:
        if bbo_dir.exists():
            for d in bbo_dir.iterdir():
                if d.is_dir() and list(d.glob("*.parquet")):
                    all_symbols.add((d.name, bbo_dir))

    print(f"\n  Symbols with BBO data: {len(all_symbols)}")

    results = []
    for i, (sym, bbo_dir) in enumerate(sorted(all_symbols), 1):
        t1 = time.time()
        r = analyze_symbol(sym, bbo_dir)
        if r:
            results.append(r)
            print(f"  [{i}/{len(all_symbols)}] {sym}: fill={r['fill_rate']:.1%}, "
                  f"spread={r['spread_median_bps']:.2f}bps, {r['n_days']}d ({time.time()-t1:.1f}s)")
        else:
            print(f"  [{i}/{len(all_symbols)}] {sym}: no data")

    if not results:
        print("\n  No BBO data found! Using defaults.")
        return

    # Sort by approximate volume
    results.sort(key=lambda x: -x["approx_daily_volume"])

    # Compute tier thresholds from actual data
    fill_rates = [r["fill_rate"] for r in results]
    spreads = [r["spread_median_bps"] for r in results]
    volumes = [r["approx_daily_volume"] for r in results]

    print(f"\n  Overall: {len(results)} symbols analyzed")
    print(f"  Fill rate: mean={np.mean(fill_rates):.1%}, "
          f"min={np.min(fill_rates):.1%}, max={np.max(fill_rates):.1%}")
    print(f"  Spread: median={np.median(spreads):.2f}bps, "
          f"p10={np.percentile(spreads, 10):.2f}bps, p90={np.percentile(spreads, 90):.2f}bps")

    # Build calibration tiers based on quartiles
    n = len(results)
    if n >= 4:
        tier1 = results[:max(2, n // 10)]  # Top 10% (mega-cap)
        tier2 = results[max(2, n // 10):n // 4]  # Next 15%
        tier3 = results[n // 4:n // 2]  # Next 25%
        tier4 = results[n // 2:]  # Bottom 50%
    else:
        tier1 = tier2 = tier3 = tier4 = results

    def tier_stats(tier_list):
        if not tier_list:
            return 0, 0.90, 1.0
        vol_min = min(r["approx_daily_volume"] for r in tier_list)
        fill = np.mean([r["fill_rate"] for r in tier_list])
        spread = np.median([r["spread_median_bps"] for r in tier_list])
        return vol_min, fill, spread

    calibration = {}
    for name, tier_data, desc in [
        ("tier1", tier1, "Mega-cap (top 10%)"),
        ("tier2", tier2, "Large-cap (top 25%)"),
        ("tier3", tier3, "Mid-cap (top 50%)"),
        ("tier4", tier4, "Small-cap (bottom 50%)"),
    ]:
        vol_min, fill, spread = tier_stats(tier_data)
        calibration[name] = {
            "volume_min": float(vol_min),
            "fill_rate": round(float(fill), 3),
            "spread_bps": round(float(spread), 2),
            "desc": desc,
            "n_symbols": len(tier_data),
        }
        print(f"\n  {name} ({desc}):")
        print(f"    Symbols: {len(tier_data)}, vol_min: ${vol_min:,.0f}")
        print(f"    Fill rate: {fill:.1%}, Spread: {spread:.2f}bps")

    # Save
    output_data = {
        "calibration": calibration,
        "per_symbol": results,
        "metadata": {
            "n_symbols": len(results),
            "bbo_sources": [str(d) for d in BBO_DIRS if d.exists()],
            "generated_at": pd.Timestamp.now().isoformat(),
        }
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")
    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
