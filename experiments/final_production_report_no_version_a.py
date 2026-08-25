"""Final Production Report without Version A.

Generates:
1. Lookback table for the common Version B test period with NCO k=8
2. Full stats report for the production config (8mo, Version B)
3. Monthly PnL breakdown
4. Bootstrap CI, DSR, VaR/CVaR, underwater analysis

This wrapper reuses the original implementation and removes the
Version A reporting sections only.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import final_production_report as base


def main():
    print("=" * 70)
    print("FINAL PRODUCTION REPORT")
    print(f"  13 features, NCO k=8 allocation, {base.W} workers")
    print(f"  Lookbacks: {base.LOOKBACKS}")
    print("  Reporting scope: Version B only")
    print("=" * 70, flush=True)

    t0 = time.time()

    print(f"\n[1] Generating signals for all lookbacks ({base.W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=base.W) as pool:
        futures = {pool.submit(base._process_asset, (a, s)): a for a, s in base.ASSETS.items()}
        done = 0
        for future in as_completed(futures):
            asset, results = future.result()
            done += 1
            all_results[asset] = results
            total = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {total} month-lookback combos", flush=True)

    sample_1m = base.load_binance_1m("BTCUSDT", [base.DATA_DIR])
    sample_15m = base.resample_ohlcv(sample_1m, "15min")
    data_start = sample_15m.index.min()
    fixed_start = str((data_start + pd.DateOffset(months=max(base.LOOKBACKS) + 1)).to_period("M"))
    print(f"\n  Signal generation: {time.time()-t0:.0f}s")
    print(f"  Version B start: {fixed_start}")

    print(f"\n{'='*70}")
    print("LOOKBACK TABLE WITH NCO k=8 ALLOCATION (VERSION B)")
    print(f"{'='*70}")
    print(f"  Common test period start: {fixed_start}")
    print(f"{'Lookback':<10s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'Trades':>7s} {'Days':>5s} {'PnL/day':>8s}")
    print("-" * 55)

    for lb in base.LOOKBACKS:
        result = base.run_lookback(lb, all_results, fixed_start)
        if result:
            print(
                f"  {lb:>2d}mo      {result['sharpe']:>7.3f} {result['total_return_pct']:>8.2f} "
                f"{result['max_dd_pct']:>7.2f} {result['n_trades']:>7d} {result['n_days']:>5d} "
                f"${result['pnl_day']:>7.2f}",
                flush=True,
            )
        else:
            print(f"  {lb:>2d}mo      NO SIGNALS", flush=True)

    print(f"\n{'='*70}")
    print("FULL STATS: 8-MONTH ROLLING + NCO k=8 (Version B)")
    print(f"{'='*70}")
    prod_result = base.run_lookback(8, all_results, fixed_start)
    base.full_stats_report("8-Month Rolling, NCO k=8, Version B", prod_result)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
