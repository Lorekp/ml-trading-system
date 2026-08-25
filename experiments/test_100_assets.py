"""Test the algorithm on ALL available assets — no cherry-picking.

Uses multiprocessing to parallelize across CPU cores.
Each asset is fully independent (own model, own walk-forward).
"""
import sys
sys.path.insert(0, "src")

import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def process_single_asset(symbol: str, data_dir: str) -> dict:
    """Process one asset end-to-end. Runs in a subprocess."""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from pathlib import Path
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.filters import apply_confidence_filter
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    from signal_engine.stats import compute_full_stats

    cfg = load_config(Path("configs/default.yaml"))
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08

    t0 = time.time()

    try:
        ohlcv_1m = load_binance_1m(symbol, [Path(data_dir)])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

        if len(ohlcv_15m) < 4000:
            return {"symbol": symbol, "status": "insufficient_data", "bars": len(ohlcv_15m)}

        features = compute_features(ohlcv_15m, cfg.features)

        wf = WalkForwardModel(cfg)
        raw = wf.generate_signals_batch(ohlcv_15m, features, symbol.replace("USDT", ""))

        if len(raw) == 0:
            return {"symbol": symbol, "status": "no_signals"}

        filtered = apply_confidence_filter(raw, cfg.confidence_filter)
        if len(filtered) < 10:
            return {"symbol": symbol, "status": "too_few_signals", "n_signals": len(filtered)}

        sim = simulate_portfolio(filtered, cfg.portfolio)
        if sim is None or sim["n_trades"] < 10:
            return {"symbol": symbol, "status": "too_few_trades"}

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 5:
            return {"symbol": symbol, "status": "insufficient_days"}

        stats = compute_full_stats(dpnl, cfg.portfolio.starting_capital,
                                    sim["n_trades"], sim["n_trades"] / len(dpnl))

        return {
            "symbol": symbol,
            "status": "ok",
            "sharpe": stats["sharpe"],
            "return_pct": stats["total_return_pct"],
            "max_dd_pct": stats["max_dd_pct"],
            "trades": sim["n_trades"],
            "n_days": stats["n_days"],
            "pnl": round(stats.get("final_equity", 10000) - 10000, 2),
            "win_rate": stats.get("win_rate_days", 0),
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Exception as e:
        return {"symbol": symbol, "status": f"error: {str(e)[:80]}",
                "elapsed_s": round(time.time() - t0, 1)}


def main():
    data_dir = Path("data/binance-1m")

    all_symbols = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and len(list(d.glob("*-1m-*.csv"))) >= 6
    ])

    n_workers = max(1, os.cpu_count() - 1)

    print("=" * 80)
    print(f"TESTING ALGORITHM ON ALL {len(all_symbols)} ASSETS ({n_workers} workers)")
    print("Config: regime_off + abs_conf_0.08 + 14 features + LogReg L1")
    print("=" * 80, flush=True)

    t_start = time.time()
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(process_single_asset, sym, str(data_dir)): sym
            for sym in all_symbols
        }

        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            completed += 1

            if r["status"] == "ok":
                print(
                    f"[{completed:>3d}/{len(all_symbols)}] {r['symbol']:<14s} "
                    f"Sharpe={r['sharpe']:>7.3f} Ret={r['return_pct']:>7.2f}% "
                    f"DD={r['max_dd_pct']:>5.2f}% Trades={r['trades']:>6d} "
                    f"PnL=${r['pnl']:>8.2f} ({r['elapsed_s']:.0f}s)",
                    flush=True,
                )
            else:
                print(
                    f"[{completed:>3d}/{len(all_symbols)}] {r['symbol']:<14s} "
                    f"{r['status']} ({r.get('elapsed_s', 0):.0f}s)",
                    flush=True,
                )

    elapsed = time.time() - t_start

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda x: x["sharpe"], reverse=True)

    print()
    print("=" * 80)
    print(f"RESULTS ({elapsed:.0f}s total, {elapsed/max(len(all_symbols),1):.1f}s/asset avg)")
    print("=" * 80)
    print(f"Tested:   {len(all_symbols)}")
    print(f"Results:  {len(ok)}")
    print(f"Skipped:  {len(results) - len(ok)} (insufficient data/errors)")
    print()

    if ok:
        sharpes = np.array([r["sharpe"] for r in ok])
        positive = sum(1 for s in sharpes if s > 0)
        strong = sum(1 for s in sharpes if s > 1.0)
        excellent = sum(1 for s in sharpes if s > 2.0)

        print(f"Sharpe > 0:   {positive}/{len(ok)} ({100*positive/len(ok):.0f}%)")
        print(f"Sharpe > 1.0: {strong}/{len(ok)} ({100*strong/len(ok):.0f}%)")
        print(f"Sharpe > 2.0: {excellent}/{len(ok)} ({100*excellent/len(ok):.0f}%)")
        print()
        print(f"Distribution: mean={sharpes.mean():.3f} median={np.median(sharpes):.3f} "
              f"std={sharpes.std():.3f} min={sharpes.min():.3f} max={sharpes.max():.3f}")
        print()

        total_pnl = sum(r["pnl"] for r in ok)
        print(f"Total PnL: ${total_pnl:,.2f}")
        print()

        print("Top 15:")
        for r in ok[:15]:
            print(f"  {r['symbol']:<14s} Sharpe={r['sharpe']:>7.3f} Ret={r['return_pct']:>7.2f}% PnL=${r['pnl']:>8.2f}")
        print()
        print("Bottom 10:")
        for r in ok[-10:]:
            print(f"  {r['symbol']:<14s} Sharpe={r['sharpe']:>7.3f} Ret={r['return_pct']:>7.2f}% PnL=${r['pnl']:>8.2f}")

    # Save
    out_path = Path("v4-results/test_100_assets.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_total": len(all_symbols),
                    "n_ok": len(ok), "elapsed_s": round(elapsed, 1)}, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
