"""Cache all assets as 15m parquet — parallel, skip existing."""
import sys, os, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance-1m")
CACHE_DIR = Path("data/cache_15m")
CACHE_DIR.mkdir(exist_ok=True)


def cache_one(sym):
    import sys; sys.path.insert(0, "src")
    import warnings; warnings.filterwarnings("ignore")
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from pathlib import Path
    cache = Path("data/cache_15m") / f"{sym}.parquet"
    if cache.exists() and cache.stat().st_size > 1000:
        return sym, 0, "cached"
    try:
        o1m = load_binance_1m(sym, [Path("data/binance-1m")])
        o15 = resample_ohlcv(o1m, "15min")
        o15.to_parquet(cache)
        return sym, len(o15), "new"
    except Exception as e:
        return sym, 0, f"err:{e}"


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    syms = [d.name for d in sorted(DATA_DIR.iterdir())
            if d.is_dir() and d.name.endswith("USDT") and len(list(d.glob("*.csv"))) >= 3]
    print(f"Caching {len(syms)} assets to parquet (10 workers)...")

    t0 = time.time()
    cached = new = failed = 0
    with ProcessPoolExecutor(max_workers=10) as pool:
        futs = {pool.submit(cache_one, s): s for s in syms}
        for f in as_completed(futs):
            sym, n, status = f.result()
            if status == "cached":
                cached += 1
            elif status.startswith("err"):
                failed += 1
            else:
                new += 1
                print(f"  {sym}: {n:,} bars", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s: {cached} cached, {new} new, {failed} failed")
