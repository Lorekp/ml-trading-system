"""Extract all configured Hyperliquid assets from each LZ4 file in one pass."""
import lz4.frame, csv, io, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
OUT_DIR = Path("v3-results")
COINS = {"BTC", "ETH", "XRP", "BNB", "SUI", "LINK"}
COLS = ["time", "funding", "open_interest", "mid_px", "impact_bid_px", "impact_ask_px"]


def extract_all_coins(fp):
    """Read one LZ4 file, extract ALL target coins at once."""
    result = {c: [] for c in COINS}
    try:
        raw = lz4.frame.decompress(open(fp, "rb").read()).decode("utf-8")
        for r in csv.DictReader(io.StringIO(raw)):
            coin = r.get("coin", "")
            if coin in COINS:
                result[coin].append({c: r.get(c, "") for c in COLS})
    except Exception:
        pass
    return result


if __name__ == "__main__":
    files = sorted(HL_DIR.glob("*.lz4"))
    print(f"{len(files)} LZ4 files, extracting {len(COINS)} coins in parallel", flush=True)

    # One pass: 16 threads, each file extracts all 6 coins
    all_data = {c: [] for c in COINS}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(extract_all_coins, fp) for fp in files]
        for i, f in enumerate(futures):
            file_result = f.result()
            for coin in COINS:
                all_data[coin].extend(file_result[coin])
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)} files processed", flush=True)

    print(f"  All files processed in {time.time()-t0:.1f}s", flush=True)

    # Save per coin
    for coin in sorted(COINS):
        df = pd.DataFrame(all_data[coin])
        out = OUT_DIR / f"hl_{coin}.csv"
        df.to_csv(out, index=False)
        print(f"  {coin}: {len(df)} rows -> {out}", flush=True)

    print(f"Done in {time.time()-t0:.1f}s total", flush=True)
