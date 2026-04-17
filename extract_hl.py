"""Extract per-asset HL data from LZ4 files using threads."""
import lz4.frame, csv, io, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
OUT_DIR = Path("v3-results")
COINS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
COLS = ["time", "funding", "open_interest", "mid_px", "impact_bid_px", "impact_ask_px"]


def extract_file(fp, coin):
    try:
        raw = lz4.frame.decompress(open(fp, "rb").read()).decode("utf-8")
        return [
            {c: r.get(c, "") for c in COLS}
            for r in csv.DictReader(io.StringIO(raw))
            if r.get("coin") == coin
        ]
    except Exception:
        return []


if __name__ == "__main__":
    files = sorted(HL_DIR.glob("*.lz4"))
    print(f"{len(files)} LZ4 files")

    for coin in COINS:
        t0 = time.time()
        all_rows = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(extract_file, fp, coin) for fp in files]
            for f in futures:
                all_rows.extend(f.result())
        df = pd.DataFrame(all_rows)
        out = OUT_DIR / f"hl_{coin}.csv"
        df.to_csv(out, index=False)
        print(f"{coin}: {len(df)} rows in {time.time()-t0:.1f}s")

    print("Done!")
