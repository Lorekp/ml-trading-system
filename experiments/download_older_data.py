"""Download older data (2022-01 to 2024-02) for 30 production assets."""
import os, time, zipfile, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

DATA_DIR = Path("C:\\Claude\\ml-trading-system\\data\\binance-1m")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ZECUSDT", "BNXUSDT",
    "BNBUSDT", "ENJUSDT", "TAOUSDT", "ADAUSDT", "ANKRUSDT", "SUIUSDT",
    "AVAXUSDT", "LINKUSDT", "NEARUSDT", "FILUSDT", "DOTUSDT", "SXPUSDT",
    "ENAUSDT", "BCHUSDT", "LTCUSDT", "UNIUSDT", "WLDUSDT", "TRXUSDT",
    "FTMUSDT", "AAVEUSDT", "FETUSDT", "APTUSDT", "HBARUSDT", "XLMUSDT",
]

# Download 2022-01 to 2024-02 (26 months of older data)
MONTHS = []
for year in range(2020, 2025):
    for month in range(1, 13):
        if year == 2024 and month > 2:
            break
        MONTHS.append((year, month))


def download_month(symbol, year, month):
    month_str = f"{year}-{month:02d}"
    symbol_dir = DATA_DIR / symbol
    symbol_dir.mkdir(exist_ok=True)
    csv_path = symbol_dir / f"{symbol}-1m-{month_str}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 1000:
        return "exists"

    filename = f"{symbol}-1m-{month_str}.zip"
    url = f"{BASE_URL}/{symbol}/1m/{filename}"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return "not_found"
        if resp.status_code != 200:
            return f"error_{resp.status_code}"

        zip_path = symbol_dir / filename
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(symbol_dir)
        zip_path.unlink()
        return "ok"
    except Exception as e:
        return f"error: {e}"


def download_symbol(symbol):
    results = []
    for year, month in MONTHS:
        r = download_month(symbol, year, month)
        results.append((f"{year}-{month:02d}", r))
        time.sleep(0.05)
    ok = sum(1 for _, r in results if r in ("ok", "exists"))
    return symbol, ok, len(MONTHS)


def main():
    print(f"Downloading older data (2022-01 to 2024-02) for {len(ASSETS)} assets")
    print(f"{len(MONTHS)} months per asset, 10 threads")
    print(flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(download_symbol, sym): sym for sym in ASSETS}
        for future in as_completed(futures):
            sym, ok, total = future.result()
            completed += 1
            print(f"  [{completed:>2d}/{len(ASSETS)}] {sym}: {ok}/{total} months", flush=True)

    # Verify
    print("\nVerification:")
    for sym in sorted(ASSETS):
        d = DATA_DIR / sym
        csvs = sorted(d.glob(f"{sym}-1m-*.csv"))
        if csvs:
            first = csvs[0].name.split("-1m-")[1].replace(".csv", "")
            last = csvs[-1].name.split("-1m-")[1].replace(".csv", "")
            print(f"  {sym}: {len(csvs)} months ({first} to {last})")


if __name__ == "__main__":
    main()
