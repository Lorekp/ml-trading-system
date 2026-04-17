#!/usr/bin/env python
"""Download historical open interest from Binance Futures API.

Endpoint: /futures/data/openInterestHist
- Periods: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
- Max 500 records per request
- Public endpoint (no API key needed)

Usage:
    python scripts/download_oi.py
    python scripts/download_oi.py --symbol ETHUSDT --period 1h
"""

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://fapi.binance.com/futures/data/openInterestHist"
PERIOD_MINUTES = {
    "5m": 5, "15m": 15, "30m": 30, "1h": 60,
    "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}


def download_oi(
    symbol: str = "BTCUSDT",
    period: str = "15m",
    start_date: str = "2023-05-01",
    end_date: str = "2024-04-01",
    output_dir: str = "data/oi",
) -> pd.DataFrame | None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    period_min = PERIOD_MINUTES[period]
    max_records = 500
    batch_td = timedelta(minutes=period_min * max_records)

    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    all_records = []
    cursor_ms = start_ms
    request_count = 0

    logger.info("Downloading %s OI (%s) from %s to %s", symbol, period, start_date, end_date)

    while cursor_ms < end_ms:
        batch_end_ms = min(cursor_ms + int(batch_td.total_seconds() * 1000), end_ms)

        params = {
            "symbol": symbol,
            "period": period,
            "startTime": cursor_ms,
            "endTime": batch_end_ms,
            "limit": max_records,
        }

        for attempt in range(5):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Rate limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == 4:
                    logger.error("Failed after 5 attempts at %s: %s",
                                 datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc), e)
                    # Save what we have and report
                    if all_records:
                        break
                    return None
                time.sleep(2 ** attempt)
                continue
        else:
            break

        if not data:
            logger.info("No data returned at %s — may have reached limit of available history",
                        datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc).date())
            break

        all_records.extend(data)
        request_count += 1

        last_ts = data[-1]["timestamp"]
        cursor_ms = last_ts + 1  # Move past last record

        if request_count % 20 == 0:
            dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            logger.info("  %d requests, %d records, up to %s", request_count, len(all_records), dt.date())

        # Rate limit: ~5 req/s max, be conservative
        time.sleep(0.3)

    if not all_records:
        logger.warning("No OI data returned for %s. Endpoint may not have historical data.", symbol)
        return None

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Convert numeric columns
    for col in ["sumOpenInterest", "sumOpenInterestValue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]

    out_file = out_path / f"{symbol}_oi_{period}.parquet"
    df.to_parquet(out_file)

    logger.info("Saved %d records to %s (range: %s to %s)",
                len(df), out_file, df.index[0].date(), df.index[-1].date())
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures OI")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--period", default="15m", choices=list(PERIOD_MINUTES.keys()))
    parser.add_argument("--start", default="2023-05-01")
    parser.add_argument("--end", default="2024-04-01")
    parser.add_argument("--output-dir", default="data/oi")
    args = parser.parse_args()

    download_oi(args.symbol, args.period, args.start, args.end, args.output_dir)


if __name__ == "__main__":
    main()
