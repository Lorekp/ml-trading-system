"""Download OHLCV candles via get_tradingview_chart_data."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from downloader.client import DeribitClient

logger = logging.getLogger(__name__)


def download_candles(client: DeribitClient, instrument: str, start: str,
                     end: str, resolution: str, output_dir: Path) -> Path:
    """Download candles for one instrument. Chunks by week (~2000 5-min candles/day)."""
    safe_name = instrument.replace("/", "_")
    out_path = output_dir / "candles" / f"{safe_name}_{resolution}m.parquet"
    if out_path.exists():
        logger.info("Candles %s already exists, skipping", instrument)
        return out_path

    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    all_rows = []
    chunk_start = s
    chunk_days = 3  # ~864 candles at 5-min per day, 3 days = ~2600 (under 5000 limit)

    while chunk_start < e:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), e)
        start_ms = int(chunk_start.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        result = client.request("get_tradingview_chart_data", {
            "instrument_name": instrument,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution,
        })

        if result and "ticks" in result:
            ticks = result["ticks"]
            opens = result.get("open", [])
            highs = result.get("high", [])
            lows = result.get("low", [])
            closes = result.get("close", [])
            volumes = result.get("volume", [])
            for j, t in enumerate(ticks):
                all_rows.append({
                    "timestamp": t,
                    "open": opens[j] if j < len(opens) else None,
                    "high": highs[j] if j < len(highs) else None,
                    "low": lows[j] if j < len(lows) else None,
                    "close": closes[j] if j < len(closes) else None,
                    "volume": volumes[j] if j < len(volumes) else None,
                })

        chunk_start = chunk_end

    if not all_rows:
        logger.warning("No candle data for %s", instrument)
        return out_path

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    logger.info("Candles %s: %d rows (%s to %s)", instrument, len(df),
                df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return out_path
