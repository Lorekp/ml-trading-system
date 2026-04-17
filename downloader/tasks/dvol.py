"""Download DVOL (implied volatility index) at configurable resolution."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from downloader.client import DeribitClient

logger = logging.getLogger(__name__)


def download_dvol(client: DeribitClient, currency: str, start: str, end: str,
                  resolution: str, output_dir: Path) -> Path:
    """Download DVOL for one currency. Chunks by day to stay under 5000-candle limit."""
    res_label = f"{int(resolution)//60}m" if resolution.isdigit() else resolution
    out_path = output_dir / "dvol" / f"{currency}_dvol_{res_label}.parquet"
    if out_path.exists():
        logger.info("DVOL %s already exists, skipping", currency)
        return out_path

    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    all_rows = []
    day = s
    while day < e:
        next_day = day + timedelta(days=1)
        start_ms = int(day.timestamp() * 1000)
        end_ms = int(next_day.timestamp() * 1000)

        result = client.request("get_volatility_index_data", {
            "currency": currency,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution,
        })

        if result and "data" in result:
            all_rows.extend(result["data"])

        day = next_day

    if not all_rows:
        logger.warning("No DVOL data for %s", currency)
        return out_path

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    logger.info("DVOL %s: %d rows (%s to %s)", currency, len(df),
                df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return out_path
