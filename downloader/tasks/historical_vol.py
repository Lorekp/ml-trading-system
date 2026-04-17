"""Download Deribit's historical realized volatility (daily)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from downloader.client import DeribitClient

logger = logging.getLogger(__name__)


def download_historical_vol(client: DeribitClient, currency: str,
                            output_dir: Path) -> Path:
    """Download full daily RV history for one currency. Single API call."""
    out_path = output_dir / "rv" / f"{currency}_rv_daily.parquet"
    if out_path.exists():
        logger.info("RV %s already exists, skipping", currency)
        return out_path

    result = client.request("get_historical_volatility", {"currency": currency})

    if not result:
        logger.warning("No RV data for %s", currency)
        return out_path

    # Result is list of [timestamp_ms, value] pairs
    df = pd.DataFrame(result, columns=["timestamp", "rv"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    logger.info("RV %s: %d days (%s to %s)", currency, len(df),
                df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return out_path
