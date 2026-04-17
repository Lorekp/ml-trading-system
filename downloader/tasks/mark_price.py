"""Download 5-min mark price history for DVOL constituent options.

Mark price can be BS-inverted to get per-strike IV, then all Greeks.
This is the key endpoint for vol-arb backtesting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from downloader.client import DeribitClient

logger = logging.getLogger(__name__)


def _date_to_ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)


def _get_expired_instruments(client: DeribitClient, currency: str) -> list[dict]:
    """Get all expired option instruments. Paginated by expiry month."""
    # get_instruments with expired=True returns all historical instruments
    result = client.request("get_instruments", {
        "currency": currency,
        "kind": "option",
        "expired": "true",
    })
    return result if isinstance(result, list) else []


def _get_active_instruments(client: DeribitClient, currency: str) -> list[dict]:
    """Get currently active option instruments."""
    result = client.request("get_instruments", {
        "currency": currency,
        "kind": "option",
        "expired": "false",
    })
    return result if isinstance(result, list) else []


def _download_instrument_marks(client: DeribitClient, instrument: str,
                               start_ms: int, end_ms: int) -> list:
    """Download mark price history for a single instrument."""
    result = client.request("get_mark_price_history", {
        "instrument_name": instrument,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
    })
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return []


def download_mark_prices(client: DeribitClient, currency: str,
                         start: str, end: str, output_dir: Path) -> int:
    """Download mark prices for all option instruments in date range.

    Returns number of instruments downloaded.
    """
    mark_dir = output_dir / "mark_price" / currency
    mark_dir.mkdir(parents=True, exist_ok=True)

    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = _date_to_ms(s)
    end_ms = _date_to_ms(e)

    # Get all instruments (expensive call: 10,000 credits)
    logger.info("Fetching %s option instruments (expired + active)...", currency)
    instruments = _get_expired_instruments(client, currency)
    instruments.extend(_get_active_instruments(client, currency))
    logger.info("  %d total instruments found", len(instruments))

    # Filter to instruments whose expiry falls within our date range
    # (or that were active during our range)
    filtered = []
    for inst in instruments:
        exp_ms = inst.get("expiration_timestamp", 0)
        created_ms = inst.get("creation_timestamp", 0)
        # Include if the instrument was alive during any part of [start, end]
        if exp_ms >= start_ms and created_ms <= end_ms:
            filtered.append(inst)

    logger.info("  %d instruments overlap with %s to %s", len(filtered), start, end)

    downloaded = 0
    skipped = 0
    empty = 0

    for i, inst in enumerate(filtered):
        name = inst["instrument_name"]
        safe_name = name.replace("/", "_")
        out_path = mark_dir / f"{safe_name}.parquet"

        if out_path.exists():
            skipped += 1
            continue

        try:
            rows = _download_instrument_marks(client, name, start_ms, end_ms)
        except Exception as ex:
            logger.debug("Failed %s: %s", name, ex)
            continue

        if not rows:
            empty += 1
            continue

        # Rows are [timestamp_ms, mark_price] pairs
        df = pd.DataFrame(rows, columns=["timestamp", "mark_price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["instrument_name"] = name

        tmp = out_path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out_path)
        downloaded += 1

        if (i + 1) % 100 == 0:
            logger.info("  Mark prices: %d/%d instruments (downloaded=%d, skipped=%d, empty=%d)",
                        i + 1, len(filtered), downloaded, skipped, empty)

    logger.info("Mark prices %s: %d downloaded, %d skipped, %d empty (of %d)",
                currency, downloaded, skipped, empty, len(filtered))
    return downloaded
