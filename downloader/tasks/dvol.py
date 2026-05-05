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
    """Download DVOL for one currency. Chunks by day to stay under 5000-candle limit.

    Append-friendly: if ``out_path`` already exists, the requested
    ``start`` is silently advanced to one resolution-step after the
    existing parquet's last timestamp, so monthly refreshes only
    download the new tail. Concat + dedupe handles overlap. The file
    is rewritten atomically via ``tmp + replace``. If the requested
    range is fully covered already, the original file is returned
    unmodified.
    """
    res_label = f"{int(resolution)//60}m" if resolution.isdigit() else resolution
    out_path = output_dir / "dvol" / f"{currency}_dvol_{res_label}.parquet"

    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    existing_df: pd.DataFrame | None = None
    if out_path.exists():
        try:
            existing_df = pd.read_parquet(out_path)
        except Exception as exc:  # noqa: BLE001 - corrupt parquet -> redownload
            logger.warning(
                "Existing DVOL %s unreadable (%s) -- redownloading from scratch",
                out_path, exc,
            )
            existing_df = None
        else:
            if "timestamp" in existing_df.columns and len(existing_df):
                last_ts = pd.to_datetime(existing_df["timestamp"]).max()
                if last_ts.tz is None:
                    last_ts = last_ts.tz_localize("UTC")
                # Download starts strictly AFTER last_ts; the dedupe
                # below covers any overlap from boundary rounding.
                advance_to = (last_ts + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0,
                )
                if advance_to >= e:
                    logger.info(
                        "DVOL %s already covers %s through %s (>= requested end %s) - no work",
                        currency, existing_df["timestamp"].min(), last_ts, end,
                    )
                    return out_path
                if advance_to > s:
                    logger.info(
                        "DVOL %s incremental: existing covers through %s; "
                        "downloading %s to %s",
                        currency, last_ts, advance_to.strftime("%Y-%m-%d"), end,
                    )
                    s = advance_to

    all_rows: list[dict] = []
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

    if not all_rows and existing_df is None:
        logger.warning("No DVOL data for %s", currency)
        return out_path
    if not all_rows:
        logger.info(
            "DVOL %s: no new data in [%s, %s) - existing file unchanged",
            currency, start, end,
        )
        return out_path

    new_df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close"],
    )
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)

    if existing_df is not None and len(existing_df):
        # Coerce existing timestamps to a tz-aware DatetimeIndex matching
        # new_df, then concat + dedupe + sort.
        existing_df["timestamp"] = pd.to_datetime(
            existing_df["timestamp"], utc=True,
        )
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    df = df.drop_duplicates(subset=["timestamp"]).sort_values(
        "timestamp",
    ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    logger.info("DVOL %s: %d rows (%s to %s)", currency, len(df),
                df["timestamp"].iloc[0], df["timestamp"].iloc[-1])
    return out_path
