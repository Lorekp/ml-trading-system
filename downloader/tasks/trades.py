"""Download option/future trades with IV (paginated, per-day files)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from downloader.client import DeribitClient

logger = logging.getLogger(__name__)


def _date_to_ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)


def download_day(client: DeribitClient, currency: str, kind: str,
                 date: datetime, output_dir: Path,
                 max_pages: int = 0) -> Path | None:
    """Download trades for one currency/kind/day. Resume-safe.

    Args:
        max_pages: Cap pagination (0 = unlimited). For spread estimation,
                   20 pages (20K trades) per sampled day is sufficient.
    """
    date_str = date.strftime("%Y-%m-%d")
    out_path = output_dir / "trades" / f"{currency}_{kind}" / f"{date_str}.parquet"
    if out_path.exists():
        return out_path

    start_ms = _date_to_ms(date)
    end_ms = _date_to_ms(date + timedelta(days=1))
    all_trades = []
    cursor_ts = start_ms
    pages = 0

    logger.info("  %s %s %s: starting%s...", currency, kind, date_str,
                f" (max {max_pages} pages)" if max_pages else "")

    while cursor_ts < end_ms:
        params = {
            "currency": currency, "kind": kind,
            "start_timestamp": cursor_ts, "end_timestamp": end_ms,
            "count": 1000, "sorting": "asc",
        }

        result = client.request("get_last_trades_by_currency_and_time",
                                params, use_history=True)
        trades = result.get("trades", [])
        if not trades:
            break

        all_trades.extend(trades)
        pages += 1

        if pages % 50 == 0:
            logger.info("  %s %s %s: page %d (%d trades so far)",
                        currency, kind, date_str, pages, len(all_trades))

        if not result.get("has_more", False):
            break
        if max_pages and pages >= max_pages:
            logger.info("  %s %s %s: capped at %d pages (%d trades)",
                        currency, kind, date_str, pages, len(all_trades))
            break

        # Advance cursor past the last trade's timestamp
        cursor_ts = trades[-1]["timestamp"] + 1

    if not all_trades:
        logger.info("  %s %s %s: no trades", currency, kind, date_str)
        return None

    df = pd.DataFrame(all_trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write (Windows: replace instead of rename)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    logger.info("  %s %s %s: %d trades (%d pages, %.1f MB)",
                currency, kind, date_str, len(all_trades), pages,
                out_path.stat().st_size / 1e6)
    return out_path


def build_work_queue(currencies: list[str], kinds: list[str],
                     start: str, end: str,
                     sample_every_n_days: int = 1) -> list[tuple]:
    """Build (currency, kind, date) tuples for parallel execution.

    Args:
        sample_every_n_days: Download 1 day per N days (7 = weekly sampling).
    """
    s = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    work = []
    d = s
    day_idx = 0
    while d < e:
        if day_idx % sample_every_n_days == 0:
            for c in currencies:
                for k in kinds:
                    work.append((c, k, d))
        d += timedelta(days=1)
        day_idx += 1
    return work
