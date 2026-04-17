"""Historical perp trade downloader — Binance Vision source.

Pulls aggressor-tagged BTCUSDT perp trades from Binance's public data
CDN (``data.binance.vision``) into per-day parquets on local disk. This
is the data source for the W5 true-VPIN pipeline: the proxy VPIN in
``vol_arb/drl_hedging/vpin.py`` is replaced with a bucketed VPIN
computed from these trades (see ``scripts/recompute_vpin.py``).

Why Binance Vision (not Bybit):
- Binance's data CDN has higher bandwidth than Bybit's public dumps.
- Binance Vision archives are well-organised by UTC day, with a fixed
  CSV schema: ``id, price, qty, quote_qty, time, is_buyer_maker``.
- ``is_buyer_maker=true`` → SELL aggressor (direction = −1).
  ``is_buyer_maker=false`` → BUY aggressor (direction = +1).
- 3 years of daily BTCUSDT perp trades ≈ 28 GB zipped, ~3M trades/day.

The downloader strips the CSVs to just the columns needed for VPIN
(``timestamp`` as UTC datetime, ``size`` in BTC, ``direction`` ±1),
writes them as per-day parquets, and resumes by skipping any day whose
parquet already exists. Failures are retried with exponential backoff.

Usage:
    PYTHONPATH=. python -m downloader.tasks.perp_trades \\
        --symbol BTCUSDT --start 2021-01-01 --end 2023-12-31

    # Short smoke test:
    PYTHONPATH=. python -m downloader.tasks.perp_trades \\
        --symbol BTCUSDT --start 2024-01-15 --end 2024-01-15
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Binance Vision public dumps — no auth required
BINANCE_VISION_BASE = (
    "https://data.binance.vision/data/futures/um/daily/trades/{symbol}"
    "/{symbol}-trades-{date}.zip"
)
DEFAULT_OUT_ROOT = Path("data/binance/perp_trades")


def _daily_url(symbol: str, date: pd.Timestamp) -> str:
    return BINANCE_VISION_BASE.format(
        symbol=symbol, date=date.strftime("%Y-%m-%d"),
    )


def _day_output_path(out_root: Path, symbol: str, date: pd.Timestamp) -> Path:
    """Parquet path: data/binance/perp_trades/<SYMBOL>/<YYYY>/<MM>/<SYMBOL>-trades-<YYYY-MM-DD>.parquet"""
    return (
        out_root
        / symbol
        / date.strftime("%Y")
        / date.strftime("%m")
        / f"{symbol}-trades-{date.strftime('%Y-%m-%d')}.parquet"
    )


def _fetch_with_retries(
    url: str, timeout_s: int, max_retries: int = 3,
) -> bytes:
    """GET the URL, retrying with exponential backoff on transient failures."""
    delay = 5.0
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_exc = exc
            code = getattr(exc, "code", None)
            # 404 is permanent — no retry, caller must handle
            if code == 404:
                raise
            logger.warning(
                "fetch attempt %d/%d failed (%s); retrying in %.1fs",
                attempt, max_retries, exc, delay,
            )
            time.sleep(delay)
            delay *= 2
    assert last_exc is not None
    raise last_exc


def _parse_zip_to_frame(raw_zip: bytes) -> pd.DataFrame:
    """Unzip the Binance trade dump and return a slim DataFrame.

    Columns returned:
        timestamp  — UTC datetime (nanosecond precision, from epoch ms)
        size       — float64, base asset qty (BTC)
        direction  — int8, +1 if aggressor was buyer, -1 if seller

    Input CSV columns (Binance Vision schema):
        id, price, qty, quote_qty, time, is_buyer_maker
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError("Binance Vision archive contained no CSV")
        with zf.open(names[0]) as f:
            df = pd.read_csv(
                f,
                usecols=["time", "qty", "is_buyer_maker"],
                dtype={"time": np.int64, "qty": np.float64},
            )

    # is_buyer_maker: True if buyer was maker → aggressor was the SELLER.
    # We want +1 when the aggressor is the buyer, -1 when the seller.
    # Both bool and string-True/False forms are possible across Binance's
    # archive generations; normalise defensively.
    ibm = df["is_buyer_maker"]
    if ibm.dtype == object:
        ibm = ibm.astype(str).str.strip().str.lower() == "true"
    else:
        ibm = ibm.astype(bool)
    direction = np.where(ibm, -1, 1).astype(np.int8)

    ts = pd.to_datetime(df["time"], unit="ms", utc=True)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "size": df["qty"].astype(np.float64),
            "direction": direction,
        },
    )
    # Binance dumps are already sorted, but a defensive sort is cheap
    # compared to the ~3M-row size and catches any future data quirks.
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def download_day(
    symbol: str,
    date: pd.Timestamp,
    out_root: Path = DEFAULT_OUT_ROOT,
    *,
    timeout_s: int = 120,
    overwrite: bool = False,
) -> Path | None:
    """Download one day of Binance perp trades and write as parquet.

    Parameters
    ----------
    symbol : str
        E.g. ``"BTCUSDT"``.
    date : pd.Timestamp
        UTC date to fetch. Time component is ignored.
    out_root : Path
        Root directory under which per-day parquets are written.
    timeout_s : int
        Per-request timeout.
    overwrite : bool
        If False (default), existing per-day parquets are skipped.

    Returns
    -------
    Path | None
        The path of the written (or already-existing) parquet, or None if
        the date was missing from Binance Vision (not yet available, or
        before the contract's listing).
    """
    date = pd.Timestamp(date).normalize()
    out_path = _day_output_path(out_root, symbol, date)
    if out_path.exists() and not overwrite:
        logger.debug("%s: cached at %s", date.date(), out_path)
        return out_path

    url = _daily_url(symbol, date)
    logger.info("%s: fetching %s", date.date(), url)
    try:
        raw = _fetch_with_retries(url, timeout_s=timeout_s)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.warning(
                "%s: 404 on Binance Vision — likely outside contract history",
                date.date(),
            )
            return None
        raise

    t0 = time.time()
    df = _parse_zip_to_frame(raw)
    parse_s = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy")

    mb_in = len(raw) / 1e6
    mb_out = out_path.stat().st_size / 1e6
    logger.info(
        "%s: %d rows, %.1f MB zip → %.1f MB parquet (parse %.1fs)",
        date.date(), len(df), mb_in, mb_out, parse_s,
    )
    return out_path


def download_range(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_root: Path = DEFAULT_OUT_ROOT,
    *,
    timeout_s: int = 120,
    overwrite: bool = False,
) -> dict:
    """Download every day in [start, end] (inclusive) for ``symbol``.

    Returns a dict summary: ``{"ok": N, "cached": N, "missing": N}``.
    """
    dates = pd.date_range(
        pd.Timestamp(start).normalize(),
        pd.Timestamp(end).normalize(),
        freq="D",
    )
    stats = {"ok": 0, "cached": 0, "missing": 0}
    for i, d in enumerate(dates, start=1):
        out_path = _day_output_path(out_root, symbol, d)
        if out_path.exists() and not overwrite:
            stats["cached"] += 1
            continue
        logger.info("[%d/%d] %s", i, len(dates), d.date())
        result = download_day(
            symbol, d, out_root,
            timeout_s=timeout_s, overwrite=overwrite,
        )
        if result is None:
            stats["missing"] += 1
        else:
            stats["ok"] += 1
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Download Binance Vision perp trades (daily ZIPs)",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out-root", default=str(DEFAULT_OUT_ROOT),
        help=f"Output root dir (default {DEFAULT_OUT_ROOT})",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=120,
        help="Per-request HTTP timeout",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if cached parquet exists",
    )
    args = parser.parse_args()

    stats = download_range(
        symbol=args.symbol,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        out_root=Path(args.out_root),
        timeout_s=args.timeout_s,
        overwrite=args.overwrite,
    )
    print(
        f"Done. ok={stats['ok']}  cached={stats['cached']}  "
        f"missing={stats['missing']}"
    )


if __name__ == "__main__":
    main()
