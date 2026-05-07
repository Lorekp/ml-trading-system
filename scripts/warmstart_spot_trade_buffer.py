"""Generate the spot-trade warm-start sidecar for the live TSRV path.

Round-4 F6/F8 (parity-cycle 2026-05-07): the live engine's
``spot_trade_buffer`` (a :class:`TradeBuffer`) is fed by the
:class:`DollarBarSpotStream` fan-out hook so :func:`compute_bar_tsrv`
reads from the same Binance Spot feed the offline pipeline trains
on. To avoid a cold first bar after engine restart, this script
extracts the last ~20 minutes of Binance Spot aggTrades into a tiny
parquet that the engine prewarms at startup.

Two source paths (mirrors ``scripts/generate_dollar_bar_cache.py``):

1. **Fast path** — read the most recent monthly parquet under
   ``data/binance-aggTrades-parquet/<asset>/<YYYY-MM>.parquet`` and
   tail-select the last ``--minutes`` of trades.
2. **REST path** — when the parquet is not present (fresh deploy),
   paginate ``BinanceRESTClient.for_spot()`` backwards from the
   latest aggTradeId until ~``--minutes`` of trades are accumulated.
   Same client / endpoint the live stream uses.

Output schema (consumed by :meth:`TradeBuffer.prewarm_from_parquet`):

    columns: [timestamp (datetime64[ns, UTC]), price (float64),
              amount (float64), is_buyer_maker (bool), a (int64)]

``is_buyer_maker`` lets the prewarm derive ``direction`` deterministically
(matching ``_process_page``'s ``is_buyer_maker → direction = -1``
convention). ``a`` is the aggTradeId — used as a stable trade_id for
dedup so the live stream can pick up *after* the warm-start window
without double-counting.

Usage::

    python scripts/warmstart_spot_trade_buffer.py --asset BTCUSDT \\
        --minutes 20 \\
        --output vol_arb/output/spot_trade_buffer_warmstart.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup so the script can be run from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))


_DEFAULT_AGGTRADES_DIR = "data/binance-aggTrades-parquet"

_BUFFER_COLUMNS: tuple[str, ...] = (
    "timestamp", "price", "amount", "is_buyer_maker", "a",
)


logger = logging.getLogger(__name__)


def _normalize_to_buffer_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast and project a Binance Vision aggTrades DataFrame into the
    buffer warm-start schema. Tolerates the column names used by the
    `data/binance-aggTrades-parquet/` archive (``transact_time``,
    ``quantity``, ``is_buyer_maker``, ``aggregate_trade_id``)."""
    if df.empty:
        return pd.DataFrame(columns=list(_BUFFER_COLUMNS))

    # Resolve timestamp column
    if "timestamp" in df.columns:
        ts = df["timestamp"]
    elif "transact_time" in df.columns:
        ts = df["transact_time"]
    elif "T" in df.columns:
        ts = df["T"]
    else:
        raise ValueError(
            "aggTrades parquet missing timestamp column; "
            "expected one of: timestamp, transact_time, T"
        )

    if pd.api.types.is_integer_dtype(ts):
        ts = pd.to_datetime(ts, unit="ms", utc=True)
    elif pd.api.types.is_datetime64_any_dtype(ts):
        if getattr(ts.dtype, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC")
    else:
        ts = pd.to_datetime(ts, utc=True)

    # Resolve price / quantity columns
    price_col = "price" if "price" in df.columns else "p"
    qty_col = (
        "amount" if "amount" in df.columns
        else ("quantity" if "quantity" in df.columns else "q")
    )
    if price_col not in df.columns or qty_col not in df.columns:
        raise ValueError(
            f"aggTrades parquet missing price/qty cols: "
            f"got {list(df.columns)[:8]}…"
        )

    # is_buyer_maker — direction sign comes from this in the prewarm path.
    if "is_buyer_maker" in df.columns:
        is_buyer_maker = df["is_buyer_maker"].astype(bool)
    elif "m" in df.columns:
        is_buyer_maker = df["m"].astype(bool)
    else:
        # Default: assume taker buys (direction +1). Logged as warning.
        logger.warning(
            "aggTrades parquet has no is_buyer_maker column — "
            "defaulting all rows to buyer (taker = buyer = direction +1)."
        )
        is_buyer_maker = pd.Series(False, index=df.index)

    # aggregate_trade_id — used as stable id for dedup.
    if "agg_trade_id" in df.columns:
        agg_id = df["agg_trade_id"].astype(np.int64)
    elif "aggregate_trade_id" in df.columns:
        agg_id = df["aggregate_trade_id"].astype(np.int64)
    elif "a" in df.columns:
        agg_id = df["a"].astype(np.int64)
    else:
        # Synthesize via row index — unique within this warm-start file.
        agg_id = pd.Series(np.arange(len(df), dtype=np.int64), index=df.index)

    out = pd.DataFrame({
        "timestamp": ts.to_numpy(),
        "price": df[price_col].astype(np.float64).to_numpy(),
        "amount": df[qty_col].astype(np.float64).to_numpy(),
        "is_buyer_maker": is_buyer_maker.to_numpy(),
        "a": agg_id.to_numpy(),
    })
    return out.sort_values("timestamp").reset_index(drop=True)


def fast_path(
    *,
    asset: str,
    minutes: float,
    aggtrades_dir: Path,
    now: pd.Timestamp,
) -> pd.DataFrame:
    """Tail-select the last ``minutes`` of trades from cached parquet."""
    asset_dir = aggtrades_dir / asset
    if not asset_dir.exists():
        raise FileNotFoundError(asset_dir)
    months = sorted(asset_dir.glob("*.parquet"))
    if not months:
        raise FileNotFoundError(f"no monthly parquets under {asset_dir}")
    cutoff = now - pd.Timedelta(minutes=minutes)
    # Walk most-recent month backwards; stop once we've covered the window.
    frames: list[pd.DataFrame] = []
    for m in reversed(months):
        df = pd.read_parquet(m)
        norm = _normalize_to_buffer_schema(df)
        norm = norm[norm["timestamp"] >= cutoff]
        if not norm.empty:
            frames.append(norm)
        if frames and frames[0]["timestamp"].min() <= cutoff:
            break
    if not frames:
        raise ValueError(
            f"no trades >= {cutoff} found under {asset_dir} "
            f"(latest parquet may be stale)"
        )
    out = pd.concat(frames[::-1], ignore_index=True)
    out = out[out["timestamp"] >= cutoff].reset_index(drop=True)
    return out


async def rest_path(
    *,
    asset: str,
    minutes: float,
    now: pd.Timestamp,
) -> pd.DataFrame:
    """Paginate Binance Spot REST backwards until window is filled."""
    from deribit_paper.client.binance_rest import BinanceRESTClient

    cutoff = now - pd.Timedelta(minutes=minutes)
    client = BinanceRESTClient.for_spot()
    await client.connect()
    try:
        # Probe latest aggTradeId via a single page from the tail.
        latest = await client.fetch_agg_trades(asset, from_id=None)
        if not latest:
            raise RuntimeError(
                "Binance Spot returned no trades on initial probe"
            )
        max_id = max(int(t["a"]) for t in latest)
        # Walk backwards 1000 trades at a time.
        all_pages: list[list[dict]] = [latest]
        cursor = max_id - len(latest)
        while cursor > 0:
            page = await client.fetch_agg_trades(
                asset, from_id=max(cursor, 0),
            )
            if not page:
                break
            all_pages.append(page)
            page_min_ts = min(int(t["T"]) for t in page)
            page_min_dt = pd.Timestamp(page_min_ts, unit="ms", tz="UTC")
            if page_min_dt <= cutoff:
                break
            cursor -= len(page)
    finally:
        await client.close()

    rows = [t for page in all_pages for t in page]
    df = pd.DataFrame(rows)
    return _normalize_to_buffer_schema(df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the live engine's spot-trade warm-start parquet.",
    )
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--minutes", type=float, default=20.0,
        help="Window of trades to capture, in minutes. Default 20.",
    )
    parser.add_argument(
        "--aggtrades-dir", default=_DEFAULT_AGGTRADES_DIR,
        help="Path to the cached aggTrades parquets (fast path).",
    )
    parser.add_argument(
        "--output",
        default="vol_arb/output/spot_trade_buffer_warmstart.parquet",
    )
    parser.add_argument(
        "--rest", action="store_true",
        help="Force REST path even if cached parquet is available.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    now = pd.Timestamp.now(tz="UTC")
    aggtrades_dir = Path(args.aggtrades_dir)
    df: pd.DataFrame | None = None
    if not args.rest:
        try:
            df = fast_path(
                asset=args.asset,
                minutes=args.minutes,
                aggtrades_dir=aggtrades_dir,
                now=now,
            )
            logger.info(
                "fast path produced %d trades from %s",
                len(df), aggtrades_dir / args.asset,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(
                "fast path unavailable (%s) — falling back to REST", e,
            )

    if df is None:
        df = asyncio.run(
            rest_path(asset=args.asset, minutes=args.minutes, now=now)
        )
        logger.info("REST path produced %d trades", len(df))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(
        "wrote %d trades to %s (window=[%s..%s], size=%.1f KB)",
        len(df), out_path,
        df["timestamp"].min(), df["timestamp"].max(),
        out_path.stat().st_size / 1024.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
