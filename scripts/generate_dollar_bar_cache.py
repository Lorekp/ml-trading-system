"""Generate the dollar-bar warm-start sidecar parquet.

F3 Tier 4 (parity-cycle round 3 — 2026-05-07): produces a tiny parquet
of the last N dollar bars matching the offline schema so the live
engine's :class:`DollarBarBuffer` can pre-warm at startup. Without
this, the engine waits ~37 minutes (250+ dollar bars) before
`parkinson_1d` and EMA50 features can fire.

Two source paths:

1. **Fast path** — read `vol_model/output/BTCUSDT_bar_series.parquet`
   and take the last N rows. This parquet is produced by
   `vol_model/features/engineer.py::_build_bars_sequential` and
   already cached.
2. **Reconstruct path** — when the cached bar-series isn't present,
   feed the most recent month's aggTrades through
   :class:`DollarBarAggregator` (mirroring Tier 1's byte-identity
   test pattern).

Output schema matches :meth:`DollarBarBuffer.as_dataframe()` exactly:

    columns: [close, open, high, low, duration_s, n_trades,
              dollar_volume, buy_volume, sell_volume, vov, imbalance,
              log_return, rv_bar]
    index:   UTC DatetimeIndex named ``timestamp``
    dtypes:  float64 for everything except n_trades (int64)

Usage::

    python scripts/generate_dollar_bar_cache.py --asset BTCUSDT \\
        --threshold 500000 --n 300 \\
        --output vol_arb/output/BTC_dollar_bar_cache.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup so the script can be run from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from deribit_paper.features.dollar_bar import DollarBarAggregator  # noqa: E402

logger = logging.getLogger(__name__)


_DEFAULT_BAR_SERIES_PATH = "vol_model/output/{asset}_bar_series.parquet"
_DEFAULT_AGGTRADES_DIR = "data/binance-aggTrades-parquet"


_BUFFER_COLUMNS: tuple[str, ...] = (
    "close", "open", "high", "low",
    "duration_s", "n_trades",
    "dollar_volume", "buy_volume", "sell_volume",
    "vov", "imbalance", "log_return", "rv_bar",
)


def _normalize_to_buffer_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast and rename a bar-series DataFrame into the buffer schema."""
    out = df[list(_BUFFER_COLUMNS)].copy()
    # buffer expects n_trades as int64 (DollarBar.n_trades is int)
    out["n_trades"] = out["n_trades"].astype(np.int64)
    # Float columns explicitly
    for col in (
        "close", "open", "high", "low", "duration_s",
        "dollar_volume", "buy_volume", "sell_volume",
        "vov", "imbalance", "log_return", "rv_bar",
    ):
        out[col] = out[col].astype(np.float64)
    out.index = pd.DatetimeIndex(out.index, tz="UTC", name="timestamp")
    return out


def _load_from_cached_bar_series(path: Path, n: int) -> pd.DataFrame | None:
    """Fast path: read the last N rows from the offline bar-series parquet."""
    if not path.exists():
        return None
    logger.info("Loading bar-series from %s ...", path)
    # Avoid loading the entire 3.7GB file: use pyarrow tail.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    n_total = pf.metadata.num_rows
    logger.info("  parquet has %d total rows; taking the last %d", n_total, n)
    # Read just the last few row-groups (each ~1M rows usually).
    target_rows = max(n * 2, 5000)  # safety margin for groupby/dedup
    cumulative = 0
    rg_indices: list[int] = []
    for i in range(pf.num_row_groups - 1, -1, -1):
        cumulative += pf.metadata.row_group(i).num_rows
        rg_indices.insert(0, i)
        if cumulative >= target_rows:
            break
    table = pf.read_row_groups(rg_indices)
    df = table.to_pandas()
    df = df.tail(n)
    return _normalize_to_buffer_schema(df)


def _reconstruct_from_aggtrades(
    asset: str,
    threshold: float,
    aggtrades_dir: Path,
    n: int,
) -> pd.DataFrame | None:
    """Fallback: replay the most recent aggTrades month through the live aggregator.

    Captures every emitted bar; returns the last ``n`` as the warmup
    cache. Sufficient to seed EMA50 + parkinson_1d at engine startup.
    """
    asset_dir = aggtrades_dir / asset
    if not asset_dir.exists():
        return None
    months = sorted(p.stem for p in asset_dir.glob("*.parquet"))
    if not months:
        return None
    most_recent = months[-1]
    src = asset_dir / f"{most_recent}.parquet"
    logger.info(
        "Reconstructing dollar bars from %s via DollarBarAggregator (threshold=%.0f) ...",
        src, threshold,
    )
    raw = pd.read_parquet(src)
    raw.index = pd.to_datetime(raw["transact_time"], unit="ms", utc=True)
    raw = raw.sort_index()

    bars: list[dict] = []
    agg = DollarBarAggregator(
        threshold=threshold,
        on_bar_close=lambda bar: bars.append({
            "timestamp": bar.timestamp,
            "close": bar.close,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "duration_s": bar.duration_s,
            "n_trades": bar.n_trades,
            "dollar_volume": bar.dollar_volume,
            "buy_volume": bar.buy_volume,
            "sell_volume": bar.sell_volume,
            "vov": bar.vov,
            "imbalance": bar.imbalance,
            "log_return": bar.log_return,
            "rv_bar": bar.rv_bar,
        }),
    )
    for ts, row in raw.iterrows():
        agg.add_trade(
            timestamp=ts,
            price=float(row["price"]),
            dollar_volume=float(row["dollar_volume"]),
            is_buyer_maker=bool(row["is_buyer_maker"]),
        )
    if not bars:
        return None
    df = pd.DataFrame(bars).set_index("timestamp")
    df = df.tail(n)
    return _normalize_to_buffer_schema(df)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--threshold", type=float, default=500_000.0)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument(
        "--output",
        default="vol_arb/output/BTC_dollar_bar_cache.parquet",
    )
    parser.add_argument(
        "--bar-series-path",
        default=None,
        help="Override the default cached bar-series path.",
    )
    parser.add_argument(
        "--aggtrades-dir",
        default=_DEFAULT_AGGTRADES_DIR,
    )
    args = parser.parse_args()

    bar_series_path = Path(
        args.bar_series_path
        or _DEFAULT_BAR_SERIES_PATH.format(asset=args.asset)
    )
    aggtrades_dir = Path(args.aggtrades_dir)

    # Fast path
    df = _load_from_cached_bar_series(bar_series_path, args.n)
    # Fallback: reconstruct from aggTrades month
    if df is None or len(df) == 0:
        logger.info(
            "Cached bar-series unavailable (path=%s exists=%s); "
            "falling back to aggTrades reconstruction.",
            bar_series_path, bar_series_path.exists(),
        )
        df = _reconstruct_from_aggtrades(
            asset=args.asset,
            threshold=args.threshold,
            aggtrades_dir=aggtrades_dir,
            n=args.n,
        )
    if df is None or len(df) == 0:
        logger.error(
            "No source available — cached bar-series missing AND "
            "aggTrades-reconstruction returned empty. Aborting.",
        )
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info(
        "Wrote %d bars to %s (window: %s → %s)",
        len(df), out_path, df.index[0], df.index[-1],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
