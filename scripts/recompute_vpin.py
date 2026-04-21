"""Recompute true VPIN from Binance aggTrades parquets (W5.2).

Reads the monthly BTCUSDT USD-M perp aggTrades parquets under
``data/binance-aggTrades-parquet/BTCUSDT/`` and computes the Easley–
Prado–O'Hara (2012) bucketed VPIN on an equal-volume bucketing scheme.
The output parquet ``vol_arb/output/vpin_btc_15m.parquet`` is aligned to
the vol_arb 15-min index and contains two columns for direct A/B:

    vpin_true  — computed from aggressor-tagged perp trades (this script)
    vpin_proxy — the returns-based proxy from
                 ``vol_arb/drl_hedging/vpin.compute_vpin_proxy_from_returns``

Streaming design
----------------
Month parquets are ~40M rows (~1 GB each). We stream month-by-month,
maintaining a small piece of state across boundaries so the rolling
VPIN window does not reset at month boundaries:

1. A *partial bucket* — the leftover cumulative volume that did not
   finish closing the last complete bucket of the previous month,
   plus the per-side (buy/sell) sums attributable to it.
2. The last ``rolling_window`` *complete* bucket imbalances — so the
   pandas rolling mean at the start of each month sees the same history
   it would see in a monolithic pass.

Per-month aggregation is vectorised via ``np.add.reduceat`` on the
within-month cumulative-volume → bucket-index map (no Python-level
iteration over the 40M rows).

Bucket sizing
-------------
Default ``bucket_volume_btc = 1000`` gives ~3 buckets per 15-min bar
on BTCUSDT perp (median daily volume ~320K BTC / 96 bars ≈ 3340 BTC
per 15m). Rolling window ``rolling_window = 50`` (the EPO canonical
value) means VPIN integrates over roughly 4 hours of flow. Both are
CLI-tunable.

Usage
-----
    # Full recompute on the canonical 2021-01..2026-03 window
    PYTHONPATH=. python scripts/recompute_vpin.py

    # Smoke test on a single month
    PYTHONPATH=. python scripts/recompute_vpin.py \\
        --start 2024-01 --end 2024-01 --output /tmp/vpin_smoke.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.drl_hedging.vpin import compute_vpin_proxy_from_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recompute_vpin")

# Defaults for BTC/15min. Overridden by --asset / --bar-minutes CLI args.
_DEFAULT_ASSET = "BTCUSDT"
_DEFAULT_BAR_MINUTES = 15


class VPINStreamer:
    """Streaming bucketed VPIN computation with rolling-window continuity.

    Maintains minimal state across monthly parquet boundaries so the
    resulting VPIN series is identical (up to floating-point rounding)
    to what a monolithic pass over all trades would produce.
    """

    def __init__(self, bucket_volume_btc: float, rolling_window: int):
        self.V = float(bucket_volume_btc)
        self.N = int(rolling_window)
        # State carried across months
        self._partial_vol = 0.0       # cumulative size since last bucket close
        self._partial_buy = 0.0       # buy-side component of partial bucket
        self._partial_sell = 0.0      # sell-side component of partial bucket
        self._recent_imbalances: list[float] = []  # last N closed-bucket imbalances
        # Output accumulators
        self._ts_parts: list[np.ndarray] = []   # end-of-bucket timestamps (ms)
        self._vpin_parts: list[np.ndarray] = []  # VPIN at each bucket close

    def process_month(self, df: pd.DataFrame) -> None:
        """Process one monthly chunk. ``df`` must have columns
        ``transact_time`` (int64 ms), ``quantity`` (float64), and
        ``is_buyer_maker`` (bool)."""
        if df.empty:
            return

        # Numpy view — avoid pandas overhead in the inner loop
        size = df["quantity"].to_numpy(dtype=np.float64, copy=False)
        ts_ms = df["transact_time"].to_numpy(dtype=np.int64, copy=False)
        # is_buyer_maker True → aggressor was seller (direction = −1)
        is_seller_aggressor = df["is_buyer_maker"].to_numpy(dtype=bool, copy=False)
        buy = np.where(is_seller_aggressor, 0.0, size)
        sell = np.where(is_seller_aggressor, size, 0.0)

        # Cumulative volume, offset by the carry-forward partial bucket
        cum = np.cumsum(size) + self._partial_vol
        bucket_idx = (cum / self.V).astype(np.int64)
        max_idx = int(bucket_idx[-1])

        if max_idx == 0:
            # Entire month falls inside the open partial bucket
            self._partial_vol = float(cum[-1])
            self._partial_buy += float(buy.sum())
            self._partial_sell += float(sell.sum())
            return

        # Find bucket-start positions via monotonic diff; reduceat sums between.
        new_bucket_mask = np.concatenate(([True], np.diff(bucket_idx) > 0))
        bucket_start = np.flatnonzero(new_bucket_mask)
        # Timestamp of each bucket's LAST trade is one position before the
        # next bucket's start (or the final row for the last bucket).
        bucket_end_positions = np.empty_like(bucket_start)
        bucket_end_positions[:-1] = bucket_start[1:] - 1
        bucket_end_positions[-1] = len(size) - 1
        bucket_end_ts = ts_ms[bucket_end_positions]

        # Per-bucket buy/sell via reduceat (C-speed, no Python loop)
        bucket_buy = np.add.reduceat(buy, bucket_start)
        bucket_sell = np.add.reduceat(sell, bucket_start)

        # The first bucket in this month merges with the carry-forward
        # partial bucket from the previous month (same bucket_idx value).
        bucket_buy[0] += self._partial_buy
        bucket_sell[0] += self._partial_sell

        # Decide whether the LAST bucket of this month is complete or
        # partial. A bucket indexed `k` is complete iff its cumulative
        # volume has reached `(k+1) * V`.
        last_bucket_closed_at = (bucket_idx[-1] + 1) * self.V
        if cum[-1] + 1e-12 < last_bucket_closed_at:
            # Last bucket is partial — carry it forward as state
            self._partial_vol = float(cum[-1] - bucket_idx[-1] * self.V)
            self._partial_buy = float(bucket_buy[-1])
            self._partial_sell = float(bucket_sell[-1])
            completed_buy = bucket_buy[:-1]
            completed_sell = bucket_sell[:-1]
            completed_end_ts = bucket_end_ts[:-1]
        else:
            # Last bucket exactly closed — nothing to carry
            self._partial_vol = 0.0
            self._partial_buy = 0.0
            self._partial_sell = 0.0
            completed_buy = bucket_buy
            completed_sell = bucket_sell
            completed_end_ts = bucket_end_ts

        if len(completed_buy) == 0:
            return

        imbalances = np.abs(completed_buy - completed_sell) / self.V

        # Rolling mean over the combined (historic + new) imbalance stream.
        # This is the only place we need the carry-forward history.
        combined = np.concatenate(
            [np.asarray(self._recent_imbalances, dtype=np.float64), imbalances],
        )
        rolling = (
            pd.Series(combined)
            .rolling(self.N, min_periods=1)
            .mean()
            .to_numpy()
        )
        # VPIN series for the NEW buckets only (slice off the historic prefix)
        n_historic = len(self._recent_imbalances)
        new_vpin = rolling[n_historic:]

        # Update the carry-forward history: last N imbalances of combined
        if len(combined) >= self.N:
            self._recent_imbalances = combined[-self.N:].tolist()
        else:
            self._recent_imbalances = combined.tolist()

        # Accumulate output
        self._ts_parts.append(completed_end_ts)
        self._vpin_parts.append(new_vpin.astype(np.float64))

    def finalize(self) -> pd.Series:
        if not self._ts_parts:
            return pd.Series(dtype=np.float64, name="vpin_true")
        ts_ms = np.concatenate(self._ts_parts)
        vpin = np.concatenate(self._vpin_parts)
        idx = pd.to_datetime(ts_ms, unit="ms", utc=True)
        s = pd.Series(vpin, index=idx, name="vpin_true")
        s = s.sort_index()
        # Clip to [0, 1] for numerical safety
        return s.clip(0.0, 1.0)

    def save_state(self, path: Path) -> None:
        """Serialize the carry-forward state to JSON for live warm-start.

        The state is tiny (~400 bytes): three floats for the partial
        bucket plus the last ``rolling_window`` imbalances.  JSON is
        human-readable and avoids pickle's arbitrary-code-execution risk.
        """
        import json

        state = {
            "version": 1,
            "bucket_volume_btc": self.V,
            "rolling_window": self.N,
            "partial_vol": self._partial_vol,
            "partial_buy": self._partial_buy,
            "partial_sell": self._partial_sell,
            "recent_imbalances": list(self._recent_imbalances),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        logger.info("Saved VPINStreamer state to %s (%d bytes)",
                     path, path.stat().st_size)

    @staticmethod
    def load_state(path: Path) -> dict:
        """Deserialize carry-forward state from JSON."""
        import json

        state = json.loads(Path(path).read_text())
        if state.get("version") != 1:
            raise ValueError(
                f"Unsupported VPINStreamer state version: {state.get('version')}. "
                f"Expected version 1."
            )
        return state


def _enumerate_months(
    start: pd.Timestamp, end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Enumerate the first day of every month in [start, end], inclusive."""
    return list(
        pd.date_range(
            pd.Timestamp(start).to_period("M").to_timestamp(),
            pd.Timestamp(end).to_period("M").to_timestamp(),
            freq="MS",
        ),
    )


def _month_parquet_path(root: Path, month: pd.Timestamp) -> Path:
    return root / f"{month.strftime('%Y-%m')}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute true VPIN from Binance aggTrades parquets",
    )
    parser.add_argument(
        "--start", default="2021-01",
        help="First month (YYYY-MM, inclusive). Default 2021-01.",
    )
    parser.add_argument(
        "--end", default="2026-03",
        help="Last month (YYYY-MM, inclusive). Default 2026-03.",
    )
    parser.add_argument(
        "--bucket-volume-btc", type=float, default=1000.0,
        help=(
            "Equal-volume bucket size in BTC. Default 1000 gives ~3 "
            "buckets per 15-min bar on BTCUSDT perp."
        ),
    )
    parser.add_argument(
        "--rolling-window", type=int, default=50,
        help="Number of buckets averaged for VPIN (EPO canonical: 50).",
    )
    parser.add_argument(
        "--asset", type=str, default=_DEFAULT_ASSET,
        help="Asset pair (e.g. BTCUSDT, ETHUSDT). Default BTCUSDT.",
    )
    parser.add_argument(
        "--bar-minutes", type=int, default=_DEFAULT_BAR_MINUTES,
        help="Bar timeframe in minutes. Default 15.",
    )
    parser.add_argument(
        "--vol-arb-path", type=str, default="",
        help="vol_arb parquet to align the output index against.",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Where to write the output parquet.",
    )
    args = parser.parse_args()

    # Resolve paths via vol_arb.paths so layout stays consistent
    from vol_arb.paths import vol_arb_path as _vol_arb_path, vpin_path as _vpin_path, CURRENCY_FOR_ASSET
    ccy = CURRENCY_FOR_ASSET[args.asset]
    # Horizon-aware vol_arb index source: reindex VPIN onto the BAR grid
    # matching the active horizon, not the default h=15 file. Missing h=5
    # vol_arb_svi parquet -> surface as FileNotFoundError (don't silently
    # fall back to h=15, which would mis-align the output).
    if not args.vol_arb_path:
        args.vol_arb_path = str(_vol_arb_path(
            ccy, svi=True, horizon_minutes=args.bar_minutes,
        ))
    if not args.output:
        args.output = str(_vpin_path(ccy, bar_minutes=args.bar_minutes))
    aggtrades_root = Path(f"data/binance-aggTrades-parquet/{args.asset}")

    months = _enumerate_months(
        pd.Timestamp(args.start + "-01"),
        pd.Timestamp(args.end + "-01"),
    )
    logger.info(
        "Processing %d months from %s to %s", len(months),
        months[0].strftime("%Y-%m"), months[-1].strftime("%Y-%m"),
    )
    logger.info(
        "VPIN params: bucket_volume=%g BTC, rolling_window=%d",
        args.bucket_volume_btc, args.rolling_window,
    )

    streamer = VPINStreamer(
        bucket_volume_btc=args.bucket_volume_btc,
        rolling_window=args.rolling_window,
    )

    t_start = time.time()
    total_rows = 0
    for i, month in enumerate(months, start=1):
        path = _month_parquet_path(aggtrades_root, month)
        if not path.exists():
            logger.warning("[%d/%d] %s: parquet missing, skipping",
                           i, len(months), month.strftime("%Y-%m"))
            continue
        t0 = time.time()
        df = pd.read_parquet(
            path, columns=["transact_time", "quantity", "is_buyer_maker"],
        )
        t_load = time.time() - t0
        t0 = time.time()
        streamer.process_month(df)
        t_proc = time.time() - t0
        total_rows += len(df)
        logger.info(
            "[%d/%d] %s: %d rows  load=%.1fs  process=%.1fs  "
            "(partial_vol=%.1f BTC)",
            i, len(months), month.strftime("%Y-%m"),
            len(df), t_load, t_proc, streamer._partial_vol,
        )
        del df  # release memory before next month

    vpin_events = streamer.finalize()
    total_elapsed = time.time() - t_start
    logger.info(
        "Streaming done: %d total trades processed, %d VPIN events, "
        "total wall %.1fs",
        total_rows, len(vpin_events), total_elapsed,
    )

    # Persist carry-forward state for live warm-start
    state_path = Path(args.output).with_name("vpin_streamer_state.json")
    streamer.save_state(state_path)

    # Resample to 15-min grid (end-of-bar convention matching vol_arb)
    vpin_15m = (
        vpin_events
        .resample(f"{args.bar_minutes}min", label="right", closed="right")
        .last()
    )

    # Align to vol_arb index via forward-fill (VPIN is a smoothed quantity;
    # carrying the last observation into bars with no new buckets is the
    # correct interpretation). Missing pre-warmup bars get the neutral 0.5
    # prior that matches the proxy path's fallback behaviour.
    logger.info("Loading vol_arb parquet for index alignment...")
    va = pd.read_parquet(args.vol_arb_path)
    vol_arb_idx = va.index
    vpin_true = vpin_15m.reindex(vol_arb_idx).ffill()

    # Count unresolved warmup bars before filling
    n_warmup = int(vpin_true.isna().sum())
    vpin_true = vpin_true.fillna(0.5)

    # Proxy for A/B
    logger.info("Computing returns-based proxy VPIN for A/B...")
    vpin_proxy = compute_vpin_proxy_from_returns(va)
    vpin_proxy = vpin_proxy.reindex(vol_arb_idx).fillna(0.5)

    out = pd.DataFrame(
        {
            "vpin_true": vpin_true.astype(np.float64),
            "vpin_proxy": vpin_proxy.astype(np.float64),
        },
        index=vol_arb_idx,
    )
    out.index.name = "timestamp"

    # Diagnostics
    corr = out[["vpin_true", "vpin_proxy"]].corr().iloc[0, 1]
    logger.info("Correlation(vpin_true, vpin_proxy) = %.4f", corr)
    logger.info(
        "vpin_true  — mean %.3f, median %.3f, std %.3f, "
        "p5/p50/p95 %.3f/%.3f/%.3f, NaN=%d (warmup=%d)",
        out["vpin_true"].mean(), out["vpin_true"].median(),
        out["vpin_true"].std(),
        out["vpin_true"].quantile(0.05),
        out["vpin_true"].quantile(0.50),
        out["vpin_true"].quantile(0.95),
        int(out["vpin_true"].isna().sum()), n_warmup,
    )
    logger.info(
        "vpin_proxy — mean %.3f, median %.3f, std %.3f, "
        "p5/p50/p95 %.3f/%.3f/%.3f",
        out["vpin_proxy"].mean(), out["vpin_proxy"].median(),
        out["vpin_proxy"].std(),
        out["vpin_proxy"].quantile(0.05),
        out["vpin_proxy"].quantile(0.50),
        out["vpin_proxy"].quantile(0.95),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    logger.info("Wrote %s (%d rows)", output_path, len(out))


if __name__ == "__main__":
    main()
