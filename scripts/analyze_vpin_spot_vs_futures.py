"""Spot-vs-Futures VPIN distribution analysis (2026-05-07).

Context
-------
The live engine reads VPIN from ``wss://fstream.binance.com/ws/btcusdt@aggTrade``
(Binance USD-M Futures). The homeserver IP is geo-blocked from that
endpoint — the WS handshake succeeds but no aggTrade frames flow,
leaving the engine on warm-start VPIN forever. We're considering
switching to ``wss://stream.binance.com:9443/ws/btcusdt@aggTrade``
(Binance Spot), which IS reachable.

Question: how different are Spot-VPIN and Futures-VPIN, on the same
window with the same bucket parameters? Specifically:
- Mean and std of the resulting VPIN distribution
- Pearson correlation between the two time series
- Time-cadence: how often does each close a new bucket?

The DRL hedger consumes VPIN as one of 8 observation slots, normalized
through VecNormalize fit on Futures VPIN. A material distribution
shift means the normalized observation goes off-center, potentially
degrading the trained policy.

Inputs
------
- Spot aggTrades:    data/binance-spot-aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-04.zip
- Futures aggTrades: data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet

Output
------
- vol_arb/output/vpin_spot_vs_futures_2026_04.parquet — joined series
- console summary with comparison stats
"""
from __future__ import annotations

import argparse
import logging
import time
import zipfile
from collections import deque
from pathlib import Path
from statistics import mean as _mean

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def stream_vpin_from_arrays(
    transact_time_ms: np.ndarray,
    quantity: np.ndarray,
    is_buyer_maker: np.ndarray,
    bucket_volume: float,
    rolling_window: int,
) -> pd.DataFrame:
    """Compute VPIN trade-by-trade on the same scalar accumulator the
    live engine uses, returning (timestamp, vpin, partial_vol) at
    every bucket close.

    Vectorised numpy makes this O(N) over trades with constant Python
    overhead per bucket (not per trade).
    """
    n = len(quantity)
    # Mark buy/sell volumes per trade
    buy = np.where(is_buyer_maker, 0.0, quantity)
    sell = np.where(is_buyer_maker, quantity, 0.0)

    # Cumulative volume tells us when each trade crosses a bucket boundary.
    cum = np.cumsum(quantity)
    # Bucket index that each trade "closes" — possibly multiple per trade
    # for very large prints. We approximate by floor(cum/V) being the
    # number of completed buckets just before this trade. A trade can
    # close at most one bucket in practice (BTC trades are tiny vs 1000-BTC
    # bucket), so np.diff gives 0 or 1 — guard with the "while" path
    # only for pathological cases.
    bucket_idx = np.floor(cum / bucket_volume).astype(np.int64)
    # Bucket close events: where bucket_idx jumps from N to N+1
    close_mask = np.concatenate(([bucket_idx[0] >= 1], np.diff(bucket_idx) >= 1))

    if not close_mask.any():
        return pd.DataFrame(columns=["timestamp", "vpin", "partial_vol"])

    # For each closing trade, the within-bucket buy/sell volume is the
    # cumsum of buy/sell since the last bucket close — vectorised via
    # np.add.reduceat on the indices of close events.
    close_idx = np.flatnonzero(close_mask)
    # reduceat needs starting indices for each segment; first segment
    # starts at 0. After the first bucket, segment k starts at close_idx[k-1]+1.
    segment_starts = np.concatenate(([0], close_idx[:-1] + 1))
    bucket_buy = np.add.reduceat(buy, segment_starts)
    bucket_sell = np.add.reduceat(sell, segment_starts)
    bucket_imbalance = np.abs(bucket_buy - bucket_sell) / bucket_volume
    bucket_close_ts = transact_time_ms[close_idx]

    # Rolling-window mean of imbalances → VPIN at each bucket close
    s = pd.Series(bucket_imbalance)
    vpin = s.rolling(window=rolling_window, min_periods=rolling_window).mean()

    return pd.DataFrame({
        "timestamp": pd.to_datetime(bucket_close_ts, unit="ms", utc=True),
        "vpin": vpin.to_numpy(),
        "partial_vol": np.zeros(len(close_idx)),  # placeholder
    }).dropna(subset=["vpin"])


def load_futures_aggtrades(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Columns expected: transact_time, quantity, is_buyer_maker
    return df


def load_spot_aggtrades(zip_path: Path) -> pd.DataFrame:
    """Stream the spot aggTrades CSV directly out of the zip.

    Note: Binance Spot aggTrades since 2025 carry MICROSECOND
    timestamps (16-digit), not milliseconds. Futures still uses ms.
    We normalise to ms here so callers can ignore the difference.
    """
    logger.info("Reading spot aggTrades zip: %s", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if not names:
            raise RuntimeError(f"empty zip: {zip_path}")
        csv_name = [n for n in names if n.endswith(".csv")][0]
        with z.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=[
                    "agg_trade_id", "price", "quantity",
                    "first_trade_id", "last_trade_id",
                    "transact_time", "is_buyer_maker", "is_best_match",
                ],
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "transact_time": "int64",
                    "is_buyer_maker": "bool",
                },
                usecols=["transact_time", "quantity", "is_buyer_maker"],
            )
    # Detect timestamp resolution by magnitude. ms timestamps in 2026 are
    # ~1.7e12; us timestamps are ~1.7e15.
    if df["transact_time"].iloc[0] > 1e14:
        logger.info("Spot timestamps look like microseconds; converting to ms")
        df["transact_time"] = df["transact_time"] // 1000
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2026-04")
    parser.add_argument("--bucket-btc", type=float, default=1000.0,
                        help="Bucket volume in BTC. Default 1000 matches "
                             "the live engine.")
    parser.add_argument("--bucket-btc-spot", type=float, default=0.0,
                        help="Optional: override Spot bucket size. "
                             "Set to ~120 BTC to match Futures cadence.")
    parser.add_argument("--rolling", type=int, default=50)
    parser.add_argument(
        "--output",
        default="vol_arb/output/vpin_spot_vs_futures.parquet",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    fut_path = repo / "data/binance-aggTrades-parquet/BTCUSDT" / f"{args.month}.parquet"
    spot_path = (
        repo / "data/binance-spot-aggTrades/BTCUSDT"
        / f"BTCUSDT-aggTrades-{args.month}.zip"
    )

    if not fut_path.exists():
        raise FileNotFoundError(f"futures parquet missing: {fut_path}")
    if not spot_path.exists():
        raise FileNotFoundError(f"spot zip missing: {spot_path}")

    # ── Futures ──────────────────────────────────────────────
    t0 = time.time()
    logger.info("Loading futures aggTrades: %s", fut_path)
    fut = load_futures_aggtrades(fut_path)
    logger.info("Futures: %d rows in %.1fs", len(fut), time.time() - t0)

    fut_vpin = stream_vpin_from_arrays(
        transact_time_ms=fut["transact_time"].to_numpy(),
        quantity=fut["quantity"].to_numpy(),
        is_buyer_maker=fut["is_buyer_maker"].to_numpy(),
        bucket_volume=args.bucket_btc,
        rolling_window=args.rolling,
    )
    fut_vol_per_min = fut["quantity"].sum() / (
        (fut["transact_time"].max() - fut["transact_time"].min()) / 60_000.0
    )
    logger.info(
        "Futures VPIN: %d bucket closes, %.1f BTC/min average flow, "
        "1000-BTC bucket fills every %.1f min",
        len(fut_vpin), fut_vol_per_min,
        args.bucket_btc / fut_vol_per_min,
    )
    del fut  # free memory

    # ── Spot ─────────────────────────────────────────────────
    t1 = time.time()
    spot = load_spot_aggtrades(spot_path)
    logger.info("Spot: %d rows in %.1fs", len(spot), time.time() - t1)

    spot_bucket = args.bucket_btc_spot if args.bucket_btc_spot > 0 else args.bucket_btc
    spot_vpin = stream_vpin_from_arrays(
        transact_time_ms=spot["transact_time"].to_numpy(),
        quantity=spot["quantity"].to_numpy(),
        is_buyer_maker=spot["is_buyer_maker"].to_numpy(),
        bucket_volume=spot_bucket,
        rolling_window=args.rolling,
    )
    spot_vol_per_min = spot["quantity"].sum() / (
        (spot["transact_time"].max() - spot["transact_time"].min()) / 60_000.0
    )
    logger.info(
        "Spot VPIN (bucket=%.1f BTC): %d bucket closes, %.1f BTC/min, "
        "fills every %.1f min",
        spot_bucket, len(spot_vpin), spot_vol_per_min,
        spot_bucket / spot_vol_per_min,
    )
    del spot

    # ── Distribution comparison ──────────────────────────────
    print("\n=== Distribution comparison ===\n")
    print(f"Window: {args.month} (1 month)")
    print(f"Futures bucket: {args.bucket_btc} BTC | Spot bucket: {spot_bucket} BTC")
    print(f"Rolling window: {args.rolling}\n")

    print(f"{'metric':<25} {'futures':>12} {'spot':>12} {'delta':>12}")
    print("-" * 65)

    f = fut_vpin["vpin"].to_numpy()
    s = spot_vpin["vpin"].to_numpy()

    for label, fv, sv in [
        ("count (bucket closes)", len(f), len(s)),
        ("mean", f.mean(), s.mean()),
        ("std", f.std(), s.std()),
        ("p5", np.percentile(f, 5), np.percentile(s, 5)),
        ("p25", np.percentile(f, 25), np.percentile(s, 25)),
        ("p50 (median)", np.median(f), np.median(s)),
        ("p75", np.percentile(f, 75), np.percentile(s, 75)),
        ("p95", np.percentile(f, 95), np.percentile(s, 95)),
        ("p99", np.percentile(f, 99), np.percentile(s, 99)),
        ("max", f.max(), s.max()),
        ("min", f.min(), s.min()),
    ]:
        if isinstance(fv, (int, np.integer)):
            print(f"{label:<25} {fv:>12d} {sv:>12d} {sv - fv:>+12d}")
        else:
            print(f"{label:<25} {fv:>12.4f} {sv:>12.4f} {sv - fv:>+12.4f}")

    # Time-aligned correlation (resample to 1h bars, take last value)
    fut_hourly = (
        fut_vpin.set_index("timestamp")["vpin"]
        .resample("1h").last().dropna()
    )
    spot_hourly = (
        spot_vpin.set_index("timestamp")["vpin"]
        .resample("1h").last().dropna()
    )
    common = fut_hourly.index.intersection(spot_hourly.index)
    fa = fut_hourly.loc[common].to_numpy()
    sa = spot_hourly.loc[common].to_numpy()
    if len(common) >= 10:
        pearson = np.corrcoef(fa, sa)[0, 1]
        spearman = (
            pd.Series(fa).rank().corr(pd.Series(sa).rank())
        )
        # Lag-1 autocorrelation per series
        ac_f = pd.Series(fa).autocorr(lag=1)
        ac_s = pd.Series(sa).autocorr(lag=1)
        print(f"\n{'metric':<25} {'futures':>12} {'spot':>12}")
        print("-" * 50)
        print(f"{'Pearson(fut, spot)':<25} {pearson:>12.4f}")
        print(f"{'Spearman(fut, spot)':<25} {spearman:>12.4f}")
        print(f"{'autocorr(lag=1 hour)':<25} {ac_f:>12.4f} {ac_s:>12.4f}")
        print(f"{'aligned hourly samples':<25} {len(common):>12d}")
    else:
        print(f"\nToo few aligned samples ({len(common)}) for correlation")

    # ── Persist for further analysis ─────────────────────────
    out_path = repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joined = pd.concat({
        "fut": fut_hourly.rename("vpin"),
        "spot": spot_hourly.rename("vpin"),
    }, axis=1)
    joined.to_parquet(out_path)
    logger.info("\nWrote joined hourly series → %s", out_path)


if __name__ == "__main__":
    main()
