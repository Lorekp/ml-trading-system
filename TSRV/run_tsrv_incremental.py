#!/usr/bin/env python
"""Incremental TSRV runner for the partial-window pipeline.

Thin wrapper around ``TSRV.tsrv_series.compute_tsrv_series`` that:

1. Loads the existing TSRV parquet (default
   ``TSRV/output/{asset}_tsrv_{tf_min}m.parquet``) and reuses its
   already-calibrated sub-bar threshold (read from a sidecar
   ``{asset}_tsrv_{tf_min}m_threshold.txt`` or, if absent, the value
   passed via ``--threshold``).
2. Calls ``compute_tsrv_series`` whose existing resume logic processes
   only months past the parquet's last fully-covered month.
3. Re-runs the current (partial) month each invocation so newly-arrived
   bookTicker days are picked up.
4. Atomically writes the extended parquet back to disk.
5. Optionally trims the output to ``--end-date`` so downstream
   consumers see a clean partial-window cutoff.

Determinism: bar T's TSRV value is computed from tick data within
[T-tf_min, T) only — no rolling state across bars — so incremental
processing is bit-identical to a full from-scratch run on the same
inputs.

Usage:
    PYTHONPATH=. python TSRV/run_tsrv_incremental.py \
        --asset BTCUSDT --tf 900 \
        --threshold 250000 \
        --end-date 2026-05-11T12:15:00
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from TSRV.config import TSRVConfig
from TSRV.tsrv_series import compute_tsrv_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TSRV-incremental")


def _resolve_threshold(
    cli_threshold: float | None, parquet_path: Path,
) -> float:
    """Resolve the sub-bar threshold from CLI, a sidecar txt, or the SSOT.

    Priority:
    1. ``--threshold`` if passed.
    2. Sidecar ``{parquet_stem}_threshold.txt`` next to the parquet.
    3. Cycle-39 (2026-05-17): the SSOT constant
       ``TSRV.config.BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD`` — the
       single canonical value the live engine's ``compute_bar_tsrv``
       also reads. Pre-cycle-39 this branch hard-failed; the SSOT
       removes the "forgot --threshold" foot-gun while still keeping
       the value pinned in exactly one place.
    """
    if cli_threshold is not None:
        return float(cli_threshold)
    sidecar = parquet_path.with_name(parquet_path.stem + "_threshold.txt")
    if sidecar.exists():
        return float(sidecar.read_text().strip())
    from TSRV.config import BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD
    logger.warning(
        "No --threshold and no sidecar %s; falling back to SSOT "
        "BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD=%.0f",
        sidecar, BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD,
    )
    return float(BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900,
                        help="Slow timeframe in seconds (default 900 = 15m)")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Dollar sub-bar threshold (REQUIRED for byte-identity with the "
             "canonical batch; do NOT recalibrate). If omitted, falls back to "
             "the threshold sidecar txt next to the parquet.",
    )
    parser.add_argument(
        "--end-date", type=str, default="",
        help="Optional ISO timestamp upper bound; rows past end_date are "
             "trimmed from the output. Default empty (keep all).",
    )
    parser.add_argument("--save-every", type=int, default=6,
                        help="Checkpoint interval (months) inside compute_tsrv_series")
    args = parser.parse_args()

    if 1440 % (args.tf // 60) != 0:
        parser.error(f"--tf={args.tf} must yield bar minutes that divide 1440")

    config = TSRVConfig(asset=args.asset, slow_timeframe_s=args.tf)
    tf_min = args.tf // 60
    out_path = Path(config.output_dir) / f"{args.asset}_tsrv_{tf_min}m.parquet"

    threshold = _resolve_threshold(args.threshold, out_path)
    logger.info("Using threshold $%s (sub-bar) — same as canonical batch", f"{threshold:,.0f}")

    print("=" * 70)
    print(f"  TSRV INCREMENTAL — {args.asset} {tf_min}m")
    print(f"  Output:    {out_path}")
    print(f"  Threshold: ${threshold:,.0f}")
    if args.end_date:
        print(f"  End date:  {args.end_date}")
    print("=" * 70)

    t0 = time.time()
    series = compute_tsrv_series(
        config.trade_dir, args.asset,
        threshold, args.tf, config,
        output_path=out_path,
        save_every=args.save_every,
    )
    elapsed_min = (time.time() - t0) / 60.0

    if len(series) == 0:
        logger.error("No data produced")
        return 1

    # Optional end_date trim
    if args.end_date:
        ed = pd.Timestamp(args.end_date)
        if series.index.tz is not None and ed.tz is None:
            ed = ed.tz_localize(series.index.tz)
        n_pre = len(series)
        series = series.loc[series.index <= ed]
        logger.info("Trimmed to end_date=%s: %d → %d rows", ed, n_pre, len(series))

    # Final atomic write
    tmp = out_path.with_suffix(".tmp")
    series.to_parquet(tmp)
    tmp.replace(out_path)
    logger.info(
        "Saved %d rows to %s (elapsed %.1f min, last bar %s)",
        len(series), out_path, elapsed_min, series.index.max(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
