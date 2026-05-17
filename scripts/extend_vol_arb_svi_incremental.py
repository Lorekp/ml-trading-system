#!/usr/bin/env python
"""Surgical incremental extension of BTC_vol_arb_svi.parquet.

Avoids the 30-90 min SVI calibration step by monkey-patching
``calibrate_surface()`` to read the already-extended SVI parquets
that ``scripts/svi_extend_incremental.py`` writes.

The rest of the pipeline (TSRV load, Greeks, slippage surface,
trailing vol, composite straddles, merges) runs normally and is
quick — most steps are bar-level or rolling-window operations.

Usage:
    PYTHONPATH=. python scripts/extend_vol_arb_svi_incremental.py \\
        --currency BTC --bar-minutes 15 --end-date 2026-05-12T18:00:00
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vol-arb-extend")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--bar-minutes", type=int, default=15)
    parser.add_argument("--svi-bucket-freq", default="15min")
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()

    # --- Monkey-patch SVI calibration to read cached parquets ---
    from vol_arb import svi as svi_mod
    from vol_arb.paths import svi_features_path, svi_surface_params_path

    feat_path = svi_features_path(args.currency, horizon_minutes=args.bar_minutes)
    params_path = svi_surface_params_path(args.currency, freq=args.svi_bucket_freq)
    logger.info("Loading cached SVI features: %s", feat_path)
    cached_features = pd.read_parquet(feat_path)
    logger.info("  %d rows, max_ts=%s", len(cached_features), cached_features["timestamp"].max())
    logger.info("Loading cached SVI params: %s", params_path)
    cached_params = pd.read_parquet(params_path)
    logger.info("  %d rows, max_ts=%s", len(cached_params), cached_params["timestamp"].max())

    original_calibrate = svi_mod.calibrate_surface

    def patched_calibrate(trades, config):
        logger.info("[PATCHED] returning cached SVI features/params (skipping calibration)")
        return cached_features, cached_params

    svi_mod.calibrate_surface = patched_calibrate
    # Also patch in pipeline namespace if already imported
    try:
        from vol_arb import pipeline as pipe_mod
        pipe_mod.calibrate_surface = patched_calibrate
    except Exception:
        pass

    # --- Run pipeline normally ---
    from vol_arb.config import VolArbConfig
    from vol_arb.pipeline import build_vol_arb_dataset, verify_alignment
    from vol_arb.paths import vol_arb_path

    config = VolArbConfig(
        currency=args.currency,
        bucket_freq=args.svi_bucket_freq,
        svi_enabled=True,
        svi_bucket_freq=args.svi_bucket_freq,
    )
    end_dt = pd.Timestamp(args.end_date)
    if end_dt.tz is None:
        end_dt = end_dt.tz_localize("UTC")

    merged = build_vol_arb_dataset(config)
    logger.info("Pipeline output: %d rows, max=%s", len(merged), merged.index.max())

    # Trim to end_date
    merged = merged[merged.index < end_dt]
    logger.info("After end_date filter: %d rows", len(merged))

    out_path = vol_arb_path(args.currency, svi=True, horizon_minutes=args.bar_minutes)
    merged.to_parquet(out_path)
    logger.info("Wrote %s (max_ts=%s)", out_path, merged.index.max())

    verify_alignment(merged)


if __name__ == "__main__":
    main()
