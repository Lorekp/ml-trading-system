#!/usr/bin/env python
"""Surgical SVI calibration extension for partial-window updates.

DO NOT use ``vol_arb/run_pipeline.py`` to add a few new bars — that
recomputes all 179K SVI buckets across 5 years.

This script:
  1. Loads existing svi_features_15m.parquet + svi_surface_params_15min.parquet.
  2. Loads option trades, filters to buckets AFTER the existing max.
  3. Runs ``calibrate_surface()`` on the new buckets only.
  4. Concats + saves.

Also rebuilds vol_arb_svi.parquet's slip surface for the new bars.

Usage:
    PYTHONPATH=. python scripts/svi_extend_incremental.py --end-date 2026-05-12T18:00:00
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.svi import calibrate_surface, SVIConfig
from vol_arb.options_loader import load_option_trades
from vol_arb.config import VolArbConfig
from vol_arb.paths import svi_features_path, svi_surface_params_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("svi-extend")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--bucket-freq", default="15min")
    parser.add_argument("--end-date", required=True, help="ISO end (exclusive cap)")
    args = parser.parse_args()

    end_dt = pd.Timestamp(args.end_date)
    if end_dt.tz is None:
        end_dt = end_dt.tz_localize("UTC")

    bucket_freq = args.bucket_freq
    # The features parquet output path is keyed by BAR cadence (15m always for
    # this pipeline), NOT the SVI bucket_freq. The bucket_freq only affects
    # the params parquet name.
    bar_minutes = 15  # h=15 production
    if bucket_freq == "1h":
        # Legacy 1h cadence has its own _1h-suffixed features parquet.
        feat_path = Path(f"vol_arb/output/svi_features_1h.parquet")
    else:
        feat_path = svi_features_path(args.currency, horizon_minutes=bar_minutes)
    params_path = svi_surface_params_path(args.currency, freq=bucket_freq)
    logger.info("Loading existing SVI features: %s", feat_path)
    feat_exist = pd.read_parquet(feat_path)
    logger.info("  %d rows, max_ts=%s", len(feat_exist), feat_exist["timestamp"].max())

    logger.info("Loading existing SVI params: %s", params_path)
    params_exist = pd.read_parquet(params_path)
    logger.info("  %d rows, max_ts=%s", len(params_exist), params_exist["timestamp"].max())

    existing_max_ts = feat_exist["timestamp"].max()

    # The features parquet uses END-of-bucket timestamps. So existing_max_ts
    # is the END of the latest covered bucket. New buckets must start at or
    # after existing_max_ts (the previous-end is also a valid bucket boundary
    # for the NEXT bucket's start).
    new_bucket_start = existing_max_ts
    new_bucket_end = end_dt  # exclusive

    # Load all option trades — vectorised loader, returns a DataFrame
    # with columns: trade_id, timestamp, strike, option_type, iv, ...
    vol_cfg = VolArbConfig()
    logger.info("Loading option trades from %s", vol_cfg.option_trade_dir)
    # We only need trades in the new window. But option_trades_loader filters
    # by date_str (YYYY-MM-DD), so we load slightly wider then trim.
    start_str = (new_bucket_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_str = (new_bucket_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    trades = load_option_trades(vol_cfg.option_trade_dir, start_str, end_str)
    logger.info("  loaded %d trades in [%s, %s]", len(trades), start_str, end_str)

    # Filter to new buckets only: bucket boundaries are at floor("15min").
    # New buckets are those that floor() to a value >= new_bucket_start.
    # Equivalently: trade.timestamp >= new_bucket_start AND < new_bucket_end.
    trades = trades[(trades["timestamp"] >= new_bucket_start) &
                    (trades["timestamp"] < new_bucket_end)].copy()
    logger.info("  after filter [%s, %s): %d trades", new_bucket_start, new_bucket_end, len(trades))

    if trades.empty:
        logger.warning("No trades in new window — nothing to extend.")
        return 0

    # Build config matching production
    svi_config = SVIConfig(
        bucket_freq=bucket_freq,
        min_strikes=vol_cfg.svi_min_strikes,
        fallback_min_strikes=vol_cfg.svi_fallback_min_strikes,
        cache_params=vol_cfg.svi_cache,
    )

    # Calibrate JUST the new buckets
    logger.info("Calibrating SVI on %d new trades...", len(trades))
    new_feat, new_params = calibrate_surface(trades, svi_config)
    logger.info("  -> %d new feature rows, %d new param rows", len(new_feat), len(new_params))
    if not new_feat.empty:
        logger.info("  new feature ts range: %s -> %s", new_feat["timestamp"].min(), new_feat["timestamp"].max())

    # Drop any new buckets whose END-stamp is at or BEFORE existing_max_ts
    # (those would overlap existing rows).
    new_feat = new_feat[new_feat["timestamp"] > existing_max_ts].copy()
    new_params = new_params[new_params["timestamp"] > existing_max_ts].copy()
    logger.info("  after dedup (> existing_max_ts): %d feats, %d params", len(new_feat), len(new_params))

    if new_feat.empty:
        logger.info("No new bars to append — pipeline already covers requested window.")
        return 0

    # Append + save
    feat_combined = pd.concat([feat_exist, new_feat], ignore_index=True)
    feat_combined = feat_combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    logger.info("Combined features: %d rows (added %d, dropped %d duplicates)",
                len(feat_combined), len(new_feat), len(feat_exist) + len(new_feat) - len(feat_combined))

    params_combined = pd.concat([params_exist, new_params], ignore_index=True)
    params_combined = params_combined.sort_values(["timestamp", "expiry_idx"]).reset_index(drop=True)

    feat_combined.to_parquet(feat_path)
    params_combined.to_parquet(params_path)
    logger.info("Wrote %s (%d rows, max_ts=%s)", feat_path, len(feat_combined), feat_combined["timestamp"].max())
    logger.info("Wrote %s (%d rows, max_ts=%s)", params_path, len(params_combined), params_combined["timestamp"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
