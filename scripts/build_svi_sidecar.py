#!/usr/bin/env python
"""Build ONLY the SVI features sidecar for a given bar/bucket cadence.

Why this exists
---------------
``vol_arb/run_pipeline.py --svi --bar-minutes {H}`` calibrates the SVI
surface at the requested cadence AND persists the
``svi_features_h{H}m.parquet`` sidecar (the file the walk-forward loader
``_load_svi_atm_1d`` reads), but it then proceeds to assemble the FULL
merged vol-arb dataset, whose trailing-1h-volume step raises
``ValueError: bucket_freq ... yields <1 bar/hour`` for any cadence >= 60min
(see vol_arb/pipeline.py:612). That crash is downstream of — and irrelevant
to — the SVI sidecar, which is all the 7-horizon vol-forecast fleet needs.

This script reuses the EXACT load + SVIConfig + calibrate + persist logic
from ``build_vol_arb_dataset`` (vol_arb/pipeline.py:518-564) and stops
cleanly once the sidecar is on disk. No new calibration logic — it calls
the same ``calibrate_surface`` and writes via the same path resolvers.

Usage:
    PYTHONPATH=. python scripts/build_svi_sidecar.py --currency BTC --bar-minutes 240

The sidecar lands at the path returned by
``vol_arb.paths.svi_features_path(currency, horizon_minutes=bar_minutes)``;
the surface params at ``svi_surface_params_path(currency, freq="{H}min")``.
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from vol_arb.config import VolArbConfig
from vol_arb.options_loader import load_option_trades
from vol_arb.paths import (
    dvol_path, option_trade_dir, svi_features_path, svi_surface_params_path,
    tsrv_path,
)
from vol_arb.pipeline import load_tsrv
from vol_arb.svi import calibrate_surface, SVIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build-svi-sidecar")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SVI features sidecar only")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--bar-minutes", type=int, required=True,
                        help="SVI calibration bucket cadence in minutes "
                             "(e.g. 60, 240, 480, 1440). Drives both the "
                             "calibration floor and the sidecar filename.")
    args = parser.parse_args()

    ccy = args.currency.upper()
    asset = f"{ccy}USDT"
    bucket_freq = f"{args.bar_minutes}min"

    # Mirror build_vol_arb_dataset's VolArbConfig construction for SVI.
    config = VolArbConfig(
        currency=ccy,
        option_trade_dir=str(option_trade_dir(ccy)),
        dvol_path=str(dvol_path(ccy)),
        tsrv_path=str(tsrv_path(asset, tf_min=args.bar_minutes)),
        svi_enabled=True,
        bucket_freq=bucket_freq,
        svi_bucket_freq=bucket_freq,
    )

    # 1. Load TSRV (only to derive the [start, end] window for option trades) —
    #    identical to build_vol_arb_dataset steps 1-2.
    logger.info("Loading TSRV from %s", config.tsrv_path)
    tsrv = load_tsrv(config.tsrv_path)
    start_date = tsrv.index.min().strftime("%Y-%m-%d")
    end_date = tsrv.index.max().strftime("%Y-%m-%d")
    logger.info("TSRV: %d rows, %s to %s", len(tsrv), start_date, end_date)

    logger.info("Loading option trades from %s", config.option_trade_dir)
    trades = load_option_trades(config.option_trade_dir, start_date, end_date)

    # 2. SVI calibration + persist — byte-identical to pipeline.py:533-564.
    t0 = time.perf_counter()
    svi_config = SVIConfig(
        bucket_freq=config.svi_bucket_freq,
        min_strikes=config.svi_min_strikes,
        fallback_min_strikes=config.svi_fallback_min_strikes,
        cache_params=config.svi_cache,
    )
    svi_features, svi_params = calibrate_surface(trades, svi_config)
    if svi_features.empty:
        logger.error("SVI calibration produced no results — aborting")
        return 1

    logger.info(
        "SVI calibration: %d buckets, %.1f%% full fit (level 0), %d param rows",
        len(svi_features),
        100 * (svi_features["svi_calibration_level"] == 0).mean(),
        len(svi_params),
    )

    _svi_bar_minutes = int(
        round(pd.Timedelta(config.bucket_freq).total_seconds() / 60.0)
    )
    feat_out = svi_features_path(config.currency, horizon_minutes=_svi_bar_minutes)
    params_out = svi_surface_params_path(config.currency, freq=config.svi_bucket_freq)
    feat_out.parent.mkdir(parents=True, exist_ok=True)
    svi_features.to_parquet(feat_out)
    svi_params.to_parquet(params_out)
    logger.info("SVI persisted: features=%s, params=%s", feat_out, params_out)

    # 3. Coverage assertion over the backtest window (2022-02 .. 2026-04).
    ts = svi_features["timestamp"]
    win = svi_features[(ts >= "2022-02-01") & (ts <= "2026-04-01")]
    cov = win["svi_atm_1d"].notna().mean() if len(win) else 0.0
    logger.info(
        "svi_atm_1d coverage over backtest window: %.1f%% (%d rows) | total rows=%d | %.1fs",
        100 * cov, len(win), len(svi_features), time.perf_counter() - t0,
    )
    if cov < 0.90:
        logger.error("COVERAGE GATE FAILED: %.1f%% < 90%%", 100 * cov)
        return 2
    logger.info("COVERAGE GATE PASSED (>=90%%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
