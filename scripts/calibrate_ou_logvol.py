"""Calibrate the bivariate OU on BTC 1-minute (log-IV, log-RV) and freeze.

Phase A artifact (plan: ``now-check-empirically-from-eager-bird.md``).

Inputs:
- ``vol_arb/output/minute_atm_iv.parquet`` → log-IV (median_iv / 100, log-decimal)
- ``TSRV/output/BTCUSDT_tsrv_1m.parquet``  → log-RV (vol_ann, log-decimal)

The calibration window (2021-04-01 → 2023-09-30 UTC, exclusive end) and the
quarantine days are pre-committed in ``vol_model/ou_logvol.py``. Output JSON
is refused-on-overwrite by ``freeze_calibration``.

Run::

    PYTHONPATH=. python scripts/calibrate_ou_logvol.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vol_model.ou_logvol import (
    CALIBRATION_WINDOW,
    QUARANTINE_DAYS,
    calibrate_bivariate_ou,
    freeze_calibration,
    load_log_iv_minute,
    load_log_rv_minute,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_ou_logvol")

DEFAULT_OUT = Path("vol_model/output/ou_logvol_calibration.json")
MINUTE_IV_PATH = Path("vol_arb/output/minute_atm_iv.parquet")
TSRV_PATH = Path("TSRV/output/BTCUSDT_tsrv_1m.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output JSON path (default {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing calibration (defeat safety)")
    args = parser.parse_args()

    logger.info("Loading minute IV from %s", MINUTE_IV_PATH)
    log_iv = load_log_iv_minute(MINUTE_IV_PATH)
    logger.info("Loaded %d log-IV observations, span %s → %s",
                len(log_iv), log_iv.index.min(), log_iv.index.max())

    logger.info("Loading TSRV from %s", TSRV_PATH)
    log_rv = load_log_rv_minute(TSRV_PATH, column="vol_ann")
    logger.info("Loaded %d log-RV observations, span %s → %s",
                len(log_rv), log_rv.index.min(), log_rv.index.max())

    cal_start, cal_end = CALIBRATION_WINDOW
    logger.info("Calibration window: %s → %s (exclusive)", cal_start, cal_end)
    logger.info("Quarantine days dropped: %d", len(QUARANTINE_DAYS))

    params = calibrate_bivariate_ou(
        log_iv, log_rv,
        calibration_window=CALIBRATION_WINDOW,
        quarantine_days=QUARANTINE_DAYS,
    )

    logger.info("Calibrated parameters:")
    logger.info("  log-IV:  kappa=%.6f, theta=%.4f, sigma=%.6f, b=%.8f",
                params.kappa_iv, params.theta_iv, params.sigma_iv, params.b_iv)
    logger.info("           a=%.6e, var_eps=%.6e",
                params.a_iv, params.var_eps_iv)
    logger.info("  log-RV:  kappa=%.6f, theta=%.4f, sigma=%.6f, b=%.8f",
                params.kappa_rv, params.theta_rv, params.sigma_rv, params.b_rv)
    logger.info("           a=%.6e, var_eps=%.6e",
                params.a_rv, params.var_eps_rv)
    logger.info("  rho:     %+.4f", params.rho)
    logger.info("  N obs:   %d", params.n_observations)

    out_path = freeze_calibration(params, args.out, force=args.force)
    logger.info("Frozen calibration written to %s", out_path)

    print()
    print("=" * 70)
    print(" OU log-vol calibration complete")
    print("=" * 70)
    print(f" log-IV:  kappa = {params.kappa_iv:8.5f},  theta = {params.theta_iv:+.4f}")
    print(f"          sigma = {params.sigma_iv:8.5f},  b     = {params.b_iv:.8f}")
    print(f" log-RV:  kappa = {params.kappa_rv:8.5f},  theta = {params.theta_rv:+.4f}")
    print(f"          sigma = {params.sigma_rv:8.5f},  b     = {params.b_rv:.8f}")
    print(f" rho:     {params.rho:+.4f}")
    print(f" N obs:   {params.n_observations:,}")
    print(f" Frozen:  {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
