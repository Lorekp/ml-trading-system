"""Generate the 5 Cycle-3 Phase P overlay sidecars from the frozen ensemble.

Runs ONCE: F1/F2/F3 boolean filters + S1/S2 size_mult scalers, all keyed
by the V2 trade log bar index. Anchors are computed on the calibration
window (2021-04-01 → 2023-09-30) ONCE and frozen.

Run::

    PYTHONPATH=. python scripts/generate_uncertainty_overlay_sidecars.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from vol_arb.uncertainty_overlays import (
    CAL_WINDOW,
    compute_anchors,
    compute_filter_mask,
    compute_scaler_S1,
    compute_scaler_S2,
    write_filter_sidecar,
    write_scaler_sidecar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_uncertainty_overlays")

ENSEMBLE_PARQUET = Path("vol_model/output/BTC_v2_har_pd_ensemble_var.parquet")
OUTPUT_DIR = Path("vol_arb/output")

SIDECAR_NAMES = {
    "F1": "BTC_v2_uncert_F1_filter01.parquet",
    "F2": "BTC_v2_uncert_F2_filter05.parquet",
    "F3": "BTC_v2_uncert_F3_filter10.parquet",
    "S1": "BTC_v2_uncert_S1_size_mult.parquet",
    "S2": "BTC_v2_uncert_S2_size_mult.parquet",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=ENSEMBLE_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    if not args.ensemble.exists():
        raise FileNotFoundError(
            f"Ensemble artifact missing: {args.ensemble}. Run "
            "scripts/train_har_pd_ensemble.py first."
        )

    # Pre-flight: refuse to clobber existing sidecars without explicit flag
    for tag, name in SIDECAR_NAMES.items():
        out = args.out_dir / name
        if out.exists() and not args.allow_overwrite:
            raise FileExistsError(
                f"Sidecar {out} already exists. Use --allow-overwrite "
                "only with a NEW plan file authorising regeneration."
            )

    logger.info("Loading ensemble: %s", args.ensemble)
    uncert = pd.read_parquet(args.ensemble)
    if not isinstance(uncert.index, pd.DatetimeIndex):
        uncert.index = pd.to_datetime(uncert.index, utc=True)
    sigma_epi_sq = (uncert["sigma_epi"] ** 2).rename("sigma_epi_sq")
    logger.info("σ²_epi: n=%d, mean=%.6f, p99=%.6f",
                len(sigma_epi_sq), float(sigma_epi_sq.mean()),
                float(sigma_epi_sq.quantile(0.99)))

    logger.info("Computing anchors on cal window %s -> %s",
                CAL_WINDOW[0].date(), CAL_WINDOW[1].date())
    anchors = compute_anchors(sigma_epi_sq, cal_window=CAL_WINDOW)
    logger.info("Anchors:")
    logger.info("  quantile_F1 (0.99): %.6f", anchors.quantile_F1)
    logger.info("  quantile_F2 (0.95): %.6f", anchors.quantile_F2)
    logger.info("  quantile_F3 (0.90): %.6f", anchors.quantile_F3)
    logger.info("  median_cal:        %.6f", anchors.median_cal)
    logger.info("  mean_cal (=σ²_train): %.6f", anchors.mean_cal)
    logger.info("  cal_n_obs: %d", anchors.cal_n_obs)

    # Generate filters
    for tag in ("F1", "F2", "F3"):
        mask = compute_filter_mask(sigma_epi_sq, anchors, tag=tag)
        out_path = args.out_dir / SIDECAR_NAMES[tag]
        write_filter_sidecar(mask, out_path)

    # Generate scalers
    s1 = compute_scaler_S1(sigma_epi_sq, anchors)
    write_scaler_sidecar(s1, args.out_dir / SIDECAR_NAMES["S1"])

    s2 = compute_scaler_S2(sigma_epi_sq, anchors)
    write_scaler_sidecar(s2, args.out_dir / SIDECAR_NAMES["S2"])

    print()
    print("=" * 70)
    print(" 5 overlay sidecars generated")
    print("=" * 70)
    for tag, name in SIDECAR_NAMES.items():
        path = args.out_dir / name
        print(f"  [{tag}] {path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
