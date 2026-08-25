"""Extract per-month aligner β* from the production aligned-bias parquet.

Inputs:
  ``vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet``
    — 145041-row parquet with columns ``y_pred``, ``y_pred_raw``, ``y_true``.
    The β* corrections are baked into ``y_pred = y_pred_raw + β*_month``.

Outputs:
  ``vol_model/output/aligner_betas_by_month.json``
    — ``{"YYYY-MM": float}`` mapping. The live predictor consumes this via
    ``vol_arb.sidecars.load_aligner_betas`` + ``lookup_beta_for_month``.

Usage:
    # from the repository root
    PYTHONPATH=. python scripts/extract_aligner_betas.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PARQUET = (
    "vol_model/output/"
    "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
)
DEFAULT_JSON = "vol_model/output/aligner_betas_by_month.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default=DEFAULT_PARQUET, type=Path)
    parser.add_argument("--output", default=DEFAULT_JSON, type=Path)
    parser.add_argument(
        "--pred-col", default="y_pred",
        help="Aligned prediction column name",
    )
    parser.add_argument(
        "--raw-col", default="y_pred_raw",
        help="Raw prediction column name (β* = mean(y_pred - y_pred_raw))",
    )
    args = parser.parse_args()

    from vol_arb.sidecars import load_aligner_betas

    if not args.input.exists():
        logger.error("Input parquet %s does not exist", args.input)
        return 2

    betas = load_aligner_betas(
        args.input, pred_col=args.pred_col, raw_col=args.raw_col
    )
    if not betas:
        logger.error(
            "No betas extracted from %s (missing pred/raw cols?)", args.input
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(betas, fh, indent=2, sort_keys=True)
    logger.info(
        "Wrote %d months of β* to %s; range [%+.4f, %+.4f]",
        len(betas), args.output,
        min(betas.values()), max(betas.values()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
