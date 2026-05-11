"""Extend ``aligner_betas_by_month.json`` to cover the current month
(and optionally a buffer of forward months) WITHOUT waiting for those
months' predictions.

Rationale (2026-05-11): the aligner's calibration slice is ``[M-3m, M)``
— strictly prior to the target month. That means β* for month M is
fully computable from already-available residuals, even when month M's
own predictions parquet rows don't exist yet (e.g. partial-month or
pre-data-publish window).

Today (2026-05-11) the canonical predictions parquet ends 2026-04-30
and aligner_betas_by_month.json ends at 2026-04 → the live engine's
month-lookup falls back to April's β* for every May bar. That's a
parity gap vs the backtest, which the user flagged. This script
closes it.

Usage:
    PYTHONPATH=. python scripts/extend_aligner_betas_forward.py
    # → extends through max(predictions_parquet_last_month + 2, current_month + 1)

    # Explicit forward horizon
    PYTHONPATH=. python scripts/extend_aligner_betas_forward.py --through 2026-06

    # Dry-run (no JSON write, just print plan + new β*)
    PYTHONPATH=. python scripts/extend_aligner_betas_forward.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

from vol_model.prediction_aligner import fit_monthly_calibrations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extend_aligner_betas")

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS_PATH = REPO_ROOT / "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"
BETAS_JSON = REPO_ROOT / "vol_model/output/aligner_betas_by_month.json"

# Safety bands — mirror vol_arb/sidecars.py ALIGNER_BETA_MIN/MAX.
BETA_MIN, BETA_MAX = -0.5, 0.5

# Aligner default — cal slice is [M-3, M).
CAL_MONTHS = 3


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """All "YYYY-MM" strings between start (inclusive) and end (inclusive)."""
    months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    return [str(m) for m in months]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--preds", default=str(PREDS_PATH), type=Path,
        help="Raw predictions parquet (must contain y_pred + y_true).",
    )
    parser.add_argument(
        "--betas", default=str(BETAS_JSON), type=Path,
        help="Target aligner_betas_by_month.json to extend in place.",
    )
    parser.add_argument(
        "--through", default="",
        help='YYYY-MM forward month to extend through (default: '
             'today + 1 month).',
    )
    parser.add_argument(
        "--cal-months", type=int, default=CAL_MONTHS,
        help="Trailing calibration window (default 3 = production).",
    )
    parser.add_argument(
        "--min-rows", type=int, default=500,
        help="Minimum cal rows; below -> pass-through β=0.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned new β* values without writing JSON.",
    )
    args = parser.parse_args()

    # ── Load existing betas (preserved as floor) ─────────────────────────
    if args.betas.exists():
        with args.betas.open("r", encoding="utf-8") as fh:
            existing = {str(k): float(v) for k, v in json.load(fh).items()}
        last_existing = max(existing) if existing else None
        logger.info(
            "Existing betas: %d months, range [%s .. %s]",
            len(existing),
            min(existing) if existing else "(none)",
            last_existing or "(none)",
        )
    else:
        existing = {}
        last_existing = None
        logger.warning("No existing %s — creating fresh.", args.betas)

    # ── Load predictions ─────────────────────────────────────────────────
    if not args.preds.exists():
        logger.error("Predictions parquet %s does not exist", args.preds)
        return 2
    preds = pd.read_parquet(args.preds)
    preds.index = pd.to_datetime(preds.index, utc=True)
    if "y_pred" not in preds.columns or "y_true" not in preds.columns:
        logger.error(
            "Predictions parquet missing y_pred/y_true (got %s)",
            list(preds.columns),
        )
        return 3
    last_pred_ts = preds.index.max()
    logger.info(
        "Predictions: %d rows, %s -> %s",
        len(preds), preds.index.min(), last_pred_ts,
    )

    # ── Resolve target month range ────────────────────────────────────────
    today = pd.Timestamp.now(tz="UTC")
    default_through = (today + pd.DateOffset(months=1)).to_period("M")
    through = pd.Period(args.through, freq="M") if args.through else default_through
    # We compute β* for every month from the data start through `through`.
    # Months already present in existing JSON are RE-COMPUTED if their cal
    # window still has data, but only OVERWRITTEN with the new value when
    # it falls within the safety band (sanity-gate).
    target_months = _months_between(
        preds.index.min(), pd.Timestamp(through.end_time, tz="UTC"),
    )
    forward_months = [m for m in target_months
                      if last_existing is None or m > last_existing]
    logger.info(
        "Target month range: %s -> %s (%d months total, %d forward)",
        target_months[0], target_months[-1],
        len(target_months), len(forward_months),
    )

    # ── Fit β* for forward months only ───────────────────────────────────
    # (skipping months that already exist preserves regression-locked
    # historic values and avoids overwriting them with whatever the
    # current alignment implementation produces.)
    if not forward_months:
        logger.info("No forward months to extend — sidecar already current.")
        return 0

    cals = fit_monthly_calibrations(
        preds,
        months=forward_months,
        cal_months=args.cal_months,
        method="bias",
        min_rows=args.min_rows,
    )

    # ── Apply sanity band + merge ────────────────────────────────────────
    new_betas = dict(existing)
    skipped: list[str] = []
    for cal in cals:
        if cal.method == "none":
            logger.warning(
                "%s: pass-through (insufficient cal rows: %d) — skipping",
                cal.month, cal.n_cal,
            )
            skipped.append(cal.month)
            continue
        b = float(cal.beta)
        if not (BETA_MIN <= b <= BETA_MAX):
            logger.warning(
                "%s: β=%+.4f outside [%+.2f, %+.2f] — skipping",
                cal.month, b, BETA_MIN, BETA_MAX,
            )
            skipped.append(cal.month)
            continue
        new_betas[cal.month] = b
        logger.info("  + %s: β=%+.4f  (n_cal=%d, %s..%s)",
                    cal.month, b, cal.n_cal,
                    cal.cal_start.strftime("%Y-%m"),
                    cal.cal_end.strftime("%Y-%m"))

    if not (set(new_betas) - set(existing)):
        logger.info("No new months were added (all skipped or already present).")
        return 0

    # ── Coverage guard ────────────────────────────────────────────────────
    dropped = set(existing) - set(new_betas)
    if dropped:
        logger.error(
            "Refusing to ship: would drop %d months %s",
            len(dropped), sorted(dropped)[:5],
        )
        return 4

    # ── Write ────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info(
            "[DRY-RUN] would write %d total months (was %d, added %d, skipped %d)",
            len(new_betas), len(existing), len(new_betas) - len(existing),
            len(skipped),
        )
        return 0

    backup = args.betas.with_suffix(f".bak.{uuid.uuid4().hex[:8]}.json")
    if args.betas.exists():
        shutil.copy2(args.betas, backup)
        logger.info("Backed up old JSON to %s", backup.name)

    tmp = args.betas.with_suffix(f".new.{uuid.uuid4().hex[:8]}.json")
    args.betas.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(new_betas, fh, indent=2, sort_keys=True)
    os.replace(tmp, args.betas)

    logger.info(
        "Wrote %d months to %s (was %d; +%d, skipped %d)",
        len(new_betas), args.betas, len(existing),
        len(new_betas) - len(existing), len(skipped),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
