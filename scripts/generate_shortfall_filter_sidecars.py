"""Generate boolean filter sidecars from Phase B's pooled-OOS shortfall scores.

Cycle 4 Phase C (CONDITIONAL on Phase B passing) — plan
``now-check-empirically-from-eager-bird.md``.

3 pre-committed cuts on the calibration-window distribution of
``p_treatment``: drop trades where pooled-OOS P(shortfall) is in top
{1%, 5%, 10%}. Sidecars use the same boolean ``drop_trade`` contract
that Cycle 3's ``vol_arb.uncertainty_overlays`` shipped, so the existing
backtester wiring at ``BacktestConfig.uncertainty_filter_path`` loads
them with no further code changes.

Pre-committed parameters (LOCKED):
- Calibration window: 2021-04-01 → 2023-10-01 UTC (exclusive end)
  — IDENTICAL to Cycle 3 ``vol_arb.uncertainty_overlays.CAL_WINDOW``.
- Cuts: {T1: 0.99 (top-1%), T2: 0.95 (top-5%), T3: 0.90 (top-10%)}.
- f_min: N/A (this is a filter, not a scaler — pure binary mask).

Outputs: ``vol_arb/output/BTC_v2_shortfall_T{1,2,3}_filter.parquet``.

Run::

    PYTHONPATH=. python scripts/generate_shortfall_filter_sidecars.py
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.uncertainty_overlays import CAL_WINDOW, write_filter_sidecar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_shortfall_filter_sidecars")


# ─── Pre-committed parameters (locked) ─────────────────────────────────

SHORTFALL_FILTER_QUANTILES: dict[str, float] = {
    "T1": 0.99,   # drop top-1% P(shortfall)
    "T2": 0.95,   # drop top-5%
    "T3": 0.90,   # drop top-10%
}

DEFAULT_PREDICTIONS_PARQUET = Path(
    "vol_arb/output/BTC_v2_shortfall_oos_predictions.parquet"
)
DEFAULT_OUT_DIR = Path("vol_arb/output")


@dataclass(frozen=True)
class ShortfallAnchors:
    """Frozen calibration-window cuts for the 3 shortfall filter strategies."""

    quantile_T1: float
    quantile_T2: float
    quantile_T3: float
    cal_n_obs: int
    cal_window_start: str
    cal_window_end: str


def compute_shortfall_anchors(
    p_treatment: pd.Series,
    *,
    cal_window: tuple[pd.Timestamp, pd.Timestamp] = CAL_WINDOW,
) -> ShortfallAnchors:
    """Compute frozen cuts on cal-window subset of pooled-OOS shortfall probabilities.

    Note: `p_treatment` typically has NaN scores in the cal window because
    walk-forward year folds 2022 are pass-through and 2023 receives scores
    from a model trained on 2022 only. So the cal-window subset of
    `p_treatment` is the union of 2023-Q1..Q3 fired-trade scores. If this
    subset is too sparse (< 200 obs), raise.
    """
    if not isinstance(p_treatment.index, pd.DatetimeIndex):
        raise TypeError("p_treatment must have a DatetimeIndex")
    cal_start, cal_end = cal_window
    cal = p_treatment.loc[
        (p_treatment.index >= cal_start) & (p_treatment.index < cal_end)
    ].dropna()
    if len(cal) < 200:
        raise ValueError(
            f"calibration slice has {len(cal)} non-NaN obs of p_treatment; "
            "need ≥ 200 for stable quantile cut. The walk-forward fold "
            "contract may have insufficient 2023-pre-October scores."
        )
    return ShortfallAnchors(
        quantile_T1=float(cal.quantile(SHORTFALL_FILTER_QUANTILES["T1"])),
        quantile_T2=float(cal.quantile(SHORTFALL_FILTER_QUANTILES["T2"])),
        quantile_T3=float(cal.quantile(SHORTFALL_FILTER_QUANTILES["T3"])),
        cal_n_obs=int(len(cal)),
        cal_window_start=cal_start.isoformat(),
        cal_window_end=cal_end.isoformat(),
    )


def compute_shortfall_filter_mask(
    p_treatment: pd.Series,
    anchors: ShortfallAnchors,
    *,
    tag: str,
) -> pd.Series:
    """Boolean mask: True where P(shortfall) ≥ anchor's frozen quantile.

    True ⇒ DROP this bar's trade. NaN scores in the eval window (e.g.,
    2022 pass-through bars) become False (do not drop).
    """
    if tag not in SHORTFALL_FILTER_QUANTILES:
        raise ValueError(
            f"tag must be one of {list(SHORTFALL_FILTER_QUANTILES)}, got {tag}"
        )
    cut = {
        "T1": anchors.quantile_T1,
        "T2": anchors.quantile_T2,
        "T3": anchors.quantile_T3,
    }[tag]
    mask = p_treatment >= cut
    return mask.fillna(False).rename(f"shortfall_filter_{tag.lower()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cycle 4 Phase C (conditional on Phase B passing): "
                    "generate boolean filter sidecars at top {1%, 5%, 10%} "
                    "of calibration-window P(shortfall|TREATMENT).",
    )
    parser.add_argument("--predictions", type=Path,
                        default=DEFAULT_PREDICTIONS_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.predictions)
    if "p_treatment" not in df.columns:
        raise KeyError(f"{args.predictions} missing 'p_treatment' column")
    p_treatment = df["p_treatment"]

    anchors = compute_shortfall_anchors(p_treatment)
    logger.info(
        "Cal-window anchors (n=%d, %s -> %s):",
        anchors.cal_n_obs, anchors.cal_window_start, anchors.cal_window_end,
    )
    logger.info("  T1 (top-1%%) = %.4f", anchors.quantile_T1)
    logger.info("  T2 (top-5%%) = %.4f", anchors.quantile_T2)
    logger.info("  T3 (top-10%%) = %.4f", anchors.quantile_T3)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_paths: dict[str, Path] = {}
    for tag in ("T1", "T2", "T3"):
        mask = compute_shortfall_filter_mask(p_treatment, anchors, tag=tag)
        out_path = args.out_dir / f"BTC_v2_shortfall_{tag}_filter.parquet"
        if out_path.exists() and not args.force:
            raise FileExistsError(
                f"{out_path} already exists. Use --force to overwrite."
            )
        write_filter_sidecar(mask, out_path)
        sidecar_paths[tag] = out_path

    print()
    print("=" * 70)
    print(" Cycle 4 Phase C: shortfall filter sidecars frozen")
    print("=" * 70)
    for tag, p in sidecar_paths.items():
        loaded = pd.read_parquet(p)
        n = len(loaded)
        n_drop = int(loaded["drop_trade"].sum())
        print(f"  {tag} ({SHORTFALL_FILTER_QUANTILES[tag]:.0%} cut): {p}")
        print(f"        n_bars={n:,}, n_drop={n_drop:,} ({100.0 * n_drop / max(n, 1):.2f}%)")
    print("=" * 70)
    print(" Next: PYTHONPATH=. python scripts/evaluate_shortfall_filter_gate.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
