"""Build the Variance Risk Premium (VRP) sidecar for Session H Candidate C.

Background
----------
Variance Risk Premium (Carr-Wu 2009, RFS; Almeida-Grith-Miftachov 2024,
arXiv:2410.15195) is the difference between ex-ante implied variance and
ex-post realized variance:

    VRP_tau = IV_tau² - RV_tau²

where IV_tau is the annualized implied volatility at maturity tau (from
option prices) and RV_tau² is the annualized realized variance over a
comparable past window.

VRP is a classic "state variable" in the volatility-risk literature:
  * **Positive VRP** = market pays a premium for vol protection (implied
    > realized). Historically associated with CALM regimes, mean-
    reverting vol.
  * **Negative VRP** = realized vol has outrun implied. Historically
    associated with CRISIS regimes or regime-flip onsets (vol expansion
    ahead).

Almeida-Grith-Miftachov 2024 documents that BTC VRP has flipped sign
multiple times across 2021-2024, with the low-vol regime producing
positive VRP and the high-vol regime producing compressed or negative
VRP — implying VRP is a useful regime-state variable for vol
forecasting.

For our 15-min BTC-vol forecasting target, we conjecture that VRP
provides information that:
  * HAR-PD alone misses (HAR-PD kernels are path-dependent on realized
    returns; they don't see IV's risk-premium signal).
  * svi_atm_1d alone misses (it's a LEVEL of IV, not the IV-vs-RV
    SPREAD. The spread is a different regime indicator — calm-IV with
    calm-RV vs calm-IV with elevated-RV are opposite regimes and
    require the SPREAD to distinguish).

Feature construction (5 columns, parity-match HAR-PD's 6)
---------------------------------------------------------
At every 15-minute grid timestamp T (end-stamped, same as other
options sidecars):
  1. ``iv_var_1m``        = (svi_atm_1m / 100)²     [annualized variance]
  2. ``rv_var_30d``       = mean(tsrv over [T-30d, T]) × 96 × 365.25
                            [annualized variance, sum over rolling window]
  3. ``vrp_30d``          = iv_var_1m − rv_var_30d  [the signal]
  4. ``vrp_30d_z30d``     = 30-day rolling z-score of vrp_30d
  5. ``vrp_30d_d1d``      = vrp_30d.diff(96) on 15m grid (24h change)

Data sources
-----------
  * IV: ``vol_arb/output/svi_features_15m.parquet`` (``svi_atm_1m``
    column; percent units → divided by 100 for decimal annualized vol).
  * RV: ``vol_arb/output/BTC_vol_arb_svi.parquet`` (``tsrv`` column;
    per-15-minute variance estimate).

Leakage safety
--------------
1. ``svi_atm_1m[T]`` is fit to options trading up to time T (production
   pattern, same as svi_atm_1d). End-stamped.
2. ``tsrv[T]`` is an estimate over [T-15m, T], strictly past.
3. Rolling mean over [T-30d, T] uses only past tsrv values (pandas
   ``rolling`` with default ``closed="right"`` and ``min_periods``).
4. ``.diff(96)`` on 15m grid = 24 hours back, strictly past.
5. Rolling z-score uses trailing 30-day window of VRP values, all past.

The sidecar emits end-stamped rows (feature[T] uses info ≤ T). The
loader at ``vol_model/signal_research.py`` applies a +15m index shift
(same convention as SVI surface / Hawkes-OFI / rough-Heston) so at
training decision time T+15m the feature is ≥15m stale — conservative.

Outputs
-------
- ``vol_arb/output/vrp_features_15m.parquet`` (5 columns, 15m grid)

CLI
---
``python -m vol_arb.build_vrp_sidecar``   (fast: ~5 seconds)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("vrp_sidecar")


SVI_PATH = Path("vol_arb/output/svi_features_15m.parquet")
TSRV_PATH = Path("vol_arb/output/BTC_vol_arb_svi.parquet")  # main sidecar w/ tsrv
OUTPUT_DIR = Path("vol_arb/output")
FEATURES_15M_PATH = OUTPUT_DIR / "vrp_features_15m.parquet"

# Annualization: (15m bars per day) × (days per year) = variance scaling
_ANNUAL_BARS_PER_DAY = 96
_DAYS_PER_YEAR = 365.25
_ANNUAL_SCALE = _ANNUAL_BARS_PER_DAY * _DAYS_PER_YEAR  # 35064

# 30-day window in 15m bars
_BARS_30D = 30 * _ANNUAL_BARS_PER_DAY  # 2880
# Z-score warmup: require at least 10 days of data
_ZSCORE_MIN_PERIODS = 10 * _ANNUAL_BARS_PER_DAY  # 960


VRP_LEVEL_FEATURES: tuple[str, ...] = (
    "iv_var_1m",
    "rv_var_30d",
    "vrp_30d",
    "vrp_30d_z30d",
)

VRP_DELTA_FEATURES: tuple[str, ...] = (
    "vrp_30d_d1d",
)

VRP_FEATURES: tuple[str, ...] = VRP_LEVEL_FEATURES + VRP_DELTA_FEATURES


def _compute_vrp_features(
    svi: pd.DataFrame,
    tsrv: pd.DataFrame,
) -> pd.DataFrame:
    """Compute VRP features on the 15m grid from aligned SVI + TSRV frames.

    Parameters
    ----------
    svi
        DataFrame indexed on 15m grid, with ``svi_atm_1m`` column (percent
        annualized vol).
    tsrv
        DataFrame indexed on same 15m grid, with ``tsrv`` column (per-bar
        realized variance).

    Returns
    -------
    DataFrame with 5 VRP feature columns on the intersected 15m grid.
    Early rows (during rolling warmup) are NaN on the rolling-dependent
    columns.
    """
    # Intersect indices — both should already be on the same grid, but
    # be defensive against bucket-boundary differences
    common = svi.index.intersection(tsrv.index).sort_values()
    logger.info(
        "intersecting grids: svi=%d tsrv=%d common=%d",
        len(svi), len(tsrv), len(common),
    )
    if len(common) == 0:
        raise ValueError("svi and tsrv have no common 15m grid timestamps")

    iv_pct = svi.loc[common, "svi_atm_1m"].astype(float)
    rv_bar = tsrv.loc[common, "tsrv"].astype(float)

    # IV in decimal annualized variance
    iv_var = (iv_pct / 100.0) ** 2

    # 30-day rolling annualized RV variance.
    # `rolling(..., closed="right")` (default) uses trailing observations
    # up to and including the current bar — strictly past-or-present.
    rv_var = rv_bar.rolling(
        window=_BARS_30D, min_periods=max(1, _BARS_30D // 4),
    ).mean() * _ANNUAL_SCALE

    vrp_30d = iv_var - rv_var

    # Rolling 30-day z-score of VRP itself
    vrp_roll = vrp_30d.rolling(
        window=_BARS_30D, min_periods=_ZSCORE_MIN_PERIODS,
    )
    vrp_mean = vrp_roll.mean()
    vrp_std = vrp_roll.std(ddof=0)
    safe_std = vrp_std.where(vrp_std > 1e-10, np.nan)
    vrp_z30d = (vrp_30d - vrp_mean) / safe_std

    # 24-hour change on 15m grid: .diff(96)
    vrp_d1d = vrp_30d.diff(_ANNUAL_BARS_PER_DAY)

    feats = pd.DataFrame(
        {
            "iv_var_1m": iv_var,
            "rv_var_30d": rv_var,
            "vrp_30d": vrp_30d,
            "vrp_30d_z30d": vrp_z30d,
            "vrp_30d_d1d": vrp_d1d,
        },
        index=common,
    )
    return feats[list(VRP_FEATURES)]


def build_sidecar(
    svi_path: Path = SVI_PATH,
    tsrv_path: Path = TSRV_PATH,
    output_path: Path = FEATURES_15M_PATH,
) -> pd.DataFrame:
    """Build + write the 15m VRP sidecar."""
    if not svi_path.exists():
        raise FileNotFoundError(f"SVI source not found: {svi_path}")
    if not tsrv_path.exists():
        raise FileNotFoundError(f"TSRV source not found: {tsrv_path}")

    logger.info("reading SVI source: %s", svi_path)
    svi = pd.read_parquet(svi_path, columns=["svi_atm_1m"])
    logger.info("reading TSRV source: %s", tsrv_path)
    tsrv = pd.read_parquet(tsrv_path, columns=["tsrv"])

    # Sanity: both should be tz-aware UTC DatetimeIndex
    for name, df in [("svi", svi), ("tsrv", tsrv)]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} index must be DatetimeIndex, got {type(df.index)}")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    feats = _compute_vrp_features(svi, tsrv)
    cov = feats.notna().mean()
    logger.info("feature coverage per column:\n%s", cov.to_string())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(output_path)
    logger.info("wrote %s (%d rows)", output_path, len(feats))
    return feats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--svi-path", default=str(SVI_PATH))
    parser.add_argument("--tsrv-path", default=str(TSRV_PATH))
    parser.add_argument("--output-path", default=str(FEATURES_15M_PATH))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    build_sidecar(
        svi_path=Path(args.svi_path),
        tsrv_path=Path(args.tsrv_path),
        output_path=Path(args.output_path),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
