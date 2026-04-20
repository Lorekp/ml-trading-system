"""Build the perp-funding-rate feature sidecar for Session H Candidate B.

Background
----------
BIS Working Paper 1087 (Bernardi-Koudijs-Ranaldo 2024, "Crypto Carry")
documents that the 8-hourly funding rate encodes directional crowding +
carry pressure in the perp market, with Sharpe 6.45 on a dedicated
carry-harvesting strategy over 2020-2024.

The researcher's thesis for our 15-min volatility forecasting target:
funding rate is a LATENT STATE variable orthogonal to realized-variance
kernels (HAR-PD) because it reflects derivatives-market positioning
rather than spot-return paths. The conjecture is that large |funding|
or sign-reversals should predict regime transitions that HAR-PD alone
cannot see — e.g. Trump-rally 2024-11/12 had extreme positive funding
with only moderate realized vol, and FTX 2022-11/12 had negative
funding amid panic unwinds.

Feature construction (5 columns, match HAR-PD parity)
-----------------------------------------------------
On the NATIVE 8-hourly funding cadence:
  1. ``fr_level``         raw funding rate (decimal, typical 1e-4 scale)
  2. ``fr_abs``           |fr| (magnitude, regime-agnostic)
  3. ``fr_ema_1d``        3-period exponential moving average (≈ 24h)
  4. ``fr_ema_7d``        21-period EMA (≈ 7 days)
  5. ``fr_zscore_30d``    rolling 90-period (30-day) z-score

Then reindexed to the 15-minute training grid via
``merge_asof(direction="backward")``: at grid timestamp T, the feature
is the most recent funding observation with ``fundingTime <= T``
(strictly causal — funding at T is known AT T because it's computed
from premium index BEFORE T and applied at T).

Leakage safety
--------------
1. The 8h rolling computations use only past funding observations
   (standard pandas ``rolling`` + ``ewm`` with ``min_periods`` set so
   early rows emit NaN rather than peek).
2. ``merge_asof(direction="backward")`` on the 15m training grid
   returns the LATEST funding row with ``fundingTime <= T_bar``. This
   is the strict causal convention (at decision time T_bar, the
   funding rate effective at timestamps ≤ T_bar is known).
3. Unlike options sidecars, no +15m loader shift is needed — the
   merge_asof-backward pattern is already conservatively aligned.
   Feature[T] uses info published strictly ≤ T, so at label-decision
   time T+15m (production convention) the feature is > 15m stale,
   which is the desired conservative margin.

Outputs
-------
- ``vol_arb/output/funding_features_15m.parquet`` (5 columns, 15m grid)

CLI
---
``python -m vol_arb.build_funding_sidecar``  (fast: ~2 seconds)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("funding_sidecar")


FUNDING_PATH = Path("data/funding-rates/BTCUSDT_funding.parquet")
OUTPUT_DIR = Path("vol_arb/output")
FEATURES_15M_PATH = OUTPUT_DIR / "funding_features_15m.parquet"

# Default 15m grid bounds (matches other sidecars: 2021-01 → 2026-03-31 end)
DEFAULT_GRID_START = pd.Timestamp("2021-01-01 00:00", tz="UTC")
DEFAULT_GRID_END = pd.Timestamp("2026-03-31 23:45", tz="UTC")


FUNDING_LEVEL_FEATURES: tuple[str, ...] = (
    "fr_level",
    "fr_abs",
    "fr_ema_1d",
    "fr_ema_7d",
    "fr_zscore_30d",
)


def _compute_funding_features_8h(funding: pd.DataFrame) -> pd.DataFrame:
    """Compute funding features on the NATIVE 8h cadence.

    Parameters
    ----------
    funding
        DataFrame indexed by ``fundingTime`` (tz-aware UTC), with
        ``fundingRate`` column (decimal; 0.0001 = 0.01 % per 8h =
        0.1095 annualized).

    Returns
    -------
    DataFrame with the 5 feature columns, indexed by ``fundingTime``.
    Early rows (during warmup of rolling statistics) are NaN.
    """
    fr = funding["fundingRate"].astype(float)

    feats = pd.DataFrame(index=funding.index)
    feats["fr_level"] = fr
    feats["fr_abs"] = fr.abs()

    # Exponential moving averages. span=3 gives ~1-day half-life
    # (3 × 8h = 24h); span=21 gives ~7-day.
    # `adjust=False` + `min_periods=span` keeps warmup NaN and uses the
    # recursive alpha = 2/(span+1) form — strict past-only aggregation.
    feats["fr_ema_1d"] = fr.ewm(span=3, adjust=False, min_periods=3).mean()
    feats["fr_ema_7d"] = fr.ewm(span=21, adjust=False, min_periods=21).mean()

    # 30-day rolling z-score. 90 funding periods = 30 days exactly.
    # min_periods=30 (10 days) lets us emit in the warmup window, but
    # stable once >= 90 periods are in-window.
    roll = fr.rolling(window=90, min_periods=30)
    roll_mean = roll.mean()
    roll_std = roll.std(ddof=0)
    # Guard against zero std at very calm periods.
    safe_std = roll_std.where(roll_std > 1e-10, np.nan)
    feats["fr_zscore_30d"] = (fr - roll_mean) / safe_std

    return feats


def _reindex_to_15m_causal(
    feats_8h: pd.DataFrame,
    grid_start: pd.Timestamp = DEFAULT_GRID_START,
    grid_end: pd.Timestamp = DEFAULT_GRID_END,
) -> pd.DataFrame:
    """Align 8h feature frame onto a 15-minute grid, strictly causal.

    Uses ``merge_asof(direction="backward")`` semantics: at grid[T], the
    value is the most recent funding row with ``fundingTime <= T``. This
    is the *conservative* causal convention — funding is known AT
    fundingTime and remains known thereafter.
    """
    # Target 15m grid
    grid_15m = pd.date_range(grid_start, grid_end, freq="15min")

    # Sort funding features by their native 8h timestamps (required by merge_asof)
    feats_sorted = feats_8h.sort_index()

    # merge_asof requires both frames sorted on the "on" column
    target = pd.DataFrame({"timestamp": grid_15m})
    source = feats_sorted.reset_index().rename(columns={feats_sorted.index.name or "index": "ftime"})
    # Ensure the source timestamp column is named consistently
    if "fundingTime" in source.columns:
        source = source.rename(columns={"fundingTime": "ftime"})

    merged = pd.merge_asof(
        left=target.sort_values("timestamp"),
        right=source.sort_values("ftime"),
        left_on="timestamp",
        right_on="ftime",
        direction="backward",
        # tolerance: up to 9h (slightly more than an 8h interval) so a
        # missing single funding period doesn't create long NaN runs
        tolerance=pd.Timedelta(hours=9),
    )
    merged = merged.set_index("timestamp")
    merged.index.name = None
    return merged[list(FUNDING_LEVEL_FEATURES)]


def build_sidecar(
    funding_path: Path = FUNDING_PATH,
    output_path: Path = FEATURES_15M_PATH,
    grid_start: pd.Timestamp = DEFAULT_GRID_START,
    grid_end: pd.Timestamp = DEFAULT_GRID_END,
) -> pd.DataFrame:
    """Build + write the 15m funding sidecar.

    Returns the feature frame.
    """
    if not funding_path.exists():
        raise FileNotFoundError(f"funding source not found: {funding_path}")
    logger.info("reading funding source: %s", funding_path)
    funding = pd.read_parquet(funding_path)

    # Sanity checks
    if "fundingRate" not in funding.columns:
        raise ValueError("funding parquet missing 'fundingRate' column")
    if not isinstance(funding.index, pd.DatetimeIndex):
        raise ValueError("funding parquet index must be a DatetimeIndex")

    # Drop duplicates that occasionally exist at exchange-side boundary jitter
    funding = funding[~funding.index.duplicated(keep="first")].sort_index()
    logger.info(
        "funding: %d rows, [%s, %s]",
        len(funding), funding.index.min(), funding.index.max(),
    )

    feats_8h = _compute_funding_features_8h(funding)
    logger.info(
        "computed 8h features: %d rows, warmup NaN: %d",
        len(feats_8h), int(feats_8h.isna().any(axis=1).sum()),
    )

    feats_15m = _reindex_to_15m_causal(feats_8h, grid_start, grid_end)
    logger.info(
        "reindexed to 15m grid: %d rows, coverage per col:\n%s",
        len(feats_15m), feats_15m.notna().mean().to_string(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats_15m.to_parquet(output_path)
    logger.info("wrote %s", output_path)
    return feats_15m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--funding-path", default=str(FUNDING_PATH))
    parser.add_argument("--output-path", default=str(FEATURES_15M_PATH))
    parser.add_argument("--start", default=str(DEFAULT_GRID_START))
    parser.add_argument("--end", default=str(DEFAULT_GRID_END))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    build_sidecar(
        funding_path=Path(args.funding_path),
        output_path=Path(args.output_path),
        grid_start=pd.Timestamp(args.start, tz="UTC"),
        grid_end=pd.Timestamp(args.end, tz="UTC"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
