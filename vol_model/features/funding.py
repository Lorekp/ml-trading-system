"""Perp-funding-rate features — Session H Candidate B.

Motivation
----------
Production (``pruned_svi_har_pd_v1``) trains on realized-variance kernels
(HAR-PD) + option-surface IV (svi_atm_1d). Both aggregate price-path
information. Funding rate is a STRUCTURALLY ORTHOGONAL channel: it is
a derivatives-market positioning signal set every 8 hours on Binance
perpetual futures, reflecting the premium that long-side holders pay
short-side holders (or vice-versa) to keep the perp price anchored to
spot.

BIS Working Paper 1087 (Bernardi-Koudijs-Ranaldo 2024, "Crypto Carry")
documents that the annualized carry (= 3 × funding rate) has delivered
Sharpe 6.45 over 2020-2024 and its time-series variation is driven by
sentiment/demand regimes, not realized vol. This implies funding
encodes a state variable that is BY CONSTRUCTION orthogonal to any
realized-variance feature.

For our 15-min BTC-vol forecasting target, we conjecture that:
  * Extreme |funding| (≥ 30d z-score 2-3) flags positioning crowding
    that historically precedes volatility expansions.
  * Funding sign reversals (capturing regime flips) help distinguish
    crisis-unwind vol (e.g. 2022-11/12 FTX, negative funding) from
    euphoria-driven vol (e.g. 2024-11/12 Trump rally, extreme positive).
  * Neither of these are captured by HAR-PD or svi_atm_1d directly.

Features emitted (5 total to parity-match HAR-PD's 6 features)
-------------------------------------------------------------
All features on the 15-min gauntlet grid via
``merge_asof(direction="backward")`` of the 8h native funding cadence:
  * ``fr_level``        raw funding rate (decimal per 8h)
  * ``fr_abs``          |funding rate| (magnitude, regime-agnostic)
  * ``fr_ema_1d``       3-period EMA on 8h cadence (≈1-day half-life)
  * ``fr_ema_7d``       21-period EMA on 8h cadence (≈7-day)
  * ``fr_zscore_30d``   rolling 90-period (30-day) z-score

Data source
-----------
Sidecar parquet at ``vol_arb/output/funding_features_15m.parquet``,
built by ``python -m vol_arb.build_funding_sidecar`` from
``data/funding-rates/BTCUSDT_funding.parquet`` (2020-01 → 2026-04).
Full 8-month regime substrate coverage (100%).

Leakage safety
--------------
The sidecar's ``merge_asof(direction="backward")`` ensures that at
grid[T], the feature uses the most recent funding observation with
``fundingTime <= T``. No forward information is used. The 8h
rolling/ewm computations use ``min_periods`` explicitly to keep
warmup rows as NaN (no look-ahead during the burn-in period).

Unlike end-stamped options sidecars, no +15m loader shift is applied
— the merge_asof-backward pattern is already conservatively aligned,
and at label-decision time T+15m the feature is ≥15m stale, which is
the desired conservative margin.

``compute_funding_features`` is a pure projection of the sidecar —
must pass ``probe_causality_by_mutation``. See
``tests/test_features_funding.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FUNDING_LEVEL_FEATURES: tuple[str, ...] = (
    "fr_level",
    "fr_abs",
    "fr_ema_1d",
    "fr_ema_7d",
    "fr_zscore_30d",
)

# No delta features — the funding cadence (8h) already captures change
# information through the EMA half-lives; adding `.diff(K)` on the 15m
# grid would mostly emit zero because consecutive 15m bars share the
# same funding value for ~32 bars.
FUNDING_FEATURES: tuple[str, ...] = FUNDING_LEVEL_FEATURES


def compute_funding_features(sidecar: pd.DataFrame) -> pd.DataFrame:
    """Project the funding sidecar into the gauntlet-facing 5-column frame.

    The sidecar already carries the 5 features; this function validates
    them and reorders to the canonical ``FUNDING_FEATURES`` order. It
    exists as a named entry point so the gauntlet depends on a contract
    rather than the raw parquet schema.

    Parameters
    ----------
    sidecar
        Full 15-minute funding sidecar from
        ``vol_arb/output/funding_features_15m.parquet``. Must contain
        all ``FUNDING_FEATURES`` columns with a DatetimeIndex.

    Returns
    -------
    DataFrame indexed by the sidecar's DatetimeIndex with columns in
    ``FUNDING_FEATURES`` order, all float64. NaN rows preserved.

    Notes
    -----
    Strictly causal by construction — the sidecar uses merge_asof-
    backward on the 8h funding cadence, and no transform here can
    introduce leakage.
    """
    if not isinstance(sidecar.index, pd.DatetimeIndex):
        raise TypeError("sidecar must have a DatetimeIndex")

    missing = [c for c in FUNDING_FEATURES if c not in sidecar.columns]
    if missing:
        raise KeyError(
            f"sidecar missing funding columns: {missing}. "
            f"Regenerate via `python -m vol_arb.build_funding_sidecar`."
        )

    out = pd.DataFrame(index=sidecar.index)
    for col in FUNDING_FEATURES:
        out[col] = sidecar[col].astype(float)
    return out[list(FUNDING_FEATURES)]


def describe_coverage(sidecar: pd.DataFrame) -> pd.Series:
    """Coverage diagnostic: per-column non-NaN fraction in the sidecar."""
    feats = compute_funding_features(sidecar)
    return feats.notna().mean()
