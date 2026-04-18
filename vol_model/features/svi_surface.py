"""SVI surface-level features — Session E Candidate A.

Motivation
----------
Production (`pruned_svi_har_pd_v1`) uses a single scalar from the SVI
constant-maturity sidecar: ``svi_atm_1d``. But the same sidecar
(``vol_arb/output/svi_features_15m.parquet``) also contains seven other
surface-structural features that are currently IGNORED by the LightGBM
stage:

    svi_atm_1w, svi_atm_2w, svi_atm_1m      -- longer-tenor ATM levels
    svi_25d_skew_1w, svi_25d_skew_1m        -- 25-delta skew per tenor
    svi_term_slope_1w_2w, svi_term_slope_1w_1m   -- term-structure slopes

These capture three structural channels that a single ATM scalar
collapses: term-structure (expected vol path by tenor), skew
(directional risk premium / crash probability), and butterfly/
curvature indirectly via the skew differences. Cont & Da Fonseca 2002
provides the foundational PCA-of-IV-surface result; Magadini et al.
2024 (SSRN:4899357) documents crypto SVI calibration stability.

On top of the seven levels, we add four **1-hour-delta** features
computed via ``.diff(4)`` on the pre-shift raw sidecar (at 15-min
cadence, 4 bars = 1h). Deltas capture regime transitions that a level
snapshot misses. The four delta-tracked features are the ones most
plausibly regime-sensitive:

    svi_atm_1w, svi_25d_skew_1m, svi_term_slope_1w_1m, svi_atm_spot

Total: 11 new features. No new data source; no new parquet needed —
the raw levels are already in the sidecar, and the deltas are derived
from them by strict past-only diffs.

Leakage safety
--------------
``.diff(periods=n)`` uses only past values; zero leakage. The compute
function is deterministic and reads no state outside ``raw``. The
caller is responsible for applying the standard +15-min alignment
shift (matches ``_load_svi_atm_1d``) AFTER computing the deltas.

This module's ``compute_svi_surface_features`` MUST pass
``probe_causality_by_mutation`` with the default mask ``idx > T``.
See ``tests/test_features_svi_surface.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# The seven level features already present in the SVI sidecar but
# unused by the production LightGBM stage.
SVI_SURFACE_LEVEL_FEATURES: tuple[str, ...] = (
    "svi_atm_1w",
    "svi_atm_2w",
    "svi_atm_1m",
    "svi_25d_skew_1w",
    "svi_25d_skew_1m",
    "svi_term_slope_1w_2w",
    "svi_term_slope_1w_1m",
)

# Four columns whose 1-hour deltas are added as regime-transition features.
# Chosen a priori (not by scanning OOS for which deltas help): one per
# structural channel — term level (svi_atm_1w), skew (svi_25d_skew_1m),
# term slope (svi_term_slope_1w_1m), spot vol (svi_atm_spot).
SVI_SURFACE_DELTA_SOURCE: tuple[str, ...] = (
    "svi_atm_1w",
    "svi_25d_skew_1m",
    "svi_term_slope_1w_1m",
    "svi_atm_spot",
)

SVI_SURFACE_DELTA_FEATURES: tuple[str, ...] = tuple(
    f"{c}_d1h" for c in SVI_SURFACE_DELTA_SOURCE
)

SVI_SURFACE_FEATURES: tuple[str, ...] = (
    SVI_SURFACE_LEVEL_FEATURES + SVI_SURFACE_DELTA_FEATURES
)

# 4 × 15 min = 1 h. Parameter set a priori.
_DELTA_LAG_BARS: int = 4


def compute_svi_surface_features(
    raw_sidecar: pd.DataFrame,
    *,
    delta_lag_bars: int = _DELTA_LAG_BARS,
) -> pd.DataFrame:
    """Build the 11-column SVI surface feature frame from the raw SVI
    sidecar (pre +15m alignment shift).

    Parameters
    ----------
    raw_sidecar
        The 15-min SVI sidecar as produced by
        ``vol_arb.run_pipeline --svi``. Must contain the columns in
        ``SVI_SURFACE_LEVEL_FEATURES`` and ``SVI_SURFACE_DELTA_SOURCE``.
        Index is a DatetimeIndex at 15-min cadence.
    delta_lag_bars
        Lag (in 15-min bars) for the rolling-change features. Default 4
        (= 1 hour). Fixed at module level; kwarg is only for tests.

    Returns
    -------
    DataFrame with the same DatetimeIndex as ``raw_sidecar`` and
    columns ``SVI_SURFACE_FEATURES`` (7 levels + 4 deltas). NaN rows are
    preserved — downstream ``dropna`` decides the warmup policy.

    Notes
    -----
    This function is strictly causal: every delta uses ``.diff(n)`` which
    is equivalent to ``x - x.shift(n)`` with ``shift(n)`` referring to
    PAST values. No forward-looking state is introduced. The caller is
    responsible for applying the +15m alignment shift used elsewhere in
    the pipeline.
    """
    if not isinstance(raw_sidecar.index, pd.DatetimeIndex):
        raise TypeError("raw_sidecar must have a DatetimeIndex")

    missing_levels = [
        c for c in SVI_SURFACE_LEVEL_FEATURES if c not in raw_sidecar.columns
    ]
    if missing_levels:
        raise KeyError(
            f"raw_sidecar missing SVI surface level columns: {missing_levels}. "
            f"Regenerate svi_features_15m.parquet via "
            f"`python -m vol_arb.run_pipeline --svi`."
        )
    missing_delta_src = [
        c for c in SVI_SURFACE_DELTA_SOURCE if c not in raw_sidecar.columns
    ]
    if missing_delta_src:
        raise KeyError(
            f"raw_sidecar missing SVI delta-source columns: {missing_delta_src}."
        )

    out = pd.DataFrame(index=raw_sidecar.index)

    # ── 7 level features (copied; no transformation) ────────────────
    for col in SVI_SURFACE_LEVEL_FEATURES:
        out[col] = raw_sidecar[col].astype(float)

    # ── 4 delta features (1-hour change via strict past diff) ───────
    for src_col in SVI_SURFACE_DELTA_SOURCE:
        delta_col = f"{src_col}_d1h"
        # .diff(n) = x - x.shift(n). shift(n) with positive n is LAGGED
        # (past values), so the result at index T is strictly a function
        # of raw[T] and raw[T-n] — both at or before T. No leakage.
        out[delta_col] = raw_sidecar[src_col].astype(float).diff(delta_lag_bars)

    # Order columns predictably (level first, then delta).
    return out[list(SVI_SURFACE_FEATURES)]


def describe_coverage(raw_sidecar: pd.DataFrame) -> pd.Series:
    """Coverage diagnostic: per-column non-NaN fraction in the raw sidecar.

    Used by the loader to warn if any new feature has < 90% coverage
    (which would block enough rows to make the gauntlet unreliable).
    """
    feats = compute_svi_surface_features(raw_sidecar)
    return feats.notna().mean()
