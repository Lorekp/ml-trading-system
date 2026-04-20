"""Rough-Heston latent-parameter features — Session F Candidate C.

Motivation
----------
Production (``pruned_svi_har_pd_v1``) uses the SVI-fitted ATM IV scalar
``svi_atm_1d`` as the only options-surface input. This is a SURFACE-LEVEL
feature — it doesn't tell the model about the ROUGH dynamics implied by
the smile curvature and term-structure.

Rough-Heston (El Euch-Rosenbaum 2019) is the canonical rough-vol model
for pricing: the variance follows a fractional Volterra SDE with Hurst
exponent H in (0, 1/2). Calibrating rough-Heston to the full options
panel at each 15-min bar extracts 4 latent-state parameters that capture
structurally-different information than a single ATM IV:

  * ``rh_v0``: instantaneous spot variance — a LATENT-STATE estimate of
    realized-vol expectation, orthogonal to a forward-looking ATM-IV.
  * ``rh_hurst``: the Hurst roughness of variance. Small H means the
    variance path is highly irregular (vol-of-vol is high at short time
    scales); large H means smooth. Crypto typically ~0.05-0.15.
  * ``rh_eta``: vol-of-vol (diffusion scale).
  * ``rh_rho``: spot-vol correlation. Sign + magnitude of leverage /
    crash-risk premium.

Plus 4 one-hour-delta features. Total: 8 features.

Data source
-----------
Sidecar parquet at ``vol_arb/output/rough_heston_features_15m.parquet``,
produced by ``python -m vol_arb.build_rough_heston_sidecar``. Full rough-
Heston calibration (El Euch-Rosenbaum fractional Riccati via
Diethelm-Ford-Freed Adams scheme, Heston Pi_1/Pi_2 pricing, SLSQP + LM
polish with differential-evolution cold-start) to the Deribit BTC options
trade panel in a **backward-only 30-minute window** ``[T - 30min, T)`` at
each 1h bucket, warm-started across buckets.

Leakage safety
--------------
Each calibration at bucket T uses ONLY option trades in ``[T-30min, T)``
(strictly past). The 15m forward-fill propagates values from past buckets
up to 45 minutes forward. The ``.diff(4)`` delta uses only past values.
The loader applies a ``+15m`` index shift, identical to SVI-ATM / SVI
surface / Hawkes-OFI convention; ensures strict causality with an
additional 15-minute safety margin.

``compute_rough_heston_features`` is a pure projection of the sidecar —
it must pass ``probe_causality_by_mutation``. See
``tests/test_features_rough_heston.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ROUGH_HESTON_LEVEL_FEATURES: tuple[str, ...] = (
    "rh_v0",
    "rh_hurst",
    "rh_eta",
    "rh_rho",
)

ROUGH_HESTON_DELTA_FEATURES: tuple[str, ...] = tuple(
    f"{c}_d1h" for c in ROUGH_HESTON_LEVEL_FEATURES
)

ROUGH_HESTON_FEATURES: tuple[str, ...] = (
    ROUGH_HESTON_LEVEL_FEATURES + ROUGH_HESTON_DELTA_FEATURES
)


def compute_rough_heston_features(sidecar: pd.DataFrame) -> pd.DataFrame:
    """Project the rough-Heston sidecar into the gauntlet-facing 8-column frame.

    The sidecar already carries the 8 columns (4 levels + 4 deltas) with
    the expected schema. This function validates them and re-orders them
    to the canonical ``ROUGH_HESTON_FEATURES`` order. It exists as a named
    entry point so the gauntlet depends on a contract rather than the
    raw parquet schema.

    Parameters
    ----------
    sidecar
        Full 15-minute rough-Heston sidecar from
        ``vol_arb/output/rough_heston_features_15m.parquet``. Must contain
        all ``ROUGH_HESTON_FEATURES`` columns with a DatetimeIndex.

    Returns
    -------
    DataFrame indexed by the sidecar's DatetimeIndex with columns in
    ``ROUGH_HESTON_FEATURES`` order, all float64. NaN rows preserved.

    Notes
    -----
    Strictly causal by construction: the sidecar is built from strictly-
    past data (see ``vol_arb.build_rough_heston_sidecar``); no transform
    here can introduce leakage.
    """
    if not isinstance(sidecar.index, pd.DatetimeIndex):
        raise TypeError("sidecar must have a DatetimeIndex")

    missing = [c for c in ROUGH_HESTON_FEATURES if c not in sidecar.columns]
    if missing:
        raise KeyError(
            f"sidecar missing rough-Heston columns: {missing}. "
            f"Regenerate via `python -m vol_arb.build_rough_heston_sidecar`."
        )

    out = pd.DataFrame(index=sidecar.index)
    for col in ROUGH_HESTON_FEATURES:
        out[col] = sidecar[col].astype(float)
    return out[list(ROUGH_HESTON_FEATURES)]


def describe_coverage(sidecar: pd.DataFrame) -> pd.Series:
    """Coverage diagnostic: per-column non-NaN fraction in the sidecar."""
    feats = compute_rough_heston_features(sidecar)
    return feats.notna().mean()
