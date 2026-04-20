"""Variance Risk Premium features — Session H Candidate C.

Motivation
----------
Variance Risk Premium (VRP = IV² − RV²) is the classic state variable
in the vol-risk literature (Carr-Wu 2009, RFS; Almeida-Grith-Miftachov
2024, arXiv:2410.15195). Intuitively: when the market pays a premium
for vol protection (IV > RV), realized vol tends to mean-revert
downward; when realized vol outruns implied (negative VRP), that
signals a regime transition ahead.

Production already includes ``svi_atm_1d`` (1-day ATM IV level) and
HAR-PD realized-variance kernels as SEPARATE features. But the LGBM
does not trivially recover their DIFFERENCE — the VRP signal is a
nonlinear combination that the tree split can't approximate cleanly
with a handful of depth levels. Supplying VRP as an explicit feature
makes this regime indicator directly usable.

For our 15-min BTC-vol forecasting target, we conjecture that VRP
provides information orthogonal to HAR-PD + svi_atm_1d because:
  * It is a SPREAD (not a level), capturing the IV-RV regime
    disagreement that neither input expresses alone.
  * Almeida-Grith-Miftachov 2024 documents regime flips in BTC VRP
    across 2021-2024 — directly analogous to our 8-month substrate
    regime breakdown.

Features emitted (5 total to parity-match HAR-PD's 6 features)
--------------------------------------------------------------
All features on the 15-min gauntlet grid (end-stamped, info ≤ T):
  * ``iv_var_1m``       = (svi_atm_1m / 100)²     [annualized variance]
  * ``rv_var_30d``      = rolling 30-day RV annualized variance
  * ``vrp_30d``         = iv_var_1m − rv_var_30d  [primary VRP signal]
  * ``vrp_30d_z30d``    = 30-day rolling z-score of vrp_30d
  * ``vrp_30d_d1d``     = 24-hour change of vrp_30d (.diff(96))

Data source
-----------
Sidecar parquet at ``vol_arb/output/vrp_features_15m.parquet``, built
by ``python -m vol_arb.build_vrp_sidecar`` from:
  * ``vol_arb/output/svi_features_15m.parquet`` (svi_atm_1m)
  * ``vol_arb/output/BTC_vol_arb_svi.parquet`` (tsrv)

Leakage safety
--------------
Each sidecar row T uses only data strictly ≤ T:
  * svi_atm_1m is fit to options trading up to T (end-stamped, same
    convention as svi_atm_1d in production).
  * tsrv at T is a per-bar variance estimate from prices in [T-15m, T].
  * 30-day rolling RV mean uses trailing 30 days of tsrv values.
  * 30-day rolling z-score uses trailing 30 days of VRP values.
  * .diff(96) on 15m grid = 24h back, strictly past.

The loader at ``vol_model/signal_research.py`` applies a +15m index
shift (same convention as SVI surface / Hawkes-OFI / rough-Heston) so
at training decision time T+15m the feature is ≥15m stale.

``compute_vrp_features`` is a pure projection of the sidecar — must
pass ``probe_causality_by_mutation``. See ``tests/test_features_vrp.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def compute_vrp_features(sidecar: pd.DataFrame) -> pd.DataFrame:
    """Project the VRP sidecar into the gauntlet-facing 5-column frame.

    The sidecar already carries the 5 features; this function validates
    them and reorders to the canonical ``VRP_FEATURES`` order. Exists as
    a named entry point so the gauntlet depends on a contract rather
    than the raw parquet schema.

    Parameters
    ----------
    sidecar
        Full 15-minute VRP sidecar from
        ``vol_arb/output/vrp_features_15m.parquet``. Must contain all
        ``VRP_FEATURES`` columns with a DatetimeIndex.

    Returns
    -------
    DataFrame indexed by the sidecar's DatetimeIndex with columns in
    ``VRP_FEATURES`` order, all float64. NaN rows preserved.

    Notes
    -----
    Strictly causal by construction — the sidecar uses only past IV
    and TSRV observations; no transform here can introduce leakage.
    """
    if not isinstance(sidecar.index, pd.DatetimeIndex):
        raise TypeError("sidecar must have a DatetimeIndex")

    missing = [c for c in VRP_FEATURES if c not in sidecar.columns]
    if missing:
        raise KeyError(
            f"sidecar missing VRP columns: {missing}. "
            f"Regenerate via `python -m vol_arb.build_vrp_sidecar`."
        )

    out = pd.DataFrame(index=sidecar.index)
    for col in VRP_FEATURES:
        out[col] = sidecar[col].astype(float)
    return out[list(VRP_FEATURES)]


def describe_coverage(sidecar: pd.DataFrame) -> pd.Series:
    """Coverage diagnostic: per-column non-NaN fraction in the sidecar."""
    feats = compute_vrp_features(sidecar)
    return feats.notna().mean()
