"""Hawkes-OFI microstructure features — Session F Candidate B.

Motivation
----------
Production (``pruned_svi_har_pd_v1``) uses OFI-derived LEVEL features: the
magnitude of buy/sell imbalance over fixed windows. What it doesn't capture
is market REFLEXIVITY — the degree to which order flow is self-reinforcing
(endogenous) vs independent (exogenous). A bivariate Hawkes process fit to
trade arrivals gives us direct estimates of both regimes via the branching
ratios.

Features (7 total: 5 levels + 2 deltas)
---------------------------------------
- ``hawkes_mu_buy``, ``hawkes_mu_sell``: exogenous intensities (events/sec).
  High when order flow is news-driven / exogenous; low when endogenous.
- ``hawkes_branching_buy``, ``hawkes_branching_sell``: total branching
  ratios = (alpha_bb + alpha_bs) and (alpha_sb + alpha_ss). Capture
  self/cross-excitation. Near 1 → critical, self-reinforcing regime (vol
  jumps); near 0 → exogenous, mean-reverting regime.
- ``hawkes_asymmetry`` = alpha_bb - alpha_ss: directional bias (buy-side
  reflexivity > sell-side).
- ``hawkes_branching_buy_d1h``, ``hawkes_branching_sell_d1h``: 1-hour
  changes in the branching ratios. Capture regime transitions.

Citations: Pham et al. 2024 (arxiv:2408.03594, Computational Economics 2025);
Filimonov & Sornette 2012; Bacry-Mastromatteo-Muzy 2015.

Data source
-----------
Sidecar parquet at ``vol_arb/output/hawkes_ofi_features_15m.parquet``,
produced by ``python -m vol_arb.build_hawkes_ofi_sidecar``. That script
fits a bivariate exponential-kernel Hawkes via MLE at every 1h bucket on
trailing-4h 1-second-binned BTCUSDT aggTrades (Binance), then forward-fills
to a 15m grid (SVI pattern). Delta features are ``.diff(4)`` on the 15m grid.

Leakage safety
--------------
The Hawkes MLE at bucket timestamp T uses ONLY trades in [T-4h, T). The
forward-fill propagates values from past buckets to the same-or-later 15m
slots. The ``.diff(4)`` delta uses only past values.

``compute_hawkes_ofi_features`` is a pure function of the sidecar — it
must pass ``probe_causality_by_mutation`` with the default mask ``idx > T``.
See ``tests/test_features_hawkes_ofi.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 5 level features copied byte-identically from the sidecar.
HAWKES_OFI_LEVEL_FEATURES: tuple[str, ...] = (
    "hawkes_mu_buy",
    "hawkes_mu_sell",
    "hawkes_branching_buy",
    "hawkes_branching_sell",
    "hawkes_asymmetry",
)

# 2 delta features already present in the 15m sidecar as .diff(4) outputs.
HAWKES_OFI_DELTA_FEATURES: tuple[str, ...] = (
    "hawkes_branching_buy_d1h",
    "hawkes_branching_sell_d1h",
)

HAWKES_OFI_FEATURES: tuple[str, ...] = (
    HAWKES_OFI_LEVEL_FEATURES + HAWKES_OFI_DELTA_FEATURES
)


def compute_hawkes_ofi_features(sidecar: pd.DataFrame) -> pd.DataFrame:
    """Project the Hawkes-OFI sidecar into the gauntlet-facing 7-column frame.

    The sidecar already carries the 7 columns (levels + deltas) with the
    right schema — this function validates them and re-orders. It exists
    as a named entry point so the gauntlet can depend on the contract
    (column names, order, dtypes) rather than on the raw parquet schema.

    Parameters
    ----------
    sidecar
        Full 15-minute Hawkes-OFI sidecar from
        ``vol_arb/output/hawkes_ofi_features_15m.parquet``. Must contain
        all ``HAWKES_OFI_FEATURES`` columns with a DatetimeIndex.

    Returns
    -------
    DataFrame indexed by the sidecar's DatetimeIndex with columns
    ``HAWKES_OFI_FEATURES`` in fixed order, all float64. NaN rows
    preserved (downstream ``dropna`` decides the warmup policy).

    Notes
    -----
    This function is strictly causal by construction: it reads the
    sidecar, which is itself built from strictly-past data (see
    ``vol_arb.build_hawkes_ofi_sidecar``). No transformation introduced
    here can add leakage.
    """
    if not isinstance(sidecar.index, pd.DatetimeIndex):
        raise TypeError("sidecar must have a DatetimeIndex")

    missing = [c for c in HAWKES_OFI_FEATURES if c not in sidecar.columns]
    if missing:
        raise KeyError(
            f"sidecar missing Hawkes-OFI columns: {missing}. "
            f"Regenerate via `python -m vol_arb.build_hawkes_ofi_sidecar`."
        )

    out = pd.DataFrame(index=sidecar.index)
    for col in HAWKES_OFI_FEATURES:
        out[col] = sidecar[col].astype(float)
    return out[list(HAWKES_OFI_FEATURES)]


def describe_coverage(sidecar: pd.DataFrame) -> pd.Series:
    """Coverage diagnostic: per-column non-NaN fraction in the sidecar."""
    feats = compute_hawkes_ofi_features(sidecar)
    return feats.notna().mean()
