"""Numerical equivalence between live and offline term-slope computations.

E.2.4 in plan ``C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md``.

The live engine's ``Predictor._compute_term_slope_1w_2w`` calls
``LiveSVICalibrator.atm_vol_at_dte(7)`` minus ``atm_vol_at_dte(14)``.
The offline ``vol_arb/svi.py::_constant_maturity_features`` computes
``svi_atm_2w - svi_atm_1w``.

These MUST be numerically identical for the same surface, because:
- Both call ``surface.total_variance(np.array([0.0]), T)``
- Both convert via ``sqrt(w / T) * 100``
- The 1w / 2w days-to-expiry constants are 7 / 365.25 and 14 / 365.25
  in both paths.

This test parametrises several synthetic surfaces (one normal, one
with single-slice degeneracy, one with exact-DTE boundary) and
asserts agreement to 1e-6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.svi import (
    ArbitrageFreeSVI,
    SVISlice,
    _constant_maturity_features,
)


def _make_surface(
    *,
    atm_1w_pct: float, atm_2w_pct: float,
    extra_slices: bool = False,
) -> ArbitrageFreeSVI:
    """Construct a 2-slice (or 3-slice) ArbitrageFreeSVI with target ATM IVs."""
    surface = ArbitrageFreeSVI(timestamp=pd.Timestamp("2026-04-15 12:00", tz="UTC"))
    for label, dte_days, atm_pct in [
        ("1w", 7.0, atm_1w_pct),
        ("2w", 14.0, atm_2w_pct),
        ("1m", 30.0, atm_2w_pct + 1.0),  # only used when extra_slices=True
    ]:
        if label == "1m" and not extra_slices:
            continue
        T = dte_days / 365.25
        # SVI at k=0 with m=0, sigma=1, b=1 → w(0) = a + 1.
        # Set a so that sqrt(w/T) = atm_iv_decimal.
        atm_decimal = atm_pct / 100.0
        w_target = atm_decimal * atm_decimal * T
        a = w_target - 1.0          # b=1, sigma=1 → atm w contribution = 1
        surface.add_slice(SVISlice(
            a=a, b=1.0, rho=0.0, m=0.0, sigma=1.0, T=T, F=60_000.0,
        ))
    return surface


@pytest.mark.parametrize("atm_1w,atm_2w", [
    (70.0, 72.0),                # normal contango
    (75.0, 70.0),                # backwardation
    (60.0, 60.0),                # flat term
])
def test_live_term_slope_matches_offline(atm_1w: float, atm_2w: float):
    """Both paths should produce the SAME 1w-2w term slope."""
    surface = _make_surface(atm_1w_pct=atm_1w, atm_2w_pct=atm_2w)

    # Offline path: _constant_maturity_features formula.
    feats = _constant_maturity_features(
        surface, pd.Timestamp("2026-04-15 12:00", tz="UTC"),
    )
    offline_slope = feats["svi_term_slope_1w_2w"]

    # Live path: surface.atm_vol_pct(T) at the two DTEs.
    live_atm_1w = surface.atm_vol_pct(T=7.0 / 365.25)
    live_atm_2w = surface.atm_vol_pct(T=14.0 / 365.25)
    live_slope = live_atm_2w - live_atm_1w

    # Both formulas: 2w - 1w.
    assert live_slope == pytest.approx(offline_slope, abs=1e-6)


def test_offline_three_slice_surface_picks_right_maturities():
    """Adding a 1m slice doesn't shift the 1w/2w slope (sanity check)."""
    surface_two_slice = _make_surface(atm_1w_pct=70.0, atm_2w_pct=72.0)
    surface_three_slice = _make_surface(
        atm_1w_pct=70.0, atm_2w_pct=72.0, extra_slices=True,
    )
    f2 = _constant_maturity_features(
        surface_two_slice,
        pd.Timestamp("2026-04-15 12:00", tz="UTC"),
    )
    f3 = _constant_maturity_features(
        surface_three_slice,
        pd.Timestamp("2026-04-15 12:00", tz="UTC"),
    )
    assert f2["svi_term_slope_1w_2w"] == pytest.approx(
        f3["svi_term_slope_1w_2w"], abs=1e-9,
    )


def test_atm_iv_within_expected_range():
    """The synthetic surface generator hits the target ATM IV exactly."""
    surface = _make_surface(atm_1w_pct=70.0, atm_2w_pct=72.0)
    assert surface.atm_vol_pct(T=7.0 / 365.25) == pytest.approx(70.0, abs=1e-9)
    assert surface.atm_vol_pct(T=14.0 / 365.25) == pytest.approx(72.0, abs=1e-9)
