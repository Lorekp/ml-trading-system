"""Regression tests for the Binance 32-asset inverse-vol sizing fix
(2026-05-18).

The live paper engine lost -$15k; ~77% of it was a sizing bug — the
per-asset inverse-vol overlay failed OPEN. ``compute_vol_mult`` returned
1.0 (full notional) whenever an asset's vol series was missing, and the
clip floor (0.3) was too high for the live universe's ~27x vol dispersion.

These tests pin the fix across all three sizing paths so the regression
cannot silently return:

  * ``live.portfolio.PortfolioManager.compute_vol_mult``
  * ``live.strategies.s8_sym15.S8Sym15Strategy.compute_vol_mult``
  * ``signal_engine.pipeline.apply_per_asset_vol_overlay``

The central invariant: an asset whose volatility is UNKNOWN must be sized
at the clip FLOOR (smallest size), never 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live.config import PortfolioConfig, VolOverlayConfig
from live.portfolio import PortfolioManager
from live.strategies.s8_sym15 import S8Sym15Strategy
from signal_engine.pipeline import apply_per_asset_vol_overlay


def _daily_vol(value: float, n: int = 40) -> pd.Series:
    """A flat daily-vol series of `value`, daily-indexed, ending 2024-02-09."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series([value] * n, index=idx)


# ── VolOverlayConfig — pin the corrected band ────────────────────────────
def test_vol_overlay_config_band_is_fixed() -> None:
    """The fix lowered the floor 0.3 -> 0.10 and raised the cap 1.5 -> 2.0."""
    vo = VolOverlayConfig()
    assert vo.floor == 0.10, "clip floor must be 0.10 (was 0.3 — too high)"
    assert vo.cap == 2.0
    assert vo.target_vol == 0.025
    assert vo.screen_max_vol == 0.20


# ── PortfolioManager.compute_vol_mult ────────────────────────────────────
def _pm() -> PortfolioManager:
    return PortfolioManager(PortfolioConfig(), 200_000.0, VolOverlayConfig())


def test_pm_failsafe_missing_series() -> None:
    """Unknown asset -> clip FLOOR, never 1.0 (the live bug)."""
    pm = _pm()
    mult = pm.compute_vol_mult("DENT", {})           # asset absent
    assert mult == pytest.approx(0.10)
    assert mult != 1.0


def test_pm_failsafe_nonfinite_vol() -> None:
    pm = _pm()
    assert pm.compute_vol_mult("X", {"X": _daily_vol(np.nan)}) == pytest.approx(0.10)
    assert pm.compute_vol_mult("X", {"X": _daily_vol(0.0)}) == pytest.approx(0.10)
    assert pm.compute_vol_mult("X", {"X": pd.Series([], dtype=float)}) == pytest.approx(0.10)


def test_pm_clip_band() -> None:
    pm = _pm()
    # high vol (25%/day) -> target/v = 0.1 -> at floor
    assert pm.compute_vol_mult("HI", {"HI": _daily_vol(0.25)}) == pytest.approx(0.10)
    # very low vol (1%/day) -> target/v = 2.5 -> clipped to cap 2.0
    assert pm.compute_vol_mult("LO", {"LO": _daily_vol(0.01)}) == pytest.approx(2.0)
    # mid vol (5%/day) -> target/v = 0.5, inside the band
    assert pm.compute_vol_mult("MID", {"MID": _daily_vol(0.05)}) == pytest.approx(0.5)


# ── S8Sym15Strategy.compute_vol_mult (the actual live path) ──────────────
def _s8(floor: float = 0.10, cap: float = 2.0,
        vol_series: dict | None = None) -> S8Sym15Strategy:
    """Build a bare S8Sym15Strategy for compute_vol_mult only — bypasses the
    heavy __init__ (model loading) since the method needs just 4 attrs."""
    s = object.__new__(S8Sym15Strategy)
    s._vol_target = 0.025
    s._vol_floor = floor
    s._vol_cap = cap
    s._vol_series = vol_series or {}
    return s


def test_s8sym15_failsafe_missing_series() -> None:
    """The live path: unknown asset -> floor, NOT 1.0 (the -$15k bug)."""
    s = _s8()
    mult = s.compute_vol_mult("PIPPIN")
    assert mult == pytest.approx(0.10)
    assert mult != 1.0


def test_s8sym15_failsafe_nonfinite() -> None:
    s = _s8(vol_series={"X": _daily_vol(np.nan)})
    assert s.compute_vol_mult("X") == pytest.approx(0.10)


def test_s8sym15_clip_band() -> None:
    s = _s8(vol_series={"HI": _daily_vol(0.25), "MID": _daily_vol(0.05)})
    assert s.compute_vol_mult("HI") == pytest.approx(0.10)   # high vol -> floor
    assert s.compute_vol_mult("MID") == pytest.approx(0.5)   # mid vol -> in band


# ── pipeline.apply_per_asset_vol_overlay (the research backtester) ───────
def _signals() -> pd.DataFrame:
    ts = pd.Timestamp("2024-02-15")  # after the vol series ends
    return pd.DataFrame({
        "timestamp": [ts, ts, ts],
        "asset": ["MID", "HI", "MISSING"],
    })


def test_pipeline_overlay_failsafe_and_clip() -> None:
    vol_series = {"MID": _daily_vol(0.05), "HI": _daily_vol(0.25)}
    out = apply_per_asset_vol_overlay(_signals(), vol_series,
                                      target_vol=0.025, floor=0.10, cap=2.0)
    by_asset = dict(zip(out["asset"], out["sizing_mult"]))
    assert by_asset["MID"] == pytest.approx(0.5)        # mid vol -> in band
    assert by_asset["HI"] == pytest.approx(0.10)        # high vol -> floor
    assert by_asset["MISSING"] == pytest.approx(0.10)   # unknown -> FAIL-SAFE floor
    assert by_asset["MISSING"] != 1.0


def test_pipeline_overlay_multiplies_existing() -> None:
    """The overlay multiplies into an existing sizing_mult, not overwrite."""
    sigs = _signals()
    sigs["sizing_mult"] = [2.0, 2.0, 2.0]
    out = apply_per_asset_vol_overlay(sigs, {"MID": _daily_vol(0.05)},
                                      target_vol=0.025, floor=0.10, cap=2.0)
    by_asset = dict(zip(out["asset"], out["sizing_mult"]))
    assert by_asset["MID"] == pytest.approx(2.0 * 0.5)   # 2.0 * vol_mult
    assert by_asset["MISSING"] == pytest.approx(2.0 * 0.10)
