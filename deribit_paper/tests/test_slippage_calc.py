"""Tests for ``deribit_paper/execution/slippage_calc.py::live_slippage_pct``.

C4 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Three contracts under test:

1. **Sidecar-only path**: with no observed spread, the function converts
   ``slip_usd_per_btc / spot`` into a percentage and clips to
   [floor_pct, cap_pct].
2. **Override path**: when ``observed_spread_usd`` is materially wider
   than the sidecar implies, return the observed pct (clipped).
3. **Defensive paths**: NaN, zero, or both-None inputs return the
   ``fallback_pct`` (legacy 0.5%).
"""
from __future__ import annotations

import math

import pytest

from deribit_paper.execution.slippage_calc import live_slippage_pct


def test_sidecar_only_typical_btc_bar():
    """BTC at $60K, sidecar median slip $9.83/BTC → ~0.0164% cap."""
    pct = live_slippage_pct(slip_usd_per_btc=9.83, spot=60_000.0)
    assert pct == pytest.approx(0.0164, abs=1e-3)


def test_floor_clip_when_sidecar_says_zero_spread():
    """A degenerate sidecar value at the noise floor clips up to floor_pct."""
    pct = live_slippage_pct(
        slip_usd_per_btc=0.001, spot=60_000.0, floor_pct=0.05,
    )
    assert pct == 0.05


def test_observed_spread_override_when_wider():
    """Observed = $200/BTC at spot=60K → 0.333%; sidecar says $10 → 0.0167%.
    Ratio is ~20x > 1.5 threshold → use observed.
    """
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0, observed_spread_usd=200.0,
    )
    assert pct == pytest.approx(0.333, abs=1e-2)


def test_observed_spread_no_override_when_within_threshold():
    """Observed = $12/BTC, sidecar = $10/BTC. Ratio 1.2 ≤ 1.5 → use sidecar."""
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0, observed_spread_usd=12.0,
    )
    sidecar_pct = 100.0 * 10.0 / 60_000.0
    assert pct == pytest.approx(sidecar_pct, abs=1e-4)


def test_cap_pct_clips_runaway_observed():
    """Observed at insane width clips to cap_pct."""
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0,
        spot=60_000.0,
        observed_spread_usd=10_000.0,             # 16.67% — clearly clipped.
        cap_pct=1.0,
    )
    assert pct == 1.0


def test_fallback_when_both_inputs_invalid():
    """No sidecar + no observed → fallback_pct."""
    pct = live_slippage_pct(
        slip_usd_per_btc=0.0, spot=60_000.0, observed_spread_usd=None,
        fallback_pct=0.5,
    )
    assert pct == 0.5


def test_fallback_when_sidecar_nan():
    """NaN sidecar treated as missing; observed-only kicks in via fallback."""
    pct = live_slippage_pct(
        slip_usd_per_btc=float("nan"), spot=60_000.0, observed_spread_usd=None,
        fallback_pct=0.42,
    )
    assert pct == 0.42


def test_observed_alone_used_when_sidecar_missing():
    """Sidecar invalid but observed provided → observed_pct (clipped)."""
    pct = live_slippage_pct(
        slip_usd_per_btc=0.0, spot=60_000.0, observed_spread_usd=60.0,
    )
    # observed_pct = 0.1; sidecar_pct=0; override_threshold * 0 = 0;
    # observed_pct > 0 → returns observed.
    assert pct == pytest.approx(0.1, abs=1e-4)


# ── E.1.9: defensive paths ───────────────────────────────────────────────


def test_zero_spot_returns_fallback_not_zero_division():
    """Pre-E.1.9 a degenerate ticker (spot=0) raised ZeroDivisionError.
    Now: returns fallback_pct."""
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=0.0, fallback_pct=0.42,
    )
    assert pct == 0.42


def test_negative_spot_returns_fallback():
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=-100.0, fallback_pct=0.42,
    )
    assert pct == 0.42


def test_nan_spot_returns_fallback():
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=float("nan"), fallback_pct=0.42,
    )
    assert pct == 0.42


# ── D7 (parity 2026-05-07): V2 size + jump-stress impact formula ──────────


def test_v2_impact_formula_size_only():
    """With ``n_btc=1.0`` and ``jump_fraction=0``, the V2 formula gives
    ``min(2 * sidecar * sqrt(max(N, 0.1)) * exp(0), 2 * sidecar)``
    = ``min(2*10*1.0, 2*10)`` = 20 USD/BTC → 20/60_000 = 0.0333%.
    Without n_btc/jf the legacy path returns 10/60_000 = 0.0167%.
    """
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0,
        n_btc=1.0, jump_fraction=0.0,
        ioc_cap_multiplier=2.0,
    )
    # 2 * 10 * 1 * exp(0) = 20; cap = 2*10=20; min=20. pct = 100*20/60K = 0.0333.
    assert pct == pytest.approx(0.0333, abs=1e-3)


def test_v2_impact_formula_size_plus_jump_stress():
    """With ``jump_fraction=0.1`` and k=3, the exp factor = exp(0.3) ≈ 1.35.
    Effective slip = min(2*10*1*1.35, 2*10) = min(27, 20) = 20 (capped).
    """
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0,
        n_btc=1.0, jump_fraction=0.1,
        spread_jump_k=3.0, ioc_cap_multiplier=2.0,
    )
    # With cap=2x sidecar, hits the ceiling at 20 → 0.0333%.
    assert pct == pytest.approx(0.0333, abs=1e-3)


def test_v2_impact_formula_size_dominates_when_n_large():
    """N=4, jf=0 → sqrt(4)=2 → 2*10*2*1 = 40, capped at 2*10=20 → 20."""
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0,
        n_btc=4.0, jump_fraction=0.0,
        ioc_cap_multiplier=2.0,
    )
    assert pct == pytest.approx(0.0333, abs=1e-3)


def test_v2_impact_formula_higher_cap_multiplier():
    """``ioc_cap_multiplier=10`` lets the size impact bind below the cap.
    N=4, jf=0 → 2*10*2 = 40, cap = 100, return 40 → 100*40/60_000 = 0.0667.
    """
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0,
        n_btc=4.0, jump_fraction=0.0,
        ioc_cap_multiplier=10.0,
    )
    assert pct == pytest.approx(0.0667, abs=1e-3)


def test_v2_impact_formula_floor_on_tiny_n():
    """N=0.01 floors at sqrt(0.1) (V2 lower bound on size mult)."""
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0,
        n_btc=0.01, jump_fraction=0.0,
        ioc_cap_multiplier=10.0,
    )
    # max(0.01, 0.1) = 0.1; sqrt(0.1) ≈ 0.316; 2 * 10 * 0.316 = 6.32
    # capped at 10*10=100, return 6.32 → 100*6.32/60K ≈ 0.0105
    assert pct == pytest.approx(0.01054, abs=1e-3)


def test_v2_impact_formula_only_when_both_args():
    """If ``n_btc`` is given but ``jump_fraction`` is None, fall back to
    the legacy bare-sidecar path. Locks the contract that BOTH must be
    supplied for the V2 formula to engage."""
    pct = live_slippage_pct(
        slip_usd_per_btc=10.0, spot=60_000.0,
        n_btc=4.0, jump_fraction=None,
    )
    # Legacy: 100 * 10 / 60K = 0.0167%
    assert pct == pytest.approx(0.01667, abs=1e-3)

