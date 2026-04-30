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
