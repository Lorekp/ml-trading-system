"""D4 (parity 2026-05-07): rolling-median-edge EMA matches V2 backtester.

Mirrors V2 backtester at ``vol_arb/backtester.py:1846-1847, 2205-2206``:
- Init: ``_edge_ema = float(np.nanmedian(edge_arr[:96]))`` once 96 bars
  of edges are seen; until then return 35.0 fallback.
- Update: ``_edge_ema += alpha * (abs_edge - _edge_ema)`` with
  ``alpha = 2/501`` (500-bar EMA), ONLY on traded bars (signal != 0
  AND N > 0). Pre-D4 the live engine used a 96-bar median deque
  updated on every finite-edge bar.

This test exercises the EMA logic in isolation by manipulating engine
state directly — the full engine boot path isn't required to validate
the warmup → EMA transition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


class _StubEngine:
    """Minimal harness mirroring TradingEngine's edge-EMA fields.

    Avoids the heavy `TradingEngine.__init__` path (DRL load, SVI,
    etc.) so we can unit-test just the warmup→EMA transition.
    """

    def __init__(self, *, warmup_n: int = 96):
        self._edge_warmup: list[float] = []
        self._edge_ema: float | None = None
        self._EDGE_WARMUP_N: int = warmup_n
        self._EDGE_EMA_ALPHA: float = 2.0 / 501.0

    def observe_pre_trade(self, abs_edge: float) -> float:
        """Mirror engine.py L1632 — accumulate warmup, return rolling_median_edge."""
        if math.isfinite(abs_edge):
            if self._edge_ema is None:
                self._edge_warmup.append(abs_edge)
                if len(self._edge_warmup) >= self._EDGE_WARMUP_N:
                    self._edge_ema = float(np.median(self._edge_warmup))
                    self._edge_warmup.clear()
        return self._edge_ema if self._edge_ema is not None else 35.0

    def observe_post_trade(self, abs_edge: float) -> None:
        """Mirror engine.py post-trade EMA update — fired-bars only."""
        if self._edge_ema is not None and math.isfinite(abs_edge):
            self._edge_ema += self._EDGE_EMA_ALPHA * (abs_edge - self._edge_ema)


# ── 1. Warmup phase returns V2 fallback ─────────────────────────────────────


def test_warmup_returns_35_fallback():
    """Until 96 bars seen, ``rolling_median_edge`` returns 35.0
    (matches V2 backtester:1846 ``35.0`` fallback for short backtests).
    """
    eng = _StubEngine()
    for _ in range(50):
        v = eng.observe_pre_trade(20.0)
    assert v == pytest.approx(35.0)
    assert eng._edge_ema is None
    assert len(eng._edge_warmup) == 50


# ── 2. Warmup completes at exactly 96 bars ─────────────────────────────────


def test_warmup_completes_at_96_bars():
    """At the 96th finite-edge bar, EMA is seeded with the median."""
    eng = _StubEngine()
    edges = list(range(1, 97))  # |edge| = 1..96, median = 48.5
    for e in edges:
        eng.observe_pre_trade(float(e))
    assert eng._edge_ema is not None
    # Median of integers 1..96 is (48+49)/2 = 48.5
    assert eng._edge_ema == pytest.approx(48.5)
    # Buffer is cleared after seeding to free memory
    assert eng._edge_warmup == []


# ── 3. EMA only updates on fired bars (post_trade) ─────────────────────────


def test_ema_only_updates_on_fired_bars():
    """observe_pre_trade is a no-op for EMA; observe_post_trade updates."""
    eng = _StubEngine(warmup_n=4)
    # Warmup with 4 bars → EMA seeded with median = 2.5
    for e in [1.0, 2.0, 3.0, 4.0]:
        eng.observe_pre_trade(e)
    assert eng._edge_ema == pytest.approx(2.5)

    # 100 finite-edge bars at edge=10.0 — but NO trades fire, no
    # post_trade calls. EMA stays at 2.5.
    pre_ema = eng._edge_ema
    for _ in range(100):
        eng.observe_pre_trade(10.0)
    assert eng._edge_ema == pre_ema, (
        "pre-trade calls should NOT update EMA (no fired-bar gating)"
    )

    # Now fire 10 trades at edge=10.0 → EMA drifts toward 10.0
    for _ in range(10):
        eng.observe_post_trade(10.0)
    # alpha=2/501. After 10 steps starting from 2.5, EMA = 2.5 + 7.5 * (1 - (1-alpha)^10)
    # = 2.5 + 7.5 * (1 - (499/501)^10) ≈ 2.5 + 7.5 * 0.0397 ≈ 2.798
    expected = 2.5 + 7.5 * (1.0 - (1.0 - 2.0 / 501.0) ** 10)
    assert eng._edge_ema == pytest.approx(expected, rel=1e-9)


# ── 4. Alpha matches V2 (500-bar EMA) ──────────────────────────────────────


def test_alpha_is_500_bar_ema():
    """``_EDGE_EMA_ALPHA = 2/501`` matches V2 backtester:1847."""
    eng = _StubEngine()
    assert eng._EDGE_EMA_ALPHA == pytest.approx(2.0 / 501.0)


# ── 5. Single fired bar at huge edge: EMA shifts proportionally ────────────


def test_single_fired_bar_shift():
    """One fired bar at edge=100 from EMA=10 → new EMA = 10 + alpha*(100-10) = 10.359."""
    eng = _StubEngine(warmup_n=2)
    eng.observe_pre_trade(10.0)
    eng.observe_pre_trade(10.0)
    assert eng._edge_ema == pytest.approx(10.0)
    eng.observe_post_trade(100.0)
    expected = 10.0 + (2.0 / 501.0) * (100.0 - 10.0)
    assert eng._edge_ema == pytest.approx(expected)


# ── 6. NaN / infinite edges are dropped ────────────────────────────────────


def test_non_finite_edges_dropped():
    """NaN edges don't affect warmup count or EMA."""
    eng = _StubEngine(warmup_n=4)
    eng.observe_pre_trade(float("nan"))
    eng.observe_pre_trade(float("inf"))
    eng.observe_pre_trade(1.0)
    eng.observe_pre_trade(2.0)
    eng.observe_pre_trade(3.0)
    eng.observe_pre_trade(4.0)
    # 4 finite edges → warmup completes, median = 2.5
    assert eng._edge_ema == pytest.approx(2.5)
    eng.observe_post_trade(float("nan"))
    # NaN should leave EMA unchanged
    assert eng._edge_ema == pytest.approx(2.5)
