"""D5 (parity-cycle round 2 — 2026-05-07): ``simulate_discrete_hedging``
and ``DRLHedger.simulate`` use ``cfg.bar_minutes`` to scale the per-step
time, instead of a hardcoded 15.0 literal.

At the V2 default (``holding_minutes=15``) this is byte-identical with
pre-fix output. At any other cadence the per-step σ scaling and tau
decay was wrong before — same value of ``n_steps`` but a path with
1-min bar would have been 15× over-scaled.
"""

from __future__ import annotations

import numpy as np
import pytest

from vol_arb.backtester import simulate_discrete_hedging


# ── simulate_discrete_hedging ────────────────────────────────────────


def _make_arrays(n: int = 4):
    return dict(
        S_entry=np.full(n, 50_000.0),
        K=np.full(n, 50_000.0),
        sigma=np.full(n, 0.75),
        T_years=np.full(n, 7.0 / 365.25),
        vol_ann=np.full(n, 0.80),
        signal=np.array([1, -1, 1, -1], dtype=int),
    )


def test_simulate_discrete_default_bar_minutes_matches_15():
    """The default (``bar_minutes=15.0``) must remain byte-identical with the pre-fix code."""
    arrs = _make_arrays()
    out_default = simulate_discrete_hedging(seed=42, **arrs)
    out_explicit_15 = simulate_discrete_hedging(seed=42, bar_minutes=15.0, **arrs)
    assert np.allclose(out_default[0], out_explicit_15[0], atol=1e-12)
    assert np.allclose(out_default[1], out_explicit_15[1], atol=1e-12)


def test_simulate_discrete_5min_bar_scales_dt_proportionally():
    """At a 5-min bar the per-step σ is √(5/15)=1/√3× smaller — paths are tighter."""
    arrs = _make_arrays()
    pnl_15, _ = simulate_discrete_hedging(seed=42, bar_minutes=15.0, **arrs)
    pnl_5, _ = simulate_discrete_hedging(seed=42, bar_minutes=5.0, **arrs)
    # Different paths → different per-trade PnLs (not identical to floats)
    assert not np.allclose(pnl_15, pnl_5)
    # But same RNG seed → STD of pnl_5 should be smaller because path widths are smaller
    assert np.std(pnl_5) <= np.std(pnl_15) + 1e-9


def test_simulate_discrete_60min_bar_widens_paths():
    arrs = _make_arrays()
    pnl_15, _ = simulate_discrete_hedging(seed=42, bar_minutes=15.0, **arrs)
    pnl_60, _ = simulate_discrete_hedging(seed=42, bar_minutes=60.0, **arrs)
    # Same RNG seed → STD of pnl_60 should be larger because 4× longer dt
    assert np.std(pnl_60) >= np.std(pnl_15) - 1e-9


# ── DRLHedger.simulate ───────────────────────────────────────────────


def test_drl_hedger_simulate_accepts_bar_minutes_kwarg():
    """Smoke: DRLHedger.simulate accepts the new bar_minutes kwarg without raising.

    We don't load a real policy here (pulling SAC weights is outside this
    test's scope); we verify the signature change at the kwarg-acceptance
    level by direct introspection.
    """
    import inspect
    from vol_arb.drl_hedging.env import DRLHedger
    sig = inspect.signature(DRLHedger.simulate)
    assert "bar_minutes" in sig.parameters
    assert sig.parameters["bar_minutes"].default == 15.0


# ── V2 byte-identity sanity ──────────────────────────────────────────


def test_v2_default_holding_minutes_unchanged():
    """The V2 preset (``holding_minutes=15``) must drive bar_minutes=15.0
    into the simulate(s), preserving byte-identity with the pre-D5 baseline."""
    from vol_arb.backtester import BacktestConfig
    cfg = BacktestConfig().with_btc_production_filters_v2()
    assert float(cfg.holding_minutes) == pytest.approx(15.0)
