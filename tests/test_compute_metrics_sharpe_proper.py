"""Regression test for ``VolArbBacktester.compute_metrics`` Sharpe (2026-05-07).

Pre-fix the headline Sharpe was computed from FIRED-BARS-ONLY
``r_t[r_t != 0]`` annualized by ``sqrt(bars_per_year)``. That overstates
Sharpe by ``~sqrt(bars_per_year / trades_per_year)`` because the
annualization factor counts every calendar bar but the std uses only
the fired-bar return distribution.

Post-fix:

* ``sharpe_ratio`` (HEADLINE) — daily-resampled Sharpe, the proper
  industry-standard quant convention. ``mean(daily_r) / std(daily_r) *
  sqrt(365.25)`` where ``daily_r`` is the sum of ``r_t`` within each
  calendar day. Matches ``vol_arb.metrics.risk_adjusted_metrics.sharpe_daily``
  for i.i.d. returns and captures intraday clustering correctly.
* ``sharpe_ratio_per_bar`` — full-series per-bar Sharpe ``mean(r_t) /
  std(r_t) * sqrt(bars_per_year)``. Diagnostic.
* ``sharpe_ratio_trades_only`` — pre-fix convention. Diagnostic for
  backwards comparison against historical logs.
"""

from __future__ import annotations

import numpy as np
import pytest

from vol_arb.backtester import VolArbBacktester


# ── Synthetic equity / r_t / trade_pnl arrays ────────────────────────────


def _build_synthetic(
    n_bars: int,
    n_trades: int,
    initial_equity: float = 100_000.0,
    seed: int = 0,
):
    """Construct a synthetic full-series r_t with known trade locations."""
    rng = np.random.default_rng(seed)
    mu = 0.001
    sigma = 0.005
    r_t = np.zeros(n_bars)
    trade_idx = rng.choice(n_bars, size=n_trades, replace=False)
    r_t[trade_idx] = rng.normal(loc=mu, scale=sigma, size=n_trades)
    trade_pnl = r_t * initial_equity
    equity = initial_equity + np.cumsum(trade_pnl)
    return equity, r_t, trade_pnl, mu, sigma


# ── 1. Headline Sharpe matches daily-resampled formula ───────────────────


def test_headline_sharpe_is_daily_resampled():
    """``sharpe_ratio`` matches mean(daily_r)/std(daily_r) * sqrt(365.25)."""
    bar_minutes = 15
    bars_per_day = 1440 // bar_minutes  # 96
    n_days = 100
    n_bars = n_days * bars_per_day
    n_trades = 400
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades, seed=7)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=bar_minutes,
    )

    daily_r = r_t.reshape(n_days, bars_per_day).sum(axis=1)
    expected = (
        float(daily_r.mean()) / float(daily_r.std(ddof=1))
        * np.sqrt(365.25)
    )
    assert metrics["sharpe_ratio"] == pytest.approx(expected, rel=1e-12)


# ── 2. Per-bar Sharpe diagnostic uses FULL r_t (zeros included) ──────────


def test_per_bar_sharpe_uses_full_series():
    """``sharpe_ratio_per_bar`` matches mean(r_t)/std(r_t)*sqrt(bars/yr)."""
    n_bars = 1000
    n_trades = 100
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=15,
    )

    bars_per_year = 365.25 * 96.0
    expected = (
        float(r_t.mean()) / float(r_t.std(ddof=1)) * np.sqrt(bars_per_year)
    )
    assert metrics["sharpe_ratio_per_bar"] == pytest.approx(expected, rel=1e-12)


# ── 3. Legacy Sharpe diagnostic uses fired bars only ─────────────────────


def test_legacy_sharpe_uses_fired_bars_only():
    """``sharpe_ratio_trades_only`` matches the pre-fix convention."""
    n_bars = 1000
    n_trades = 100
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=15,
    )

    fired = r_t[r_t != 0]
    bars_per_year = 365.25 * 96.0
    expected_legacy = (
        float(fired.mean()) / float(fired.std(ddof=1))
        * np.sqrt(bars_per_year)
    )
    assert metrics["sharpe_ratio_trades_only"] == pytest.approx(
        expected_legacy, rel=1e-12,
    )


# ── 4. Legacy overstates per-bar by expected factor ──────────────────────


def test_legacy_overstates_per_bar_by_expected_factor():
    """The legacy fired-only Sharpe exceeds the per-bar full-series
    Sharpe by roughly ``sqrt(n_bars / n_trades)``.
    """
    n_bars = 10_000
    n_trades = 1_000
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades, seed=42)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=15,
    )

    legacy = metrics["sharpe_ratio_trades_only"]
    per_bar = metrics["sharpe_ratio_per_bar"]
    ratio = legacy / per_bar
    expected = np.sqrt(n_bars / n_trades)
    assert 0.5 * expected < ratio < 2.0 * expected, (
        f"legacy/per-bar ratio = {ratio:.2f}, expected ≈ {expected:.2f}"
    )


# ── 5. Daily and per-bar Sharpes agree for i.i.d. returns ────────────────


def test_daily_and_per_bar_agree_for_iid_returns():
    """For an i.i.d. return process the daily and per-bar Sharpes should
    agree to within sampling noise. Real strategies (with intra-day
    clustering) violate this; the gap signals autocorrelation."""
    bar_minutes = 15
    bars_per_day = 1440 // bar_minutes
    n_days = 200
    n_bars = n_days * bars_per_day
    # Spread n_trades uniformly across days — i.i.d. signal locations.
    n_trades = 800
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades, seed=11)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=bar_minutes,
    )
    sharpe_daily = metrics["sharpe_ratio"]
    sharpe_per_bar = metrics["sharpe_ratio_per_bar"]
    # Within 30% — generous for sampling noise on 200 days.
    assert sharpe_daily == pytest.approx(sharpe_per_bar, rel=0.30)


# ── 6. Annualization scales correctly for non-15m bars ───────────────────


@pytest.mark.parametrize("bar_minutes", [5, 15, 60])
def test_daily_resample_scales_with_bar_minutes(bar_minutes):
    """Daily Sharpe should be invariant to bar_minutes for the SAME
    underlying daily-return distribution. Verify by constructing
    synthetic data on each bar grid that produces the same daily totals.
    """
    bars_per_day = 1440 // bar_minutes
    n_days = 50
    n_bars = n_days * bars_per_day
    rng = np.random.default_rng(seed=3)
    daily_pnl = rng.normal(loc=100.0, scale=200.0, size=n_days)
    # Distribute each day's PnL uniformly across its bars.
    r_t = np.repeat(daily_pnl / 100_000.0 / bars_per_day, bars_per_day)
    pnl = r_t * 100_000.0
    eq = 100_000.0 + np.cumsum(pnl)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=bar_minutes,
    )

    # Expected daily Sharpe is invariant — depends only on the daily series.
    daily_r = daily_pnl / 100_000.0
    expected = (
        float(daily_r.mean()) / float(daily_r.std(ddof=1))
        * np.sqrt(365.25)
    )
    assert metrics["sharpe_ratio"] == pytest.approx(expected, rel=1e-9)


# ── 7. Sortino daily uses the same convention ────────────────────────────


def test_sortino_uses_daily_resampled_downside():
    """``sortino_ratio`` uses the std of NEGATIVE daily returns."""
    bar_minutes = 15
    bars_per_day = 96
    n_days = 100
    n_bars = n_days * bars_per_day
    n_trades = 400
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades, seed=13)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=bar_minutes,
    )

    daily_r = r_t.reshape(n_days, bars_per_day).sum(axis=1)
    downside = daily_r[daily_r < 0]
    expected = (
        float(daily_r.mean()) / float(downside.std(ddof=1))
        * np.sqrt(365.25)
    )
    assert metrics["sortino_ratio"] == pytest.approx(expected, rel=1e-12)


# ── 8. Default bar_minutes preserves V2 convention ───────────────────────


def test_default_bar_minutes_is_15():
    """Default ``bar_minutes=15`` yields V2-canonical bars_per_day=96."""
    n_bars = 96 * 50
    n_trades = 200
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades)
    m = VolArbBacktester.compute_metrics(eq, r_t, pnl, 100_000.0)
    m_explicit = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=15,
    )
    assert m["sharpe_ratio"] == pytest.approx(m_explicit["sharpe_ratio"], rel=1e-12)


# ── 9. Edge cases ────────────────────────────────────────────────────────


def test_zero_trades_returns_minimal_dict():
    """No fired bars → minimal dict (preserves prior contract)."""
    eq = np.full(100, 100_000.0)
    r_t = np.zeros(100)
    pnl = np.zeros(100)
    metrics = VolArbBacktester.compute_metrics(eq, r_t, pnl, 100_000.0)
    assert metrics["n_trades"] == 0


def test_single_day_returns_zero_daily_sharpe():
    """With < 2 full days of data the daily Sharpe collapses to 0
    (std over a single sample is undefined). Per-bar diagnostic is
    still computed.
    """
    bar_minutes = 15
    bars_per_day = 96
    # 1 full day — std(ddof=1) on 1 sample is undefined.
    n_bars = bars_per_day
    n_trades = 5
    eq, r_t, pnl, _, _ = _build_synthetic(n_bars, n_trades, seed=99)
    metrics = VolArbBacktester.compute_metrics(
        eq, r_t, pnl, 100_000.0, bar_minutes=bar_minutes,
    )
    assert metrics["sharpe_ratio"] == 0.0
    # Per-bar still reportable
    assert metrics["sharpe_ratio_per_bar"] != 0.0
