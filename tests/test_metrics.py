"""Tests for vol_arb/metrics.py — compound and non-compound paths.

These tests use a **deterministic synthetic trade log** with known
``net_pnl`` and ``net_pnl_nc`` columns.  Every metric value is
analytically computable from the fixture, so we can assert exact (or
near-exact) equalities rather than "it runs and doesn't crash".

The synthetic fixture is 90 daily bars with a PnL pattern chosen so
that the compound and non-compound curves diverge materially — that's
what exercises the ``is_nc`` branch points in:

- ``_daily_returns``         (denominator: drifting vs fixed)
- ``_drawdown_series``       (percent-of-peak vs percent-of-initial)
- ``return_metrics``          (CAGR vs None)
- ``risk_adjusted_metrics``   (Calmar/Martin numerator)
- ``stability_metrics``       (log-regression vs raw)

The compound vs non-compound outputs MUST differ on these metrics
(otherwise one of the branches isn't firing), and the NC values must
match an independent numpy reference computation defined inline in
each test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.metrics import (
    _daily_pnl,
    _daily_returns,
    _drawdown_series,
    build_nc_view,
    drawdown_metrics,
    full_metrics,
    full_metrics_both,
    return_metrics,
    risk_adjusted_metrics,
    stability_metrics,
    top_drawdown_episodes,
)


INITIAL_EQUITY = 100_000.0


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_synthetic_trade_log(
    n_days: int = 90,
    seed: int = 1337,
) -> pd.DataFrame:
    """Build a deterministic trade log with matched compound + NC columns.

    The per-day PnL pattern is a mixture of small wins and occasional
    losses with enough magnitude to create non-trivial drawdowns.  For
    each "day" we emit a single 12:00 UTC bar — the metrics toolkit
    normalises timestamps to day keys, so one bar per day is enough.
    """
    rng = np.random.default_rng(seed)
    # Mixture: small positive drift + occasional fat tails.
    daily_pnl = rng.normal(loc=400.0, scale=2_500.0, size=n_days)
    # Inject 3 deterministic drawdown clusters so max_dd is non-trivial.
    daily_pnl[15] = -15_000.0
    daily_pnl[16] = -6_000.0
    daily_pnl[45] = -20_000.0
    daily_pnl[72] = -8_000.0

    ts = pd.date_range("2025-01-01 12:00:00", periods=n_days, freq="D", tz="UTC")

    # Compound: each day's return reinvests. Build running equity curve.
    equity_compound = np.empty(n_days)
    eq = INITIAL_EQUITY
    r_t_compound = np.empty(n_days)
    for i in range(n_days):
        r = daily_pnl[i] / eq
        r_t_compound[i] = r
        eq = eq * (1.0 + r)
        equity_compound[i] = eq

    # Non-compound: PnL is computed on a fixed reference notional, so
    # the additive equity curve is initial + cumsum(pnl_nc).  To keep
    # the test honest we use the SAME ``daily_pnl`` series but treat
    # it as the NC column (matches what the backtester does: the NC
    # sizing block uses the fixed notional for sizing so net_pnl and
    # net_pnl_nc coincide to within rounding for small moves).
    equity_nc = INITIAL_EQUITY + np.cumsum(daily_pnl)
    r_t_nc = daily_pnl / INITIAL_EQUITY

    return pd.DataFrame(
        {
            "net_pnl": daily_pnl,
            "equity": equity_compound,
            "r_t": r_t_compound,
            "signal": np.ones(n_days, dtype=int),  # so trade_stats does something
            "net_pnl_nc": daily_pnl,
            "equity_nc": equity_nc,
            "r_t_nc": r_t_nc,
        },
        index=ts,
    )


@pytest.fixture(scope="module")
def trade_log() -> pd.DataFrame:
    return _make_synthetic_trade_log()


@pytest.fixture(scope="module")
def nc_view(trade_log) -> pd.DataFrame:
    return build_nc_view(trade_log)


# ── _daily_returns ────────────────────────────────────────────────────


def test_daily_returns_compound_uses_pct_change(trade_log):
    """Compound daily returns must equal ``equity.pct_change().dropna()``."""
    r = _daily_returns(trade_log, INITIAL_EQUITY, is_nc=False)
    expected = trade_log["equity"].pct_change().dropna()
    np.testing.assert_allclose(r.values, expected.values, rtol=1e-12)


def test_daily_returns_nc_uses_fixed_reference(trade_log, nc_view):
    """NC daily returns must equal ``daily_pnl / initial_equity``.

    This is the key fix — the old ``pct_change`` path used a drifting
    denominator (early days ÷$100K, late days ÷$689K), which isn't a
    true percent return of a fixed-notional strategy.
    """
    r_nc = _daily_returns(nc_view, INITIAL_EQUITY, is_nc=True)
    expected = (trade_log["net_pnl"] / INITIAL_EQUITY).dropna()
    # Indexes differ (daily vs bar); compare on overlapping days
    np.testing.assert_allclose(
        r_nc.sort_index().values,
        expected.sort_index().values,
        rtol=1e-12,
    )


def test_daily_returns_nc_and_compound_differ(trade_log, nc_view):
    """Sanity: after a few weeks of drift, the two views must differ.

    If they coincide, one of the branches isn't firing and we're not
    actually testing anything meaningful.  The compound path drops
    the first day via ``pct_change().dropna()`` while the NC path
    keeps it, so compare on the overlapping index only.
    """
    r_comp = _daily_returns(trade_log, INITIAL_EQUITY, is_nc=False)
    r_nc = _daily_returns(nc_view, INITIAL_EQUITY, is_nc=True)
    common = r_comp.index.intersection(r_nc.index)
    assert len(common) > 0
    assert not np.allclose(
        r_comp.loc[common].values,
        r_nc.loc[common].values,
        rtol=1e-3,
    )


# ── _drawdown_series ──────────────────────────────────────────────────


def test_drawdown_series_compound_is_percent_of_peak(trade_log):
    _, dd = _drawdown_series(trade_log, INITIAL_EQUITY, is_nc=False)
    eq = trade_log["equity"].groupby(
        trade_log.index.normalize(),
    ).last()
    running_max = eq.cummax()
    expected = (eq - running_max) / running_max
    np.testing.assert_allclose(dd.values, expected.values, rtol=1e-12)


def test_drawdown_series_nc_is_percent_of_initial_equity(trade_log, nc_view):
    """NC drawdown must be ``(eq - running_max) / initial_equity``."""
    _, dd_nc = _drawdown_series(nc_view, INITIAL_EQUITY, is_nc=True)
    pnl = trade_log["net_pnl"]
    daily_pnl = pnl.groupby(pnl.index.normalize()).sum()
    eq_nc = INITIAL_EQUITY + daily_pnl.cumsum()
    running_max = eq_nc.cummax()
    expected = (eq_nc - running_max) / INITIAL_EQUITY
    np.testing.assert_allclose(dd_nc.values, expected.values, rtol=1e-12)


# ── drawdown_metrics ──────────────────────────────────────────────────


def test_drawdown_metrics_nc_max_dd_matches_independent_reference(nc_view, trade_log):
    """NC max DD must equal ``min(eq_nc - running_max) / initial_equity``."""
    stats = drawdown_metrics(nc_view, INITIAL_EQUITY, is_nc=True)

    daily_pnl = trade_log["net_pnl"].groupby(
        trade_log.index.normalize(),
    ).sum()
    eq_nc = INITIAL_EQUITY + daily_pnl.cumsum()
    running_max = eq_nc.cummax()
    expected_max_dd_pct = float((eq_nc - running_max).min() / INITIAL_EQUITY) * 100.0

    assert stats["max_drawdown_pct"] == pytest.approx(expected_max_dd_pct, rel=1e-9)


def test_drawdown_metrics_comp_and_nc_differ(trade_log, nc_view):
    """Compound vs NC max DD should be materially different.

    Compound max DD is a percent-of-peak on a curve that has
    (hopefully) grown; NC max DD is a percent-of-initial on an
    additive curve.  They must NOT coincide.
    """
    comp = drawdown_metrics(trade_log, INITIAL_EQUITY, is_nc=False)
    nc = drawdown_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    assert comp["max_drawdown_pct"] != pytest.approx(nc["max_drawdown_pct"], rel=1e-3)


# ── return_metrics ────────────────────────────────────────────────────


def test_return_metrics_compound_reports_cagr(trade_log):
    stats = return_metrics(trade_log, INITIAL_EQUITY, is_nc=False)
    assert stats["cagr_pct"] is not None
    assert isinstance(stats["cagr_pct"], float)


def test_return_metrics_nc_omits_cagr(nc_view):
    """In NC mode CAGR is undefined and must be returned as None."""
    stats = return_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    assert stats["cagr_pct"] is None
    # The arithmetic annualized return must still be present + finite.
    assert np.isfinite(stats["ann_return_arith_pct"])


def test_return_metrics_nc_arith_return_matches_formula(nc_view, trade_log):
    """``ann_return_arith_pct`` must equal ``total_pnl / initial / years``."""
    stats = return_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    total_pnl = float(trade_log["net_pnl"].sum())
    span = (
        trade_log.index[-1] - trade_log.index[0]
    ).total_seconds() / 86400.0
    years = span / 365.25
    # mean(daily_pnl / initial) * 365.25 = total_pnl / initial / days_in_span * 365.25
    # Use a small tolerance since the daily resample aggregates by day.
    expected = (total_pnl / INITIAL_EQUITY) / years * 100.0
    # This isn't exactly equal because of daily grouping, but should be
    # within a few percent for a 90-day window of daily bars.
    assert stats["ann_return_arith_pct"] == pytest.approx(expected, rel=0.02)


# ── risk_adjusted_metrics ─────────────────────────────────────────────


def test_risk_adjusted_metrics_nc_calmar_uses_arith_return(nc_view):
    """NC Calmar numerator must be the arithmetic annualized return.

    In compound mode the doc states Calmar = CAGR / |Max DD|.  Since
    CAGR is undefined in NC, we substitute the arithmetic annualized
    return (which is the honest fixed-notional equivalent).  Check
    that the value is consistent with that substitution.
    """
    ret = return_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    dd = drawdown_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    ra = risk_adjusted_metrics(nc_view, INITIAL_EQUITY, is_nc=True)

    max_dd = abs(dd["max_drawdown_pct"]) / 100.0
    expected_calmar = (ret["ann_return_arith_pct"] / 100.0) / max_dd
    assert ra["calmar"] == pytest.approx(expected_calmar, rel=1e-9)


def test_risk_adjusted_metrics_nc_martin_uses_additive_ulcer(nc_view):
    """NC Ulcer Index must be RMS of the additive drawdown series."""
    _, dd_series = _drawdown_series(nc_view, INITIAL_EQUITY, is_nc=True)
    expected_ulcer = float(np.sqrt(np.mean(dd_series.values ** 2)))
    ra = risk_adjusted_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    assert ra["ulcer_index"] == pytest.approx(expected_ulcer * 100.0, rel=1e-9)


def test_risk_adjusted_metrics_sharpe_nc_equals_manual(nc_view):
    """NC Sharpe must equal mean/std of ``daily_pnl / initial_equity``."""
    r = _daily_returns(nc_view, INITIAL_EQUITY, is_nc=True)
    expected_sharpe = float(
        (r.mean() / r.std(ddof=1)) * np.sqrt(365.25),
    )
    ra = risk_adjusted_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    assert ra["sharpe_daily"] == pytest.approx(expected_sharpe, rel=1e-9)


# ── stability_metrics ─────────────────────────────────────────────────


def test_stability_metrics_nc_fits_raw_curve(nc_view, trade_log):
    """NC R² must be the linregress fit to the RAW daily equity curve."""
    stats = stability_metrics(nc_view, INITIAL_EQUITY, is_nc=True)

    daily_pnl = trade_log["net_pnl"].groupby(
        trade_log.index.normalize(),
    ).sum()
    eq_nc = INITIAL_EQUITY + daily_pnl.cumsum()
    from scipy import stats as spstats
    x = np.arange(len(eq_nc))
    slope, intercept, r_value, _, _ = spstats.linregress(x, eq_nc.values)
    expected_r2 = float(r_value ** 2)
    assert stats["equity_curve_r_squared"] == pytest.approx(
        expected_r2, rel=1e-9,
    )


def test_stability_metrics_compound_fits_log_curve(trade_log):
    """Compound R² must be the linregress fit to log(daily_eq)."""
    stats = stability_metrics(trade_log, INITIAL_EQUITY, is_nc=False)

    daily_eq = trade_log["equity"].groupby(
        trade_log.index.normalize(),
    ).last()
    from scipy import stats as spstats
    log_eq = np.log(daily_eq.clip(lower=1e-9))
    x = np.arange(len(log_eq))
    slope, intercept, r_value, _, _ = spstats.linregress(x, log_eq.values)
    expected_r2 = float(r_value ** 2)
    assert stats["equity_curve_r_squared"] == pytest.approx(
        expected_r2, rel=1e-9,
    )


# ── full_metrics_both ──────────────────────────────────────────────────


def test_full_metrics_both_produces_both_views(trade_log):
    out = full_metrics_both(trade_log, INITIAL_EQUITY)
    assert "compound" in out
    assert "non_compound" in out
    # CAGR is only defined on the compound view.
    assert out["compound"]["cagr_pct"] is not None
    assert out["non_compound"]["cagr_pct"] is None


def test_full_metrics_both_sharpes_differ(trade_log):
    """Sanity: Sharpe on the two views must differ at materially large magnitudes.

    If both Sharpes coincide, one of the ``is_nc`` branches is silent.
    """
    out = full_metrics_both(trade_log, INITIAL_EQUITY)
    comp_sharpe = out["compound"]["sharpe_daily"]
    nc_sharpe = out["non_compound"]["sharpe_daily"]
    # At 90 days with drift they should differ by > 1% (the drifting
    # denominator bug was large — the real-world gap is ~30%).
    assert abs(comp_sharpe - nc_sharpe) / max(abs(comp_sharpe), 1e-9) > 0.01


def test_build_nc_view_does_not_set_attrs(trade_log):
    """``build_nc_view`` must NOT rely on ``.attrs`` — that was the old,
    fragile signalling channel we deliberately removed."""
    view = build_nc_view(trade_log)
    assert "is_nc_view" not in view.attrs


def test_nc_metrics_robust_to_boolean_indexing(trade_log):
    """A boolean-filtered NC view must still produce correct NC metrics.

    The old attrs-based flag did NOT survive boolean indexing (pandas
    drops ``.attrs`` in many operations), so any caller that masked the
    view before passing it to metrics silently got compound formulas.
    The new explicit ``is_nc`` kwarg is immune — this test verifies it.
    """
    view = build_nc_view(trade_log)
    # Filter out the first 10 rows — a legal operation that used to
    # silently strip the attrs flag.
    filtered = view.iloc[10:]
    stats = full_metrics(filtered, INITIAL_EQUITY, is_nc=True)
    # Sanity: max DD is a plausible % of initial equity (<50%).
    assert -50.0 < stats["max_drawdown_pct"] <= 0.0
    # CAGR must still be None under the explicit kwarg.
    assert stats["cagr_pct"] is None


# ── top_drawdown_episodes ─────────────────────────────────────────────


def test_top_dd_episodes_deepest_matches_max_drawdown(nc_view):
    """The deepest episode's ``depth_pct`` must equal aggregate max_drawdown_pct.

    If they disagree, either the episode-walking loop has a bug or the
    aggregate computation does — both helpers read the same
    ``_drawdown_series`` output, so they must coincide.
    """
    eps = top_drawdown_episodes(nc_view, INITIAL_EQUITY, is_nc=True, k=10)
    agg = drawdown_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    assert len(eps) >= 1
    # Episodes sorted ascending by depth_pct → index 0 is deepest.
    deepest = eps[0]["depth_pct"]
    assert deepest == pytest.approx(agg["max_drawdown_pct"], rel=1e-9)
    # Sort invariant: depth_pct non-decreasing (deepest first, then
    # shallower episodes).
    for prev, cur in zip(eps, eps[1:]):
        assert prev["depth_pct"] <= cur["depth_pct"] + 1e-12


def test_top_dd_episodes_event_count_matches_aggregate(nc_view):
    """With k=large, number of returned episodes must equal n_drawdown_events."""
    eps = top_drawdown_episodes(nc_view, INITIAL_EQUITY, is_nc=True, k=10_000)
    agg = drawdown_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    assert len(eps) == agg["n_drawdown_events"]


def test_top_dd_episodes_k_clips_result(nc_view):
    """``k=2`` must return at most 2 episodes, sorted deepest-first."""
    eps2 = top_drawdown_episodes(nc_view, INITIAL_EQUITY, is_nc=True, k=2)
    assert len(eps2) <= 2
    # If 2 or more episodes exist, index 0 depth <= index 1 depth.
    if len(eps2) == 2:
        assert eps2[0]["depth_pct"] <= eps2[1]["depth_pct"] + 1e-12


def test_top_dd_episodes_known_synthetic_trough():
    """On a hand-built fixture with a known trough, the deepest episode
    must find it exactly.

    Build a 20-day NC log where day 5 injects -$10K and the rest are
    small positives.  Peak is day 4 at +$2K; trough is day 5 at
    -$10K + $2K = -$8K.  depth_pct on NC is
    (eq - running_max) / initial_equity → -10_000 / 100_000 = -10.00%.
    (Note: using 0.5K daily positive means trough_val = -9_500 / 100K.)
    """
    n = 20
    ts = pd.date_range("2025-06-01 12:00:00", periods=n, freq="D", tz="UTC")
    pnl = np.full(n, 500.0)  # +$500/day baseline
    pnl[5] = -10_000.0       # single-day shock

    # NC equity = initial + cumsum
    equity_nc = INITIAL_EQUITY + np.cumsum(pnl)
    # Compound equity (not the subject of this test but the fixture
    # needs both columns so build_nc_view works).
    equity_comp = np.empty(n)
    eq = INITIAL_EQUITY
    for i in range(n):
        eq = eq * (1.0 + pnl[i] / eq)
        equity_comp[i] = eq

    tl = pd.DataFrame(
        {
            "net_pnl": pnl,
            "equity": equity_comp,
            "r_t": pnl / INITIAL_EQUITY,
            "signal": np.ones(n, dtype=int),
            "net_pnl_nc": pnl,
            "equity_nc": equity_nc,
            "r_t_nc": pnl / INITIAL_EQUITY,
        },
        index=ts,
    )
    view = build_nc_view(tl)
    eps = top_drawdown_episodes(view, INITIAL_EQUITY, is_nc=True, k=5)

    assert len(eps) >= 1
    deepest = eps[0]
    # Days 0..4 add +500 each → cumsum through index 4 = +2_500,
    # eq_nc[4] = 102_500 (the running-max peak for the DD that follows).
    # Day 5 adds -10_000 → cumsum[5] = -7_500, eq_nc[5] = 92_500.
    # depth_pct = (92_500 - 102_500) / 100_000 * 100 = -10.00%.
    assert deepest["depth_pct"] == pytest.approx(-10.0, abs=0.01)
    # drawdown_metrics / top_drawdown_episodes work on the DAILY
    # equity curve, which normalizes index to midnight — compare dates.
    assert deepest["trough_ts"].normalize() == ts[5].normalize()
    # trades_count must span from start of DD through recovery (if recovered)
    # — at 500/day after trough it recovers in roughly 10_000/500 = 20 days,
    # so this DD likely stays open through series end.
    assert deepest["trades_count"] >= 1


def test_top_dd_episodes_no_drawdown_returns_empty():
    """A strictly-increasing equity curve has zero drawdown episodes."""
    n = 10
    ts = pd.date_range("2025-01-01 12:00:00", periods=n, freq="D", tz="UTC")
    pnl = np.full(n, 100.0)  # always positive
    equity_comp = np.empty(n)
    eq = INITIAL_EQUITY
    for i in range(n):
        eq = eq * (1.0 + pnl[i] / eq)
        equity_comp[i] = eq
    tl = pd.DataFrame(
        {
            "net_pnl": pnl,
            "equity": equity_comp,
            "r_t": pnl / INITIAL_EQUITY,
            "signal": np.ones(n, dtype=int),
            "net_pnl_nc": pnl,
            "equity_nc": INITIAL_EQUITY + np.cumsum(pnl),
            "r_t_nc": pnl / INITIAL_EQUITY,
        },
        index=ts,
    )
    view = build_nc_view(tl)
    eps = top_drawdown_episodes(view, INITIAL_EQUITY, is_nc=True, k=10)
    assert eps == []


def test_top_dd_episodes_open_episode_has_none_recovery():
    """If equity ends below its running peak, recovery_ts must be None."""
    n = 15
    ts = pd.date_range("2025-03-01 12:00:00", periods=n, freq="D", tz="UTC")
    pnl = np.full(n, 500.0)
    pnl[10] = -5_000.0  # shock late in the series; won't recover
    equity_comp = np.empty(n)
    eq = INITIAL_EQUITY
    for i in range(n):
        eq = eq * (1.0 + pnl[i] / eq)
        equity_comp[i] = eq
    tl = pd.DataFrame(
        {
            "net_pnl": pnl,
            "equity": equity_comp,
            "r_t": pnl / INITIAL_EQUITY,
            "signal": np.ones(n, dtype=int),
            "net_pnl_nc": pnl,
            "equity_nc": INITIAL_EQUITY + np.cumsum(pnl),
            "r_t_nc": pnl / INITIAL_EQUITY,
        },
        index=ts,
    )
    view = build_nc_view(tl)
    eps = top_drawdown_episodes(view, INITIAL_EQUITY, is_nc=True, k=5)
    assert len(eps) == 1
    assert eps[0]["recovery_ts"] is None
