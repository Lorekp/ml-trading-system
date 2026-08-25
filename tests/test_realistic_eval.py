"""Unit tests for vol_arb/realistic_eval.py — the real-path gamma leg + brackets.

The data-dependent end-to-end tests are guarded by file existence so the pure
algorithm tests always run.
"""
import os

import numpy as np
import pandas as pd
import pytest

from vol_arb import realistic_eval as re
from vol_arb.realistic_eval import (
    PricePanel,
    RealisticEvalConfig,
    compute_gamma_leg_realpath,
    _closed_form_gamma_relerr,
    _YEAR_MIN,
)

_TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
_HAS_DATA = os.path.exists(_TRADE_LOG) and os.path.exists("data/binance-1m/BTCUSDT")


def _synthetic_panel(t0_ns, n_bars, S0, rets):
    """Build a PricePanel from explicit minute returns starting at t0_ns."""
    ts = t0_ns + np.arange(n_bars, dtype=np.int64) * 60_000_000_000
    close = np.empty(n_bars)
    close[0] = S0
    close[1:] = S0 * np.exp(np.cumsum(rets))
    return PricePanel(ts.astype(np.int64), close)


def test_gamma_closed_form():
    # Pure algebra cross-check, machine precision.
    assert _closed_form_gamma_relerr() < 1e-12


def test_gamma_leg_matches_analytic_on_real_panel():
    # One trade over a known minute path; the engine's Ito gamma must equal the
    # hand-computed 0.5*Gamma*sum(r^2 - IV^2 dt)*S^2 with the short-sign flip.
    t0 = np.datetime64("2022-02-01T00:00:00", "ns").astype("int64")
    rets = np.array([0.001, -0.002, 0.0015, 0.0005])  # 5 bars total (anchor + 4 moves window of 4)
    panel = _synthetic_panel(t0, 20, 40000.0, np.concatenate([rets, np.zeros(15)]))

    entry = pd.to_datetime([pd.Timestamp("2022-02-01 00:00:00", tz="UTC")])
    exit_ = pd.to_datetime([pd.Timestamp("2022-02-01 00:04:00", tz="UTC")])  # 4 bars
    cfg = RealisticEvalConfig()
    g, hf, nreb, mah, pos, resid = compute_gamma_leg_realpath(
        entry.values, exit_.values,
        S_entry=np.array([40000.0]),
        delta_long=np.array([0.0]),
        gamma_long=np.array([1e-5]),
        signal=np.array([-1.0]),
        iv_used=np.array([0.6]),
        T_years=np.array([10 / 365.25]),
        panel=panel, cfg=cfg,
    )
    # Hand analytic over the same 4 minute returns.
    lo = np.searchsorted(panel.ts_i8, entry.values.astype("datetime64[ns]").astype("int64"), "left")[0]
    hi = np.searchsorted(panel.ts_i8, exit_.values.astype("datetime64[ns]").astype("int64"), "right")[0]
    Sw = panel.close[lo:hi + 0]
    # anchor first to S_entry (engine does Smat[:,0]=S_entry)
    Sw = Sw.copy(); Sw[0] = 40000.0
    r = np.diff(np.log(Sw))
    dt = 1.0 / _YEAR_MIN
    # sign factor is `signal` (short=-1): long-gamma form 0.5*Γ*Σ(r²−IV²dt)*S²
    analytic = (-1.0) * 0.5 * 1e-5 * (Sw[:-1] ** 2 * (r**2 - 0.6**2 * dt)).sum()
    assert np.isclose(g[0], analytic, rtol=1e-9, atol=1e-9)


def test_no_lookahead_mutation():
    # Shuffling bars BEYOND exit_ts must not change the gamma leg (causality).
    t0 = np.datetime64("2022-02-01T00:00:00", "ns").astype("int64")
    rng = np.random.RandomState(1)
    rets = rng.normal(0, 0.001, 30)
    panel_a = _synthetic_panel(t0, 31, 40000.0, rets)
    rets_b = rets.copy()
    rets_b[10:] = rng.normal(0, 0.005, 20)   # mutate everything after bar 10
    panel_b = _synthetic_panel(t0, 31, 40000.0, rets_b)

    entry = pd.to_datetime([pd.Timestamp("2022-02-01 00:00:00", tz="UTC")])
    exit_ = pd.to_datetime([pd.Timestamp("2022-02-01 00:05:00", tz="UTC")])  # within first 10 bars
    cfg = RealisticEvalConfig()
    kw = dict(S_entry=np.array([40000.0]), delta_long=np.array([0.0]),
              gamma_long=np.array([1e-5]), signal=np.array([-1.0]),
              iv_used=np.array([0.6]), T_years=np.array([10 / 365.25]), cfg=cfg)
    ga = compute_gamma_leg_realpath(entry.values, exit_.values, panel=panel_a, **kw)[0]
    gb = compute_gamma_leg_realpath(entry.values, exit_.values, panel=panel_b, **kw)[0]
    assert np.isclose(ga[0], gb[0])


def test_zero_fee_implied_path_gamma_near_zero():
    # Implied path: GBM at sigma=IV so RV==IV -> integrand (r^2 - IV^2 dt) mean-0.
    # On a finite draw the gamma leg should be SMALL relative to a real-vol-gap path.
    t0 = np.datetime64("2022-02-01T00:00:00", "ns").astype("int64")
    panel = _synthetic_panel(t0, 100, 40000.0, np.zeros(99))  # placeholder; implied path overrides
    n = 500
    entry = pd.to_datetime([pd.Timestamp("2022-02-01 00:00:00", tz="UTC")] * n)
    exit_ = pd.to_datetime([pd.Timestamp("2022-02-01 00:15:00", tz="UTC")] * n)
    cfg = RealisticEvalConfig(use_implied_path=True, zero_fees=True)
    g, *_ = compute_gamma_leg_realpath(
        entry.values, exit_.values,
        S_entry=np.full(n, 40000.0), delta_long=np.zeros(n),
        gamma_long=np.full(n, 1e-5), signal=np.full(n, -1.0),
        iv_used=np.full(n, 0.6), T_years=np.full(n, 10 / 365.25),
        panel=panel, cfg=cfg,
    )
    # Per-trade gamma has mean ~0; the magnitude per trade is tiny vs a 1% real move.
    # Reference: a single 1% move over a bar contributes ~0.5*1e-5*S^2*1e-4 ~ 0.8.
    # The implied-path average should be far below that scale.
    assert abs(g.mean()) < 0.5
    # And the CI should straddle 0:
    se = g.std() / np.sqrt(n)
    assert abs(g.mean()) < 3 * se + 0.1


def test_ito_short_sign():
    # Short straddle profits when realized var < implied var.
    t0 = np.datetime64("2022-02-01T00:00:00", "ns").astype("int64")
    # Calm path (tiny moves) but high IV -> short gamma should be POSITIVE.
    panel = _synthetic_panel(t0, 20, 40000.0, np.full(19, 1e-6))
    entry = pd.to_datetime([pd.Timestamp("2022-02-01 00:00:00", tz="UTC")])
    exit_ = pd.to_datetime([pd.Timestamp("2022-02-01 00:10:00", tz="UTC")])
    cfg = RealisticEvalConfig()
    g_short = compute_gamma_leg_realpath(
        entry.values, exit_.values, S_entry=np.array([40000.0]),
        delta_long=np.array([0.0]), gamma_long=np.array([1e-5]),
        signal=np.array([-1.0]), iv_used=np.array([1.0]),
        T_years=np.array([10 / 365.25]), panel=panel, cfg=cfg)[0]
    assert g_short[0] > 0   # calm realized, high implied -> short gamma wins
    # Long flips sign.
    g_long = compute_gamma_leg_realpath(
        entry.values, exit_.values, S_entry=np.array([40000.0]),
        delta_long=np.array([0.0]), gamma_long=np.array([1e-5]),
        signal=np.array([1.0]), iv_used=np.array([1.0]),
        T_years=np.array([10 / 365.25]), panel=panel, cfg=cfg)[0]
    assert np.isclose(g_long[0], -g_short[0])


def test_determinism():
    t0 = np.datetime64("2022-02-01T00:00:00", "ns").astype("int64")
    rng = np.random.RandomState(2)
    panel = _synthetic_panel(t0, 50, 40000.0, rng.normal(0, 0.001, 49))
    entry = pd.to_datetime([pd.Timestamp("2022-02-01 00:00:00", tz="UTC")])
    exit_ = pd.to_datetime([pd.Timestamp("2022-02-01 00:15:00", tz="UTC")])
    cfg = RealisticEvalConfig()
    kw = dict(S_entry=np.array([40000.0]), delta_long=np.array([0.0]),
              gamma_long=np.array([1e-5]), signal=np.array([-1.0]),
              iv_used=np.array([0.6]), T_years=np.array([10 / 365.25]),
              panel=panel, cfg=cfg)
    a = compute_gamma_leg_realpath(entry.values, exit_.values, **kw)[0]
    b = compute_gamma_leg_realpath(entry.values, exit_.values, **kw)[0]
    assert np.array_equal(a, b)


# ── Data-dependent end-to-end (guarded) ──
@pytest.mark.skipif(not _HAS_DATA, reason="trade log / kline data not present")
def test_v2_byte_identity():
    import hashlib
    h_before = hashlib.md5(open(_TRADE_LOG, "rb").read()).hexdigest()
    tl = pd.read_parquet(_TRADE_LOG)
    cfg = RealisticEvalConfig(start="2024-01-01", end="2024-01-04", fill_tol_minutes=60)
    re.evaluate_realistic(tl, cfg)
    h_after = hashlib.md5(open(_TRADE_LOG, "rb").read()).hexdigest()
    assert h_before == h_after  # engine must not mutate the input parquet


@pytest.mark.skipif(not _HAS_DATA, reason="trade log / kline data not present")
def test_bracket_and_gate_smoke():
    tl = pd.read_parquet(_TRADE_LOG)
    cfg = RealisticEvalConfig(start="2024-01-01", end="2024-01-08", fill_tol_minutes=60)
    result = re.evaluate_realistic(tl, cfg)
    bracket = re.assert_brackets(tl, result, cfg)
    # pess <= realistic <= optim (with tolerance); not a build failure.
    assert not bracket.bracket_high_is_failure
    assert bracket.closed_form_gamma_max_relerr < 1e-9
