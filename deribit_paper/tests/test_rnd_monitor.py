"""Tests for deribit_paper/risk/rnd_monitor.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.risk.rnd_monitor import RNDConfig, RNDMonitor, RNDSnapshot


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_params_df(
    *,
    timestamp: pd.Timestamp,
    expiry_dt: pd.Timestamp,
    F: float = 50_000.0,
    T: float | None = None,
    a: float = 0.001,
    b: float = 0.1,
    rho: float = -0.3,
    m: float = 0.0,
    sigma: float = 0.1,
) -> pd.DataFrame:
    """Build a minimal SVI params DataFrame with one slice row."""
    if T is None:
        T = (expiry_dt - timestamp).total_seconds() / (365.25 * 24 * 3600)
    return pd.DataFrame([{
        "timestamp": timestamp,
        "expiry_dt": expiry_dt,
        "T": T,
        "F": F,
        "a": a,
        "b": b,
        "rho": rho,
        "m": m,
        "sigma": sigma,
        "level": 2,
        "rmse": 0.001,
        "n_strikes": 10,
    }])


def _base_time() -> pd.Timestamp:
    return pd.Timestamp("2026-04-10 12:00:00", tz="UTC")


# ── Construction ────────────────────────────────────────────────────────

def test_monitor_construct_defaults():
    m = RNDMonitor()
    assert m.n_samples == 0
    assert m.latest is None
    assert m.should_halt() is False
    assert m.size_scale() == 1.0


def test_monitor_no_params_returns_none():
    m = RNDMonitor()
    snapshot = m.update(
        timestamp=_base_time(),
        spot=50_000.0,
        params_df=None,
    )
    assert snapshot is None
    assert m.n_samples == 0


def test_monitor_empty_params_returns_none():
    m = RNDMonitor()
    snapshot = m.update(
        timestamp=_base_time(),
        spot=50_000.0,
        params_df=pd.DataFrame(),
    )
    assert snapshot is None


# ── Front slice selection ──────────────────────────────────────────────

def test_front_slice_filtered_by_dte_band():
    """Only slices within [min_dte_days, max_dte_days] are eligible."""
    t = _base_time()
    # Build 3 slices: too close (12h), in-band (5d), too far (45d)
    params = pd.concat([
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(hours=12)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=5)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=45)),
    ], ignore_index=True)

    m = RNDMonitor()
    snapshot = m.update(timestamp=t, spot=50_000.0, params_df=params)

    assert snapshot is not None
    # The 5-day expiry should be selected
    assert abs(snapshot.front_T_years * 365.25 - 5.0) < 0.01


def test_front_slice_picks_closest_to_target_in_band():
    """When multiple slices are eligible, pick the one closest to target_dte_days.

    Default target is 7 days. Given {4d, 7d, 12d} all in the [3, 14]
    band, the 7d slice should win because |7-7| < |4-7| = |12-7| = 3.
    """
    t = _base_time()
    params = pd.concat([
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=4)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=7)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=12)),
    ], ignore_index=True)

    m = RNDMonitor()
    snapshot = m.update(timestamp=t, spot=50_000.0, params_df=params)

    assert snapshot is not None
    assert abs(snapshot.front_T_years * 365.25 - 7.0) < 0.01


def test_front_slice_tie_break_picks_lower_dte():
    """Ties on |dte - target| break deterministically via idxmin → lower DTE."""
    t = _base_time()
    # 5d and 9d are equidistant from target=7; idxmin returns the first
    # occurrence, which is the 5d row (ordered first in the DataFrame).
    params = pd.concat([
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=5)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=9)),
    ], ignore_index=True)

    m = RNDMonitor()
    snapshot = m.update(timestamp=t, spot=50_000.0, params_df=params)

    assert snapshot is not None
    assert abs(snapshot.front_T_years * 365.25 - 5.0) < 0.01


def test_front_slice_rejects_out_of_band():
    """DTEs outside the [3, 14] default band must all be rejected."""
    t = _base_time()
    params = pd.concat([
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=1.5)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=15)),
        _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=30)),
    ], ignore_index=True)

    m = RNDMonitor()
    snapshot = m.update(timestamp=t, spot=50_000.0, params_df=params)
    assert snapshot is None


def test_front_slice_returns_none_when_all_outside_band():
    t = _base_time()
    params = _make_params_df(
        timestamp=t, expiry_dt=t + pd.Timedelta(hours=6),  # too close
    )
    m = RNDMonitor()
    snapshot = m.update(timestamp=t, spot=50_000.0, params_df=params)
    assert snapshot is None


# ── RND computation ────────────────────────────────────────────────────

def test_snapshot_basic_shape():
    t = _base_time()
    params = _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=7))
    m = RNDMonitor()
    snapshot = m.update(timestamp=t, spot=50_000.0, params_df=params)

    assert snapshot is not None
    assert isinstance(snapshot, RNDSnapshot)
    # Tail probs should be in [0, 1]
    assert 0.0 <= snapshot.tail_prob_down <= 1.0
    assert 0.0 <= snapshot.tail_prob_up <= 1.0
    assert 0.0 <= snapshot.tail_prob_total <= 1.0
    # Integral of a well-calibrated SVI density should be ~1
    assert 0.8 <= snapshot.rnd_integral <= 1.2
    # Tail probabilities should not be zero for a reasonable slice
    assert snapshot.tail_prob_total > 0


def test_tail_threshold_is_vol_regime_insensitive_under_sigma_mult():
    """Sigma-mult tail threshold should give similar tail probabilities
    for slices with the same RND SHAPE but different overall vol levels.

    This is the whole point of the sigma-multiple design: the
    threshold scales with sqrt(w_ATM), so a 2-sigma tail is always a
    2-sigma tail regardless of whether implied vol is at 30% or 80%.
    The earlier fixed-percentage and sqrt(T)-scaled-percent designs
    did not have this property and produced either vanishing or
    saturated tail probabilities depending on the vol regime.
    """
    t = _base_time()
    cfg = RNDConfig(tail_sigma_mult=2.0)
    m1 = RNDMonitor(config=cfg)
    m2 = RNDMonitor(config=cfg)

    # Same shape parameters (a, b, rho, m, sigma) → same RND shape.
    # Different spot → different absolute vol but same k-space shape.
    params_low = _make_params_df(
        timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        F=30_000.0, b=0.10, sigma=0.10,
    )
    params_high = _make_params_df(
        timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        F=120_000.0, b=0.10, sigma=0.10,
    )
    s_low = m1.update(timestamp=t, spot=30_000.0, params_df=params_low)
    s_high = m2.update(timestamp=t, spot=120_000.0, params_df=params_high)

    assert s_low is not None and s_high is not None
    # Tail probability should be nearly identical — within 10%.
    rel_diff = abs(s_low.tail_prob_total - s_high.tail_prob_total) / max(
        s_low.tail_prob_total, 1e-9,
    )
    assert rel_diff < 0.10, (
        f"sigma-mult should normalize vol regime; got {s_low.tail_prob_total:.4f} vs "
        f"{s_high.tail_prob_total:.4f} (rel diff {rel_diff:.3f})"
    )


def test_baseline_is_max_of_short_and_long_medians():
    """Baseline must equal max(short_window_median, long_window_median).

    Construct a history where the short-window median drifts higher
    than the long-window median, then the same in reverse, and check
    that ``_baseline_tail_prob`` always returns the max of the two.
    """
    from deribit_paper.risk.rnd_monitor import RNDSnapshot
    import numpy as np

    cfg = RNDConfig(
        baseline_window_bars=5,
        baseline_long_window_bars=10,
        baseline_warmup_bars=3,
    )
    m = RNDMonitor(config=cfg)
    t0 = _base_time()

    # Pattern: 5 low samples (0.01) then 5 high samples (0.10).
    # long-window (last 10) median = 0.05 (average of lows and highs);
    # short-window (last 5) median = 0.10 (the recent block).
    # Baseline = max(0.10, 0.05) = 0.10.
    vals_low = [0.01] * 5
    vals_high = [0.10] * 5
    values = vals_low + vals_high

    for i, v in enumerate(values):
        snap = RNDSnapshot(
            timestamp=t0 + pd.Timedelta(minutes=15 * i),
            front_expiry=t0 + pd.Timedelta(days=7),
            front_T_years=7 / 365.25,
            spot=50_000.0, forward=50_000.0,
            tail_prob_down=v / 2, tail_prob_up=v / 2,
            tail_prob_total=v,
            rnd_integral=1.0, rnd_skewness=0.0, rnd_kurtosis=0.0,
        )
        m._history.append(snap)

    baseline = m._baseline_tail_prob()
    assert baseline is not None
    assert baseline == pytest.approx(0.10, abs=1e-12)

    # Reverse: 5 highs then 5 lows.
    # long-window (all 10): sorted [0.01]×5 + [0.10]×5, even count
    # → median = (0.01 + 0.10) / 2 = 0.055.
    # short-window (last 5) = [0.01]×5 → median = 0.01.
    # Baseline = max(0.01, 0.055) = 0.055.
    m2 = RNDMonitor(config=cfg)
    values_rev = vals_high + vals_low
    for i, v in enumerate(values_rev):
        snap = RNDSnapshot(
            timestamp=t0 + pd.Timedelta(minutes=15 * i),
            front_expiry=t0 + pd.Timedelta(days=7),
            front_T_years=7 / 365.25,
            spot=50_000.0, forward=50_000.0,
            tail_prob_down=v / 2, tail_prob_up=v / 2,
            tail_prob_total=v,
            rnd_integral=1.0, rnd_skewness=0.0, rnd_kurtosis=0.0,
        )
        m2._history.append(snap)

    baseline_rev = m2._baseline_tail_prob()
    assert baseline_rev is not None
    assert baseline_rev == pytest.approx(0.055, abs=1e-12)


def test_skewed_slice_has_asymmetric_tail_mass():
    """A slice with strongly negative skew (rho) has more downside tail mass.

    Under the sigma-mult design, absolute tail probability is
    normalized out of vol level — but RND SHAPE still matters.  A
    strongly right-skewed density (large negative rho) concentrates
    probability asymmetrically; the total 2-sigma tail probability
    shifts accordingly because the density is fatter on one side.
    """
    t = _base_time()
    # Symmetric baseline (rho = 0)
    symmetric = _make_params_df(
        timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        rho=0.0, b=0.15, sigma=0.15, m=0.0,
    )
    # Strongly negatively-skewed — market prices a put premium
    skewed = _make_params_df(
        timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        rho=-0.8, b=0.15, sigma=0.15, m=0.0,
    )

    m1 = RNDMonitor()
    m2 = RNDMonitor()
    s_sym = m1.update(timestamp=t, spot=50_000.0, params_df=symmetric)
    s_skew = m2.update(timestamp=t, spot=50_000.0, params_df=skewed)

    assert s_sym is not None and s_skew is not None
    # The skewed slice should have asymmetric tail mass: more down-tail
    # than up-tail (market prices negative returns more heavily).
    assert s_skew.tail_prob_down > s_skew.tail_prob_up
    # Skew should be reflected in the RND skewness statistic
    assert s_skew.rnd_skewness < s_sym.rnd_skewness


# ── Rolling baseline & halt ────────────────────────────────────────────

def test_halt_inactive_during_warmup():
    """No halt while baseline has fewer than baseline_warmup_bars samples."""
    t0 = _base_time()
    cfg = RNDConfig(baseline_warmup_bars=10)
    m = RNDMonitor(config=cfg)

    # 5 snapshots — below warmup threshold
    for i in range(5):
        t = t0 + pd.Timedelta(minutes=15 * i)
        params = _make_params_df(
            timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        )
        m.update(timestamp=t, spot=50_000.0, params_df=params)

    assert not m.should_halt()
    assert m.size_scale() == 1.0
    assert m._baseline_tail_prob() is None


def test_halt_fires_on_fat_tailed_rnd_shift():
    """A shift from smooth to fat-tailed (jumpy) RND must halt.

    Under the sigma-mult design, vol-level shifts alone do not change
    the 2σ tail mass (they scale out of the threshold).  What DOES
    change it is the RND SHAPE — specifically, high kurtosis (peaked
    density with fat tails, i.e., the market pricing in jumps).  A
    smooth RND has tail mass ≈ 3-4% (close to the normal-distribution
    4.6% at 2σ); a peaked jumpy RND can reach 5-8%.

    Baseline: wide, smooth slices (sigma=0.15) → tail ≈ 3%.
    Spike: tight, jumpy slice (sigma=0.05) → tail ≈ 5% plus high kurt.
    Halt at halt_mult=1.4× baseline fires on this shift.
    """
    t0 = _base_time()
    cfg = RNDConfig(
        baseline_warmup_bars=5,
        tail_halt_mult=1.4,
    )
    m = RNDMonitor(config=cfg)

    # Baseline: smooth, non-jumpy RND slices.
    for i in range(10):
        t = t0 + pd.Timedelta(minutes=15 * i)
        calm = _make_params_df(
            timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
            rho=-0.3, b=0.15, sigma=0.15, m=0.0,
        )
        m.update(timestamp=t, spot=50_000.0, params_df=calm)

    baseline = m._baseline_tail_prob()
    assert baseline is not None

    # Spike: the ATM curvature tightens while fat tails appear — high
    # kurtosis, peaked density, characteristic of the market pricing
    # an impending jump.
    t = t0 + pd.Timedelta(minutes=15 * 11)
    spike = _make_params_df(
        timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        rho=-0.3, b=0.15, sigma=0.05, m=0.0,
    )
    snap = m.update(timestamp=t, spot=50_000.0, params_df=spike)
    assert snap is not None
    # Verify the spike actually has fat tails (kurtosis dramatically higher)
    assert snap.rnd_kurtosis > 2.0

    assert m.should_halt()
    assert m.halt_reason() is not None
    assert "RND tail prob" in m.halt_reason()


def test_size_scale_linear_ramp_between_baseline_and_halt():
    """Size scale ramps linearly from 1.0 (at baseline) to size_min (at halt)."""
    t0 = _base_time()
    cfg = RNDConfig(
        baseline_warmup_bars=3,
        tail_halt_mult=2.0,
        size_min=0.25,
    )
    m = RNDMonitor(config=cfg)

    # Baseline
    for i in range(5):
        t = t0 + pd.Timedelta(minutes=15 * i)
        calm = _make_params_df(
            timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
            b=0.05, sigma=0.06,
        )
        m.update(timestamp=t, spot=50_000.0, params_df=calm)

    baseline = m._baseline_tail_prob()
    assert baseline is not None
    assert baseline > 0

    # Force a synthetic snapshot RIGHT AT the baseline → scale == 1
    from deribit_paper.risk.rnd_monitor import RNDSnapshot
    m._latest = RNDSnapshot(
        timestamp=t0, front_expiry=t0 + pd.Timedelta(days=7),
        front_T_years=7 / 365.25, spot=50_000.0, forward=50_000.0,
        tail_prob_down=baseline / 2, tail_prob_up=baseline / 2,
        tail_prob_total=baseline,
        rnd_integral=1.0, rnd_skewness=0.0, rnd_kurtosis=0.0,
    )
    assert abs(m.size_scale() - 1.0) < 1e-9

    # Halfway → scale ~ 0.625 (midpoint between 1.0 and 0.25)
    m._latest.tail_prob_total = 1.5 * baseline
    assert abs(m.size_scale() - 0.625) < 1e-9

    # At halt level → scale = size_min
    m._latest.tail_prob_total = 2.0 * baseline
    assert abs(m.size_scale() - 0.25) < 1e-9

    # Beyond halt level → still size_min (clamped)
    m._latest.tail_prob_total = 5.0 * baseline
    assert abs(m.size_scale() - 0.25) < 1e-9


# ── History & telemetry ────────────────────────────────────────────────

def test_history_caps_at_longer_of_baseline_windows():
    """The rolling history deque bounds at max(short, long) baseline windows.

    The monitor needs enough history to compute BOTH the short-window
    median and the long-window median, so the deque must be sized to
    the longer of the two.
    """
    cfg = RNDConfig(baseline_window_bars=5, baseline_long_window_bars=10)
    m = RNDMonitor(config=cfg)
    t0 = _base_time()

    for i in range(30):
        t = t0 + pd.Timedelta(minutes=15 * i)
        params = _make_params_df(
            timestamp=t, expiry_dt=t + pd.Timedelta(days=7),
        )
        m.update(timestamp=t, spot=50_000.0, params_df=params)

    assert m.n_samples == 10  # capped at the longer window


def test_to_dict_includes_all_telemetry_keys():
    t = _base_time()
    params = _make_params_df(timestamp=t, expiry_dt=t + pd.Timedelta(days=7))
    m = RNDMonitor()
    m.update(timestamp=t, spot=50_000.0, params_df=params)

    d = m.to_dict()
    assert set(d.keys()) >= {
        "n_samples", "is_halted", "halt_reason", "size_scale",
        "latest_tail_prob", "latest_skew", "latest_kurt",
        "baseline_tail_prob",
    }
    assert d["n_samples"] == 1
    assert d["latest_tail_prob"] is not None
