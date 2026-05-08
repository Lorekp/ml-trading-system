"""S-1d (parity-r3 cycle 7 — 2026-05-08) — TSRV-on-dollar-bar-grid.

Locks `TSRVDollarBarFeatures` against an inline pandas reference
that replicates the offline `_ffill_to_bars + _window_and_aggregate`
pipeline on a synthetic (TSRV buffer, dollar-bar buffer) fixture.

Strategy: build the SAME computation both ways, evaluate at the
latest dollar-bar position (which equals offline's
`groupby(window_labels).last().iloc[-1]` for the most recent 15-min
window), and assert equality to numerical-precision tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.tsrv_dollar_bar_grid import TSRVDollarBarFeatures
from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
from deribit_paper.features.tsrv_live import BarTSRV


# ── Fixtures ───────────────────────────────────────────────────────


def _make_bar_tsrv(
    *,
    vol_ann: float = 0.5,
    up_jump: float = 0.05,
    down_jump: float = 0.04,
    bpv: float = 0.045,
    rv_slow: float = 0.05,
    rv_fast: float = 0.04,
    n_ticks: int = 1000,
) -> BarTSRV:
    return BarTSRV(
        tsrv=vol_ann ** 2 * 1e-4, rv_fast=rv_fast, rv_slow=rv_slow,
        bpv=bpv, medrv=0.045, continuous_var=0.04, jump_var=0.005,
        n_ticks=n_ticks, n_sub_bars=10, n_dash=20.0,
        rs_down=0.02, rs_up=0.02,
        jump_fraction=up_jump + down_jump,
        down_jump_fraction=down_jump,
        up_jump_fraction=up_jump,
        vol_ann=vol_ann, vol_ann_continuous=vol_ann * 0.99,
    )


def _make_dollar_bar(ts: pd.Timestamp, close: float = 70_000.0) -> DollarBar:
    return DollarBar(
        timestamp=ts,
        open=close, close=close, high=close, low=close,
        duration_s=60.0, n_trades=10,
        dollar_volume=500_000.0, buy_volume=250_000.0, sell_volume=250_000.0,
        vov=0.0, imbalance=0.0,
        log_return=0.0, rv_bar=0.0,
    )


def _seed_tsrv_buffer(n: int, *, seed: int = 0) -> TSRVHistoryBuffer:
    """N TSRV bars at 15-min cadence with realistic varying values."""
    rng = np.random.default_rng(seed)
    buf = TSRVHistoryBuffer(max_size=max(n + 10, 200))
    base = pd.Timestamp("2026-04-25", tz="UTC")
    for i in range(n):
        ts = base + pd.Timedelta(minutes=15 * i)
        buf.append_bar(ts, _make_bar_tsrv(
            vol_ann=float(0.4 + 0.2 * rng.random()),
            up_jump=float(0.01 + 0.04 * rng.random()),
            down_jump=float(0.01 + 0.04 * rng.random()),
            bpv=float(0.03 + 0.04 * rng.random()),
            rv_slow=float(0.04 + 0.04 * rng.random()),
            rv_fast=float(0.02 + 0.06 * rng.random()),
            n_ticks=int(500 + 1500 * rng.random()),
        ))
    return buf


def _seed_dollar_bar_buffer(
    n: int, *, base: pd.Timestamp, gap_seconds: int = 60,
    max_size: int | None = None,
) -> DollarBarBuffer:
    """N dollar bars at ``gap_seconds`` cadence starting at ``base``.

    ``gap_seconds=60`` gives 60 dollar bars per hour (1440 per day),
    inside the typical BTC bpd range so 1d-window rolling is stable.
    """
    buf = DollarBarBuffer(max_size=max_size if max_size is not None else max(2 * n, 500))
    for i in range(n):
        buf.append(_make_dollar_bar(base + pd.Timedelta(seconds=gap_seconds * i)))
    return buf


@pytest.fixture
def feat_pair():
    """Standard fixture: 200 TSRV bars (15-min) + 6000 dollar bars
    spaced at 60-second cadence aligned with the TSRV grid."""
    n_tsrv = 200
    tsrv_buf = _seed_tsrv_buffer(n_tsrv)
    base_db = pd.Timestamp("2026-04-25", tz="UTC")
    db_buf = _seed_dollar_bar_buffer(6000, base=base_db, gap_seconds=60)
    feats = TSRVDollarBarFeatures(tsrv_buf, db_buf)
    return tsrv_buf, db_buf, feats


# ── _ffill_to_dollar_bars primitive ────────────────────────────────


def test_ffill_to_dollar_bars_matches_inline_reference(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    live = feats._ffill_to_dollar_bars("up_jump_fraction")
    # Reference: same computation done inline.
    tsrv_series = tsrv_buf.as_series("up_jump_fraction")
    db_idx = db_buf.as_index()
    lagged = tsrv_series.shift(1)
    union_idx = lagged.dropna().index.union(db_idx).sort_values()
    expected = lagged.reindex(union_idx).ffill().reindex(db_idx)
    pd.testing.assert_series_equal(live, expected, check_freq=False)


def test_ffill_returns_none_for_empty_buffers():
    tsrv_buf = TSRVHistoryBuffer(max_size=10)
    db_buf = DollarBarBuffer(max_size=10)
    feats = TSRVDollarBarFeatures(tsrv_buf, db_buf)
    assert feats._ffill_to_dollar_bars("up_jump_fraction") is None


def test_ffill_caches_and_invalidates_on_buffer_advance(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    s1 = feats._ffill_to_dollar_bars("up_jump_fraction")
    s2 = feats._ffill_to_dollar_bars("up_jump_fraction")
    assert s1 is s2  # cached
    # Append a new dollar bar -> cache invalidates
    base = pd.Timestamp("2026-04-30", tz="UTC")
    db_buf.append(_make_dollar_bar(base + pd.Timedelta(hours=1)))
    s3 = feats._ffill_to_dollar_bars("up_jump_fraction")
    assert s3 is not s1


# ── tsrv_log_vol_at_db (used by tsrv_mean_*, tsrv_ffd) ────────────


def test_tsrv_log_vol_at_db_matches_inline(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    live = feats._tsrv_log_vol_at_db()
    # Inline reference
    vol_ann = tsrv_buf.as_series("vol_ann")
    ln = np.log(vol_ann.clip(lower=1e-10))
    lagged = ln.shift(1)
    db_idx = db_buf.as_index()
    union_idx = lagged.dropna().index.union(db_idx).sort_values()
    expected = lagged.reindex(union_idx).ffill().reindex(db_idx)
    pd.testing.assert_series_equal(live, expected, check_freq=False)


# ── tsrv_mean_1d ──────────────────────────────────────────────────


def test_tsrv_mean_1d_matches_offline_formula(feat_pair):
    """tsrv_mean_1d at the latest dollar bar = inline rolling-mean
    of log(vol_ann) ffilled-to-db."""
    tsrv_buf, db_buf, feats = feat_pair
    bpd = feats._bpd()
    live = feats.tsrv_mean_1d()
    expected_series = feats._tsrv_log_vol_at_db().rolling(
        bpd * 1, min_periods=max(bpd // 4, 10),
    ).mean()
    expected = float(expected_series.iloc[-1])
    if np.isnan(expected):
        # If the rolling window's min_periods isn't met, both should be None.
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1d_lag1_matches_offline_formula(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    bpd = feats._bpd()
    live = feats.tsrv_mean_1d_lag(1)
    expected_series = feats._tsrv_log_vol_at_db().rolling(
        bpd * 1, min_periods=max(bpd // 4, 10),
    ).mean()
    expected = float(expected_series.iloc[-2])
    if np.isnan(expected):
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1d_ema10_matches_offline_formula(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    bpd = feats._bpd()
    rolling = feats._tsrv_log_vol_at_db().rolling(
        bpd * 1, min_periods=max(bpd // 4, 10),
    ).mean()
    expected = float(rolling.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1])
    live = feats.tsrv_mean_1d_ema(10)
    if np.isnan(expected):
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=12)


# ── tsrv_mean_1w ──────────────────────────────────────────────────


def test_tsrv_mean_1w_matches_offline_formula(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    bpd = feats._bpd()
    expected_series = feats._tsrv_log_vol_at_db().rolling(
        min(max(bpd * 7, 10), 50_000),
        min_periods=max(min(max(bpd * 7, 10), 50_000) // 4, 10),
    ).mean()
    expected = float(expected_series.iloc[-1])
    live = feats.tsrv_mean_1w()
    if np.isnan(expected):
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1w_lag3_matches_offline_formula(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    bpd = feats._bpd()
    expected_series = feats._tsrv_log_vol_at_db().rolling(
        min(max(bpd * 7, 10), 50_000),
        min_periods=max(min(max(bpd * 7, 10), 50_000) // 4, 10),
    ).mean()
    expected = float(expected_series.iloc[-4])
    live = feats.tsrv_mean_1w_lag(3)
    if np.isnan(expected):
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1w_ema50_matches_offline_formula(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    bpd = feats._bpd()
    rolling = feats._tsrv_log_vol_at_db().rolling(
        min(max(bpd * 7, 10), 50_000),
        min_periods=max(min(max(bpd * 7, 10), 50_000) // 4, 10),
    ).mean()
    expected = float(rolling.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1])
    live = feats.tsrv_mean_1w_ema(50)
    if np.isnan(expected):
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=12)


# ── tsrv_ffd (frac_diff_ffd on dollar-bar grid) ──────────────────


def test_tsrv_ffd_matches_inline_frac_diff(feat_pair):
    """tsrv_ffd on dollar-bar grid = frac_diff_ffd(log_vol_at_db)."""
    from src.afml.fracdiff import frac_diff_ffd
    tsrv_buf, db_buf, feats = feat_pair
    ln_at_db = feats._tsrv_log_vol_at_db()
    expected_df = frac_diff_ffd(
        ln_at_db.to_frame("tsrv"), d=0.4, threshold=1e-5,
    )
    live = feats.tsrv_ffd()
    if expected_df["tsrv"].empty:
        assert live is None
    else:
        # Reindex onto the full ln_at_db index for the latest position
        expected_full = expected_df["tsrv"].reindex(ln_at_db.index)
        expected = float(expected_full.iloc[-1])
        if np.isnan(expected):
            assert live is None
        else:
            np.testing.assert_almost_equal(live, expected, decimal=10)


def test_tsrv_ffd_lag1_matches_inline(feat_pair):
    from src.afml.fracdiff import frac_diff_ffd
    tsrv_buf, db_buf, feats = feat_pair
    ln_at_db = feats._tsrv_log_vol_at_db()
    expected_full = frac_diff_ffd(
        ln_at_db.to_frame("tsrv"), d=0.4, threshold=1e-5,
    )["tsrv"].reindex(ln_at_db.index)
    live = feats.tsrv_ffd_lag(1)
    expected = float(expected_full.iloc[-2])
    if np.isnan(expected):
        assert live is None
    else:
        np.testing.assert_almost_equal(live, expected, decimal=10)


# ── down_jump_frac / up_jump_frac (ffilled directly) ──────────────


def test_down_jump_frac_raw_at_latest_db(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    expected_series = feats._down_jump_frac_at_db()
    expected = float(expected_series.iloc[-1])
    live = feats.down_jump_frac()
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_down_jump_frac_lag3(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    expected_series = feats._down_jump_frac_at_db()
    expected = float(expected_series.iloc[-4])
    live = feats.down_jump_frac_lag(3)
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_down_jump_frac_ema50(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    series = feats._down_jump_frac_at_db()
    expected = float(series.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1])
    live = feats.down_jump_frac_ema(50)
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_up_jump_frac_raw(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    series = feats._up_jump_frac_at_db()
    expected = float(series.iloc[-1])
    live = feats.up_jump_frac()
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_up_jump_frac_ema_returns_finite_values(feat_pair):
    """On the dollar-bar grid, ema10 and ema50 of an ffilled TSRV
    series collapse near-identically because the same TSRV value is
    repeated across many dollar bars between TSRV refreshes.
    Sanity-check finiteness only."""
    tsrv_buf, db_buf, feats = feat_pair
    e10 = feats.up_jump_frac_ema(10)
    e50 = feats.up_jump_frac_ema(50)
    assert e10 is not None and e50 is not None
    assert np.isfinite(e10) and np.isfinite(e50)


# ── noise_level ────────────────────────────────────────────────────


def test_noise_level_raw_matches_offline_formula(feat_pair):
    """noise_level_at_db = ffilled (rv_fast / (2 × max(n_ticks, 1)))."""
    tsrv_buf, db_buf, feats = feat_pair
    rv_fast = tsrv_buf.as_series("rv_fast")
    n_ticks = tsrv_buf.as_series("n_ticks")
    tsrv_scalar = rv_fast / (2.0 * n_ticks.clip(lower=1.0))
    db_idx = db_buf.as_index()
    lagged = tsrv_scalar.shift(1)
    union_idx = lagged.dropna().index.union(db_idx).sort_values()
    expected_series = lagged.reindex(union_idx).ffill().reindex(db_idx)
    expected = float(expected_series.iloc[-1])
    live = feats.noise_level()
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_noise_level_lag1_matches_inline(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    series = feats._noise_level_at_db()
    expected = float(series.iloc[-2])
    live = feats.noise_level_lag(1)
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_noise_level_ema10_matches_inline(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    series = feats._noise_level_at_db()
    expected = float(series.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1])
    live = feats.noise_level_ema(10)
    np.testing.assert_almost_equal(live, expected, decimal=12)


# ── bpv_tsrv_ratio ─────────────────────────────────────────────────


def test_bpv_tsrv_ratio_raw_matches_offline_formula(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    bpv = tsrv_buf.as_series("bpv")
    rv_slow = tsrv_buf.as_series("rv_slow")
    tsrv_scalar = bpv / rv_slow.clip(lower=1e-20)
    db_idx = db_buf.as_index()
    lagged = tsrv_scalar.shift(1)
    union_idx = lagged.dropna().index.union(db_idx).sort_values()
    expected_series = lagged.reindex(union_idx).ffill().reindex(db_idx)
    expected = float(expected_series.iloc[-1])
    live = feats.bpv_tsrv_ratio()
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_bpv_tsrv_ratio_lag1_matches_inline(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    series = feats._bpv_tsrv_ratio_at_db()
    expected = float(series.iloc[-2])
    live = feats.bpv_tsrv_ratio_lag(1)
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_bpv_tsrv_ratio_ema50_matches_inline(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    series = feats._bpv_tsrv_ratio_at_db()
    expected = float(series.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1])
    live = feats.bpv_tsrv_ratio_ema(50)
    np.testing.assert_almost_equal(live, expected, decimal=12)


# ── Cache invalidation ────────────────────────────────────────────


def test_cache_invalidates_on_db_buffer_advance(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    s1 = feats._down_jump_frac_at_db()
    s2 = feats._down_jump_frac_at_db()
    assert s1 is s2  # cached
    base = pd.Timestamp("2026-05-01", tz="UTC")
    db_buf.append(_make_dollar_bar(base + pd.Timedelta(hours=2)))
    s3 = feats._down_jump_frac_at_db()
    assert s3 is not s1


def test_cache_invalidates_on_tsrv_buffer_advance(feat_pair):
    tsrv_buf, db_buf, feats = feat_pair
    s1 = feats._down_jump_frac_at_db()
    base = pd.Timestamp("2026-05-01", tz="UTC")
    tsrv_buf.append_bar(base, _make_bar_tsrv())
    s3 = feats._down_jump_frac_at_db()
    assert s3 is not s1


# ── Defensive: empty buffers ─────────────────────────────────────


def test_empty_buffers_return_none():
    tsrv_buf = TSRVHistoryBuffer(max_size=10)
    db_buf = DollarBarBuffer(max_size=10)
    feats = TSRVDollarBarFeatures(tsrv_buf, db_buf)
    assert feats.tsrv_mean_1d() is None
    assert feats.tsrv_mean_1w() is None
    assert feats.tsrv_ffd() is None
    assert feats.up_jump_frac() is None
    assert feats.down_jump_frac() is None
    assert feats.noise_level() is None
    assert feats.bpv_tsrv_ratio() is None


def test_db_buffer_only_returns_none():
    tsrv_buf = TSRVHistoryBuffer(max_size=10)
    base = pd.Timestamp("2026-04-25", tz="UTC")
    db_buf = _seed_dollar_bar_buffer(100, base=base)
    feats = TSRVDollarBarFeatures(tsrv_buf, db_buf)
    # No TSRV data → all should be None.
    assert feats.tsrv_mean_1d() is None
    assert feats.up_jump_frac() is None
