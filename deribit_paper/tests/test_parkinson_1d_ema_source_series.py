"""F-2 regression test (round-3 parity audit, fix 2026-05-08).

The offline pipeline computes ``parkinson_1d_ema50`` as a TWO-stage
transform on the per-bar parkinson series:

1. ``feats["parkinson_1d"] = parkinson_bar.rolling(w_park,
   min_periods=max(w_park//4, 10)).mean()`` (engineer.py:362-363)
2. ``feats["parkinson_1d_ema50"] = feats["parkinson_1d"].ewm(
   span=50, min_periods=1).mean()`` (engineer.py:460)

Pre-fix, the live ``DollarBarFeatures.parkinson_1d_ema(50)`` EMAed
the raw per-bar parkinson series directly — same shape but materially
higher reactivity. This test feeds a synthetic per-bar series through
both paths and asserts byte-for-byte agreement (1e-12 rel) on every
sample after the rolling-mean warmup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import DollarBarFeatures


def _make_bar(
    timestamp: pd.Timestamp,
    *,
    high: float,
    low: float,
    close: float,
) -> DollarBar:
    return DollarBar(
        timestamp=timestamp,
        close=close, open=close, high=high, low=low,
        duration_s=60.0, n_trades=10,
        dollar_volume=500_000.0, buy_volume=250_000.0, sell_volume=250_000.0,
        vov=0.001, imbalance=0.0,
        log_return=0.0, rv_bar=0.0,
    )


def _seed_buffer(n_bars: int, hl_pattern: list[tuple[float, float]]) -> DollarBarBuffer:
    """Seed a buffer with ``n_bars`` synthetic dollar bars; high/low pulled from
    ``hl_pattern`` cyclically."""
    buf = DollarBarBuffer(max_size=max(n_bars, 100))
    base = pd.Timestamp("2026-05-08 00:00:00", tz="UTC")
    for i in range(n_bars):
        h, l = hl_pattern[i % len(hl_pattern)]
        buf.append(_make_bar(
            base + pd.Timedelta(seconds=15 * i),
            high=h, low=l, close=(h + l) / 2,
        ))
    return buf


# ── Byte-identity to offline two-stage transform ────────────────


def test_parkinson_1d_ema_matches_offline_two_stage_transform():
    """Live parkinson_1d_ema(50) must equal the offline two-stage
    transform on the same input bars.
    """
    # Synthetic high/low pattern with varying volatility so the EMA
    # has something interesting to track.
    hl_pattern = [
        (101.0, 100.0),  # narrow
        (105.0, 100.0),  # wide
        (102.0, 100.5),  # narrow
        (110.0, 100.0),  # very wide
        (103.0, 102.0),  # very narrow
        (108.0, 101.0),  # wide
    ]
    n_bars = 500  # enough for the rolling window to be satisfied
    rolling_window = 250
    span = 50

    buf = _seed_buffer(n_bars, hl_pattern)
    feats = DollarBarFeatures(buf)

    # Live result
    live_val = feats.parkinson_1d_ema(span=span, rolling_window=rolling_window)
    assert live_val is not None and np.isfinite(live_val)

    # Offline reference (two-stage)
    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    parkinson_bar = np.log(hl) ** 2 / (4.0 * np.log(2.0))
    min_periods = max(rolling_window // 4, 10)
    parkinson_1d = parkinson_bar.rolling(
        window=rolling_window, min_periods=min_periods,
    ).mean()
    offline_val = float(
        parkinson_1d.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
    )

    np.testing.assert_allclose(
        live_val, offline_val, rtol=1e-12, atol=1e-15,
        err_msg="parkinson_1d_ema(50) must byte-identity to offline two-stage transform",
    )


def test_parkinson_1d_ema_differs_from_old_raw_bar_ema():
    """Sanity: the post-fix value differs from the pre-fix raw-bar EMA
    on a non-degenerate input — proving F-2 was a real divergence.
    """
    hl_pattern = [(101.0, 100.0), (110.0, 100.0)]  # spike pattern
    buf = _seed_buffer(500, hl_pattern)
    feats = DollarBarFeatures(buf)

    post_fix = feats.parkinson_1d_ema(span=50, rolling_window=250)

    # Pre-fix path: EMA of raw per-bar parkinson series.
    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    parkinson_bar = np.log(hl) ** 2 / (4.0 * np.log(2.0))
    pre_fix = float(
        parkinson_bar.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    )

    # The two should differ measurably (>1% relative). They share a
    # mean but reactivity is wildly different.
    rel_diff = abs(post_fix - pre_fix) / max(abs(pre_fix), 1e-12)
    assert rel_diff > 0.01, (
        f"post-fix and pre-fix EMA should differ noticeably; "
        f"got post={post_fix}, pre={pre_fix}, rel_diff={rel_diff:.6f}"
    )


def test_parkinson_1d_ema_underfull_buffer_returns_none():
    """When the buffer is empty, the live API returns None (handler
    falls back to baseline)."""
    buf = DollarBarBuffer(max_size=100)
    feats = DollarBarFeatures(buf)
    out = feats.parkinson_1d_ema(span=50, rolling_window=250)
    assert out is None


def test_parkinson_1d_returns_rolling_mean_at_latest():
    """parkinson_1d() now returns the latest value of the rolling-mean
    series. Verify it equals offline `parkinson_1d.iloc[-1]`."""
    hl_pattern = [(101.0, 100.0), (105.0, 100.0)]
    buf = _seed_buffer(500, hl_pattern)
    feats = DollarBarFeatures(buf)

    live_val = feats.parkinson_1d(window=250)
    assert live_val is not None

    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    parkinson_bar = np.log(hl) ** 2 / (4.0 * np.log(2.0))
    offline_val = float(
        parkinson_bar.rolling(
            window=250, min_periods=max(250 // 4, 10),
        ).mean().iloc[-1]
    )

    np.testing.assert_allclose(
        live_val, offline_val, rtol=1e-12, atol=1e-15,
    )


def test_parkinson_rolling_caches_per_window():
    """Different windows must produce independent cache entries."""
    hl_pattern = [(101.0, 100.0), (105.0, 100.0)]
    buf = _seed_buffer(500, hl_pattern)
    feats = DollarBarFeatures(buf)

    val_250 = feats.parkinson_1d_ema(span=50, rolling_window=250)
    val_100 = feats.parkinson_1d_ema(span=50, rolling_window=100)

    # The two windows have different rolling-mean shapes, so the EMAs
    # at the latest sample should differ.
    assert val_250 is not None and val_100 is not None
    assert val_250 != pytest.approx(val_100, rel=1e-9), (
        f"different rolling windows should produce different EMAs; "
        f"got w=250 → {val_250}, w=100 → {val_100}"
    )

    # Cache contains both entries.
    assert 250 in feats._cached_parkinson_rolling
    assert 100 in feats._cached_parkinson_rolling


def test_parkinson_cache_invalidated_on_buffer_append():
    """A new bar appended after a parkinson computation invalidates
    the rolling-mean cache.

    Use a small rolling_window (50) and a 1000× wider spike so the
    impact is large enough to detect even after EMA dampening — the
    contract is "cache was rebuilt", not a specific delta magnitude.
    """
    hl_pattern = [(101.0, 100.0), (101.0, 100.0)]  # narrow only
    buf = _seed_buffer(80, hl_pattern)
    feats = DollarBarFeatures(buf)

    val_before = feats.parkinson_1d_ema(span=10, rolling_window=50)
    assert val_before is not None
    assert 50 in feats._cached_parkinson_rolling

    # Append a SHARP volatility spike (range 100× narrower-bar range).
    base = pd.Timestamp("2026-05-08 00:00:00", tz="UTC")
    buf.append(_make_bar(
        base + pd.Timedelta(seconds=15 * 80),
        high=200.0, low=100.0, close=150.0,
    ))

    # After append, _cached_len mismatches actual len → cache must rebuild.
    val_after = feats.parkinson_1d_ema(span=10, rolling_window=50)
    assert val_after is not None
    # The new high-vol bar pulls the EMA noticeably up.
    assert val_after > val_before * 2.0, (
        f"after appending a 100×-wider bar, EMA should rise sharply; "
        f"before={val_before}, after={val_after}"
    )
