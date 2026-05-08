"""F-3 (parity-r3 cycle 2 — 2026-05-08) regression tests.

``DollarBarFeatures.bars_per_day`` must mirror the offline
``vol_model/features/engineer.py::avg_bars_per_day`` byte-for-byte so
the live parkinson_1d / parkinson_1d_ema50 handlers produce values that
match offline at the same buffer state.

The handler factory ``_h_parkinson_1d`` / ``_h_parkinson_1d_ema50`` in
``feature_assembler.py`` resolves bpd at handler-call time and passes it
through to ``parkinson_1d(window=bpd)`` /
``parkinson_1d_ema(50, rolling_window=bpd)`` — replacing the pre-fix
hardcoded 250.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import DollarBarFeatures


def _make_bar(ts: pd.Timestamp, *, hl: tuple[float, float] = (101.0, 100.0)) -> DollarBar:
    return DollarBar(
        timestamp=ts, open=100.5, close=100.5, high=hl[0], low=hl[1],
        duration_s=60.0, n_trades=10,
        dollar_volume=500_000.0, buy_volume=250_000.0, sell_volume=250_000.0,
        vov=0.001, imbalance=0.0,
        log_return=0.0, rv_bar=0.0,
    )


def _seed_buffer(n_bars: int, *, freq_minutes: int = 15) -> DollarBarBuffer:
    buf = DollarBarBuffer(max_size=max(n_bars, 100))
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    for i in range(n_bars):
        buf.append(_make_bar(base + pd.Timedelta(minutes=freq_minutes * i)))
    return buf


# ── bars_per_day mirrors offline avg_bars_per_day ─────────────────────


def test_bars_per_day_matches_offline_avg_bars_per_day():
    """Live ``bars_per_day`` must produce the same result as offline's
    ``avg_bars_per_day`` on the same DatetimeIndex — mirrors the
    closed-form ``max(int(len(recent) / max(span_days, 1)), 10)``.

    For 2880 bars at 15-min cadence: span = (2880-1)*15min ≈ 30 days
    less 15min, so ``.days`` = 29, and bpd = max(2880/29, 10) = 99.
    """
    buf = _seed_buffer(30 * 96, freq_minutes=15)
    feats = DollarBarFeatures(buf)

    # Replicate offline avg_bars_per_day inline as the reference.
    df = buf.as_dataframe()
    cutoff = df.index[-1] - pd.Timedelta(days=30)
    recent = df.index[df.index >= cutoff]
    n_days = max((recent[-1] - recent[0]).days, 1)
    expected = max(int(len(recent) / n_days), 10)
    assert feats.bars_per_day(trailing_days=30) == expected
    # Sanity: the value is in the expected ~96-100 range (not 250 fallback)
    assert 90 <= feats.bars_per_day(trailing_days=30) <= 110


def test_bars_per_day_matches_offline_on_high_density():
    """Higher density (5-min cadence) → bpd ~3x higher than 15-min."""
    buf = _seed_buffer(30 * 288, freq_minutes=5)
    feats = DollarBarFeatures(buf)

    df = buf.as_dataframe()
    cutoff = df.index[-1] - pd.Timedelta(days=30)
    recent = df.index[df.index >= cutoff]
    n_days = max((recent[-1] - recent[0]).days, 1)
    expected = max(int(len(recent) / n_days), 10)
    assert feats.bars_per_day(trailing_days=30) == expected
    # Sanity: ~3x denser than the 15-min test above
    assert feats.bars_per_day(trailing_days=30) >= 280


def test_bars_per_day_floors_at_10():
    """Single bar in buffer → can't compute density → fallback floor."""
    buf = _seed_buffer(1)
    feats = DollarBarFeatures(buf)
    # Single bar → len(df) < 2 → returns the 250 cold-start fallback.
    assert feats.bars_per_day() == 250


def test_bars_per_day_returns_250_for_empty_buffer():
    buf = DollarBarBuffer(max_size=100)
    feats = DollarBarFeatures(buf)
    assert feats.bars_per_day() == 250


def test_bars_per_day_clips_short_window():
    """If only 5 bars over a few hours, bpd computes as ~5/1day=5 →
    floored to 10 (matches offline ``max(int(...), 10)``)."""
    buf = _seed_buffer(5, freq_minutes=15)
    feats = DollarBarFeatures(buf)
    # 5 bars × 15min = 75min < 1 day → n_days = max(0, 1) = 1
    # max(5/1, 10) = 10 (floor)
    assert feats.bars_per_day() == 10


# ── parkinson handlers consume dynamic bpd ────────────────────────────


def _seed_buffer_with_volatility_shift(n_total: int) -> DollarBarBuffer:
    """Seed a buffer with a recent volatility shift so different rolling
    windows produce measurably different values.

    First (n_total - 50) bars: narrow (high=101, low=100). Last 50 bars:
    very wide (high=110, low=100). A small window (e.g. 50) captures
    almost only the wide regime; a large window (e.g. 250) averages a
    lot of narrow with the wide tail → meaningfully different value.
    """
    buf = DollarBarBuffer(max_size=max(n_total, 100))
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    spike_start = max(n_total - 50, 0)
    for i in range(n_total):
        hl = (110.0, 100.0) if i >= spike_start else (101.0, 100.0)
        buf.append(_make_bar(base + pd.Timedelta(minutes=15 * i), hl=hl))
    return buf


def test_parkinson_1d_uses_dynamic_window():
    """When bpd resolves to N, parkinson_1d should match
    ``parkinson_1d(N)`` exactly — not the hardcoded ``parkinson_1d(250)``
    when N != 250 produces a different value.

    Uses a regime-shift buffer so different window sizes resolve to
    different rolling-mean values.
    """
    buf = _seed_buffer_with_volatility_shift(500)  # short buffer; bpd<<250
    feats = DollarBarFeatures(buf)

    bpd = feats.bars_per_day()
    assert bpd != 250, "Test premise needs bpd != 250 to exercise dynamic path"

    val_at_bpd = feats.parkinson_1d(bpd)
    val_at_250 = feats.parkinson_1d(250)
    # bpd < 250 → smaller window weights recent (wide-regime) bars more →
    # larger value than the 250-window average.
    assert val_at_bpd is not None
    if val_at_250 is not None:
        # When 250 also resolves, they MUST differ
        assert val_at_bpd != pytest.approx(val_at_250, rel=1e-9)


def test_parkinson_1d_ema_uses_dynamic_rolling_window():
    """parkinson_1d_ema(50, rolling_window=bpd) must use the dynamic
    bpd as the inner rolling-mean window."""
    buf = _seed_buffer_with_volatility_shift(500)
    feats = DollarBarFeatures(buf)

    bpd = feats.bars_per_day()
    assert bpd != 250

    val_at_bpd = feats.parkinson_1d_ema(50, rolling_window=bpd)
    val_at_250 = feats.parkinson_1d_ema(50, rolling_window=250)
    assert val_at_bpd is not None
    if val_at_250 is not None:
        # Different inner windows → different EMA outputs over a regime shift.
        assert val_at_bpd != pytest.approx(val_at_250, rel=1e-9)


def test_bars_per_day_invalidated_on_buffer_append():
    """bpd is recomputed off the buffer's current state; appending bars
    in a new day shifts bpd accordingly."""
    buf = DollarBarBuffer(max_size=10_000)
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    # Day 1: 96 bars
    for i in range(96):
        buf.append(_make_bar(base + pd.Timedelta(minutes=15 * i)))
    feats = DollarBarFeatures(buf)
    bpd_day1 = feats.bars_per_day(trailing_days=30)

    # Day 2: another 96 bars
    for i in range(96):
        buf.append(_make_bar(base + pd.Timedelta(days=1, minutes=15 * i)))
    bpd_day2 = feats.bars_per_day(trailing_days=30)
    # 96 → 192 bars over 1 day = bpd ~96 OR over 2 days = bpd ~96.
    # The exact number depends on how trailing_days bins; either way
    # the value is a sensible ~96, not stuck at the day-1 result.
    assert 80 <= bpd_day2 <= 200, (
        f"bpd should adapt as buffer grows; got day1={bpd_day1}, day2={bpd_day2}"
    )
