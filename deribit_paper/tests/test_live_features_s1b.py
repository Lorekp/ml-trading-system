"""S-1b (parity-r3 cycle 5 — 2026-05-08) regression tests.

Live equivalents of three offline TransVol-input feature groups:

- ``hurst_100`` — rolling Hurst exponent on log-returns (port of
  ``vol_model/features/engineer.py:_hurst_subsampled``).
- ``ffd_rv`` — fractionally-differenced log-close
  (port of ``frac_diff_ffd(log(close), d=0.4)``).
- ``rv_zscore_1d/1w/1m`` — rolling z-scores of realised variance
  (port of ``engineer.py:233-240``).

Each test verifies byte-equality (or strict ε-tolerance) against the
offline implementation on a fixed numerical slice. Pre-S-1b the live
engine had no path to these features — the encoder's ``tv_X``
embeddings were silently filled from a stale baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import (
    DollarBarFeatures,
    _ffd_weights,
    _hurst_subsampled_window,
)


# ── Fixture: synthetic dollar-bar buffer with realistic features ────


def _make_bar(
    ts: pd.Timestamp,
    *,
    close: float,
    high: float,
    low: float,
    log_return: float,
    rv_bar: float,
) -> DollarBar:
    return DollarBar(
        timestamp=ts,
        open=close, close=close, high=high, low=low,
        duration_s=60.0, n_trades=10,
        dollar_volume=500_000.0, buy_volume=250_000.0, sell_volume=250_000.0,
        vov=0.001, imbalance=0.0,
        log_return=log_return, rv_bar=rv_bar,
    )


def _seed_buffer(n: int, *, seed: int = 0, max_size: int | None = None) -> DollarBarBuffer:
    """Synthetic buffer: GBM log-prices + per-bar rv from squared log-returns.

    ``max_size`` defaults to ``2*n`` so subsequent appends in the same
    test don't trigger FIFO eviction (which would mask cache
    invalidation bugs in DollarBarFeatures).
    """
    rng = np.random.default_rng(seed)
    log_rets = rng.standard_normal(n) * 0.001
    log_prices = np.cumsum(log_rets)
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    buf = DollarBarBuffer(max_size=max_size if max_size is not None else max(2 * n, 200))
    for i in range(n):
        close = float(np.exp(log_prices[i]))
        # Synthetic high/low: ±0.5*|return| around close
        spread = max(abs(log_rets[i]) * 0.5, 1e-6)
        hi = close * np.exp(spread)
        lo = close * np.exp(-spread)
        rv = float(log_rets[i] ** 2)
        buf.append(_make_bar(
            base + pd.Timedelta(minutes=15 * i),
            close=close, high=hi, low=lo,
            log_return=float(log_rets[i]),
            rv_bar=rv,
        ))
    return buf


# ── ffd_rv ─────────────────────────────────────────────────────────


def test_ffd_weights_cached_per_d_threshold():
    w1 = _ffd_weights(0.4, 1e-5)
    w2 = _ffd_weights(0.4, 1e-5)
    assert w1 is w2  # cached


def test_ffd_weights_first_value_is_one():
    w = _ffd_weights(0.4, 1e-5)
    assert w[0] == pytest.approx(1.0)


def test_ffd_weights_matches_offline_recurrence():
    """Compare against the offline ``get_weights_ffd`` implementation.

    Offline stores weights in REVERSED order (``tuple(weights[::-1])``
    at ``fracdiff.py:42``) so that ``np.convolve(values, weights[::-1])``
    re-reverses them to chronological order before sliding. Live stores
    chronological directly because it does a single dot-product against
    a reversed values slice. Both encode the same recurrence; the test
    compares the live array to the reversed offline array.
    """
    from src.afml.fracdiff import get_weights_ffd
    offline = get_weights_ffd(0.4, 1e-5).ravel()
    live = _ffd_weights(0.4, 1e-5)
    assert live.shape == offline.shape
    np.testing.assert_array_almost_equal(live, offline[::-1], decimal=15)


def test_ffd_rv_byte_equal_to_offline_frac_diff_ffd():
    """Live `ffd_rv` reproduces offline `frac_diff_ffd(log(close))` at
    the latest bar within numerical precision."""
    from src.afml.fracdiff import frac_diff_ffd
    n = 2500
    buf = _seed_buffer(n, seed=42)
    feats = DollarBarFeatures(buf)

    log_close = np.log(buf.as_dataframe()["close"].to_numpy(dtype=np.float64))
    offline = frac_diff_ffd(
        pd.DataFrame({"p": log_close}), d=0.4, threshold=1e-5,
    )
    live = feats.ffd_rv()
    assert live is not None
    # Strict tolerance — same float ops should give bit-equality up to
    # floating-point reduction order.
    assert abs(live - float(offline["p"].iloc[-1])) < 1e-10


def test_ffd_rv_returns_none_on_underfull_buffer():
    """When buffer has fewer rows than the FFD window, return None."""
    buf = _seed_buffer(100)  # < 1458 weights for d=0.4
    feats = DollarBarFeatures(buf)
    assert feats.ffd_rv() is None


def test_ffd_rv_advances_with_new_bar():
    """A new bar shifts the trailing window; result must change."""
    n = 2500
    buf = _seed_buffer(n, seed=1)
    feats = DollarBarFeatures(buf)
    val_before = feats.ffd_rv()
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    buf.append(_make_bar(
        base + pd.Timedelta(minutes=15 * (n + 1)),
        close=70_000.0, high=71_000.0, low=69_000.0,
        log_return=0.005, rv_bar=2.5e-5,
    ))
    val_after = feats.ffd_rv()
    assert val_before != pytest.approx(val_after, abs=1e-12)


# ── hurst_100 ──────────────────────────────────────────────────────


def test_hurst_subsampled_window_in_unit_interval_for_random_walk():
    """Hurst of N(0,1) returns should be ≈0.5 (mean reversion neutral).
    Tolerate a wide band — 100-bar Hurst is noisy."""
    rng = np.random.default_rng(1)
    h = _hurst_subsampled_window(rng.standard_normal(200), window=100)
    assert h is not None
    assert 0.0 <= h <= 1.0
    # Statistical: Hurst of pure random walk should be in [0.3, 0.7]
    # for window=100. Wider band to guard against single-seed flakiness.
    assert 0.2 <= h <= 0.85


def test_hurst_subsampled_window_returns_none_on_short_input():
    rng = np.random.default_rng(0)
    assert _hurst_subsampled_window(rng.standard_normal(50), window=100) is None


def test_hurst_subsampled_window_returns_none_on_constant_returns():
    """Zero std → degenerate R/S. Returns None."""
    constant = np.zeros(200)
    assert _hurst_subsampled_window(constant, window=100) is None


def test_hurst_100_via_dollar_bar_features():
    """End-to-end: feature wrapper computes hurst on the buffer's
    log_return column, returns a finite value in [0, 1]."""
    buf = _seed_buffer(200, seed=7)
    feats = DollarBarFeatures(buf)
    h = feats.hurst_100()
    assert h is not None
    assert 0.0 <= h <= 1.0


def test_hurst_100_returns_none_when_buffer_too_short():
    buf = _seed_buffer(50)  # < window=100
    feats = DollarBarFeatures(buf)
    assert feats.hurst_100() is None


def test_hurst_100_byte_equal_to_offline_at_window_position():
    """The offline `_hurst_subsampled` computes Hurst at every position
    `pos` ∈ {window, window+step, ...}. Live computes at the LAST bar.
    When the buffer length equals exactly `window + k*step` for some k,
    live's value equals offline's value at that position (no
    interpolation needed)."""
    from vol_model.features.engineer import _hurst_subsampled
    rng = np.random.default_rng(11)
    returns = rng.standard_normal(200)  # window=100, step=50 → positions = [100, 150]
    offline = _hurst_subsampled(returns, window=100, step=50)
    # offline[199] is interpolated between positions 100 & 150; not a
    # direct match. Compare offline[150] (the second R/S evaluation)
    # vs live evaluated on returns[:150].
    live = _hurst_subsampled_window(returns[:150], window=100)
    assert live is not None
    # The second offline evaluation point IS the value at position 150.
    # Offline reports it at index 150 (positions[1]).
    if not np.isnan(offline[150]):
        np.testing.assert_almost_equal(live, offline[150], decimal=10)


# ── rv_zscore_1d / 1w / 1m ────────────────────────────────────────


def test_rv_zscore_horizon_validation():
    buf = _seed_buffer(200)
    feats = DollarBarFeatures(buf)
    with pytest.raises(ValueError):
        feats.rv_zscore("invalid_horizon")


def test_rv_zscore_returns_none_on_short_buffer():
    buf = _seed_buffer(15)
    feats = DollarBarFeatures(buf)
    assert feats.rv_zscore("1d") is None


def test_rv_zscore_1d_matches_offline_formula():
    """Live `rv_zscore_1d` equals offline byte-for-byte when both have
    the same buffer + bpd."""
    n = 2000
    buf = _seed_buffer(n, seed=33)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day(trailing_days=30)

    # Offline: rv_bar.rolling(w).sum() then mu + sigma rolling
    rv_bar = buf.as_dataframe()["rv_bar"].astype(np.float64)
    w = min(max(bpd * 1, 10), 50_000)
    mp = max(w // 4, 10)
    rv_roll = rv_bar.rolling(w, min_periods=mp).sum()
    mu = rv_roll.rolling(w, min_periods=mp).mean()
    sigma = rv_roll.rolling(w, min_periods=mp).std()
    expected = (rv_roll - mu) / sigma.clip(lower=1e-15)

    live = feats.rv_zscore("1d", bpd=bpd)
    assert live is not None
    np.testing.assert_almost_equal(
        live, float(expected.iloc[-1]), decimal=12,
    )


def test_rv_zscore_horizons_have_increasing_windows():
    """1d uses bpd × 1; 1w uses bpd × 7; 1m uses bpd × 30. The wider
    the window, the smaller the z-score's reactivity. Sanity-check
    that the three values aren't trivially identical."""
    n = 50_000  # need enough history for the 30× window
    buf = _seed_buffer(n, seed=99)
    feats = DollarBarFeatures(buf)
    z1d = feats.rv_zscore("1d")
    z1w = feats.rv_zscore("1w")
    z1m = feats.rv_zscore("1m")
    # All three should be finite if buffer is wide enough.
    assert z1d is not None and z1w is not None and z1m is not None
    # And they shouldn't all be identical (different windows ⇒ different smoothing)
    assert not (z1d == pytest.approx(z1w) and z1w == pytest.approx(z1m))
