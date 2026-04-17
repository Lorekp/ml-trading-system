"""Tests for signal_engine.filters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.config import ConfidenceFilterConfig, RegimeFilterConfig
from signal_engine.filters import apply_confidence_filter, apply_regime_filter


def test_percentile_filter_keeps_fraction(sample_signals: pd.DataFrame) -> None:
    """Percentile 70 filter keeps approximately 30% of signals."""
    cfg = ConfidenceFilterConfig(mode="percentile", threshold=70.0)
    filtered = apply_confidence_filter(sample_signals, cfg)

    kept_frac = len(filtered) / len(sample_signals)
    # Should keep ~30% (70th percentile = top 30%)
    assert 0.20 <= kept_frac <= 0.40, f"Kept {kept_frac:.2%}, expected ~30%"


def test_absolute_filter_symmetry(sample_signals: pd.DataFrame) -> None:
    """Absolute filter keeps signals with |prob - 0.5| >= threshold."""
    # threshold=0.10 means keep signals where confidence (|prob-0.5|) >= 0.10
    # i.e., prob > 0.60 OR prob < 0.40
    cfg = ConfidenceFilterConfig(mode="absolute", threshold=0.10)
    filtered = apply_confidence_filter(sample_signals, cfg)

    # Should have both long (prob > 0.6) and short (prob < 0.4) signals
    has_long = (filtered["probability"] > 0.60).any()
    has_short = (filtered["probability"] < 0.40).any()
    assert has_long, "Should keep high-probability signals"
    assert has_short, "Should keep low-probability signals"

    # No signals in the middle band (confidence < 0.10)
    mid_band = filtered["confidence"] < 0.10
    assert not mid_band.any(), "Should have no low-confidence signals"


def test_regime_filter_direction_agreement() -> None:
    """Regime filter only keeps signals agreeing with the 4H trend."""
    # Create a clear bullish 4H regime (close above EMA)
    idx_4h = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
    close_4h = pd.Series(np.linspace(100, 200, 100), index=idx_4h)  # uptrend
    ohlcv_4h = pd.DataFrame(
        {
            "open": close_4h,
            "high": close_4h * 1.01,
            "low": close_4h * 0.99,
            "close": close_4h,
            "volume": np.ones(100),
        },
        index=idx_4h,
    )

    # Create signals: half long, half short, in the middle of the 4H range
    n = 40
    timestamps = pd.date_range("2024-01-05", periods=n, freq="h", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": np.tile([1.0, -1.0], n // 2),
            "fwd_return": np.random.default_rng(42).normal(0, 0.01, n),
            "confidence": np.full(n, 0.1),
            "probability": np.tile([0.6, 0.4], n // 2),
            "signal_strength": np.full(n, 0.3),
            "asset": "TEST",
        }
    )

    cfg = RegimeFilterConfig(enabled=True, ema_span=20)
    filtered = apply_regime_filter(signals, ohlcv_4h, cfg)

    # In a strong uptrend, should mostly keep longs
    if len(filtered) > 0:
        long_frac = (filtered["direction"] == 1.0).mean()
        assert long_frac > 0.8, f"Expected mostly longs in uptrend, got {long_frac:.2%}"


def test_empty_dataframe_passthrough() -> None:
    """Empty DataFrame returns empty DataFrame for both filters."""
    empty = pd.DataFrame()

    cfg_conf = ConfidenceFilterConfig(mode="percentile", threshold=70.0)
    assert len(apply_confidence_filter(empty, cfg_conf)) == 0

    ohlcv_4h = pd.DataFrame(
        {"close": [100], "open": [100], "high": [101], "low": [99], "volume": [1]},
        index=pd.date_range("2024-01-01", periods=1, freq="4h", tz="UTC"),
    )
    cfg_regime = RegimeFilterConfig(enabled=True, ema_span=20)
    assert len(apply_regime_filter(empty, ohlcv_4h, cfg_regime)) == 0


def test_none_passthrough() -> None:
    """None input returns empty DataFrame."""
    cfg = ConfidenceFilterConfig(mode="percentile", threshold=70.0)
    result = apply_confidence_filter(None, cfg)  # type: ignore[arg-type]
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
