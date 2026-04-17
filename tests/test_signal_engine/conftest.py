"""Shared fixtures for signal_engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.config import SystemConfig


@pytest.fixture()
def synthetic_ohlcv() -> pd.DataFrame:
    """2000 bars of realistic crypto OHLCV data (seed 42).

    Generates a random-walk close price starting at 50_000 with ~0.1%
    per-bar volatility, realistic high/low spread, and log-normal volume.
    """
    rng = np.random.RandomState(42)
    n = 2000

    # Random-walk close price
    returns = rng.normal(0, 0.001, size=n)
    close = 50_000.0 * np.exp(np.cumsum(returns))

    # High/low from close with realistic spread
    spread = close * rng.uniform(0.0005, 0.003, size=n)
    high = close + spread * rng.uniform(0.3, 1.0, size=n)
    low = close - spread * rng.uniform(0.3, 1.0, size=n)

    # Open = previous close + small gap
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.0002, size=n))
    open_[0] = close[0]

    # Ensure OHLC consistency: high >= max(open,close), low <= min(open,close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.lognormal(mean=15, sigma=1.0, size=n)

    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


@pytest.fixture()
def sample_config() -> SystemConfig:
    """SystemConfig with test-appropriate values."""
    return SystemConfig(
        assets={"TEST": "TESTUSDT"},
        portfolio={
            "starting_capital": 10000.0,
            "max_dd": 0.15,
            "delever_start": 0.075,
            "slippage": 0.0001,
            "max_exposure": 0.80,
            "max_per_asset": 0.20,
            "base_sizing_pct": 0.02,
            "max_signal_mult": 2.0,
        },
    )


@pytest.fixture()
def sample_signals() -> pd.DataFrame:
    """DataFrame of pre-computed signals for filter/portfolio tests.

    Contains 500 signals spanning 100 days with realistic distributions.
    """
    rng = np.random.RandomState(42)
    n = 500

    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    probabilities = rng.beta(2, 2, size=n)  # clustered around 0.5
    directions = np.where(probabilities > 0.5, 1.0, -1.0)
    confidences = np.abs(probabilities - 0.5)
    fwd_returns = rng.normal(0.0005, 0.005, size=n)
    signal_strengths = np.abs(rng.normal(0, 0.3, size=n))

    assets = rng.choice(["BTC", "ETH", "XRP"], size=n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": directions,
            "fwd_return": fwd_returns,
            "confidence": confidences,
            "probability": probabilities,
            "signal_strength": signal_strengths,
            "asset": assets,
        }
    )
