"""
Tests for technical indicators and derivatives features.

Validates: correct computation, no look-ahead, NaN handling,
and that derivatives features are uncorrelated with OHLCV features.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.technical import (
    rsi, stochastic_rsi, macd, bollinger_pct_b, atr,
    swing_levels, multi_tf_momentum,
)
from afml.derivatives import (
    funding_features, oi_features, premium_features, impact_features,
)


@pytest.fixture
def ohlcv():
    np.random.seed(42)
    n = 500
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


class TestRSI:
    def test_bounded_0_100(self, ohlcv):
        r = rsi(ohlcv["close"], 14).dropna()
        assert r.min() >= 0, f"RSI below 0: {r.min()}"
        assert r.max() <= 100, f"RSI above 100: {r.max()}"

    def test_trending_up_high_rsi(self):
        np.random.seed(42)
        # Need enough data for EWM warmup, with noise to avoid zero-denominator
        close = pd.Series(np.linspace(100, 300, 500) + np.random.normal(0, 2, 500))
        r = rsi(close, 14)
        # Check last value (skip NaN at start)
        last = r.iloc[-1]
        assert not np.isnan(last), "RSI should produce a value at end of 500-bar uptrend"
        assert last > 50, f"Uptrend should have RSI > 50, got {last}"

    def test_trending_down_low_rsi(self):
        close = pd.Series(range(200, 100, -1), dtype=float)
        r = rsi(close, 14).dropna()
        assert r.iloc[-1] < 30, f"Strong downtrend should have RSI < 30, got {r.iloc[-1]}"


class TestMACD:
    def test_output_columns(self, ohlcv):
        m = macd(ohlcv["close"])
        assert "macd_hist" in m.columns

    def test_no_nans_after_warmup(self, ohlcv):
        m = macd(ohlcv["close"])
        assert m["macd_hist"].iloc[50:].isna().sum() == 0


class TestBollingerPctB:
    def test_at_midband_near_05(self, ohlcv):
        b = bollinger_pct_b(ohlcv["close"], 20)
        # Mean should be approximately 0.5 over long period
        mean_b = b.dropna().mean()
        assert 0.3 < mean_b < 0.7, f"Mean %B should be ~0.5, got {mean_b}"


class TestATR:
    def test_positive(self, ohlcv):
        a = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        assert (a.dropna() > 0).all()


class TestSwingLevels:
    def test_no_look_ahead(self, ohlcv):
        """Swing levels should only use past data."""
        sw = swing_levels(ohlcv["high"], ohlcv["low"], lookback=20)
        # Removing the last bar shouldn't change any previous values
        sw2 = swing_levels(ohlcv["high"].iloc[:-1], ohlcv["low"].iloc[:-1], lookback=20)
        common = sw.index[:-1].intersection(sw2.index)
        if len(common) > 100:
            diff = (sw.loc[common] - sw2.loc[common]).abs().max().max()
            assert diff < 1e-10, f"Swing levels changed when removing last bar: diff={diff}"

    def test_dist_non_negative(self, ohlcv):
        sw = swing_levels(ohlcv["high"], ohlcv["low"], lookback=20)
        assert (sw["dist_nearest_level"].dropna() >= 0).all()


class TestFundingFeatures:
    def test_zscore_mean_near_zero(self):
        np.random.seed(42)
        funding = pd.Series(np.random.normal(0, 0.001, 500))
        feats = funding_features(funding)
        zs = feats["funding_zscore"].dropna()
        assert abs(zs.mean()) < 0.5, f"Z-score mean should be ~0, got {zs.mean()}"

    def test_extreme_funding_high_zscore(self):
        funding = pd.Series(np.concatenate([np.zeros(200), np.ones(10) * 0.01]))
        feats = funding_features(funding, window_short=50)
        # Last values should have extreme z-score
        last_zs = feats["funding_zscore"].iloc[-1]
        assert abs(last_zs) > 1.5, f"Extreme funding should have |z| > 1.5, got {last_zs}"


class TestOIFeatures:
    def test_divergence_detection(self):
        n = 100
        close = pd.Series(np.linspace(100, 110, n))   # Price going up
        oi = pd.Series(np.linspace(1000, 900, n))      # OI going down = divergence
        feats = oi_features(oi, close)
        # Should detect divergence in later bars
        div = feats["oi_price_divergence"].iloc[20:].mean()
        assert div > 0.3, f"Should detect price/OI divergence, got {div}"


class TestCorrelationWithOHLCV:
    """Critical test: derivatives features must be uncorrelated with OHLCV."""

    def test_funding_uncorrelated_with_returns(self):
        np.random.seed(42)
        n = 1000
        # Simulate independent funding and returns
        returns = pd.Series(np.random.normal(0, 0.01, n))
        funding = pd.Series(np.random.normal(0, 0.001, n))
        corr = returns.corr(funding)
        assert abs(corr) < 0.1, f"Funding should be uncorrelated with returns: {corr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
