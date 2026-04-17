"""
Tests for CUSUM filter and event sampling.

Verifies:
1. CUSUM fires on significant moves, not noise
2. Higher threshold = fewer events
3. Dynamic threshold adapts to volatility
4. Z-score filter catches statistical anomalies
5. get_daily_vol produces sensible volatility estimates
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.filters import cusum_filter, z_score_filter, get_daily_vol


@pytest.fixture
def trending_series():
    """Price series with a clear trend: flat, then spike, then flat."""
    np.random.seed(42)
    n = 500
    # Flat period, then spike at bar 200, then flat again
    prices = np.ones(n) * 100.0
    prices[200:] += 10.0  # $10 jump at bar 200
    # Add small noise
    prices += np.random.normal(0, 0.1, n).cumsum() * 0.01
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.Series(prices, index=idx)


@pytest.fixture
def random_walk():
    """Standard random walk."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=1000, freq="4h")
    return pd.Series(prices, index=idx)


class TestCusumFilter:
    def test_detects_spike(self, trending_series):
        """CUSUM should fire around the spike at bar 200."""
        events = cusum_filter(trending_series, threshold=5.0)
        assert len(events) > 0, "Should detect the $10 spike"
        # At least one event should be near bar 200
        spike_time = trending_series.index[200]
        near_spike = events[(events >= spike_time - pd.Timedelta(hours=20)) &
                           (events <= spike_time + pd.Timedelta(hours=20))]
        assert len(near_spike) > 0, "Should fire near the spike"

    def test_higher_threshold_fewer_events(self, random_walk):
        """Higher threshold should produce fewer events."""
        events_low = cusum_filter(random_walk, threshold=0.5)
        events_high = cusum_filter(random_walk, threshold=2.0)
        assert len(events_low) > len(events_high), (
            f"Low threshold should fire more. Got {len(events_low)} vs {len(events_high)}"
        )

    def test_zero_threshold_fires_every_bar(self, random_walk):
        """Threshold of 0 should fire on every non-zero change."""
        events = cusum_filter(random_walk, threshold=0.0)
        # Should fire very frequently (almost every bar)
        assert len(events) > len(random_walk) * 0.8

    def test_dynamic_threshold(self, random_walk):
        """Dynamic (Series) threshold should work and adapt."""
        vol = get_daily_vol(random_walk, span=50)
        # CRITICAL: vol is in percentage terms (e.g., 0.01 = 1%).
        # CUSUM operates on raw price diffs. Must scale: threshold = vol * price.
        threshold = vol * random_walk
        events = cusum_filter(random_walk, threshold=threshold)
        assert len(events) > 0, "Dynamic threshold should produce some events"
        # Event rate should be reasonable with vol-scaled threshold
        event_rate = len(events) / len(random_walk)
        assert 0.01 < event_rate < 0.8, f"Event rate {event_rate:.2f} seems extreme"

    def test_returns_datetimeindex(self, random_walk):
        """Output should be a DatetimeIndex."""
        events = cusum_filter(random_walk, threshold=1.0)
        assert isinstance(events, pd.DatetimeIndex)

    def test_events_within_data_range(self, random_walk):
        """All events should be within the data's time range."""
        events = cusum_filter(random_walk, threshold=1.0)
        if len(events) > 0:
            assert events.min() >= random_walk.index[1]  # First diff at index[1]
            assert events.max() <= random_walk.index[-1]


class TestGetDailyVol:
    def test_positive_output(self, random_walk):
        """Volatility should be positive."""
        vol = get_daily_vol(random_walk, span=50)
        assert (vol.dropna() > 0).all()

    def test_higher_vol_during_volatile_period(self):
        """Vol estimate should be higher during volatile periods."""
        np.random.seed(42)
        n = 500
        # Low vol first half, high vol second half
        returns = np.concatenate([
            np.random.normal(0, 0.005, n // 2),
            np.random.normal(0, 0.05, n // 2),
        ])
        prices = 100 * np.exp(np.cumsum(returns))
        series = pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="4h"))

        vol = get_daily_vol(series, span=50)
        vol_first = vol.iloc[60:n // 2].mean()
        vol_second = vol.iloc[n // 2 + 60:].mean()
        assert vol_second > vol_first * 3, "Second half should be much more volatile"


class TestZScoreFilter:
    def test_detects_outliers(self):
        """Z-score filter should detect injected outliers."""
        np.random.seed(42)
        values = np.random.normal(100, 1, 500)
        # Inject outliers
        values[200] = 120  # 20 sigma outlier
        values[350] = 80   # -20 sigma outlier
        series = pd.Series(values, index=pd.date_range("2024-01-01", periods=500, freq="4h"))

        events = z_score_filter(series, mean_window=50, std_window=50, z_threshold=3.0)
        assert len(events) >= 2, "Should detect at least the 2 injected outliers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
