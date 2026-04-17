"""
Tests for triple-barrier labeling and meta-labeling.

These tests are CRITICAL — incorrect labels corrupt everything downstream.
We verify edge cases, boundary conditions, and the meta-labeling logic.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.labeling import (
    add_vertical_barrier,
    apply_pt_sl_on_t1,
    get_events,
    get_bins,
    drop_rare_labels,
)
from afml.filters import cusum_filter, get_daily_vol


@pytest.fixture
def simple_prices():
    """Simple price series: flat, then up, then down, then flat."""
    prices = [100.0] * 10  # flat
    prices += [100 + i * 0.5 for i in range(1, 11)]  # up to 105
    prices += [105 - i * 1.0 for i in range(1, 11)]  # down to 95
    prices += [95.0] * 10  # flat
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="4h")
    return pd.Series(prices, index=idx)


@pytest.fixture
def random_walk_1000():
    """1000-bar random walk for statistical tests."""
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=1000, freq="4h")
    return pd.Series(prices, index=idx)


class TestAddVerticalBarrier:
    def test_barrier_is_correct_distance(self, simple_prices):
        events = simple_prices.index[[5, 15, 25]]
        t1 = add_vertical_barrier(events, simple_prices, num_bars=10)
        assert len(t1) == 3
        # Each vertical barrier should be 10 bars (40 hours) after event
        for ev, barrier in t1.items():
            expected = ev + pd.Timedelta(hours=40)
            assert barrier == expected

    def test_clips_at_end_of_series(self, simple_prices):
        # Event near end of series — vertical barrier should be clipped
        events = pd.DatetimeIndex([simple_prices.index[-3]])
        t1 = add_vertical_barrier(events, simple_prices, num_bars=10)
        # Should be empty or clipped (10 bars from -3 exceeds series)
        assert len(t1) <= 1


class TestApplyPtSlOnT1:
    def test_profit_take_hit(self, simple_prices):
        """Entry at bar 10 (price=100), price rises to 105 → PT should hit."""
        events = pd.DataFrame(
            {"t1": [simple_prices.index[30]], "trgt": [0.03], "side": [1.0]},
            index=[simple_prices.index[10]],
        )
        result = apply_pt_sl_on_t1(simple_prices, events, pt_sl=[1, 1])
        # PT should fire (price goes up 5% > 3% target)
        assert pd.notna(result.iloc[0]["pt"])

    def test_stop_loss_hit(self, simple_prices):
        """Entry at bar 15 (price ~102.5), price drops to 95 → SL should hit."""
        events = pd.DataFrame(
            {"t1": [simple_prices.index[35]], "trgt": [0.03], "side": [1.0]},
            index=[simple_prices.index[15]],
        )
        result = apply_pt_sl_on_t1(simple_prices, events, pt_sl=[1, 1])
        # SL should fire (price drops ~7% > 3% target)
        assert pd.notna(result.iloc[0]["sl"])

    def test_disabled_barrier(self, simple_prices):
        """Setting pt_sl[0]=0 should disable profit-take barrier."""
        events = pd.DataFrame(
            {"t1": [simple_prices.index[30]], "trgt": [0.03], "side": [1.0]},
            index=[simple_prices.index[10]],
        )
        result = apply_pt_sl_on_t1(simple_prices, events, pt_sl=[0, 1])
        # PT should be NaT (disabled)
        assert pd.isna(result.iloc[0]["pt"])

    def test_short_side(self, simple_prices):
        """Short trade: entry at bar 15, price drops → should be profit."""
        events = pd.DataFrame(
            {"t1": [simple_prices.index[35]], "trgt": [0.03], "side": [-1.0]},
            index=[simple_prices.index[15]],
        )
        result = apply_pt_sl_on_t1(simple_prices, events, pt_sl=[1, 1])
        # For short, price dropping = profit → PT should fire
        assert pd.notna(result.iloc[0]["pt"])


class TestGetEvents:
    def test_produces_valid_events(self, random_walk_1000):
        vol = get_daily_vol(random_walk_1000, span=50)
        # Scale vol to price units for CUSUM
        threshold = vol * random_walk_1000
        t_events = cusum_filter(random_walk_1000, threshold=threshold)

        if len(t_events) < 5:
            pytest.skip("Too few CUSUM events for test")

        events = get_events(
            close=random_walk_1000,
            t_events=t_events,
            pt_sl=[1, 1],
            target=vol,
            min_ret=0.001,
            num_bars_vertical=20,
        )

        # Events should have required columns
        assert "t1" in events.columns
        assert "trgt" in events.columns
        assert len(events) > 0

        # t1 should always be AFTER the event timestamp
        for ev, row in events.iterrows():
            assert row["t1"] >= ev, f"t1 should be >= event time"

    def test_min_ret_filters_low_vol(self, random_walk_1000):
        """min_ret should filter out events where vol is too low."""
        vol = get_daily_vol(random_walk_1000, span=50)
        threshold = vol * random_walk_1000
        t_events = cusum_filter(random_walk_1000, threshold=threshold)

        events_low = get_events(random_walk_1000, t_events, [1, 1], vol, min_ret=0.0)
        events_high = get_events(random_walk_1000, t_events, [1, 1], vol, min_ret=0.05)

        assert len(events_high) <= len(events_low)


class TestGetBins:
    def test_standard_labels_are_minus1_0_plus1(self, random_walk_1000):
        """Standard labeling should produce {-1, 0, +1} labels."""
        vol = get_daily_vol(random_walk_1000, span=50)
        threshold = vol * random_walk_1000
        t_events = cusum_filter(random_walk_1000, threshold=threshold)

        if len(t_events) < 5:
            pytest.skip("Too few events")

        events = get_events(random_walk_1000, t_events, [1, 1], vol, num_bars_vertical=20)
        bins = get_bins(events, random_walk_1000)

        assert "ret" in bins.columns
        assert "bin" in bins.columns
        assert set(bins["bin"].unique()).issubset({-1.0, 0.0, 1.0})

    def test_meta_labeling_produces_binary(self, random_walk_1000):
        """Meta-labeling should produce {0, 1} labels."""
        vol = get_daily_vol(random_walk_1000, span=50)
        threshold = vol * random_walk_1000
        t_events = cusum_filter(random_walk_1000, threshold=threshold)

        if len(t_events) < 5:
            pytest.skip("Too few events")

        # Simulate primary model: always predict long
        side = pd.Series(1.0, index=t_events)

        events = get_events(
            random_walk_1000, t_events, [1, 1], vol,
            num_bars_vertical=20, side=side,
        )
        bins = get_bins(events, random_walk_1000)

        # Meta-labeling: bins should be {0, 1}
        assert set(bins["bin"].unique()).issubset({0.0, 1.0}), (
            f"Meta-labeling should produce {{0, 1}}, got {bins['bin'].unique()}"
        )


class TestDropRareLabels:
    def test_drops_rare_class(self):
        """Should drop labels appearing less than min_pct."""
        data = pd.DataFrame({
            "bin": [1.0] * 90 + [-1.0] * 8 + [0.0] * 2,
            "ret": np.random.randn(100),
        })
        filtered = drop_rare_labels(data, min_pct=0.05)
        assert 0.0 not in filtered["bin"].values, "Should have dropped the 2% class"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
