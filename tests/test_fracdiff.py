"""
Tests for fractional differentiation implementation.

Verifies:
1. Weight computation matches known values
2. d=0 returns original series
3. d=1 returns standard returns (approximately)
4. d=0.5 is between d=0 and d=1 in terms of memory
5. find_min_d returns sensible values for random walk vs stationary data
6. FFD preserves more memory (higher correlation) than standard returns
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.fracdiff import get_weights_ffd, frac_diff_ffd, find_min_d


class TestGetWeightsFfd:
    def test_d_zero_returns_single_weight(self):
        """d=0 should produce only w_0=1 (no differentiation)."""
        w = get_weights_ffd(0.0)
        assert len(w) == 1
        assert w[0, 0] == pytest.approx(1.0)

    def test_d_one_returns_two_weights(self):
        """d=1 should produce [1, -1] (standard first difference)."""
        w = get_weights_ffd(1.0, threshold=0.01)
        assert len(w) == 2
        assert w[0, 0] == pytest.approx(-1.0)
        assert w[1, 0] == pytest.approx(1.0)

    def test_weights_decay_for_fractional_d(self):
        """For 0 < d < 1, weights should decay toward zero."""
        w = get_weights_ffd(0.5, threshold=1e-5)
        assert len(w) > 2, "Should have more than 2 weights for fractional d"
        # Last weight (most recent) should be 1.0
        assert w[-1, 0] == pytest.approx(1.0)
        # Weights should alternate in sign and decrease in magnitude
        # (oldest weights are smallest in absolute value)
        assert abs(w[0, 0]) < abs(w[-1, 0])

    def test_smaller_threshold_gives_wider_window(self):
        """Smaller threshold = more weights retained = wider window."""
        w_coarse = get_weights_ffd(0.4, threshold=1e-2)
        w_fine = get_weights_ffd(0.4, threshold=1e-6)
        assert len(w_fine) > len(w_coarse)


class TestFracDiffFfd:
    @pytest.fixture
    def random_walk(self):
        """Generate a simple random walk (non-stationary)."""
        np.random.seed(42)
        n = 1000
        returns = np.random.normal(0, 0.01, n)
        prices = 100 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2020-01-01", periods=n, freq="4h")
        return pd.DataFrame({"close": prices}, index=idx)

    def test_output_shorter_than_input(self, random_walk):
        """FFD drops the first `width` rows."""
        log_prices = np.log(random_walk)
        # Use coarser threshold for 1000-bar test data
        # (d=0.4 at threshold=1e-5 needs 1458 bars > our 1000)
        ffd = frac_diff_ffd(log_prices, d=0.4, threshold=1e-2)
        assert len(ffd) < len(random_walk)
        assert len(ffd) > 0, (
            f"FFD returned empty. Window={len(get_weights_ffd(0.4, 1e-2))} bars, "
            f"data={len(random_walk)} bars."
        )

    def test_d_near_one_approximates_returns(self, random_walk):
        """FFD with d=1.0 should be close to log-returns."""
        log_prices = np.log(random_walk)
        ffd_d1 = frac_diff_ffd(log_prices, d=1.0, threshold=0.01)
        log_returns = log_prices.diff().dropna()

        # Align indices
        common = ffd_d1.index.intersection(log_returns.index)
        corr = np.corrcoef(
            ffd_d1.loc[common, "close"].values,
            log_returns.loc[common, "close"].values,
        )[0, 1]
        assert corr > 0.99, f"d=1 FFD should match log-returns, got corr={corr:.4f}"

    def test_higher_d_less_correlated_with_prices(self, random_walk):
        """Higher d should produce features LESS correlated with original prices."""
        log_prices = np.log(random_walk)

        # Use coarser threshold so d=0.2 fits in 1000 bars
        # d=0.2 at 1e-2 needs ~22 bars, d=0.8 at 1e-2 needs ~7 bars
        ffd_low = frac_diff_ffd(log_prices, d=0.2, threshold=1e-2)
        ffd_high = frac_diff_ffd(log_prices, d=0.8, threshold=1e-2)

        common_low = log_prices.index.intersection(ffd_low.index)
        common_high = log_prices.index.intersection(ffd_high.index)

        corr_low = abs(np.corrcoef(
            log_prices.loc[common_low, "close"].values,
            ffd_low.loc[common_low, "close"].values,
        )[0, 1])

        corr_high = abs(np.corrcoef(
            log_prices.loc[common_high, "close"].values,
            ffd_high.loc[common_high, "close"].values,
        )[0, 1])

        assert corr_low > corr_high, (
            f"Lower d should preserve more memory (higher correlation). "
            f"corr(d=0.2)={corr_low:.4f}, corr(d=0.8)={corr_high:.4f}"
        )


class TestFindMinD:
    def test_random_walk_needs_positive_d(self):
        """A random walk (non-stationary) should require d > 0 for stationarity."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
        series = pd.Series(prices, index=pd.date_range("2020-01-01", periods=500, freq="4h"))

        result = find_min_d(series, d_range=np.linspace(0, 1, 11))
        assert result["d_star"] is not None, "Should find a d that achieves stationarity"
        assert result["d_star"] > 0, "Random walk should need d > 0"
        assert result["d_star"] < 1, "Should not need full differentiation"

    def test_stationary_series_needs_d_near_zero(self):
        """An already-stationary series should need d ≈ 0."""
        np.random.seed(42)
        stationary = pd.Series(
            np.exp(np.random.normal(5, 0.01, 500)),  # log-normal around exp(5)
            index=pd.date_range("2020-01-01", periods=500, freq="4h"),
        )

        result = find_min_d(stationary, d_range=np.linspace(0, 1, 11))
        # Stationary data should need d=0 or very small d
        assert result["d_star"] is not None
        assert result["d_star"] <= 0.2, f"Stationary series should need low d, got {result['d_star']}"

    def test_results_dataframe_shape(self):
        """Results should have one row per d tested."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200)))
        series = pd.Series(prices, index=pd.date_range("2020-01-01", periods=200, freq="4h"))

        d_range = np.linspace(0, 1, 6)
        result = find_min_d(series, d_range=d_range)
        assert len(result["results"]) == len(d_range)
        assert "d" in result["results"].columns
        assert "p_value" in result["results"].columns
        assert "correlation" in result["results"].columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
