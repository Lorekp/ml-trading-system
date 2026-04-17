"""
Tests for features, evaluation metrics, and bet sizing.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.features import (
    parkinson_vol, corwin_schultz_spread, lempel_ziv_complexity,
    rolling_entropy, rolling_volume_zscore, hour_of_day, day_of_week,
)
from afml.evaluation import (
    rank_ic, directional_accuracy, net_sharpe,
    deflated_sharpe_ratio, max_drawdown, hhi_concentration,
)
from afml.bet_sizing import get_signal, discretize_signal, kelly_size


@pytest.fixture
def ohlcv():
    np.random.seed(42)
    n = 500
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.exponential(1000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({"open": close * 0.999, "high": high, "low": low,
                          "close": close, "volume": volume}, index=idx)


class TestParkinsonVol:
    def test_positive(self, ohlcv):
        vol = parkinson_vol(ohlcv["high"], ohlcv["low"], window=20)
        assert (vol.dropna() > 0).all()

    def test_higher_vol_when_range_wider(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="4h")
        high_narrow = pd.Series(101.0, index=idx)
        low_narrow = pd.Series(99.0, index=idx)
        high_wide = pd.Series(110.0, index=idx)
        low_wide = pd.Series(90.0, index=idx)
        vol_narrow = parkinson_vol(high_narrow, low_narrow, 20).dropna().mean()
        vol_wide = parkinson_vol(high_wide, low_wide, 20).dropna().mean()
        assert vol_wide > vol_narrow


class TestCorwinSchultz:
    def test_non_negative(self, ohlcv):
        spread = corwin_schultz_spread(ohlcv["high"], ohlcv["low"], 20)
        assert (spread.dropna() >= 0).all()


class TestLempelZiv:
    def test_constant_string_low_complexity(self):
        # Constant string: patterns are {0, 00, 000, ...} → ~sqrt(2n)/n
        # For n=100: ~14 patterns / 100 = 0.14
        assert lempel_ziv_complexity("0" * 100) < 0.2

    def test_random_string_higher_complexity(self):
        np.random.seed(42)
        random_str = "".join(str(np.random.randint(0, 5)) for _ in range(100))
        assert lempel_ziv_complexity(random_str) > 0.3

    def test_constant_lower_than_random(self):
        const = lempel_ziv_complexity("111111111111111111111111")
        rand = lempel_ziv_complexity("01234012340123401234012340")
        assert const < rand


class TestRollingEntropy:
    def test_returns_series(self, ohlcv):
        ent = rolling_entropy(ohlcv["close"], window=50)
        assert isinstance(ent, pd.Series)
        assert len(ent) == len(ohlcv) - 1  # one less due to pct_change


class TestRankIC:
    def test_perfect_prediction(self):
        pred = pd.Series([1, 2, 3, 4, 5], index=range(5))
        ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=range(5))
        assert rank_ic(pred, ret) == pytest.approx(1.0)

    def test_inverse_prediction(self):
        pred = pd.Series([5, 4, 3, 2, 1], index=range(5))
        ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=range(5))
        assert rank_ic(pred, ret) == pytest.approx(-1.0)

    def test_random_near_zero(self):
        np.random.seed(42)
        pred = pd.Series(np.random.randn(1000))
        ret = pd.Series(np.random.randn(1000))
        ic = rank_ic(pred, ret)
        assert abs(ic) < 0.1, f"Random signals should have IC near 0, got {ic}"


class TestDirectionalAccuracy:
    def test_perfect(self):
        pred = pd.Series([1, -1, 1, -1, 1])
        actual = pd.Series([0.1, -0.1, 0.1, -0.1, 0.1])
        assert directional_accuracy(pred, actual) == 1.0

    def test_random_near_half(self):
        np.random.seed(42)
        pred = pd.Series(np.random.choice([-1, 1], 1000))
        actual = pd.Series(np.random.randn(1000))
        acc = directional_accuracy(pred, actual)
        assert 0.45 < acc < 0.55, f"Random should be ~0.5, got {acc}"


class TestNetSharpe:
    def test_costs_reduce_sharpe(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        sharpe_gross = net_sharpe(returns, cost_per_trade=0, annualization=252)
        sharpe_net = net_sharpe(returns, cost_per_trade=0.002, annualization=252)
        assert sharpe_net < sharpe_gross


class TestDSR:
    def test_many_trials_high_threshold(self):
        """With 100 trials, DSR should be very skeptical of moderate Sharpe."""
        dsr = deflated_sharpe_ratio(
            observed_sr=1.5, num_trials=100, num_observations=500,
        )
        assert dsr < 0.95, f"100 trials should deflate SR=1.5 below significance"

    def test_few_trials_easier(self):
        """With 5 trials, moderate Sharpe should be more credible."""
        dsr_few = deflated_sharpe_ratio(
            observed_sr=2.0, num_trials=5, num_observations=500,
        )
        dsr_many = deflated_sharpe_ratio(
            observed_sr=2.0, num_trials=100, num_observations=500,
        )
        assert dsr_few > dsr_many, "Fewer trials → higher DSR"


class TestMaxDrawdown:
    def test_known_drawdown(self):
        equity = pd.Series([100, 110, 90, 100])
        dd = max_drawdown(equity)
        # Peak 110, trough 90 → dd = 20/110 ≈ 0.1818
        assert dd == pytest.approx(20 / 110, abs=0.001)


class TestHHI:
    def test_single_trade_max(self):
        returns = pd.Series([10.0])
        assert hhi_concentration(returns) == pytest.approx(1.0)

    def test_equal_distribution_low(self):
        returns = pd.Series([1.0] * 100)
        hhi = hhi_concentration(returns)
        assert hhi < 0.05, f"100 equal trades should have low HHI, got {hhi}"


class TestBetSizing:
    def test_fifty_percent_zero_signal(self):
        assert get_signal(0.5) == pytest.approx(0.0, abs=0.01)

    def test_high_prob_positive_signal(self):
        assert get_signal(0.9) > 0.5

    def test_low_prob_negative_signal(self):
        assert get_signal(0.1) < -0.5

    def test_signal_bounded(self):
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            s = get_signal(p)
            assert -1 <= s <= 1, f"Signal out of bounds for p={p}: {s}"


class TestDiscretize:
    def test_rounds_correctly(self):
        assert discretize_signal(0.34, step_size=0.1) == pytest.approx(0.3)
        assert discretize_signal(0.36, step_size=0.1) == pytest.approx(0.4)


class TestKelly:
    def test_edge_positive(self):
        f = kelly_size(win_prob=0.6, win_return=0.10, loss_return=-0.10)
        assert f > 0, "Positive edge should give positive Kelly"
        assert f == pytest.approx(0.2, abs=0.01)  # (1*0.6 - 0.4)/1 = 0.2

    def test_no_edge_zero(self):
        f = kelly_size(win_prob=0.5, win_return=0.10, loss_return=-0.10)
        assert f == pytest.approx(0.0, abs=0.01)

    def test_negative_edge(self):
        f = kelly_size(win_prob=0.3, win_return=0.10, loss_return=-0.10)
        assert f < 0, "Negative edge should give negative Kelly (don't bet)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
