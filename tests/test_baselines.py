"""
Tests for baseline models.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from models.baselines import (
    AlwaysFlat, BuyAndHold, MomentumBaseline, SARIMABaseline, evaluate_baseline,
)


@pytest.fixture
def trending_up():
    """Clearly trending up price series."""
    idx = pd.date_range("2024-01-01", periods=500, freq="4h")
    prices = 100 + np.linspace(0, 50, 500) + np.random.normal(0, 1, 500)
    return pd.Series(prices, index=idx)


@pytest.fixture
def random_walk():
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=500, freq="4h")
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 500)))
    return pd.Series(prices, index=idx)


class TestAlwaysFlat:
    def test_predicts_zero(self, random_walk):
        model = AlwaysFlat()
        pred = model.predict(random_walk)
        assert (pred == 0).all()


class TestBuyAndHold:
    def test_predicts_one(self, random_walk):
        model = BuyAndHold()
        pred = model.predict(random_walk)
        assert (pred == 1).all()

    def test_profitable_in_uptrend(self, trending_up):
        model = BuyAndHold()
        pred = model.predict(trending_up)
        returns = trending_up.pct_change().dropna()
        gross_ret = (pred.iloc[1:] * returns).mean()
        assert gross_ret > 0, "Buy-and-hold should profit in uptrend"


class TestMomentum:
    def test_predicts_direction(self, random_walk):
        model = MomentumBaseline(lookback=7)
        pred = model.predict(random_walk)
        # Should have +1 and -1 values
        assert pred.dropna().abs().mean() == pytest.approx(1.0, abs=0.2)

    def test_longer_lookback_smoother(self, random_walk):
        pred_short = MomentumBaseline(lookback=3).predict(random_walk)
        pred_long = MomentumBaseline(lookback=14).predict(random_walk)
        # Longer lookback should change direction less frequently
        changes_short = pred_short.diff().abs().dropna().sum()
        changes_long = pred_long.diff().abs().dropna().sum()
        assert changes_long < changes_short


class TestSARIMA:
    def test_fast_predict_runs(self, random_walk):
        """SARIMA fast predict should run without error."""
        model = SARIMABaseline(order=(1, 0, 0), seasonal_order=(0, 0, 0, 6))
        pred = model.predict_fast(random_walk, eval_points=10)
        non_nan = pred.dropna()
        assert len(non_nan) >= 5, f"Should have some predictions, got {len(non_nan)}"


class TestEvaluateBaseline:
    def test_perfect_predictor(self):
        np.random.seed(42)
        n = 100
        returns = pd.Series(np.random.choice([0.01, -0.01], n), name="actual")
        predictions = pd.Series(np.sign(returns), name="perfect")
        result = evaluate_baseline(predictions, returns, cost_per_trade=0)
        assert result["accuracy"] == 1.0
        assert result["gross_mean_return"] > 0

    def test_costs_reduce_returns(self):
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        predictions = pd.Series(np.sign(np.random.randn(100)), name="random")
        result_no_cost = evaluate_baseline(predictions, returns, cost_per_trade=0)
        result_with_cost = evaluate_baseline(predictions, returns, cost_per_trade=0.002)
        assert result_with_cost["net_mean_return"] <= result_no_cost["gross_mean_return"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
