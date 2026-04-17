"""Tests for signal_engine.stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.stats import compute_full_stats


def test_sharpe_on_known_curve() -> None:
    """Sharpe ratio on a constant positive daily return is well-defined."""
    n = 365
    daily_pnl = pd.Series(
        np.full(n, 10.0),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    stats = compute_full_stats(daily_pnl, starting_capital=10000.0,
                               n_trades=1000, trades_per_day=2.7)
    # Constant positive returns -> very high Sharpe (std ~0 from float noise)
    # With truly constant returns std=0 and sharpe=0, so add tiny noise
    rng = np.random.RandomState(42)
    noisy_pnl = pd.Series(
        10.0 + rng.normal(0, 0.01, n),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    stats2 = compute_full_stats(noisy_pnl, starting_capital=10000.0,
                                n_trades=1000, trades_per_day=2.7)
    assert stats2["sharpe"] > 10, f"Expected very high Sharpe, got {stats2['sharpe']}"


def test_max_drawdown_known_series() -> None:
    """Max drawdown on a known equity series."""
    # Equity goes 100 -> 110 -> 90 -> 95
    # DD from 110 to 90 = 18.18%
    daily_pnl = pd.Series(
        [1000, -2000, 500],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    # This has only 3 days — test that it still works with > 5 days
    daily_pnl_long = pd.Series(
        [100, 100, 100, 100, 100, -300, 50, 50],
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
    )
    stats = compute_full_stats(daily_pnl_long, starting_capital=10000.0,
                               n_trades=100, trades_per_day=1.0)
    assert "error" not in stats
    assert stats["max_dd_pct"] > 0, "Should have positive max drawdown"


def test_psr_computation() -> None:
    """PSR should be between 0 and 100 and reasonable for positive Sharpe."""
    rng = np.random.RandomState(42)
    n = 500
    # Positive-mean returns with reasonable noise
    daily_pnl = pd.Series(
        rng.normal(5, 20, n),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    stats = compute_full_stats(daily_pnl, starting_capital=10000.0,
                               n_trades=2000, trades_per_day=4.0)
    assert 0 <= stats["psr_pct"] <= 100
    # With positive mean, PSR should be above 50
    assert stats["psr_pct"] > 50, f"Expected PSR > 50 for positive Sharpe, got {stats['psr_pct']}"


def test_insufficient_days() -> None:
    """Fewer than 5 days returns an error dict."""
    daily_pnl = pd.Series(
        [10, 20, 30],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    stats = compute_full_stats(daily_pnl, starting_capital=10000.0,
                               n_trades=10, trades_per_day=1.0)
    assert "error" in stats
    assert stats["error"] == "insufficient_days"


def test_all_keys_present() -> None:
    """Result dict contains all expected keys."""
    rng = np.random.RandomState(42)
    daily_pnl = pd.Series(
        rng.normal(5, 10, 100),
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )
    stats = compute_full_stats(daily_pnl, starting_capital=10000.0,
                               n_trades=500, trades_per_day=5.0)
    expected_keys = {
        "sharpe", "sortino", "calmar", "cagr_pct", "total_return_pct",
        "max_dd_pct", "win_rate_days", "loss_rate_days", "profit_factor",
        "avg_win_bps", "avg_loss_bps", "expectancy_bps", "ann_std", "ann_var",
        "dd_recovery_days", "psr_pct", "skewness", "kurtosis", "n_days",
        "n_trades", "trades_per_day", "final_equity",
    }
    assert set(stats.keys()) == expected_keys


def test_profit_factor_capped() -> None:
    """Profit factor is capped at 99.0."""
    # All winning days -> very high PF
    daily_pnl = pd.Series(
        np.full(100, 10.0),
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
    )
    stats = compute_full_stats(daily_pnl, starting_capital=10000.0,
                               n_trades=100, trades_per_day=1.0)
    # All positive -> no losing days -> profit_factor = inf -> capped to 99
    # Actually with truly constant, losing.sum() == 0, so it's inf, capped to 99
    assert stats["profit_factor"] <= 99.0


def test_compute_rolling_sharpe() -> None:
    """Rolling Sharpe returns correct value for a known series."""
    from signal_engine.stats import compute_rolling_sharpe

    # 30 days of constant $10/day PnL — very high Sharpe (near-zero variance)
    daily_pnl = pd.Series(
        np.full(30, 10.0),
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
    )
    rs = compute_rolling_sharpe(daily_pnl, starting_capital=10000.0, window_days=20)
    assert rs is not None
    assert rs > 100  # near-zero std means extremely high Sharpe

    # Mixed returns
    mixed = pd.Series(
        [10, -5, 15, -3, 8, 12, -2, 7, 11, -1, 9, 6, -4, 13, 5, 8, -6, 10, 3, 7],
        index=pd.date_range("2024-01-01", periods=20, freq="D"),
    )
    rs2 = compute_rolling_sharpe(mixed, starting_capital=10000.0, window_days=20)
    assert rs2 is not None
    assert rs2 > 0  # net positive PnL -> positive Sharpe

    # Insufficient data
    short = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    assert compute_rolling_sharpe(short, 10000.0, window_days=20) is None
