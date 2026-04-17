"""Regression test — runs the full pipeline on real data and checks results.

Requires actual Binance 1m data in data/binance-1m.  Marked @slow because
it processes 16 assets through walk-forward training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from signal_engine.config import load_config
from signal_engine.pipeline import run_backtest

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_full_backtest_regression() -> None:
    """Run full backtest with default config and assert against reference metrics.

    Reference values from the expanded_portfolio_test (16-asset Approach C):
        Sharpe ~ 6.787, Return ~ 105.13%, MaxDD ~ 0.64%, Trades = 56285
    """
    config_path = Path("C:/Claude/ml-trading-system/configs/default.yaml")
    if not config_path.exists():
        pytest.skip("Config file not found")

    cfg = load_config(config_path)

    # Verify real data exists
    data_dir = Path(cfg.data.data_dirs[0])
    if not data_dir.exists():
        pytest.skip(f"Data directory not found: {data_dir}")

    results = run_backtest(cfg)
    stats = results["stats"]

    if "error" in stats:
        pytest.fail(f"Backtest returned error: {stats['error']}")

    # Log actual values for debugging
    logger.info("Regression results: %s", stats)

    # Assert against reference with 5% relative tolerance.
    # These are the HONEST numbers with the regime filter shift(1) fix
    # (no look-ahead bias).  Previous inflated values (Sharpe 6.787,
    # Return 105%) were caused by the 4H regime filter using the
    # current (unfinished) bar instead of the previous completed bar.
    assert stats["sharpe"] == pytest.approx(2.076, rel=0.05), (
        f"Sharpe: expected ~2.076, got {stats['sharpe']}"
    )
    assert stats["total_return_pct"] == pytest.approx(22.59, rel=0.05), (
        f"Return: expected ~22.59%, got {stats['total_return_pct']}"
    )
    assert stats["max_dd_pct"] == pytest.approx(2.80, rel=0.10), (
        f"MaxDD: expected ~2.80%, got {stats['max_dd_pct']}"
    )
    assert stats["n_trades"] == pytest.approx(78386, rel=0.05), (
        f"Trades: expected ~78386, got {stats['n_trades']}"
    )
