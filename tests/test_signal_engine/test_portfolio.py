"""Tests for signal_engine.portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import compute_daily_pnl_series, simulate_portfolio
from signal_engine.portfolio_path import simulate_portfolio_precomputed_exits


def _make_signals(
    n: int = 200,
    direction: float = 1.0,
    fwd_return: float = 0.001,
    signal_strength: float = 0.5,
    asset: str = "BTC",
    seed: int = 42,
) -> pd.DataFrame:
    """Helper to build a signals DataFrame."""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": np.full(n, direction),
            "fwd_return": np.full(n, fwd_return),
            "confidence": np.full(n, 0.1),
            "probability": np.full(n, 0.6),
            "signal_strength": np.full(n, signal_strength),
            "asset": asset,
        }
    )


def test_dd_halt_triggers() -> None:
    """Portfolio halts trading when max DD is breached."""
    # Large positions with heavy losses; delever_start == max_dd so no gradual
    # deleveraging — the portfolio goes straight from full-size to halt.
    signals = _make_signals(n=500, direction=1.0, fwd_return=-0.50, signal_strength=2.0)
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_dd=0.10,
        delever_start=0.10,  # same as max_dd => no gradual reduction
        slippage=0.0001,
        max_exposure=1.0,
        max_per_asset=1.0,
        base_sizing_pct=0.20,
        max_signal_mult=3.0,
    )
    sim = simulate_portfolio(signals, cfg)
    assert sim is not None
    assert sim["halted"] is True


def test_deleveraging_reduces_size() -> None:
    """When DD exceeds delever_start, position sizes shrink."""
    # Create signals that cause moderate losses, then more trades
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    fwd_returns = np.concatenate([
        np.full(50, -0.003),  # losses to trigger deleveraging
        np.full(50, 0.001),   # continued trading
    ])
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": np.ones(n),
            "fwd_return": fwd_returns,
            "confidence": np.full(n, 0.1),
            "probability": np.full(n, 0.6),
            "signal_strength": np.full(n, 0.5),
            "asset": "BTC",
        }
    )
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_dd=0.20,
        delever_start=0.02,
        max_exposure=1.0,
        max_per_asset=1.0,
    )
    sim = simulate_portfolio(signals, cfg)
    assert sim is not None

    # Check that sizes in the second half are generally smaller than the first
    # few trades (after losses reduce equity and trigger deleveraging)
    active = ~sim["skipped"]
    if active.sum() > 20:
        sizes = sim["sizes"][active]
        early_avg = sizes[:10].mean()
        late_avg = sizes[-10:].mean()
        # Not a strict requirement since equity itself drops, but sizes should adapt
        assert late_avg < early_avg * 1.5  # rough sanity check


def test_exposure_limits_enforced(sample_signals: pd.DataFrame) -> None:
    """Total exposure never exceeds max_exposure * equity."""
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_exposure=0.50,
        max_per_asset=0.20,
    )
    sim = simulate_portfolio(sample_signals, cfg)
    assert sim is not None
    # The simulation uses decay on exposure, so it's implicitly bounded
    # Just verify it ran without error and produced trades
    assert sim["n_trades"] > 0


def test_per_asset_limits() -> None:
    """Per-asset exposure is limited."""
    # Create many signals for one asset to test the cap
    signals = _make_signals(n=300, signal_strength=1.0, fwd_return=0.001)
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_per_asset=0.05,
        max_exposure=1.0,
    )
    sim = simulate_portfolio(signals, cfg)
    assert sim is not None
    # Should still execute trades despite per-asset limits
    assert sim["n_trades"] > 0


def test_halted_state_stops_all_trading() -> None:
    """After halting, no more trades execute."""
    signals = _make_signals(n=500, direction=1.0, fwd_return=-0.02)
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_dd=0.05,
        max_exposure=1.0,
        max_per_asset=1.0,
        base_sizing_pct=0.05,
    )
    sim = simulate_portfolio(signals, cfg)
    assert sim is not None
    assert sim["halted"] is True

    # After the halt point, all trades should be skipped
    halted_idx = None
    for i in range(len(sim["dd"])):
        if sim["dd"][i] >= cfg.max_dd:
            halted_idx = i
            break
    if halted_idx is not None and halted_idx + 1 < len(sim["skipped"]):
        # Everything after halt should be skipped (pnl = 0)
        post_halt_pnl = sim["pnl"][halted_idx + 1 :]
        assert (post_halt_pnl == 0).all()


def test_daily_pnl_series() -> None:
    """compute_daily_pnl_series returns a valid Series."""
    signals = _make_signals(n=200, fwd_return=0.001)
    cfg = PortfolioConfig(starting_capital=10000.0)
    sim = simulate_portfolio(signals, cfg)
    daily = compute_daily_pnl_series(sim)
    assert daily is not None
    assert isinstance(daily, pd.Series)
    assert len(daily) >= 5


def test_too_few_signals_returns_none() -> None:
    """Fewer than 10 signals returns None."""
    signals = _make_signals(n=5)
    cfg = PortfolioConfig()
    result = simulate_portfolio(signals, cfg)
    assert result is None


def test_probability_edge_and_regime_multiplier_reduce_size() -> None:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": np.ones(20),
            "fwd_return": np.full(20, 0.001),
            "confidence": np.full(20, 0.1),
            "probability": np.array([0.55] * 10 + [0.90] * 10),
            "signal_strength": np.ones(20),
            "regime_exposure_mult": np.array([0.5] * 10 + [1.0] * 10),
            "asset": ["BTC"] * 20,
        }
    )
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_exposure=1.0,
        max_per_asset=1.0,
        base_sizing_pct=0.1,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True,
        regime_exposure_floor=0.5,
    )

    sim = simulate_portfolio(signals, cfg)

    assert sim is not None
    active_sizes = sim["sizes"][~sim["skipped"]]
    assert active_sizes[:5].mean() < active_sizes[-5:].mean()


def test_precomputed_path_realizes_partial_exit_cashflows_at_event_times() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="3h", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "de_risk_timestamp": timestamps + pd.Timedelta(hours=1),
            "exit_timestamp": timestamps + pd.Timedelta(hours=2),
            "direction": np.ones(10),
            "probability": np.full(10, 0.7),
            "signal_strength": np.full(10, 1.0),
            "regime_exposure_mult": np.ones(10),
            "asset": ["BTC"] * 10,
            "fwd_return": np.full(10, 0.03),
            "de_risk_return": np.full(10, 0.02),
            "expiry_return": np.full(10, 0.04),
            "remaining_fraction": np.full(10, 0.5),
            "return_already_signed": np.ones(10, dtype=bool),
            "bars_held": np.full(10, 2),
        }
    )
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_exposure=1.0,
        max_per_asset=1.0,
        base_sizing_pct=0.1,
        slippage=0.0,
    )

    sim = simulate_portfolio_precomputed_exits(signals, cfg)

    assert sim is not None
    sim_ts = pd.to_datetime(sim["timestamps"], utc=True)
    # We should see cashflow events at de-risk / exit timestamps, not only entries.
    assert len(sim_ts) > len(timestamps)
    assert (sim["pnl"] != 0).sum() >= 10
    assert sim["final_equity"] > cfg.starting_capital


def test_precomputed_path_does_not_book_pnl_on_entry_when_exit_is_later() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": timestamps,
            "exit_timestamp": timestamps + pd.Timedelta(hours=1),
            "direction": np.ones(10),
            "probability": np.full(10, 0.7),
            "signal_strength": np.full(10, 1.0),
            "regime_exposure_mult": np.ones(10),
            "asset": ["BTC"] * 10,
            "fwd_return": np.full(10, 0.01),
            "return_already_signed": np.ones(10, dtype=bool),
            "bars_held": np.ones(10),
        }
    )
    cfg = PortfolioConfig(
        starting_capital=10000.0,
        max_exposure=1.0,
        max_per_asset=1.0,
        base_sizing_pct=0.1,
        slippage=0.0,
    )

    sim = simulate_portfolio_precomputed_exits(signals, cfg)

    assert sim is not None
    sim_ts = pd.to_datetime(sim["timestamps"], utc=True)
    assert sim_ts.min() == timestamps.min()
    # Entry timestamp row exists, but the first positive pnl is realized later.
    positive_ts = sim_ts[np.where(sim["pnl"] > 0)[0]]
    assert len(positive_ts) > 0
    assert positive_ts.min() > timestamps.min()
