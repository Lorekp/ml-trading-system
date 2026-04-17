"""Aggregator parity tests — Phase A of the fix-all plan.

Verifies:
- `Strategy.preview_size` produces the same value as `compute_size` for a
  spread of (probability, drawdown, vol_mult, breaker) states.
- `SignalAggregator` output is deterministic across input permutations.
- Aggregator sort key `(asset, timeframe)` matches the backtest's
  post-`assemble_signals` iteration order within a single timestamp.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from live.core.aggregator import SignalAggregator
from live.core.interfaces import PortfolioView, Position, Signal
from live.strategies.s8_sym15 import S8Sym15Strategy


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_strategy(base_sizing_pct: float = 0.30, **overrides: Any) -> S8Sym15Strategy:
    """Instantiate S8Sym15Strategy with test-friendly config (no model load)."""
    cfg: dict[str, Any] = {
        "timeframes": ["15m", "1h"],
        "holds": [2, 4],
        "portfolio": {
            "base_sizing_pct": base_sizing_pct,
            "max_signal_mult": 2.0,
            "max_dd": 0.10,
            "delever_start": 0.05,
            "regime_exposure_floor": 0.50,
        },
        "breaker": {"window": 10, "threshold": -0.01,
                    "floor": 0.10, "up_threshold": 0.01, "cap": 1.5},
        "s8": {"core": ["BTC"], "max_total": 30,
               "prep_cache_path": "does/not/exist.pkl"},
        "model_dir": "does/not/exist",
        "vol_overlay": {"target_vol": 0.025, "floor": 0.3, "cap": 1.5},
        "signal_generator": {"conf_threshold": 0.04, "regime_ema_span": 20},
    }
    cfg.update(overrides)
    # SignalGenerator.__init__ tolerates missing model dirs (logs error, loads 0).
    return S8Sym15Strategy(config=cfg)


def _make_signal(asset: str, tf: str = "15m", *,
                 probability: float = 0.70, strength: float = 0.4,
                 direction: float = 1.0, hold: int = 2,
                 strategy_id: str = "s8_sym15") -> Signal:
    """Construct a Signal; strategy_id matches the aggregator's strategy_id."""
    return Signal(
        asset=asset, symbol=f"{asset}USDT", timeframe=tf,
        direction=direction, probability=probability,
        confidence=abs(probability - 0.5),
        signal_strength=strength, hold_bars=hold,
        strategy_id=strategy_id,
    )


def _make_portfolio(equity: float = 200_000.0, drawdown: float = 0.0) -> PortfolioView:
    return PortfolioView(
        equity=equity, peak_equity=equity, drawdown=drawdown,
        open_positions=[], n_positions_by_asset={},
        exposure_by_asset={}, total_exposure=0.0,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_preview_size_equals_compute_size_across_spread():
    """For any (prob, drawdown), preview_size must equal compute_size(equity, dd, vol_mult)
    when vol_mult is whatever the strategy would compute itself."""
    s = _make_strategy()
    for prob in (0.45, 0.52, 0.60, 0.70, 0.85, 0.95):
        for dd in (0.0, 0.03, 0.06, 0.09):
            sig = _make_signal("BTC", probability=prob, strength=abs(prob - 0.5) * 2)
            portfolio = _make_portfolio(equity=200_000.0, drawdown=dd)
            expected = s.compute_size(
                sig, equity=portfolio.equity, drawdown=portfolio.drawdown,
                vol_mult=s.compute_vol_mult(sig.asset),
            )
            actual = s.preview_size(sig, portfolio)
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-9), \
                f"preview_size mismatch at prob={prob} dd={dd}: {actual} vs {expected}"


def test_preview_size_returns_zero_above_max_dd():
    """At or above max_dd, compute_size returns 0 — preview must too."""
    s = _make_strategy()
    sig = _make_signal("BTC", probability=0.90)
    portfolio = _make_portfolio(drawdown=0.10)
    assert s.preview_size(sig, portfolio) == 0.0


def test_preview_size_tracks_breaker_state():
    """Breaker losing-streak down-scales preview_size the same way as compute_size."""
    s = _make_strategy()
    sig = _make_signal("BTC", probability=0.80)
    portfolio = _make_portfolio()

    # Before any breaker history, base sizing applies
    before = s.preview_size(sig, portfolio)

    # Record a losing streak large enough to trip the breaker
    for _ in range(10):
        s.on_trade("BTC", pnl=-500.0, notional=10_000.0)

    after = s.preview_size(sig, portfolio)
    # Breaker floor=0.10 — size must shrink materially
    assert after < before
    assert after <= before * 0.35  # well below, but not exactly floor due to ratio math


def test_aggregator_sort_is_deterministic():
    """Permuting the input order must not change the output order."""
    strat = _make_strategy()
    portfolio = _make_portfolio()
    base_sigs = [
        _make_signal("BTC", "15m"),
        _make_signal("ETH", "15m"),
        _make_signal("BTC", "1h"),
        _make_signal("ADA", "15m"),
    ]

    agg = SignalAggregator(max_signals_per_bar=10,
                            max_per_asset_pct=0.50, max_exposure_pct=0.95)

    rng = random.Random(42)
    first_out = None
    for _ in range(20):
        perm = base_sigs[:]
        rng.shuffle(perm)
        out = agg.aggregate([(strat, s) for s in perm], portfolio)
        key = [(s.asset, s.timeframe) for _, s in out]
        if first_out is None:
            first_out = key
        else:
            assert key == first_out, f"Non-deterministic aggregator output: {key} vs {first_out}"


def test_aggregator_sort_matches_backtest_order():
    """Given the same set of signals, aggregator order must equal what a pandas
    DataFrame sorted by (asset, timeframe) would produce."""
    strat = _make_strategy()
    portfolio = _make_portfolio()
    # 4 signals across 2 assets and 2 TFs
    signals = [
        _make_signal("ETH", "1h"),
        _make_signal("BTC", "1h"),
        _make_signal("ETH", "15m"),
        _make_signal("BTC", "15m"),
    ]
    agg = SignalAggregator(max_signals_per_bar=10,
                            max_per_asset_pct=0.50, max_exposure_pct=0.95)
    out = agg.aggregate([(strat, s) for s in signals], portfolio)

    # What the backtest would produce on identical signals:
    df = pd.DataFrame({
        "asset": [s.asset for s in signals],
        "timeframe": [s.timeframe for s in signals],
    })
    df_sorted = df.sort_values(["asset", "timeframe"]).reset_index(drop=True)

    live_order = [(s.asset, s.timeframe) for _, s in out]
    backtest_order = list(zip(df_sorted["asset"], df_sorted["timeframe"]))
    assert live_order == backtest_order


def test_aggregator_uses_preview_size_for_caps():
    """The aggregator must enforce caps based on strategy.preview_size, not a
    fixed edge × base formula."""
    # Strategy with unusually small base_sizing_pct — so preview_size is small
    # relative to the old est_size = equity × 0.30 × edge (which is the default).
    strat = _make_strategy(base_sizing_pct=0.02)  # 15× smaller than default
    portfolio = _make_portfolio(equity=10_000.0)
    # Five BTC signals at the same bar; each preview = ~$10k × 0.02 × 1.0 = ~$200.
    # Under the old formula the aggregator would have estimated $10k × 0.30 × 1.0 = $3k each,
    # hitting per-asset cap after just 1 signal (max_per_asset_pct=0.06 → $600).
    # With preview_size, five signals of ~$200 each fit well under $600.
    sigs = [_make_signal("BTC", "15m", probability=1.0, strength=1.0)
            for _ in range(5)]
    agg = SignalAggregator(
        max_signals_per_bar=20,
        max_per_asset_pct=0.06,
        max_exposure_pct=0.90,
    )
    out = agg.aggregate([(strat, s) for s in sigs], portfolio)
    # At least 2 should pass — the old aggregator would have allowed only 1 (est_size > cap)
    # or possibly 0 depending on exact ordering. The new one must let 3+ through.
    assert len(out) >= 3, f"Expected preview_size to loosen cap; got {len(out)}"


def test_aggregator_empty_and_conflicts():
    """Empty input → empty output. Conflicting directions on same asset → dropped."""
    strat = _make_strategy()
    portfolio = _make_portfolio()
    agg = SignalAggregator(max_signals_per_bar=10)
    assert agg.aggregate([], portfolio) == []

    # Two opposite BTC signals from the same strategy → both dropped
    s_long = _make_signal("BTC", "15m", direction=1.0)
    s_short = _make_signal("BTC", "15m", direction=-1.0)
    out = agg.aggregate([(strat, s_long), (strat, s_short)], portfolio)
    assert out == []
