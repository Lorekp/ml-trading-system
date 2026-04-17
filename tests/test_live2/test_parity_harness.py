"""End-to-end parity harness.

Cross-cuts the aggregator → strategy → execution pipeline to catch regressions
that individual unit tests miss:

  - Aggregator's `preview_size` delegation exactly tracks `compute_size`
    across a spread of portfolio states (drawdown, breaker, vol overlay).
  - Aggregator order on a multi-bar, multi-asset, multi-TF signal stream
    reproduces the backtest's post-`assemble_signals` iteration order on the
    same data.
  - CalibratedPaperExecution's fill prices, given the strategy's computed
    size, match the backtest's `compute_slippage`-based math to 1e-9 relative.

No models loaded; all signals are synthetic so the harness is fast and
runs offline.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest

from live.core.aggregator import SignalAggregator
from live.core.interfaces import PortfolioView, Signal
from live.execution.calibrated import CalibratedPaperExecution
from live.strategies.s8_sym15 import S8Sym15Strategy
from signal_engine.slippage import compute_slippage, load_per_symbol_calibration


# ── Shared fixtures ──────────────────────────────────────────────────────────


VIP2 = ("VIP2", 1.60, 3.75)


def _make_strategy(**overrides: Any) -> S8Sym15Strategy:
    cfg: dict[str, Any] = {
        "timeframes": ["15m", "1h", "3h", "4h", "24h"],
        "holds": [2, 4, 16, 6, 12],
        "portfolio": {
            "base_sizing_pct": 0.30,
            "max_signal_mult": 2.0,
            "max_dd": 0.10,
            "delever_start": 0.05,
            "regime_exposure_floor": 0.50,
        },
        "breaker": {"window": 10, "threshold": -0.01,
                    "floor": 0.10, "up_threshold": 0.01, "cap": 1.5},
        "s8": {"core": ["BTC"], "prep_cache_path": "does/not/exist.pkl"},
        "model_dir": "does/not/exist",
        "signal_generator": {"conf_threshold": 0.04, "regime_ema_span": 20},
    }
    cfg.update(overrides)
    return S8Sym15Strategy(config=cfg)


def _sig(asset: str, tf: str, *, probability: float = 0.75,
         direction: float = 1.0, strength: float | None = None) -> Signal:
    if strength is None:
        strength = abs(2 * probability - 1)
    hold = {"15m": 2, "1h": 4, "3h": 16, "4h": 6, "24h": 12}[tf]
    return Signal(
        asset=asset, symbol=f"{asset}USDT", timeframe=tf,
        direction=direction, probability=probability,
        confidence=abs(probability - 0.5),
        signal_strength=strength, hold_bars=hold, strategy_id="s8_sym15",
    )


def _portfolio(*, equity: float = 200_000.0, drawdown: float = 0.0,
               total_exposure: float = 0.0,
               exposure_by_asset: dict[str, float] | None = None) -> PortfolioView:
    return PortfolioView(
        equity=equity, peak_equity=equity, drawdown=drawdown,
        open_positions=[], n_positions_by_asset={},
        exposure_by_asset=exposure_by_asset or {},
        total_exposure=total_exposure,
    )


# ── End-to-end: preview_size tracks compute_size across many states ──────────


@pytest.mark.parametrize("drawdown", [0.0, 0.04, 0.07, 0.095])
@pytest.mark.parametrize("probability", [0.55, 0.72, 0.88, 0.97])
@pytest.mark.parametrize("breaker_trades", [0, 5, 10])
def test_preview_size_matches_compute_size_full_matrix(
        drawdown, probability, breaker_trades):
    strat = _make_strategy()
    # Warm breaker with a mix of wins/losses so its mult is something
    # non-trivial — e.g., 5 losses in a row trips the scale-down.
    for _ in range(breaker_trades):
        strat.on_trade("BTC", pnl=-250.0, notional=10_000.0)

    sig = _sig("BTC", "15m", probability=probability)
    port = _portfolio(drawdown=drawdown)

    expected = strat.compute_size(
        sig, equity=port.equity, drawdown=port.drawdown,
        vol_mult=strat.compute_vol_mult("BTC"),
    )
    actual = strat.preview_size(sig, port)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-9)


# ── Multi-bar, multi-asset, multi-TF iteration order ─────────────────────────


def test_multi_bar_order_matches_backtest():
    """Feed the aggregator a synthetic signal stream from three bar-times
    with different asset/TF combinations, and assert the per-bar output
    matches what the backtest would iterate on (sorted by asset, tf)."""
    # Use tiny base_sizing_pct so the portfolio cap never binds — this test
    # is about ORDER, not cap enforcement (that's tested separately).
    strat = _make_strategy(portfolio={"base_sizing_pct": 0.001,
                                       "max_signal_mult": 2.0,
                                       "max_dd": 0.10, "delever_start": 0.05,
                                       "regime_exposure_floor": 0.50})

    assets = ["BTC", "DENT", "ETH", "CHZ", "XRP"]
    tfs = ["15m", "1h", "3h", "4h", "24h"]

    # Build a "prepared" DataFrame as if it came out of assemble_signals.
    import itertools, random
    rng = random.Random(0)
    rows = []
    for ts in pd.date_range("2026-04-17", periods=3, freq="15min", tz="UTC"):
        # Pick a random subset of (asset, tf) at each bar
        for a, tf in itertools.product(assets, tfs):
            if rng.random() < 0.5:
                rows.append({"timestamp": ts, "asset": a, "timeframe": tf,
                             "probability": 0.6 + rng.random() * 0.3})
    df = pd.DataFrame(rows)

    sorted_df = df.sort_values(
        ["timestamp", "asset", "timeframe"], kind="stable"
    ).reset_index(drop=True)

    # For each bar, run the live aggregator on the corresponding signals
    agg = SignalAggregator(max_signals_per_bar=100,
                            max_per_asset_pct=0.5, max_exposure_pct=0.95)
    for ts, bar in sorted_df.groupby("timestamp", sort=True):
        sigs = [(strat, _sig(r["asset"], r["timeframe"],
                              probability=r["probability"]))
                for _, r in bar.iterrows()]
        out = agg.aggregate(sigs, _portfolio())
        live_order = [(s.asset, s.timeframe) for _, s in out]
        backtest_order = list(zip(bar["asset"], bar["timeframe"]))
        assert live_order == backtest_order, (
            f"order mismatch at {ts}: live={live_order}  backtest={backtest_order}")


# ── Execution pipeline: preview_size → compute_size → fill_price ─────────────


class _FakeExchange:
    """ExchangeAdapter stub with deterministic ticker output."""
    def __init__(self, *, price: float, quote_vol: float):
        self._price = price
        self._qv = quote_vol

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {
            "last_price": self._price,
            "bid": self._price * 0.9999,
            "ask": self._price * 1.0001,
            "volume_24h": 0.0,
            "quote_volume_24h": self._qv,
        }


def test_full_pipeline_fill_price_parity():
    """Strategy computes size → CalibratedPaperExecution fills at the exact
    price the backtest math would predict."""
    strat = _make_strategy()
    # Pre-warm strategy's vol cache for the participation gate (won't bind here)
    strat._vol_cache_usd["BTC"] = 5_000_000_000.0
    strat._participation_rate = 0.02
    strat._guard_pct = 0.50
    strat._slip_cal = load_per_symbol_calibration()
    strat._vip_tier = VIP2

    sig = _sig("BTC", "15m", probability=0.85)
    port = _portfolio(equity=200_000.0)

    size_usd = strat.preview_size(sig, port)
    assert size_usd > 0

    # Execute via calibrated — reuse the same VIP + calibration
    mid = 1000.0
    exchange = _FakeExchange(price=mid, quote_vol=5_000_000_000.0)
    exe = CalibratedPaperExecution(
        exchange, state=object(), vip_tier=VIP2,
    )
    qty = size_usd / mid
    result = asyncio.run(exe.execute_entry("BTCUSDT", "Buy", qty))
    assert result is not None

    # Expected fill from backtest math
    expected_slip = compute_slippage(
        "BTC", size_usd, 5_000_000_000.0,
        per_symbol_cal=load_per_symbol_calibration(),
        vip_tier=VIP2,
    )
    # CalibratedPaperExecution uses the bid/ask-implied mid; our ticker has
    # bid=0.9999*mid, ask=1.0001*mid, so (bid+ask)/2 = mid exactly.
    expected_price = mid * (1.0 + expected_slip)
    assert result.fill_price == pytest.approx(expected_price, rel=1e-9)
