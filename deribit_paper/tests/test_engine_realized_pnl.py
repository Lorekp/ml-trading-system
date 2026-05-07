"""Tests for the realized-PnL recipe in :meth:`TradingEngine._execute_and_persist`.

E.1.10 in plan ``C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md``.

The recipe must mirror V2 backtester's ``net_pnl_nc`` term:
    net = option_pnl + hedge_pnl - option_fees - hedge_fees

Pre-E.1.10 the live engine's ``realized_pnl_usd`` was option-premium-only;
hedge PnL and fees were silently dropped, biasing the equity tracker
(used by drawdown / kill-switch / sizing-DD-floor) and the harness PnL
diff (used by I4 alignment analysis).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from deribit_paper.config import PaperConfig
from deribit_paper.engine import TradingEngine
from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.perp_hedger import HedgeRunSummary


def _filled(price: float) -> OrderResult:
    return OrderResult(
        client_order_id="x",
        order_id="o",
        instrument_name="BTC-OPT",
        side="buy",
        amount_requested=1.0,
        filled_amount=1.0,
        average_price=price,
        state="filled",
        is_dry_run=True,
    )


def _make_outcome(
    *,
    direction: int,
    n_contracts_btc: float,
    entry_call: float,
    entry_put: float,
    exit_call: float,
    exit_put: float,
    hedge_pnl_usd: float = 0.0,
    total_usd_traded: float = 0.0,
) -> SimpleNamespace:
    straddle = SimpleNamespace(
        call_instrument="BTC-CALL",
        put_instrument="BTC-PUT",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-22 08:00:00", tz="UTC"),
        index_price=60_000.0,
    )
    signal = SimpleNamespace(
        direction=direction,
        edge_vol_pts=10.0,
        rv_pred_pct=72.0,
        iv_used_pct=60.0,
        straddle=straddle,
        term_slope_1w_2w=None,
        bar_tsrv=SimpleNamespace(jump_fraction=0.0),
    )
    return SimpleNamespace(
        trade_id="t",
        signal=signal,
        entry_call=_filled(entry_call),
        entry_put=_filled(entry_put),
        exit_call=_filled(exit_call),
        exit_put=_filled(exit_put),
        n_contracts_btc=n_contracts_btc,
        status="completed",
        hedge_summary=HedgeRunSummary(
            trade_id="t",
            n_ticks_executed=10,
            n_ticks_skipped_dust=0,
            total_usd_traded=total_usd_traded,
            final_inventory_btc=0.0,
            realized_pnl_usd=hedge_pnl_usd,
        ),
        error=None,
        started_at=pd.Timestamp("2026-04-15 12:00:00", tz="UTC"),
        ended_at=pd.Timestamp("2026-04-15 12:15:00", tz="UTC"),
    )


@pytest.fixture
def engine(tmp_path):
    cfg = PaperConfig()
    cfg.client_id = "x"
    cfg.client_secret = "y"
    cfg.state_dir = str(tmp_path / "state")
    cfg.logs_dir = str(tmp_path / "logs")
    cfg.sqlite_db = str(tmp_path / "paper.db")
    cfg.metrics_parquet = str(tmp_path / "metrics.parquet")
    cfg.option_maker_fee = 0.0003
    cfg.perp_taker_fee = 0.0005
    cfg.ensure_dirs()
    e = TradingEngine(cfg)

    # Stub the bits _execute_and_persist touches so we can drive it directly.
    e.straddle_flow = MagicMock()
    e.straddle_flow.execute = AsyncMock()
    e.kill_switch = MagicMock()
    e.equity_tracker = MagicMock()
    e.equity_tracker.snapshots = MagicMock(return_value=[MagicMock()])
    e.store = MagicMock()
    e.store.persist_trade = AsyncMock()
    e.store.persist_equity_snapshot = AsyncMock()
    e.predictor = None
    e.telegram = None
    e._harness_writer = MagicMock()
    e._harness_writer.consolidate_and_write = AsyncMock()
    return e


@pytest.mark.asyncio
async def test_short_straddle_pnl_recipe_includes_hedge_and_fees(engine):
    """SHORT: sold premium 0.10 BTC × 1.0 BTC × $60K = $6000 entry, bought
    back at 0.08 BTC × 1.0 × $60K = $4800. Option PnL = +$1200.
    Hedge PnL = +$50 (gain from delta hedging).
    Option fees = 4 × 0.0003 × 60000 × 1.0 = $72.

    E1 (parity-cycle round 2 — 2026-05-07): hedge cost rate is now
    ``perp_taker_fee + perp_slippage_bps/1e4`` = 0.0005 + 0.0002 = 0.0007
    (V2 backtester recipe, backtester.py:803). Hedge cost on $20K
    traded = $14 (was $10 pre-E1).
    Net = 1200 + 50 - 72 - 14 = $1164.
    """
    outcome = _make_outcome(
        direction=-1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.04, exit_put=0.04,
        hedge_pnl_usd=50.0, total_usd_traded=20_000.0,
    )
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=-1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    # Equity tracker should have received the net PnL.
    args, _ = engine.equity_tracker.on_trade_closed.call_args
    assert args[0] == pytest.approx(1164.0, abs=1.0)


@pytest.mark.asyncio
async def test_long_straddle_pnl_sign_matches_backtest(engine):
    """LONG: bought 0.05+0.05=0.10 BTC × 1.0 × $60K = $6000 entry,
    sold at 0.06+0.06=0.12 × 1.0 × $60K = $7200 exit. Option PnL = +$1200.
    No hedge PnL/fees in this case.
    Option fees = $72.
    Net = 1200 - 72 = $1128.
    """
    outcome = _make_outcome(
        direction=1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.06, exit_put=0.06,
        hedge_pnl_usd=0.0, total_usd_traded=0.0,
    )
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    args, _ = engine.equity_tracker.on_trade_closed.call_args
    assert args[0] == pytest.approx(1128.0, abs=1.0)


@pytest.mark.asyncio
async def test_realized_pnl_zero_when_legs_missing(engine):
    """Defensive: missing exit fills → option_pnl = 0; hedge PnL still
    contributes if the hedge ran."""
    outcome = _make_outcome(
        direction=1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.06, exit_put=0.06,
        hedge_pnl_usd=20.0, total_usd_traded=10_000.0,
    )
    outcome.exit_call = None
    outcome.exit_put = None
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    args, _ = engine.equity_tracker.on_trade_closed.call_args
    # E1 (parity-cycle round 2): cost_rate = perp_taker_fee +
    # perp_slippage_bps/1e4 = 0.0005 + 0.0002 = 0.0007.
    # Hedge fee on $10K = $7. Hedge PnL 20 - 7 = $13.
    assert args[0] == pytest.approx(13.0, abs=1.0)


# ── E.1.15: index_price fallback removal ─────────────────────────────────


@pytest.mark.asyncio
async def test_skip_equity_update_when_straddle_missing(engine):
    """E.1.15: degenerate trade context (no straddle) → skip equity update,
    don't fall back to hardcoded $60K."""
    outcome = _make_outcome(
        direction=1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.06, exit_put=0.06,
    )
    outcome.signal.straddle = None
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    # Equity tracker NEVER called.
    engine.equity_tracker.on_trade_closed.assert_not_called()


@pytest.mark.asyncio
async def test_skip_equity_update_when_index_price_zero(engine):
    """E.1.15: zero index_price treated as degenerate."""
    outcome = _make_outcome(
        direction=1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.06, exit_put=0.06,
    )
    outcome.signal.straddle.index_price = 0.0
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    engine.equity_tracker.on_trade_closed.assert_not_called()


@pytest.mark.asyncio
async def test_skip_equity_update_when_index_price_nan(engine):
    """E.1.15: NaN index_price treated as degenerate."""
    outcome = _make_outcome(
        direction=1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.06, exit_put=0.06,
    )
    outcome.signal.straddle.index_price = float("nan")
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    engine.equity_tracker.on_trade_closed.assert_not_called()


# ── 12.5%-of-premium fee cap (Plan: Part 1b, 2026-05-07) ───────────────


@pytest.mark.asyncio
async def test_fee_cap_binds_for_low_premium_legs(engine):
    """Per-leg cap = MIN(0.0003 BTC, 0.125 · leg_premium_btc).
    Pre-fix the engine charged 4 × 0.0003 × index unconditionally; post-fix
    it must apply 0.125·premium when that's tighter (deep-OTM / near-expiry).

    Entry leg cap = 0.125·0.001 = 0.000125 BTC ⇒ $7.5/leg.
    Exit  leg cap = 0.125·0.0008 = 0.0001 BTC ⇒ $6.0/leg.
    Total fee = 2·$7.5 + 2·$6.0 = $27 (vs $72 pre-fix flat).

    Option PnL: SHORT, entry premium 0.001+0.001=0.002 BTC × 1 × $60K = $120,
    exit 0.0008+0.0008=0.0016 × 1 × $60K = $96 ⇒ option_pnl = +$24.
    Net = 24 - 27 - 0 (no hedge) = -$3.
    """
    outcome = _make_outcome(
        direction=-1, n_contracts_btc=1.0,
        entry_call=0.001, entry_put=0.001,
        exit_call=0.0008, exit_put=0.0008,
        hedge_pnl_usd=0.0, total_usd_traded=0.0,
    )
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=-1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    args, _ = engine.equity_tracker.on_trade_closed.call_args
    # 24 - 27 = -3. Pre-fix would have given 24 - 72 = -48.
    assert args[0] == pytest.approx(-3.0, abs=0.5), (
        "Per-fill 12.5% cap not applied — engine.py:1960 may have "
        "regressed to the uncapped flat-fee formula."
    )


@pytest.mark.asyncio
async def test_fee_cap_unbounded_at_typical_premiums(engine):
    """Sanity counterpart to the cap-binds test: at typical V2 premiums
    (≥ 0.005 BTC) the cap does NOT bind, so the post-fix fee equals the
    pre-fix flat-fee value — proves the fix is non-destructive in the
    regime where the strategy actually trades.
    """
    outcome = _make_outcome(
        direction=-1, n_contracts_btc=1.0,
        entry_call=0.05, entry_put=0.05,
        exit_call=0.04, exit_put=0.04,
        hedge_pnl_usd=0.0, total_usd_traded=0.0,
    )
    engine.straddle_flow.execute.return_value = outcome

    fake_signal = SimpleNamespace(direction=-1, edge_vol_pts=10.0)
    await engine._execute_and_persist(
        fake_signal, n_contracts_btc=1.0,
    )

    args, _ = engine.equity_tracker.on_trade_closed.call_args
    # SHORT: option_pnl = (0.10-0.08)·1·60000 = +$1200. Fee = 4·0.0003·60000 = $72.
    # No hedge ⇒ net = 1200 - 72 = $1128.
    assert args[0] == pytest.approx(1128.0, abs=1.0)
