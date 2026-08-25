"""Historical false-profit sign smoke checks using local Deribit option fills.

These slow integration checks replay a short slice of the invalidated BTC V2
experiment. They verify only the direction of the known failure:

* honest vega marking does not claim profit when real fills lose;
* legacy SVI-difference marking can reproduce the false-profit signature.

This is forensic regression coverage, not a PnL-realism gate or evidence that
the strategy is deployable. The module skips when its local datasets are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRADES_DIR = Path("data/deribit/trades/BTC_option")
# A two-month window with useful tape coverage and bounded runtime.
START, END = "2024-09-01", "2024-11-01"


def _has_data() -> bool:
    if not TRADES_DIR.exists() or not any(TRADES_DIR.glob("*.parquet")):
        return False
    from vol_arb.backtester import BacktestConfig

    return Path(BacktestConfig().vol_arb_path).exists()


def _run_and_mark(honest: bool):
    from vol_arb.backtester import BacktestConfig, VolArbBacktester
    from vol_arb.honest_pnl import mark_trade_log_to_real_premium
    from vol_arb.options_loader import load_option_trades

    cfg = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    cfg.kelly_mult_path = None
    cfg.start_dt, cfg.end_dt = START, END
    cfg.honest_vega_iv = honest
    _, trade_log = VolArbBacktester(cfg).run_backtest()
    fired = trade_log[trade_log["N_contracts_nc"] > 0].copy()
    trades = load_option_trades(
        str(TRADES_DIR),
        start_date=START,
        end_date="2024-11-02",
    )
    return mark_trade_log_to_real_premium(fired, trades, tol_minutes=60.0)


@pytest.mark.skipif(not _has_data(), reason="option tape / merged dataset absent")
def test_honest_marking_does_not_create_false_profit():
    from vol_arb.honest_pnl import gate_no_fake_profit

    marked = _run_and_mark(honest=True)
    gate = gate_no_fake_profit(marked.trade_log)
    assert gate.passed, gate.message
    assert gate.detail["real_fill_total"] < 0
    assert gate.detail["model_total"] <= 0 or not gate.detail["false_profit"]


@pytest.mark.skipif(not _has_data(), reason="option tape / merged dataset absent")
def test_legacy_svi_marking_reproduces_false_profit_signature():
    from vol_arb.honest_pnl import gate_no_fake_profit

    marked = _run_and_mark(honest=False)
    gate = gate_no_fake_profit(marked.trade_log)
    assert not gate.passed, (
        "historical honest_vega_iv=False should reproduce the false-profit "
        f"signature, but the sign smoke check passed: {gate.message}"
    )
    assert gate.detail["false_profit"] is True
    assert gate.detail["model_total"] > 0 > gate.detail["real_fill_total"]
