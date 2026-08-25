"""Regression for one-way fill costs versus a round-trip reference."""

from __future__ import annotations

import pytest

from src.live.testnet.reconciler import reconcile


def _fills() -> list[dict]:
    return [
        {
            "symbol": "BTCUSDT",
            "side": "Buy",
            "intent_side": "open_long",
            "qty": 0.01,
            "decision_mid": 100_000.0,
            "fill_price": 100_019.5,
            "method": "taker",
            "slippage_bps": 0.97,
            "fee_bps": 0.98,
            "fee_usd": 0.0975,
            "notional_usd": 1_000.0,
        }
        for _ in range(100)
    ]


def test_reconcile_compares_round_trip_costs() -> None:
    summary = reconcile(_fills())

    assert summary["avg_total_cost_bps_one_way"] == pytest.approx(1.95)
    assert summary["avg_total_cost_bps_round_trip"] == pytest.approx(3.90)
    assert summary["backtest_avg_cost_rt_bps"] == pytest.approx(3.90)
    assert summary["cost_delta_vs_backtest_bps"] == pytest.approx(0.0)


def test_reconcile_uses_explicit_round_trip_reference() -> None:
    summary = reconcile(_fills(), backtest_avg_cost_rt_bps=4.25)

    assert summary["cost_delta_vs_backtest_bps"] == pytest.approx(-0.35)
