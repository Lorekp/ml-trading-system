"""Test for B-002: NaN-poisoned sum_slippage_usd in reconciler.

Bug: FillMetrics.add silently accepts NaN notional/slip_bps, poisoning the
running sum and producing NaN cost_delta_bps -> gate check fails cryptically.

Fix: guarded add returns early on non-finite inputs and tracks
n_skipped_poisoned.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.live.testnet.reconciler import FillMetrics, aggregate, reconcile


def test_fill_metrics_skips_nan_slippage():
    fm = FillMetrics()
    fm.add({"symbol": "BTCUSDT", "side": "Buy", "intent_side": "open_long",
            "qty": 1.0, "decision_mid": 100.0, "fill_price": 100.5,
            "method": "taker", "slippage_bps": float("nan"),
            "fee_bps": 3.75, "fee_usd": 3.75, "notional_usd": 100.0})
    assert fm.sum_slippage_usd == 0.0
    assert fm.n_fills == 0
    assert fm.n_skipped_poisoned == 1


def test_fill_metrics_skips_nan_notional():
    fm = FillMetrics()
    fm.add({"symbol": "BTCUSDT", "side": "Buy", "intent_side": "open_long",
            "qty": 1.0, "decision_mid": 100.0, "fill_price": 100.5,
            "method": "taker", "slippage_bps": 1.0,
            "fee_bps": 3.75, "fee_usd": 3.75, "notional_usd": float("nan")})
    assert fm.sum_slippage_usd == 0.0
    assert fm.n_fills == 0
    assert fm.n_skipped_poisoned == 1


def test_reconcile_returns_finite_on_poisoned_fills():
    fills = [
        {"symbol": "BTCUSDT", "side": "Buy", "intent_side": "open_long",
         "qty": 1.0, "decision_mid": 100.0, "fill_price": 100.5,
         "method": "taker", "slippage_bps": 1.0,
         "fee_bps": 3.75, "fee_usd": 3.75, "notional_usd": float("nan")},
        {"symbol": "ETHUSDT", "side": "Sell", "intent_side": "open_short",
         "qty": 10.0, "decision_mid": 3000.0, "fill_price": 3000.5,
         "method": "taker", "slippage_bps": 0.5,
         "fee_bps": 3.75, "fee_usd": 11.25, "notional_usd": 30_000.0},
    ]
    s = reconcile(fills)
    assert s["n_fills"] == 1
    assert s["n_skipped_poisoned"] == 1
    # Cost delta must be finite, not NaN
    import math
    assert math.isfinite(s["cost_delta_vs_backtest_bps"]), \
        "cost_delta must be finite even with poisoned fills"
    assert math.isfinite(s["adjusted_live_sharpe"]), \
        "adjusted_live_sharpe must be finite"


if __name__ == "__main__":
    test_fill_metrics_skips_nan_slippage()
    test_fill_metrics_skips_nan_notional()
    test_reconcile_returns_finite_on_poisoned_fills()
    print("B-002 PASS")
