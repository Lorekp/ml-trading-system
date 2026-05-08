"""E-7 (parity-r3 cycle 3 — 2026-05-08) regression tests.

``StraddleFlow`` now charges a maker adverse-selection penalty
mirroring ``vol_arb/backtester.py:2256-2272``:
``vega_s × cfg.maker_adverse_selection_iv`` per BTC, scaled by N.

Live charges proportionally based on the per-leg maker fill fraction —
``raw["execution_type"]`` ∈ {"maker", "partial_maker", "taker"}.
Engine deducts the result from ``realized_pnl_usd`` at trade close.

Pre-fix the live engine charged nothing on its maker entries.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.profit_scale import bsm_vega_per_vol_pt
from deribit_paper.execution.straddle_flow import StraddleFlow


def _make_flow(*, maker_adverse_selection_iv: float = 0.5) -> StraddleFlow:
    """Construct a StraddleFlow with stub dependencies — only the
    maker-adverse-selection helper is exercised."""
    return StraddleFlow(
        order_manager=MagicMock(),
        live_drl_hedger=MagicMock(),
        maker_adverse_selection_iv=maker_adverse_selection_iv,
    )


def _result(
    *,
    execution_type: str,
    filled_amount: float = 0.5,
    maker_filled: float | None = None,
) -> OrderResult:
    """Synthesise an OrderResult with the same `raw` shape that
    ``place_maker_then_taker`` produces (`execution_type` + `maker_filled`).
    """
    raw: dict[str, Any] = {"execution_type": execution_type}
    if maker_filled is not None:
        raw["maker_filled"] = float(maker_filled)
    return OrderResult(
        client_order_id="X",
        order_id="Y",
        instrument_name="BTC-X",
        side="sell",
        amount_requested=filled_amount,
        filled_amount=float(filled_amount),
        average_price=0.05,
        state="filled",
        is_dry_run=False,
        raw=raw,
    )


# ── Both legs maker → full straddle vega × penalty × N ────────────────


def test_both_legs_maker_charge_equals_straddle_vega_times_penalty():
    """When call+put fully fill as maker, the live charge equals the
    backtester's ``vega_s × cfg.maker_adverse_selection_iv × N`` exactly
    (vega_s being the COMPOSITE/straddle vega = 2× per-leg vega)."""
    flow = _make_flow(maker_adverse_selection_iv=0.5)
    n = 0.5
    spot, K, iv_dec, tau = 60_000.0, 60_000.0, 0.65, 2.0 / 365.25
    leg_vega = bsm_vega_per_vol_pt(
        spot=spot, strike=K, iv_decimal=iv_dec, tau_years=tau,
    )
    expected_charge = leg_vega * 0.5 * (n + n)  # call + put both maker
    charge = flow._maker_adverse_selection_charge(
        entry_call=_result(execution_type="maker", filled_amount=n),
        entry_put=_result(execution_type="maker", filled_amount=n),
        n_contracts=n,
        spot=spot, strike=K,
        iv_entry_decimal=iv_dec,
        tau_entry_years=tau,
    )
    assert charge == pytest.approx(expected_charge, rel=1e-9)
    # And this composite-vega × penalty × N agrees with the backtester
    # formula: 2 × leg_vega × penalty × n_contracts.
    composite_form = (2 * leg_vega) * 0.5 * n
    assert charge == pytest.approx(composite_form, rel=1e-9)


# ── Both legs taker → zero charge ─────────────────────────────────────


def test_both_legs_taker_no_charge():
    """No maker fills → no penalty (matches backtester's
    ``maker_filled_arr[i] == False`` short-circuit)."""
    flow = _make_flow()
    charge = flow._maker_adverse_selection_charge(
        entry_call=_result(execution_type="taker", filled_amount=0.5),
        entry_put=_result(execution_type="taker", filled_amount=0.5),
        n_contracts=0.5, spot=60_000.0, strike=60_000.0,
        iv_entry_decimal=0.65, tau_entry_years=2.0 / 365.25,
    )
    assert charge == 0.0


# ── Mixed: one leg maker, one taker → half charge ────────────────────


def test_mixed_legs_charges_only_maker_leg():
    """Call=maker, put=taker → only the call leg is charged. Result
    equals one leg's vega × penalty × n_contracts."""
    flow = _make_flow(maker_adverse_selection_iv=0.5)
    n = 0.5
    spot, K, iv_dec, tau = 60_000.0, 60_000.0, 0.65, 2.0 / 365.25
    leg_vega = bsm_vega_per_vol_pt(
        spot=spot, strike=K, iv_decimal=iv_dec, tau_years=tau,
    )
    # Charge = leg_vega × penalty × n_call_maker (only call) = leg_vega × 0.5 × 0.5
    expected = leg_vega * 0.5 * n
    charge = flow._maker_adverse_selection_charge(
        entry_call=_result(execution_type="maker", filled_amount=n),
        entry_put=_result(execution_type="taker", filled_amount=n),
        n_contracts=n, spot=spot, strike=K,
        iv_entry_decimal=iv_dec, tau_entry_years=tau,
    )
    assert charge == pytest.approx(expected, rel=1e-9)


# ── Partial maker fill → proportional charge ────────────────────────


def test_partial_maker_charge_uses_maker_filled_amount():
    """When `execution_type=='partial_maker'`, the `raw["maker_filled"]`
    field is the BTC of the leg that filled at maker price; the
    remainder filled at taker. Charge scales by maker_filled, not the
    total filled_amount."""
    flow = _make_flow(maker_adverse_selection_iv=0.5)
    n = 0.5
    spot, K, iv_dec, tau = 60_000.0, 60_000.0, 0.65, 2.0 / 365.25
    leg_vega = bsm_vega_per_vol_pt(
        spot=spot, strike=K, iv_decimal=iv_dec, tau_years=tau,
    )
    # Call: 0.3 maker + 0.2 taker; put: full taker.
    # Charge = leg_vega × 0.5 × (0.3 + 0) = leg_vega × 0.15
    expected = leg_vega * 0.5 * 0.3
    charge = flow._maker_adverse_selection_charge(
        entry_call=_result(
            execution_type="partial_maker",
            filled_amount=n, maker_filled=0.3,
        ),
        entry_put=_result(execution_type="taker", filled_amount=n),
        n_contracts=n, spot=spot, strike=K,
        iv_entry_decimal=iv_dec, tau_entry_years=tau,
    )
    assert charge == pytest.approx(expected, rel=1e-9)


# ── Disabled (penalty=0) → zero charge regardless of fill type ──────


def test_disabled_penalty_short_circuits():
    """``maker_adverse_selection_iv <= 0`` disables the charge entirely."""
    flow = _make_flow(maker_adverse_selection_iv=0.0)
    charge = flow._maker_adverse_selection_charge(
        entry_call=_result(execution_type="maker", filled_amount=0.5),
        entry_put=_result(execution_type="maker", filled_amount=0.5),
        n_contracts=0.5, spot=60_000.0, strike=60_000.0,
        iv_entry_decimal=0.65, tau_entry_years=2.0 / 365.25,
    )
    assert charge == 0.0


# ── Degenerate inputs → zero charge (defensive) ─────────────────────


def test_degenerate_vega_returns_zero_charge():
    """If BSM vega computation degenerates (e.g. tau<=0), no charge."""
    flow = _make_flow(maker_adverse_selection_iv=0.5)
    charge = flow._maker_adverse_selection_charge(
        entry_call=_result(execution_type="maker", filled_amount=0.5),
        entry_put=_result(execution_type="maker", filled_amount=0.5),
        n_contracts=0.5, spot=60_000.0, strike=60_000.0,
        iv_entry_decimal=0.65,
        tau_entry_years=0.0,  # degenerate
    )
    assert charge == 0.0


# ── TradeOutcome carries the field ──────────────────────────────────


def test_tradeoutcome_has_maker_adverse_selection_charge_field():
    """The dataclass exposes `maker_adverse_selection_charge_usd`
    (defaults to 0.0) so the engine can deduct it from realized PnL."""
    from deribit_paper.execution.straddle_flow import TradeOutcome
    out = TradeOutcome(
        trade_id="t1", signal=MagicMock(),
        entry_call=None, entry_put=None,
        exit_call=None, exit_put=None,
        n_contracts_btc=0.5, status="completed",
    )
    assert hasattr(out, "maker_adverse_selection_charge_usd")
    assert out.maker_adverse_selection_charge_usd == 0.0
    out2 = TradeOutcome(
        trade_id="t2", signal=MagicMock(),
        entry_call=None, entry_put=None,
        exit_call=None, exit_put=None,
        n_contracts_btc=0.5, status="completed",
        maker_adverse_selection_charge_usd=12.34,
    )
    assert out2.maker_adverse_selection_charge_usd == pytest.approx(12.34)
