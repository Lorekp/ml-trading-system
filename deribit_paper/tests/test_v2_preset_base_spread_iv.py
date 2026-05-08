"""S-8 (parity-r3 cycle 2 — 2026-05-08) regression test.

The canonical V2 trade log is generated via
``BacktestConfig.with_btc_production_filters_v2().with_realistic_costs()``,
where ``with_realistic_costs()`` flips ``base_spread_iv 1.5 → 2.0``
(``vol_arb/backtester.py:464``). Pre-fix the live engine's
``PaperConfig.with_v2_preset()`` did NOT flip ``base_spread_iv`` —
operators calling the live preset got 1.5 (the default), producing a
tighter half-spread than V2 canon.

This test locks the cross-engine alignment.
"""

from __future__ import annotations

import pytest

from deribit_paper.config import PaperConfig


def test_with_v2_preset_sets_base_spread_iv_to_2():
    cfg = PaperConfig.with_v2_preset()
    assert cfg.base_spread_iv == pytest.approx(2.0), (
        "S-8 regression: with_v2_preset must set base_spread_iv=2.0 to "
        "match BacktestConfig.with_realistic_costs() (the trade-log "
        "generator's path)."
    )


def test_default_preset_keeps_base_spread_iv_at_1_5():
    """Sanity: bare PaperConfig default stays at 1.5 (S-8 only flips
    the V2 path). Default users get the legacy half-spread."""
    cfg = PaperConfig()
    assert cfg.base_spread_iv == pytest.approx(1.5)


def test_v2_and_realistic_costs_agree_on_base_spread_iv():
    """Cross-engine sanity: the live V2 preset agrees with the
    backtester's realistic-costs leg of the V2 trade-log generator."""
    from vol_arb.backtester import BacktestConfig
    bt_cfg = BacktestConfig().with_btc_production_filters_v2().with_realistic_costs()
    paper_cfg = PaperConfig.with_v2_preset()
    assert paper_cfg.base_spread_iv == pytest.approx(bt_cfg.base_spread_iv), (
        f"base_spread_iv mismatch: paper={paper_cfg.base_spread_iv} "
        f"vs backtest={bt_cfg.base_spread_iv}"
    )
