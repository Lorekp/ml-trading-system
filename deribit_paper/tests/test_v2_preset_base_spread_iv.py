"""Historical V2 replay must preserve ``base_spread_iv=2.0``.

The canonical V2 trade log is generated via
``BacktestConfig.with_historical_btc_v2().with_realistic_costs()``,
where ``with_realistic_costs()`` flips ``base_spread_iv 1.5 → 2.0``
(``vol_arb/backtester.py:464``). An earlier reconstruction omitted that
override and produced a tighter half-spread. This is forensic parity,
not evidence that the historical results are valid.

This test locks the cross-engine alignment.
"""

from __future__ import annotations

import pytest

from deribit_paper.config import PaperConfig


def test_historical_v2_preset_sets_base_spread_iv_to_2():
    cfg = PaperConfig.historical_v2_preset()
    assert cfg.base_spread_iv == pytest.approx(2.0), (
        "Historical V2 replay must set base_spread_iv=2.0 to "
        "match BacktestConfig.with_realistic_costs() (the trade-log "
        "generator's path)."
    )


def test_default_preset_keeps_base_spread_iv_at_1_5():
    """Sanity: bare PaperConfig default stays at 1.5 (S-8 only flips
    the V2 path). Default users get the legacy half-spread."""
    cfg = PaperConfig()
    assert cfg.base_spread_iv == pytest.approx(1.5)


def test_v2_and_realistic_costs_agree_on_base_spread_iv():
    """The historical reconstruction matches the archived generator."""
    from vol_arb.backtester import BacktestConfig
    bt_cfg = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    paper_cfg = PaperConfig.historical_v2_preset()
    assert paper_cfg.base_spread_iv == pytest.approx(bt_cfg.base_spread_iv), (
        f"base_spread_iv mismatch: paper={paper_cfg.base_spread_iv} "
        f"vs backtest={bt_cfg.base_spread_iv}"
    )
