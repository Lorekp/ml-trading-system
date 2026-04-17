"""Tests for PaperConfig — loading, defaults, safety guarantees."""

from __future__ import annotations

import os

import pytest

from deribit_paper.config import DEFAULT_CONFIG, PaperConfig


def test_defaults_are_safe():
    """Default config MUST be dry-run and point at testnet."""
    cfg = PaperConfig()
    assert cfg.dry_run is True
    assert cfg.venue == "testnet"
    assert "test.deribit.com" in cfg.rest_base_url
    assert "test.deribit.com" in cfg.ws_url


def test_default_config_is_singleton_like():
    """DEFAULT_CONFIG is shared; mutating it should not be the norm."""
    assert DEFAULT_CONFIG.dry_run is True
    assert DEFAULT_CONFIG.venue == "testnet"


def test_strategy_knobs_match_backtester_convention():
    cfg = PaperConfig()
    assert cfg.holding_minutes == 15
    assert cfg.bar_minutes == 15
    assert cfg.base_spread_iv == 1.5
    assert cfg.entry_threshold == 25.0
    assert cfg.entry_threshold_long == 30.0
    assert cfg.target_margin_utilization == 0.05
    assert cfg.hedge_sim_steps == 10


def test_composite_v2_sizing_fields_present():
    """composite_v2 sizing knobs must be present and match BacktestConfig defaults."""
    cfg = PaperConfig()
    assert cfg.sizing_edge_cap == 3.0
    assert cfg.sizing_dd_floor == 0.25
    assert cfg.baseline_target_dvol == 50.0
    assert cfg.sizing_liq_cap == 1.5
    assert cfg.short_sizing_penalty == 0.50


def test_paper_initial_equity_is_demo_btc():
    cfg = PaperConfig()
    assert cfg.initial_equity == 1.0


def test_from_env_without_credentials(monkeypatch):
    """from_env() should not crash when Deribit credentials are absent.

    We monkeypatch ``_load_env`` to a no-op so the real ``.env`` file
    (which now contains live testnet credentials) doesn't leak into
    this test.  Then we explicitly clear the env vars and call
    ``from_env()``.
    """
    monkeypatch.setattr("deribit_paper.config._load_env", lambda: None)
    for key in (
        "DERIBIT_TESTNET_CLIENT_ID",
        "DERIBIT_TESTNET_CLIENT_SECRET",
        "DERIBIT_TESTNET_BASE_URL",
        "DERIBIT_TESTNET_WS_URL",
        "PAPER_DRY_RUN",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = PaperConfig.from_env()
    assert cfg.client_id == ""
    assert cfg.client_secret == ""
    assert cfg.dry_run is True


def test_from_env_with_credentials(monkeypatch):
    monkeypatch.setattr("deribit_paper.config._load_env", lambda: None)
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_ID", "test_id")
    monkeypatch.setenv("DERIBIT_TESTNET_CLIENT_SECRET", "test_secret")
    cfg = PaperConfig.from_env()
    assert cfg.client_id == "test_id"
    assert cfg.client_secret == "test_secret"


def test_paper_dry_run_env_var_false(monkeypatch):
    """PAPER_DRY_RUN=false should disable dry_run in from_env()."""
    monkeypatch.setattr("deribit_paper.config._load_env", lambda: None)
    monkeypatch.setenv("PAPER_DRY_RUN", "false")
    cfg = PaperConfig.from_env()
    assert cfg.dry_run is False


def test_paper_dry_run_env_var_default_is_true(monkeypatch):
    monkeypatch.setattr("deribit_paper.config._load_env", lambda: None)
    monkeypatch.delenv("PAPER_DRY_RUN", raising=False)
    cfg = PaperConfig.from_env()
    assert cfg.dry_run is True


def test_switch_to_prod_without_creds_raises(monkeypatch):
    """switch_to_prod() must refuse to reuse testnet credentials."""
    monkeypatch.delenv("DERIBIT_PROD_CLIENT_ID", raising=False)
    monkeypatch.delenv("DERIBIT_PROD_CLIENT_SECRET", raising=False)
    cfg = PaperConfig()
    with pytest.raises(RuntimeError, match="DERIBIT_PROD_CLIENT_ID"):
        cfg.switch_to_prod()


def test_switch_to_prod_with_creds(monkeypatch):
    monkeypatch.setenv("DERIBIT_PROD_CLIENT_ID", "prod_id")
    monkeypatch.setenv("DERIBIT_PROD_CLIENT_SECRET", "prod_secret")
    cfg = PaperConfig()
    cfg.switch_to_prod()
    assert cfg.venue == "prod"
    assert "test.deribit.com" not in cfg.rest_base_url
    assert cfg.client_id == "prod_id"


# ── New config fields for live trading ─────────────────────────────


def test_new_safety_config_fields_present():
    """All new safety config fields must exist with sensible defaults."""
    cfg = PaperConfig()
    assert cfg.max_orphan_close_retries == 3
    assert cfg.svi_max_stale_hours == 4.0
    assert cfg.equity_reconcile_bars == 96


def test_deribit_fee_and_margin_defaults():
    """Config defaults must match Deribit's actual fee schedule and margin formula."""
    cfg = PaperConfig()
    # Deribit option fee: 0.03% of underlying (same for maker and taker)
    assert cfg.option_maker_fee == 0.0003
    # Deribit perp taker fee: 0.05% (5bps)
    assert cfg.perp_taker_fee == 0.0005
    # Deribit ATM IM ≈ max(0.15, 0.10) + mark_price ≈ 0.20
    assert cfg.im_fraction == 0.20


def test_for_small_account_sets_safe_limits(monkeypatch):
    """for_small_account() must set conservative limits for small capital."""
    monkeypatch.setattr("deribit_paper.config._load_env", lambda: None)
    for key in ("DERIBIT_TESTNET_CLIENT_ID", "DERIBIT_TESTNET_CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    cfg = PaperConfig.for_small_account(equity_usd=750.0)
    assert cfg.max_open_positions == 1
    assert cfg.kill_switch_daily_dd_pct == 5.0
    assert cfg.min_trade_floor is True
    assert cfg.dry_run is True  # must still default to safe
    # max_single_trade_btc must NOT be hardcoded — formula scales with equity
    assert cfg.max_single_trade_btc == 2.0  # default, not overridden
