"""Tests for the ``--paper`` umbrella flag.

The ``--paper`` flag flips the engine into "real mainnet quotes,
simulated fills, no credentials" mode:

- Implies ``--venue prod`` (mainnet WS + REST URLs).
- Implies ``cfg.paper_mode = True`` and ``dry_run = True``.
- Mutually exclusive with ``--live`` (the safety belt is dry-run;
  --live would defeat it).
- Does NOT require ``DERIBIT_PROD_*`` environment variables.

The flag's job is purely to plumb through CLI → config; the engine
internally honours ``cfg.paper_mode`` to skip private REST calls and
to use 100ms (unauthenticated) trade channels instead of .raw.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from deribit_paper import cli
from deribit_paper.config import PaperConfig


@pytest.fixture(autouse=True)
def _reset_dry_run():
    from deribit_paper.risk.dry_run import DryRunState
    DryRunState().reset()
    yield
    DryRunState().reset()


def _patch_argv(*args: str):
    return patch.object(sys, "argv", ["deribit_paper", *args])


def test_paper_flag_sets_paper_mode_and_prod_venue():
    """--paper flips paper_mode=True, venue=prod, and skips passphrase."""
    confirm = MagicMock(return_value="token")
    with _patch_argv("--paper"), \
            patch("deribit_paper.cli._confirm_live", confirm), \
            patch("deribit_paper.cli.PaperConfig") as PC:
        cfg = MagicMock(spec=PaperConfig)
        cfg.client_id = ""
        cfg.client_secret = ""
        cfg.venue = "testnet"  # starts as testnet, switch_to_prod flips it
        cfg.paper_mode = False

        def _switch_to_prod(*, paper_mode: bool = False) -> None:
            cfg.venue = "prod"
            cfg.paper_mode = paper_mode
        cfg.switch_to_prod = MagicMock(side_effect=_switch_to_prod)

        PC.from_env = MagicMock(return_value=cfg)
        cli.main()

    confirm.assert_not_called()
    cfg.switch_to_prod.assert_called_once_with(paper_mode=True)


def test_paper_and_live_are_mutually_exclusive():
    """--paper + --live should exit with code 2 (user error)."""
    with _patch_argv("--paper", "--live"), \
            patch("deribit_paper.cli.PaperConfig") as PC:
        cfg = MagicMock(spec=PaperConfig)
        cfg.client_id = ""
        cfg.client_secret = ""
        cfg.venue = "testnet"
        cfg.paper_mode = False
        cfg.switch_to_prod = MagicMock()
        PC.from_env = MagicMock(return_value=cfg)
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 2


def test_switch_to_prod_paper_mode_skips_credential_check(monkeypatch):
    """``switch_to_prod(paper_mode=True)`` does NOT require DERIBIT_PROD_*."""
    monkeypatch.delenv("DERIBIT_PROD_CLIENT_ID", raising=False)
    monkeypatch.delenv("DERIBIT_PROD_CLIENT_SECRET", raising=False)
    cfg = PaperConfig()
    # No raise expected.
    cfg.switch_to_prod(paper_mode=True)
    assert cfg.venue == "prod"
    assert cfg.paper_mode is True
    assert cfg.rest_base_url == "https://www.deribit.com/api/v2"
    assert cfg.ws_url == "wss://www.deribit.com/ws/api/v2"


def test_switch_to_prod_non_paper_still_requires_credentials(monkeypatch):
    """The original credential-required path is preserved for live runs."""
    monkeypatch.delenv("DERIBIT_PROD_CLIENT_ID", raising=False)
    monkeypatch.delenv("DERIBIT_PROD_CLIENT_SECRET", raising=False)
    cfg = PaperConfig()
    with pytest.raises(RuntimeError, match="DERIBIT_PROD_CLIENT_ID"):
        cfg.switch_to_prod(paper_mode=False)


def test_paper_mode_flag_default_is_false():
    """``PaperConfig.paper_mode`` must default to False — the safety
    invariant is that paper mode is opt-in via CLI."""
    cfg = PaperConfig()
    assert cfg.paper_mode is False


def test_paper_initial_equity_default_matches_v2_backtest():
    """Synthetic starting equity defaults to $100K to align per-trade
    PnL distribution with vol_arb/output/BTC_v2_trade_log.parquet
    conventions."""
    cfg = PaperConfig()
    assert cfg.paper_initial_equity_usd == pytest.approx(100_000.0)
