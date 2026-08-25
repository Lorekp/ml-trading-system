"""CLI safety tests for testnet and mainnet-quote paper modes.

The application CLI has no real-order or historical-V2 configuration
route. ``--paper`` uses public mainnet data with dry-run forced on.
"""
from __future__ import annotations

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
    """--paper selects public mainnet quotes and forces dry-run."""
    with _patch_argv("--paper"), \
            patch("deribit_paper.cli.PaperConfig") as PC:
        cfg = MagicMock(spec=PaperConfig)
        cfg.client_id = ""
        cfg.client_secret = ""
        cfg.venue = "testnet"  # starts as testnet, switch_to_prod flips it
        cfg.paper_mode = False
        cfg.dry_run = False

        def _switch_to_prod(*, paper_mode: bool = False) -> None:
            cfg.venue = "prod"
            cfg.paper_mode = paper_mode
        cfg.switch_to_prod = MagicMock(side_effect=_switch_to_prod)

        PC.from_env = MagicMock(return_value=cfg)
        cli.main()

    cfg.switch_to_prod.assert_called_once_with(paper_mode=True)
    assert cfg.dry_run is True
    from deribit_paper.risk.dry_run import get_dry_run
    assert get_dry_run() is True


@pytest.mark.parametrize(
    "removed_args",
    [
        ("--live",),
        ("--venue", "prod"),
        ("--config-preset", "v2"),
        ("--dry-run",),
    ],
)
def test_removed_unsafe_or_obsolete_cli_routes_are_rejected(removed_args):
    """Argparse rejects removed real-order, prod, and V2 preset routes."""
    with _patch_argv(*removed_args):
        with pytest.raises(SystemExit) as exc_info:
            cli._parse_args()
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


def test_paper_mode_flag_default_is_false():
    """``PaperConfig.paper_mode`` must default to False — the safety
    invariant is that paper mode is opt-in via CLI."""
    cfg = PaperConfig()
    assert cfg.paper_mode is False


def test_paper_initial_equity_retains_historical_comparison_convention():
    """The synthetic $100K convention is comparison metadata, not capital."""
    cfg = PaperConfig()
    assert cfg.paper_initial_equity_usd == pytest.approx(100_000.0)
