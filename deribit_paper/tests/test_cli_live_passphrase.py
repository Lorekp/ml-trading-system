"""Tests for the ``--live`` passphrase invariant.

E.1.12 in plan ``C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md``.

Pre-E.1.12, ``cli.py`` had a carve-out::

    if args.venue == "testnet":
        token = secrets.token_hex(16)        # skip passphrase
    else:
        token = _confirm_live()

This eroded the ``--live`` invariant: an operator could forget
``--venue prod`` and unknowingly route to live testnet orders without
seeing the confirmation prompt. The carve-out is removed; the
passphrase is now required regardless of venue.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from deribit_paper import cli


@pytest.fixture(autouse=True)
def _reset_dry_run():
    from deribit_paper.risk.dry_run import DryRunState
    DryRunState().reset()
    yield
    DryRunState().reset()


def _patch_argv(*args: str):
    """Convenience: replace sys.argv for one test."""
    return patch.object(sys, "argv", ["deribit_paper", *args])


def test_live_on_testnet_requires_passphrase():
    """Pre-E.1.12 this skipped _confirm_live; post-E.1.12 it's required."""
    confirm = MagicMock(return_value="token")
    with _patch_argv("--live", "--venue", "testnet"), \
            patch("deribit_paper.cli._confirm_live", confirm), \
            patch("deribit_paper.cli.PaperConfig") as PC:
        cfg = MagicMock()
        cfg.client_id = "x"
        cfg.client_secret = "y"
        cfg.venue = "testnet"
        PC.from_env = MagicMock(return_value=cfg)
        # Call main() but don't actually run any sub-command (no
        # --test-auth/--smoke/--run-paper).
        cli.main()
    confirm.assert_called_once()


def test_live_on_prod_requires_passphrase():
    """Sanity: prod ALWAYS required passphrase. Behaviour unchanged but
    asserted for completeness."""
    confirm = MagicMock(return_value="token")
    with _patch_argv("--live", "--venue", "prod"), \
            patch("deribit_paper.cli._confirm_live", confirm), \
            patch("deribit_paper.cli.PaperConfig") as PC:
        cfg = MagicMock()
        cfg.client_id = "x"
        cfg.client_secret = "y"
        cfg.venue = "prod"
        cfg.switch_to_prod = MagicMock()
        PC.from_env = MagicMock(return_value=cfg)
        cli.main()
    confirm.assert_called_once()


def test_dry_run_does_not_call_passphrase():
    """No ``--live`` → no passphrase prompt at all."""
    confirm = MagicMock(return_value="token")
    with _patch_argv("--venue", "testnet"), \
            patch("deribit_paper.cli._confirm_live", confirm), \
            patch("deribit_paper.cli.PaperConfig") as PC:
        cfg = MagicMock()
        cfg.client_id = "x"
        cfg.client_secret = "y"
        cfg.venue = "testnet"
        PC.from_env = MagicMock(return_value=cfg)
        cli.main()
    confirm.assert_not_called()
