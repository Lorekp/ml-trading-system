"""Tests for the global DRY_RUN safety flag."""

from __future__ import annotations

import pytest

from deribit_paper.risk.dry_run import DryRunState, get_dry_run, set_dry_run


@pytest.fixture(autouse=True)
def _reset_dry_run():
    """Ensure every test starts with dry_run = True (default)."""
    DryRunState().reset()
    yield
    DryRunState().reset()


def test_default_is_true():
    assert get_dry_run() is True


def test_singleton_state():
    a = DryRunState()
    b = DryRunState()
    assert a is b


def test_enable_is_idempotent():
    assert get_dry_run() is True
    set_dry_run(True)
    assert get_dry_run() is True


def test_disable_requires_token():
    with pytest.raises(RuntimeError, match="confirmation_token"):
        set_dry_run(False)


def test_disable_rejects_default_token():
    with pytest.raises(RuntimeError, match="confirmation_token"):
        set_dry_run(False, confirmation_token="default")


def test_disable_with_valid_token():
    set_dry_run(False, confirmation_token="abc123xyz")
    assert get_dry_run() is False


def test_re_enable_after_disable():
    set_dry_run(False, confirmation_token="token")
    assert get_dry_run() is False
    set_dry_run(True)
    assert get_dry_run() is True
