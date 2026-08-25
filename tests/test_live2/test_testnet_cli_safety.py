"""Safety invariants for the Binance paper/testnet CLI."""

import pytest

from live.testnet.cli import TESTNET_BASE_URL, _credentials_for_mode


def test_paper_mode_ignores_all_environment_credentials(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "production-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "production-secret")
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "testnet-key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "testnet-secret")

    assert _credentials_for_mode("paper") == ("", "")


def test_live_testnet_reads_only_testnet_credentials(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "production-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "production-secret")
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "testnet-key")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "testnet-secret")

    assert _credentials_for_mode("live-testnet") == (
        "testnet-key",
        "testnet-secret",
    )


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported testnet mode"):
        _credentials_for_mode("live-real")


def test_endpoint_is_testnet_only():
    assert TESTNET_BASE_URL == "https://testnet.binancefuture.com"
