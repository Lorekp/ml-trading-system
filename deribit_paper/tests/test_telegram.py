"""Tests for TelegramNotifier — queueing, throttling, disabled mode."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from deribit_paper.monitoring.telegram import TelegramNotifier


def test_disabled_when_credentials_missing():
    n = TelegramNotifier(bot_token="", chat_id="")
    assert n.enabled is False


def test_disabled_notifier_send_is_noop():
    n = TelegramNotifier(bot_token="", chat_id="")
    n.send("trade_open", "hello")
    assert len(n._queue) == 0


def test_enabled_with_valid_credentials():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    assert n.enabled is True


def test_send_enqueues_message():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    n.send("trade_open", "hello")
    assert len(n._queue) == 1
    msg = n._queue[0]
    assert msg.category == "trade_open"
    assert msg.text == "hello"


def test_error_category_throttled():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    n.send("error", "first")
    n.send("error", "second within throttle window")
    assert len(n._queue) == 1  # second dropped


def test_non_error_categories_not_throttled():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    n.send("trade_open", "a")
    n.send("trade_open", "b")
    n.send("trade_open", "c")
    assert len(n._queue) == 3


def test_queue_bounded_by_max():
    n = TelegramNotifier(bot_token="tok", chat_id="123", queue_max=3)
    for i in range(10):
        n.send("trade_open", f"msg_{i}")
    assert len(n._queue) == 3


@pytest.mark.asyncio
async def test_flush_once_sends_via_http():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    # Inject a mocked httpx client
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = lambda: None
    mock_client.post = AsyncMock(return_value=mock_resp)
    n._client = mock_client

    n.send("trade_open", "hello world")
    sent = await n.flush_once()
    assert sent == 1
    assert len(n._queue) == 0
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_flush_requeues_on_failure():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("network down"))
    n._client = mock_client

    n.send("trade_open", "msg1")
    n.send("trade_open", "msg2")
    sent = await n.flush_once()
    assert sent == 0
    # Messages should still be in the queue
    assert len(n._queue) == 2


def test_trade_open_formatter():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    n.trade_open(
        trade_id="live_abcdef0123",
        direction=-1,
        strike=60_000.0,
        n_contracts_btc=1.5,
        expiry_dt="2026-04-20",
        iv_used_pct=75.3,
    )
    assert len(n._queue) == 1
    msg = n._queue[0]
    assert "SHORT" in msg.text
    assert "live_abc" in msg.text  # first 8 chars of trade_id
    assert "60,000" in msg.text


def test_kill_switch_formatter():
    n = TelegramNotifier(bot_token="tok", chat_id="123")
    n.kill_switch("drawdown exceeded")
    msg = n._queue[0]
    assert "KILL SWITCH" in msg.text
    assert "drawdown" in msg.text
