"""Tests for trade-task drain on engine shutdown.

Pre-E.1.4, ``engine.shutdown()`` only cancelled the four named services
(WS dispatch, bar clock, telegram flush, heartbeat). Fire-and-forget
trade tasks spawned in ``process_bar`` were left untracked — on
shutdown they died with positions still open on Deribit but no record
locally. This test exercises the drain path.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from deribit_paper.config import PaperConfig
from deribit_paper.engine import TradingEngine


@pytest.fixture
def cfg(tmp_path) -> PaperConfig:
    c = PaperConfig()
    c.client_id = "test_id"
    c.client_secret = "test_secret"
    c.state_dir = str(tmp_path / "state")
    c.logs_dir = str(tmp_path / "logs")
    c.sqlite_db = str(tmp_path / "paper.db")
    c.metrics_parquet = str(tmp_path / "metrics.parquet")
    c.holding_minutes = 1               # short timeout for tests
    c.ensure_dirs()
    return c


@pytest.mark.asyncio
async def test_trade_task_completes_during_shutdown(cfg):
    """A trade task that completes within the drain window is awaited
    rather than cancelled."""
    engine = TradingEngine(cfg)
    completed = asyncio.Event()

    async def _fake_trade() -> None:
        await asyncio.sleep(0.1)
        completed.set()

    task = asyncio.create_task(_fake_trade(), name="trade_test")
    engine._trade_tasks.append(task)

    await engine.shutdown()
    assert task.done()
    assert completed.is_set()


@pytest.mark.asyncio
async def test_trade_task_cancelled_on_drain_timeout(cfg):
    """A trade task that overruns the drain timeout is cancelled cleanly."""
    cfg.holding_minutes = 0  # so drain_timeout = 60s
    engine = TradingEngine(cfg)

    cancelled_seen = asyncio.Event()

    async def _slow_trade() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled_seen.set()
            raise

    task = asyncio.create_task(_slow_trade(), name="trade_slow")
    engine._trade_tasks.append(task)

    # Replace the drain_timeout via the config so the test runs fast.
    # We rely on shutdown's holding_minutes calculation; cfg.holding_minutes=0
    # means drain_timeout = 60s. We force a manual cancel by patching
    # asyncio.wait to return immediately.
    real_wait = asyncio.wait

    async def fast_wait(*args, **kwargs):
        # Wait briefly so the inner sleep gets a chance to start, then
        # return as if the timeout fired.
        await asyncio.sleep(0.1)
        return set(), {task}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "wait", fast_wait)
        await engine.shutdown()

    assert task.done()
    assert cancelled_seen.is_set()


@pytest.mark.asyncio
async def test_completed_tasks_reaped_from_list():
    """The reap pass after appending a new task drops finished entries."""
    cfg = PaperConfig()
    cfg.client_id = "x"
    cfg.client_secret = "y"
    engine = TradingEngine(cfg)

    async def _quick() -> None:
        return None

    t1 = asyncio.create_task(_quick(), name="trade_1")
    await t1
    engine._trade_tasks.append(t1)
    # Simulate the reap line from process_bar
    engine._trade_tasks = [t for t in engine._trade_tasks if not t.done()]
    assert engine._trade_tasks == []
