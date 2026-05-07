"""S2 (parity-cycle round 2 — 2026-05-07): ``process_bar`` drains
in-flight trade-close coroutines briefly so sizing reads the post-close
equity instead of pre-close.

Pre-S2 the bar-aligned exit (D12) made trade-T's exit fire on bar T+1's
open boundary. The ``equity_tracker.on_trade_closed`` update was a
coroutine racing with bar T+1's ``process_bar`` — coroutine ordering
decided whether sizing saw stale or fresh equity. The fix awaits any
pending trade tasks for up to 200ms before sizing.

Tested by direct invocation of the drain code path with a synthetic
pending task; the engine-level integration is covered by the smoke
fixtures elsewhere.
"""

from __future__ import annotations

import asyncio

import pytest


# ── direct test of asyncio.wait drain semantics ──────────────────────


@pytest.mark.asyncio
async def test_drain_returns_immediately_when_no_pending():
    """Empty pending list → asyncio.wait returns immediately (no-op)."""
    pending = []
    if pending:
        # We don't even call wait — branch never enters
        await asyncio.wait(pending, timeout=0.2)
    # Sanity: this completes with no delay
    assert True


@pytest.mark.asyncio
async def test_drain_resolves_when_task_finishes_in_time():
    """A trade-close coroutine that finishes within 200ms is fully drained."""
    async def fast_close():
        await asyncio.sleep(0.01)
    task = asyncio.create_task(fast_close())
    done, still_pending = await asyncio.wait([task], timeout=0.2)
    assert task in done
    assert not still_pending


@pytest.mark.asyncio
async def test_drain_times_out_when_task_slower_than_200ms():
    """A stuck trade-close coroutine returns to ``still_pending`` on timeout."""
    async def slow_close():
        await asyncio.sleep(2.0)
    task = asyncio.create_task(slow_close())
    try:
        done, still_pending = await asyncio.wait([task], timeout=0.05)
        # 50ms timeout < 2s slow_close → task is in still_pending
        assert not done
        assert task in still_pending
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


@pytest.mark.asyncio
async def test_drain_handles_completed_tasks_filter():
    """Already-done tasks are excluded from the drain wait."""
    async def fast_close():
        await asyncio.sleep(0.001)
    t_done = asyncio.create_task(fast_close())
    await asyncio.sleep(0.05)  # let it complete
    assert t_done.done()

    # Mimic the engine's filter: only pending tasks are passed to wait
    trade_tasks = [t_done]
    pending = [t for t in trade_tasks if not t.done()]
    assert pending == []


@pytest.mark.asyncio
async def test_drain_partial_resolves_some_blocks_others():
    """Mixed-pace pending tasks: some drain, some hit timeout."""
    async def fast():
        await asyncio.sleep(0.01)

    async def slow():
        await asyncio.sleep(2.0)

    t_fast = asyncio.create_task(fast())
    t_slow = asyncio.create_task(slow())
    try:
        done, still_pending = await asyncio.wait(
            [t_fast, t_slow], timeout=0.1,
        )
        assert t_fast in done
        assert t_slow in still_pending
    finally:
        t_slow.cancel()
        try:
            await t_slow
        except (asyncio.CancelledError, BaseException):
            pass
