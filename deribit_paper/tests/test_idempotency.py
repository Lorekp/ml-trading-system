"""Tests for IdempotencyGuard + client order ID generation."""

from __future__ import annotations

import time

from deribit_paper.execution.idempotency import (
    IdempotencyGuard,
    new_client_order_id,
)


def test_new_client_order_id_has_prefix():
    oid = new_client_order_id("paper")
    assert oid.startswith("paper_")
    assert len(oid) > len("paper_")


def test_new_client_order_id_is_unique():
    ids = {new_client_order_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_guard_initially_empty():
    g = IdempotencyGuard()
    assert g.was_sent("foo") is False


def test_guard_mark_then_was_sent():
    g = IdempotencyGuard()
    g.mark_sent("foo")
    assert g.was_sent("foo") is True
    assert g.was_sent("bar") is False


def test_guard_double_mark_logs_warning():
    g = IdempotencyGuard()
    g.mark_sent("foo")
    g.mark_sent("foo")  # should not raise, just log
    assert g.was_sent("foo") is True


def test_guard_ttl_eviction():
    g = IdempotencyGuard(ttl_seconds=0.05)  # 50ms TTL for fast test
    g.mark_sent("foo")
    assert g.was_sent("foo") is True
    time.sleep(0.1)
    assert g.was_sent("foo") is False


def test_guard_max_entries_hard_cap():
    g = IdempotencyGuard(max_entries=3)
    for i in range(5):
        g.mark_sent(f"oid_{i}")
    # Oldest two should be evicted
    assert g.was_sent("oid_0") is False
    assert g.was_sent("oid_1") is False
    assert g.was_sent("oid_2") is True
    assert g.was_sent("oid_3") is True
    assert g.was_sent("oid_4") is True


def test_guard_clear():
    g = IdempotencyGuard()
    g.mark_sent("foo")
    g.mark_sent("bar")
    g.clear()
    assert g.was_sent("foo") is False
    assert g.was_sent("bar") is False
