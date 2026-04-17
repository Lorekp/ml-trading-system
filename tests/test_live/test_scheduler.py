"""Tests for live.scheduler — bar timing and alignment."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from live.config import SchedulerConfig
from live.scheduler import BarScheduler


def _make_scheduler(
    interval_m: int = 15, offset_s: int = 5
) -> BarScheduler:
    """Create a BarScheduler with the given parameters."""
    config = SchedulerConfig(
        bar_interval_m=interval_m,
        poll_offset_s=offset_s,
    )
    return BarScheduler(config)


class TestNextBarAlignment:
    """Test that next_bar_time aligns to bar boundaries."""

    def test_next_bar_alignment(self) -> None:
        """At 12:07, next 15m bar is at 12:15:05."""
        scheduler = _make_scheduler(15, 5)
        mock_now = datetime(2025, 6, 15, 12, 7, 0, tzinfo=timezone.utc)

        with patch("live.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            target = scheduler.next_bar_time()

        assert target.minute == 15
        assert target.second == 5
        assert target.hour == 12

    def test_next_bar_past_boundary(self) -> None:
        """At 12:16, next 15m bar is at 12:30:05."""
        scheduler = _make_scheduler(15, 5)
        mock_now = datetime(2025, 6, 15, 12, 16, 0, tzinfo=timezone.utc)

        with patch("live.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            target = scheduler.next_bar_time()

        assert target.minute == 30
        assert target.second == 5


class TestNextBarWithOffset:
    """Test that the poll offset is correctly applied."""

    def test_next_bar_with_offset(self) -> None:
        """Offset of 10s shifts the target from :15:00 to :15:10."""
        scheduler = _make_scheduler(15, 10)
        mock_now = datetime(2025, 6, 15, 12, 7, 0, tzinfo=timezone.utc)

        with patch("live.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            target = scheduler.next_bar_time()

        assert target.minute == 15
        assert target.second == 10


class TestEdgeAtBoundary:
    """Test behaviour at or near exact bar boundaries."""

    def test_edge_at_boundary(self) -> None:
        """At exactly 12:45:00, next bar should be 13:00:05 (not 12:45:05)."""
        scheduler = _make_scheduler(15, 5)
        mock_now = datetime(2025, 6, 15, 12, 45, 0, tzinfo=timezone.utc)

        with patch("live.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            target = scheduler.next_bar_time()

        assert target.hour == 13
        assert target.minute == 0
        assert target.second == 5
