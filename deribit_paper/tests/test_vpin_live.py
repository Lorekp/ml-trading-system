"""Tests for LiveVPINStream and _LiveBucketAccumulator.

Four groups:
  A. Bucket accumulator unit tests (no I/O)
  B. State serialization round-trip
  C. WebSocket message processing
  D. Integration with StraddleFlow
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from deribit_paper.features.vpin_live import (
    LiveVPINStream,
    _LiveBucketAccumulator,
)


# ── Group A: Bucket accumulator unit tests ─────────────────────────


class TestLiveBucketAccumulator:
    """Pure scalar arithmetic — no I/O, no async."""

    def test_single_bucket_close(self):
        """Feed exactly V BTC of trades; one bucket should close."""
        acc = _LiveBucketAccumulator(bucket_volume_btc=10.0, rolling_window=5)
        # 10 BTC all buys
        acc.process_trade(10.0, is_seller_aggressor=False)
        assert acc._buckets_closed == 1
        assert acc._latest_vpin is not None
        assert acc._latest_vpin == pytest.approx(1.0)  # all buys = imbalance 1.0

    def test_partial_bucket_carryforward(self):
        """0.7V then 0.5V: first bucket closes, 0.2V carries forward."""
        acc = _LiveBucketAccumulator(bucket_volume_btc=10.0, rolling_window=5)
        acc.process_trade(7.0, is_seller_aggressor=False)
        assert acc._buckets_closed == 0
        assert acc._partial_vol == pytest.approx(7.0)

        acc.process_trade(5.0, is_seller_aggressor=True)
        assert acc._buckets_closed == 1
        # 2.0 BTC should carry forward
        assert acc._partial_vol == pytest.approx(2.0)

    def test_rolling_window_caps_at_n(self):
        """Feed 60 buckets with N=5; deque stays at 5."""
        acc = _LiveBucketAccumulator(bucket_volume_btc=1.0, rolling_window=5)
        for _ in range(60):
            acc.process_trade(1.0, is_seller_aggressor=False)
        assert acc._buckets_closed == 60
        assert len(acc._recent_imbalances) == 5

    def test_all_buys_vpin_equals_one(self):
        """Pure buy flow: every bucket imbalance = |V-0|/V = 1.0."""
        acc = _LiveBucketAccumulator(bucket_volume_btc=5.0, rolling_window=10)
        for _ in range(20):
            acc.process_trade(5.0, is_seller_aggressor=False)
        assert acc._latest_vpin == pytest.approx(1.0)

    def test_balanced_flow_vpin_near_zero(self):
        """Alternating equal buy/sell within each bucket -> VPIN near 0."""
        acc = _LiveBucketAccumulator(bucket_volume_btc=10.0, rolling_window=10)
        for _ in range(20):
            # Each bucket: 5 buy + 5 sell = imbalance |5-5|/10 = 0
            acc.process_trade(5.0, is_seller_aggressor=False)
            acc.process_trade(5.0, is_seller_aggressor=True)
        assert acc._latest_vpin == pytest.approx(0.0, abs=0.01)

    def test_vpin_always_in_01(self):
        """VPIN must be in [0, 1] under arbitrary trade sequences."""
        rng = np.random.default_rng(42)
        acc = _LiveBucketAccumulator(bucket_volume_btc=5.0, rolling_window=10)
        for _ in range(200):
            qty = float(rng.uniform(0.1, 3.0))
            is_sell = bool(rng.random() < 0.5)
            acc.process_trade(qty, is_seller_aggressor=is_sell)
        if acc._latest_vpin is not None:
            assert 0.0 <= acc._latest_vpin <= 1.0


# ── Group B: State serialization round-trip ───���────────────────────


class TestStateSerialization:

    def test_save_load_roundtrip(self):
        """VPINStreamer state -> JSON -> LiveBucketAccumulator produces same VPIN."""
        from scripts.recompute_vpin import VPINStreamer

        # Create a streamer with some state
        streamer = VPINStreamer(bucket_volume_btc=10.0, rolling_window=5)
        # Simulate some data via direct state manipulation
        streamer._partial_vol = 3.5
        streamer._partial_buy = 2.0
        streamer._partial_sell = 1.5
        streamer._recent_imbalances = [0.1, 0.3, 0.5, 0.2, 0.4]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            streamer.save_state(path)

            # Verify file exists and is valid JSON
            state = json.loads(path.read_text())
            assert state["version"] == 1
            assert state["bucket_volume_btc"] == 10.0
            assert len(state["recent_imbalances"]) == 5

            # Load into accumulator
            acc = _LiveBucketAccumulator(bucket_volume_btc=10.0, rolling_window=5)
            loaded = VPINStreamer.load_state(path)
            acc.load_state(loaded)

            assert acc._partial_vol == pytest.approx(3.5)
            assert acc._partial_buy == pytest.approx(2.0)
            assert acc._partial_sell == pytest.approx(1.5)
            assert len(acc._recent_imbalances) == 5
            assert acc._latest_vpin == pytest.approx(0.3)  # mean of [0.1, 0.3, 0.5, 0.2, 0.4]

    def test_state_version_check(self):
        """Loading a state dict with wrong version raises ValueError."""
        from scripts.recompute_vpin import VPINStreamer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"version": 99, "data": "bad"}))
            with pytest.raises(ValueError, match="Unsupported VPINStreamer state version"):
                VPINStreamer.load_state(path)


# ── Group C: WebSocket message processing ──────────────────────────


class TestMessageProcessing:

    def test_process_aggtrade_message(self):
        """Canned aggTrade JSON updates the accumulator correctly."""
        stream = LiveVPINStream(
            bucket_volume_btc=10.0, rolling_window=5, fallback_vpin=0.5,
        )
        msg = json.dumps({
            "e": "aggTrade", "s": "BTCUSDT",
            "p": "65000.00", "q": "5.0", "m": True, "T": 1712345678901,
        })
        stream._process_message(msg)
        assert stream._trades_processed == 1
        assert stream._acc._partial_vol == pytest.approx(5.0)
        assert stream._acc._partial_sell == pytest.approx(5.0)  # m=True -> seller aggressor

    def test_fallback_on_no_data(self):
        """Before any trades arrive, latest_vpin returns the fallback."""
        stream = LiveVPINStream(fallback_vpin=0.42)
        assert stream.latest_vpin == pytest.approx(0.42)

    def test_malformed_message_does_not_crash(self):
        """Garbage JSON logs warning and continues."""
        stream = LiveVPINStream(fallback_vpin=0.5)
        stream._process_message("not json at all")
        stream._process_message("{}")  # missing q and m
        assert stream._trades_processed == 0
        assert stream.latest_vpin == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_reconnect_preserves_state(self):
        """Simulated disconnect does not reset VPIN state."""
        stream = LiveVPINStream(
            bucket_volume_btc=5.0, rolling_window=3, fallback_vpin=0.5,
        )
        # Feed some trades to establish VPIN state
        for _ in range(3):
            msg = json.dumps({"q": "5.0", "m": False, "T": 1})
            stream._process_message(msg)
        assert stream._acc._buckets_closed == 3
        vpin_before = stream.latest_vpin

        # Simulate reconnect by stopping and checking state preserved
        await stream.stop()
        assert stream.latest_vpin == pytest.approx(vpin_before)


# ── Group D: Integration ───────────────────────────────────────────


class TestStraddleFlowIntegration:

    def test_straddle_flow_uses_vpin_provider(self):
        """StraddleFlow passes vpin_provider() to begin_trade."""
        from deribit_paper.execution.straddle_flow import StraddleFlow
        from deribit_paper.signal.drl_hedger_live import LiveDRLHedger

        drl_mock = MagicMock(spec=LiveDRLHedger)
        drl_mock.begin_trade.return_value = "trade_001"

        order_manager = MagicMock()
        flow = StraddleFlow(
            order_manager=order_manager,
            live_drl_hedger=drl_mock,
            vpin_provider=lambda: 0.195,
        )
        assert flow._vpin_provider is not None
        assert flow._vpin_provider() == pytest.approx(0.195)

    def test_straddle_flow_no_provider_fallback(self):
        """Without vpin_provider, StraddleFlow uses None (begin_trade gets 0.5)."""
        from deribit_paper.execution.straddle_flow import StraddleFlow

        drl_mock = MagicMock()
        order_manager = MagicMock()
        flow = StraddleFlow(
            order_manager=order_manager,
            live_drl_hedger=drl_mock,
        )
        assert flow._vpin_provider is None


# ── Group E: Staleness detection (C4) ─────────────────────────────────


class TestVPINStaleness:

    def test_not_stale_within_warmup_grace_period(self):
        """Immediately after construction, before grace period elapses,
        is_stale returns False. The grace period equals ``stale_seconds``
        — same threshold the live-data-flowing case uses, so a
        permanently-silent feed transitions to stale at the same
        deadline a previously-flowing feed would."""
        stream = LiveVPINStream(fallback_vpin=0.5, stale_seconds=60.0)
        assert stream.is_stale is False

    def test_stale_after_warmup_with_zero_messages(self):
        """Bug fix (2026-05-07): a feed that has been running longer than
        ``stale_seconds`` but never delivered a single message must
        transition to ``is_stale=True``. Pre-fix returned False forever
        (treating "never received" as "warmup"), masking a real
        permanent-disconnect / geo-block scenario."""
        import time
        stream = LiveVPINStream(
            bucket_volume_btc=10.0, fallback_vpin=0.5, stale_seconds=0.01,
        )
        # No _process_message call — last_update_ts stays None
        time.sleep(0.02)  # wait for grace period to elapse
        assert stream.is_stale is True, (
            "is_stale must transition to True after stale_seconds "
            "even if the feed never delivered a message"
        )

    def test_not_stale_after_recent_update(self):
        """Shortly after a trade, is_stale returns False."""
        import time
        stream = LiveVPINStream(
            bucket_volume_btc=10.0, fallback_vpin=0.5, stale_seconds=60.0,
        )
        msg = json.dumps({"q": "5.0", "m": False, "T": 1})
        stream._process_message(msg)
        assert stream.is_stale is False

    def test_stale_after_timeout(self):
        """After stale_seconds with no update, is_stale returns True."""
        import time
        stream = LiveVPINStream(
            bucket_volume_btc=10.0, fallback_vpin=0.5, stale_seconds=0.01,
        )
        msg = json.dumps({"q": "5.0", "m": False, "T": 1})
        stream._process_message(msg)
        time.sleep(0.02)  # wait for stale threshold
        assert stream.is_stale is True
