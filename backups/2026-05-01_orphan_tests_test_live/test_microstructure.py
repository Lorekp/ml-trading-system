"""Tests for microstructure features — OFI and VPIN kernels."""

import numpy as np
import pandas as pd
import pytest

from vol_model.features.microstructure import (
    _ofi_multi_level,
    _vpin_ring_buffer,
    align_lob_to_bars,
)


# ── VPIN Tests ──

class TestVPIN:

    def test_warmup_nan(self):
        """First n_buckets-1 bars should be NaN."""
        buy = np.full(100, 300_000.0)
        sell = np.full(100, 200_000.0)
        out = np.empty(100)
        _vpin_ring_buffer(buy, sell, 50, out)
        assert np.isnan(out[:49]).all()
        assert np.isfinite(out[49:]).all()

    def test_all_buy_vpin_one(self):
        """Pure buy flow -> VPIN = 1.0."""
        buy = np.full(100, 500_000.0)
        sell = np.zeros(100)
        out = np.empty(100)
        _vpin_ring_buffer(buy, sell, 50, out)
        assert out[-1] == pytest.approx(1.0)

    def test_balanced_flow_vpin_zero(self):
        """Equal buy/sell per bar -> VPIN = 0.0."""
        buy = np.full(100, 250_000.0)
        sell = np.full(100, 250_000.0)
        out = np.empty(100)
        _vpin_ring_buffer(buy, sell, 50, out)
        assert out[-1] == pytest.approx(0.0)

    def test_matches_naive_rolling(self):
        """Compare ring buffer VPIN to naive rolling computation."""
        np.random.seed(42)
        n = 500
        buy = np.random.uniform(100_000, 400_000, n)
        sell = np.random.uniform(100_000, 400_000, n)
        n_buckets = 50

        out = np.empty(n)
        _vpin_ring_buffer(buy, sell, n_buckets, out)

        # Naive computation
        abs_imb = np.abs(buy - sell)
        total_vol = buy + sell
        naive = np.full(n, np.nan)
        for i in range(n_buckets - 1, n):
            window = slice(i - n_buckets + 1, i + 1)
            naive[i] = abs_imb[window].sum() / total_vol[window].sum()

        valid = np.isfinite(out) & np.isfinite(naive)
        np.testing.assert_allclose(out[valid], naive[valid], rtol=1e-10)

    def test_no_floating_point_drift(self):
        """After many bars, VPIN should still be correct."""
        np.random.seed(123)
        n = 100_000
        buy = np.random.uniform(200_000, 300_000, n)
        sell = np.random.uniform(200_000, 300_000, n)
        out = np.empty(n)
        _vpin_ring_buffer(buy, sell, 50, out)

        # Recompute last value naively
        abs_imb = np.abs(buy[-50:] - sell[-50:]).sum()
        total_vol = (buy[-50:] + sell[-50:]).sum()
        expected = abs_imb / total_vol

        assert out[-1] == pytest.approx(expected, rel=1e-8)


# ── OFI Tests ──

class TestOFI:

    def _make_ordered_book(self, n, base_bid=100.0, spread=2.0, level_gap=0.5):
        """Create properly ordered book data."""
        bid_px = np.zeros((n, 10))
        ask_px = np.zeros((n, 10))
        bid_sz = np.random.uniform(0.5, 2.0, (n, 10))
        ask_sz = np.random.uniform(0.5, 2.0, (n, 10))
        for i in range(n):
            shift = np.random.uniform(-1, 1)
            for L in range(10):
                bid_px[i, L] = base_bid + shift - L * level_gap
                ask_px[i, L] = base_bid + shift + spread + L * level_gap
        return (np.ascontiguousarray(bid_px), np.ascontiguousarray(bid_sz),
                np.ascontiguousarray(ask_px), np.ascontiguousarray(ask_sz))

    def test_first_snapshot_zero_ofi(self):
        """First snapshot has no delta -> OFI = 0."""
        bid_px, bid_sz, ask_px, ask_sz = self._make_ordered_book(10)
        out_ofi = np.empty(10)
        out_wmid = np.empty(10)
        _ofi_multi_level(bid_px, bid_sz, ask_px, ask_sz, out_ofi, out_wmid)
        assert out_ofi[0] == 0.0

    def test_bid_rise_positive_ofi(self):
        """Bid price increase -> positive OFI."""
        bp = np.array([[100.0] * 10, [101.0] * 10], dtype=np.float64)
        bs = np.full((2, 10), 1.0)
        ap = np.array([[102.0] * 10, [102.0] * 10], dtype=np.float64)
        as_ = np.full((2, 10), 1.0)
        o = np.empty(2)
        w = np.empty(2)
        _ofi_multi_level(bp, bs, ap, as_, o, w)
        assert o[1] > 0

    def test_bid_fall_negative_ofi(self):
        """Bid price decrease -> negative OFI."""
        bp = np.array([[101.0] * 10, [100.0] * 10], dtype=np.float64)
        bs = np.full((2, 10), 1.0)
        ap = np.array([[102.0] * 10, [102.0] * 10], dtype=np.float64)
        as_ = np.full((2, 10), 1.0)
        o = np.empty(2)
        w = np.empty(2)
        _ofi_multi_level(bp, bs, ap, as_, o, w)
        assert o[1] < 0

    def test_symmetric_book_zero_ofi(self):
        """Identical consecutive snapshots -> OFI = 0."""
        bp = np.tile([100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5], (5, 1))
        bs = np.full((5, 10), 1.0)
        ap = np.tile([102.0, 102.5, 103.0, 103.5, 104.0, 104.5, 105.0, 105.5, 106.0, 106.5], (5, 1))
        as_ = np.full((5, 10), 1.0)
        bp = np.ascontiguousarray(bp)
        ap = np.ascontiguousarray(ap)
        o = np.empty(5)
        w = np.empty(5)
        _ofi_multi_level(bp, bs, ap, as_, o, w)
        # After first snapshot, all OFI should be 0 (identical books)
        np.testing.assert_allclose(o[1:], 0.0, atol=1e-15)

    def test_wmid_between_bbo(self):
        """With ordered book, wmid should be between best bid and best ask."""
        bid_px, bid_sz, ask_px, ask_sz = self._make_ordered_book(100)
        out_ofi = np.empty(100)
        out_wmid = np.empty(100)
        _ofi_multi_level(bid_px, bid_sz, ask_px, ask_sz, out_ofi, out_wmid)
        for i in range(100):
            assert bid_px[i, 0] <= out_wmid[i] <= ask_px[i, 0], \
                f"wmid {out_wmid[i]:.4f} outside [{bid_px[i,0]:.4f}, {ask_px[i,0]:.4f}] at {i}"

    def test_ofi_bounded(self):
        """OFI should not diverge with realistic data."""
        bid_px, bid_sz, ask_px, ask_sz = self._make_ordered_book(1000)
        out_ofi = np.empty(1000)
        out_wmid = np.empty(1000)
        _ofi_multi_level(bid_px, bid_sz, ask_px, ask_sz, out_ofi, out_wmid)
        assert np.all(np.isfinite(out_ofi))
        assert np.abs(out_ofi).max() < 10.0  # normalized, should be bounded


# ── Alignment Tests ──

class TestAlignment:

    def test_forward_fill(self):
        """Bar close should map to most recent LOB snapshot."""
        lob_ts = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
        lob_ofi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        lob_wmid = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        bar_ts = np.array([2500, 3500, 4500], dtype=np.int64) * 1_000_000  # ns

        bar_ofi, bar_wmid = align_lob_to_bars(lob_ts, lob_ofi, lob_wmid, bar_ts)
        # wmid at bar 2500ms should be the snapshot at 2000ms
        assert bar_wmid[0] == pytest.approx(101.0)
        assert bar_wmid[1] == pytest.approx(102.0)
        assert bar_wmid[2] == pytest.approx(103.0)

    def test_no_future_data(self):
        """LOB snapshot timestamp must be <= bar close timestamp."""
        lob_ts = np.array([1000, 2000, 3000], dtype=np.int64)
        lob_wmid = np.array([100.0, 101.0, 102.0])
        lob_ofi = np.array([0.1, 0.2, 0.3])
        # Bar at 500ms — before any LOB data
        bar_ts = np.array([500], dtype=np.int64) * 1_000_000
        _, bar_wmid = align_lob_to_bars(lob_ts, lob_ofi, lob_wmid, bar_ts)
        assert np.isnan(bar_wmid[0])

    def test_outside_coverage_nan(self):
        """Bars outside LOB window should be NaN."""
        lob_ts = np.array([1000, 2000, 3000], dtype=np.int64)
        lob_ofi = np.array([0.1, 0.2, 0.3])
        lob_wmid = np.array([100.0, 101.0, 102.0])
        bar_ts = np.array([500, 1500, 5000], dtype=np.int64) * 1_000_000
        bar_ofi, bar_wmid = align_lob_to_bars(lob_ts, lob_ofi, lob_wmid, bar_ts)
        assert np.isnan(bar_ofi[0])  # before LOB start
        assert np.isnan(bar_ofi[2])  # after LOB end

    def test_empty_lob(self):
        """No LOB data -> all NaN."""
        bar_ts = np.array([1000, 2000], dtype=np.int64) * 1_000_000
        bar_ofi, bar_wmid = align_lob_to_bars(
            np.array([], dtype=np.int64),
            np.array([]),
            np.array([]),
            bar_ts,
        )
        assert np.isnan(bar_ofi).all()
        assert np.isnan(bar_wmid).all()
