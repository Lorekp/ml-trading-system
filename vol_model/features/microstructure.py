"""Microstructure features — Order Flow Imbalance (OFI) and VPIN.

OFI: Multi-level order flow imbalance from L2 snapshots (Cont, Kukanov & Stoikov 2014).
VPIN: Volume-synchronized probability of informed trading (Easley, Lopez de Prado & O'Hara 2012).

Vectorized numpy implementations — no Numba dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# VPIN — numpy vectorized with cumsum trick
# ────────────────────────────────────────────────────────────

def _vpin_rolling(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """Compute rolling VPIN using numpy cumsum (vectorized, no loops).

    VPIN_t = sum|V_B - V_S| / sum(V_B + V_S) over rolling n_buckets window.
    Returns array of shape (N,). First (n_buckets-1) entries are NaN.
    """
    abs_imb = np.abs(buy_volume - sell_volume)
    total_vol = buy_volume + sell_volume

    # Cumsum trick for rolling sum
    cs_abs = np.cumsum(abs_imb)
    cs_vol = np.cumsum(total_vol)

    n = len(buy_volume)
    out = np.full(n, np.nan, dtype=np.float64)

    # Prepend 0 for the shifted cumsum
    cs_abs_pad = np.concatenate([[0.0], cs_abs])
    cs_vol_pad = np.concatenate([[0.0], cs_vol])

    # For i >= n_buckets-1: sum over [i-n_buckets+1, i]
    # = cs_pad[i+1] - cs_pad[i+1-n_buckets]
    idx_end = np.arange(n_buckets, n + 1)  # i+1 for padded array
    idx_start = idx_end - n_buckets
    sum_abs = cs_abs_pad[idx_end] - cs_abs_pad[idx_start]
    sum_vol = cs_vol_pad[idx_end] - cs_vol_pad[idx_start]

    safe_vol = np.where(sum_vol > 0, sum_vol, 1.0)
    out[n_buckets - 1:] = sum_abs / safe_vol
    out[n_buckets - 1:][sum_vol == 0] = 0.0

    return out


# ────────────────────────────────────────────────────────────
# OFI — numpy vectorized
# ────────────────────────────────────────────────────────────

def _ofi_multi_level(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-level OFI and depth-weighted mid-price (vectorized).

    Parameters:
        bid_px, bid_sz, ask_px, ask_sz: (N, n_levels) float64 arrays

    Returns:
        ofi: (N,) normalized OFI per snapshot
        wmid: (N,) depth-weighted mid-price
    """
    n_snaps, n_levels = bid_px.shape

    # ── Depth-weighted mid-price ──
    # wmid = sum(ask_px * bid_sz + bid_px * ask_sz) / sum(bid_sz + ask_sz)
    num = np.sum(ask_px * bid_sz + bid_px * ask_sz, axis=1)
    den = np.sum(bid_sz + ask_sz, axis=1)
    wmid = np.where(den > 0, num / den, 0.5 * (bid_px[:, 0] + ask_px[:, 0]))

    # ── OFI per level ──
    # Bid side: compare bid_px[t] vs bid_px[t-1]
    bp_curr = bid_px[1:]   # (N-1, L)
    bp_prev = bid_px[:-1]
    bs_curr = bid_sz[1:]
    bs_prev = bid_sz[:-1]

    bid_up = bp_curr > bp_prev
    bid_same = bp_curr == bp_prev
    bid_down = ~bid_up & ~bid_same

    dw_b = np.where(bid_up, bs_curr,
            np.where(bid_same, bs_curr - bs_prev, -bs_prev))

    # Ask side: compare ask_px[t] vs ask_px[t-1]
    ap_curr = ask_px[1:]
    ap_prev = ask_px[:-1]
    as_curr = ask_sz[1:]
    as_prev = ask_sz[:-1]

    ask_down = ap_curr < ap_prev
    ask_same = ap_curr == ap_prev
    ask_up = ~ask_down & ~ask_same

    dw_a = np.where(ask_down, -as_curr,
            np.where(ask_same, -(as_curr - as_prev), as_prev))

    # Inverse-depth weighting: w_L = 1/(L+1)
    weights = 1.0 / (np.arange(n_levels, dtype=np.float64) + 1.0)  # (L,)

    # Weighted OFI per snapshot
    raw_ofi = np.sum(weights[None, :] * (dw_b - dw_a), axis=1)  # (N-1,)
    total_depth = np.sum(bs_curr + as_curr, axis=1)  # (N-1,)

    ofi = np.zeros(n_snaps, dtype=np.float64)
    safe_depth = np.where(total_depth > 0, total_depth, 1.0)
    ofi[1:] = raw_ofi / safe_depth
    ofi[1:][total_depth == 0] = 0.0

    return ofi, wmid


# ────────────────────────────────────────────────────────────
# WRAPPER CLASS
# ────────────────────────────────────────────────────────────

class MicrostructureFeatures:
    """Orchestrates OFI + VPIN computation aligned to dollar-bar grid.

    Usage:
        ms = MicrostructureFeatures(config)
        vpin = ms.compute_vpin(bars)              # full 63-month history
        ofi, wmid = ms.compute_ofi(bars)          # NaN outside L2 coverage
    """

    def __init__(self, config):
        self.config = config
        self._l2_data_dir = Path(getattr(config, "l2_data_dir", "data/hl-bbo-raw"))
        self._coin = config.asset.replace("USDT", "")
        self._vpin_window = getattr(config, "vpin_window", 50)

    def compute_vpin(self, bars: pd.DataFrame) -> pd.Series:
        """Compute VPIN via numpy rolling cumsum on full bar history."""
        buy_vol = bars["buy_volume"].values.astype(np.float64)
        sell_vol = bars["sell_volume"].values.astype(np.float64)
        out = _vpin_rolling(buy_vol, sell_vol, self._vpin_window)
        return pd.Series(out, index=bars.index, name="vpin")

    def compute_ofi(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Compute multi-level OFI aligned to bar grid (chunk-based).

        Processes L2 data one day at a time to avoid OOM.
        Returns (ofi, wmid) Series indexed like bars.
        """
        from vol_model.features.lob_loader import iter_l2_chunks, get_l2_date_range

        n_bars = len(bars)
        bar_ofi = np.full(n_bars, np.nan, dtype=np.float64)
        bar_wmid = np.full(n_bars, np.nan, dtype=np.float64)
        bar_ts_ms = bars.index.values.astype("int64") // 1_000_000

        date_range = get_l2_date_range(self._l2_data_dir)
        if date_range is None:
            logger.warning("No L2 chunks found — OFI will be NaN")
            return (pd.Series(bar_ofi, index=bars.index, name="ofi"),
                    pd.Series(bar_wmid, index=bars.index, name="wmid"))

        lob_start_ms, lob_end_ms = date_range
        logger.info("    L2 range: %s to %s",
                     pd.Timestamp(lob_start_ms, unit="ms", tz="UTC").date(),
                     pd.Timestamp(lob_end_ms, unit="ms", tz="UTC").date())

        n_chunks = 0
        n_bars_filled = 0

        for ts_ms, bid_px, bid_sz, ask_px, ask_sz in iter_l2_chunks(
            self._l2_data_dir, self._coin
        ):
            n_snaps = len(ts_ms)

            # Compute per-snapshot OFI (vectorized numpy)
            snap_ofi, snap_wmid = _ofi_multi_level(bid_px, bid_sz, ask_px, ask_sz)

            # Find bars within this chunk's time range
            day_start = ts_ms[0]
            day_end = ts_ms[-1]
            bar_mask = (bar_ts_ms >= day_start) & (bar_ts_ms <= day_end)

            if not bar_mask.any():
                n_chunks += 1
                continue

            # Vectorized alignment
            bar_indices = np.where(bar_mask)[0]
            bars_in_day_ms = bar_ts_ms[bar_indices]
            idx = np.searchsorted(ts_ms, bars_in_day_ms, side="right")

            # wmid: forward-fill from last snapshot
            wmid_idx = np.clip(idx - 1, 0, n_snaps - 1)
            valid_wmid = idx > 0
            bar_wmid[bar_indices[valid_wmid]] = snap_wmid[wmid_idx[valid_wmid]]

            # OFI: cumsum trick for inter-bar sums
            ofi_cumsum = np.concatenate([[0.0], np.cumsum(snap_ofi)])
            ends = np.clip(idx, 0, n_snaps)
            starts = np.concatenate([[0], ends[:-1]])
            bar_ofi[bar_indices] = ofi_cumsum[ends] - ofi_cumsum[starts]

            n_bars_filled += bar_mask.sum()
            n_chunks += 1

            if n_chunks % 100 == 0:
                logger.info("    OFI: %d chunks, %d bars filled", n_chunks, n_bars_filled)

        logger.info("    OFI: %d chunks, %d / %d bars filled (%.1f%%)",
                     n_chunks, n_bars_filled, n_bars,
                     100 * n_bars_filled / n_bars if n_bars > 0 else 0)

        return (pd.Series(bar_ofi, index=bars.index, name="ofi"),
                pd.Series(bar_wmid, index=bars.index, name="wmid"))

    def verify_alignment(self, bars: pd.DataFrame) -> dict:
        """Diagnostic: check temporal alignment quality between LOB and bars."""
        from vol_model.features.lob_loader import get_l2_date_range

        date_range = get_l2_date_range(self._l2_data_dir)
        if date_range is None:
            return {"status": "no_l2_data"}

        lob_start, lob_end = date_range
        bar_ts_ms = bars.index.values.astype("int64") // 1_000_000
        in_range = (bar_ts_ms >= lob_start) & (bar_ts_ms <= lob_end)

        return {
            "status": "ok",
            "coverage_pct": float(100 * in_range.sum() / len(bars)),
            "n_bars_covered": int(in_range.sum()),
            "n_bars_total": len(bars),
            "lob_start": pd.Timestamp(lob_start, unit="ms", tz="UTC").isoformat(),
            "lob_end": pd.Timestamp(lob_end, unit="ms", tz="UTC").isoformat(),
        }
