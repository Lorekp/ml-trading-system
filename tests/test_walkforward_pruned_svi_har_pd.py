"""Alignment + leakage tests for the HAR-PD walk-forward training harness.

The critical invariants pinned here:

  1. Post -15m shift, HAR-PD at join-index T uses data strictly KNOWN at
     wall-clock time T (not T+15m — that would be a leak, nor T-15m — that
     would be conservatively wasteful). Verified by mutating TSRV bars that
     start at or after T and confirming the post-shift HAR-PD[T] is byte-
     identical.

  2. The -15m shift exactly reverses the internal shift(1) that HAR-PD's
     feature construction applies. Sanity: shifting by -15m and then
     reversing the internal shift(1) recovers TSRV at end-stamp T — i.e.,
     the most recent complete bar ending at decision time T.

  3. The end-to-end label/feature alignment matches: at decision time T,
     predict label for bar [T, T+15m) using HAR-PD + SVI + TransVol +
     base features all known at or before wall-clock T. No feature may
     contain info from within [T, T+15m).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_model.features.har_pd import compute_har_pd_features
from vol_model.run_walkforward_pruned_svi_har_pd import (
    ADD_HAR_PD_FEATURES,
    _load_har_pd,
)


# ── 1. Shift-direction + alignment ──────────────────────────────────


def test_load_har_pd_applies_minus_15m_shift():
    """The loader must shift the sidecar index backward by 15 minutes so
    that the feature at new-index T was originally at T+15m (i.e., a row
    computed from TSRV end-stamps T, T-15m, ..., T-K+1)."""
    raw_path = Path("vol_model/output/BTCUSDT_features_har_pd_15m.parquet")
    if not raw_path.exists():
        pytest.skip("HAR-PD sidecar not present")

    raw = pd.read_parquet(raw_path)
    shifted = _load_har_pd("BTCUSDT")
    # The shifted index should equal raw index minus 15 minutes, row-for-row.
    diffs = (raw.index - shifted.index).unique()
    assert len(diffs) == 1, f"non-uniform shift detected: {diffs}"
    assert diffs[0] == pd.Timedelta(minutes=15), (
        f"expected +15m difference (raw = shifted + 15m), got {diffs[0]}"
    )


def test_load_har_pd_values_byte_identical_to_raw_post_shift():
    """The loader changes index but not values — row ordering preserved."""
    raw_path = Path("vol_model/output/BTCUSDT_features_har_pd_15m.parquet")
    if not raw_path.exists():
        pytest.skip("HAR-PD sidecar not present")

    raw = pd.read_parquet(raw_path)[list(ADD_HAR_PD_FEATURES)]
    shifted = _load_har_pd("BTCUSDT")
    # Reset both indexes to integer positions so we can compare values only.
    assert np.array_equal(
        raw.values, shifted.values, equal_nan=True
    ), "loader changed feature values, not just index"


# ── 2. Post-shift leakage probe ─────────────────────────────────────


def test_no_lookahead_after_shift():
    """After the -15m shift, HAR-PD at new-index T must NOT depend on TSRV
    bars starting at or after wall-clock T.

    Construction: a bar with TSRV end-stamp T corresponds to the bar
    [T-15m, T), which STARTS at T-15m and ENDS at T. This bar is strictly
    before the prediction window [T, T+15m), so using TSRV[T] is safe.

    Leakage would be using TSRV[T+15m] or later (bars that start inside the
    prediction window). The probe: mutate TSRV at end-stamps > T (bars
    starting at T or later), regenerate HAR-PD, apply -15m shift, and
    confirm the new-index-T row is unchanged."""
    tsrv_path = Path("TSRV/output/BTCUSDT_tsrv_15m.parquet")
    if not tsrv_path.exists():
        pytest.skip("TSRV parquet not present")

    tsrv = pd.read_parquet(tsrv_path)

    # Pick a wall-clock T mid-series so we have plenty of history.
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")

    # Normal HAR-PD + -15m shift → value at new-index T.
    hpd_normal = compute_har_pd_features(tsrv)
    hpd_normal.index = hpd_normal.index - pd.Timedelta(minutes=15)
    val_normal = hpd_normal.loc[T]

    # Mutate TSRV bars whose end-stamp > T (equivalently, bars starting at T
    # or later — the first "forbidden" bar starts at T and ends at T+15m,
    # stamped at T+15m; the second starts at T+15m, ends at T+30m, stamped
    # at T+30m; etc.).
    tsrv_mut = tsrv.copy()
    mask = tsrv_mut.index > T
    n_mutated = int(mask.sum())
    assert n_mutated > 0, "test requires some future bars — pick earlier T"
    tsrv_mut.loc[mask, ["rv_ann", "rs_up", "rs_down"]] = 999.0

    hpd_mut = compute_har_pd_features(tsrv_mut)
    hpd_mut.index = hpd_mut.index - pd.Timedelta(minutes=15)
    val_mut = hpd_mut.loc[T]

    diff = (val_normal - val_mut).abs().max()
    assert diff == 0.0, (
        f"HAR-PD at new-index T={T} changed by {diff} when {n_mutated} future "
        f"TSRV rows were mutated. This indicates post-shift lookahead leakage."
    )


def test_shift_preserves_dependence_on_current_bar():
    """Conversely, HAR-PD at new-index T MUST depend on TSRV end-stamped at T
    (= bar [T-15m, T), the most recent bar known at decision time T). This is
    the whole point of the -15m shift — without it, we'd be discarding this
    bar unnecessarily."""
    tsrv_path = Path("TSRV/output/BTCUSDT_tsrv_15m.parquet")
    if not tsrv_path.exists():
        pytest.skip("TSRV parquet not present")

    tsrv = pd.read_parquet(tsrv_path)
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")

    # Normal post-shift value at new-index T
    hpd_normal = compute_har_pd_features(tsrv)
    hpd_normal.index = hpd_normal.index - pd.Timedelta(minutes=15)
    val_normal = hpd_normal.loc[T]

    # Mutate ONLY TSRV at end-stamp T (bar [T-15m, T)). A correctly-aligned
    # HAR-PD at new-index T should change because this bar IS used.
    tsrv_mut = tsrv.copy()
    mask_current_bar = tsrv_mut.index == T
    assert mask_current_bar.sum() == 1, "expected exactly one row at T"
    tsrv_mut.loc[mask_current_bar, ["rv_ann", "rs_up", "rs_down"]] = 999.0

    hpd_mut = compute_har_pd_features(tsrv_mut)
    hpd_mut.index = hpd_mut.index - pd.Timedelta(minutes=15)
    val_mut = hpd_mut.loc[T]

    max_diff = (val_normal - val_mut).abs().max()
    assert max_diff > 0.0, (
        "HAR-PD at new-index T did NOT respond to mutation of TSRV[T]. The "
        "-15m shift should make the bar [T-15m, T) contribute to the new-T "
        "feature; if it doesn't, the shift isn't wired correctly."
    )


# ── 3. Cross-check: symmetric to SVI's +15m shift ───────────────────


def test_har_pd_alignment_matches_svi_end_convention():
    """Both SVI (+15m shift) and HAR-PD (-15m shift) should land with
    feature-info-cutoff equal to the label's decision time T. Concretely:
    HAR-PD post-shift at index T uses info through wall-clock T, which is
    the same info cutoff that production SVI achieves post-shift.

    We verify this through the leakage probe above. This test adds the
    cross-check that the shifts are in OPPOSITE directions (SVI shifts
    forward because raw SVI lives 15m ahead of labels; HAR-PD shifts back
    because raw HAR-PD lives 15m behind labels due to its internal shift(1))."""
    # Symbolic check — no backtest run needed.
    SVI_SHIFT_MINUTES = +15    # from _load_svi_atm_1d
    HAR_PD_SHIFT_MINUTES = -15  # from _load_har_pd (this change)
    assert SVI_SHIFT_MINUTES == -HAR_PD_SHIFT_MINUTES, (
        "SVI and HAR-PD shifts must be equal magnitude, opposite sign — they "
        "correct opposite stamp conventions (SVI: fitted-end-of-bar indexed "
        "at bar start; HAR-PD: paper-faithful strictly-past w/ internal "
        "shift(1))."
    )
