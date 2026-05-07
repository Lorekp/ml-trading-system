"""HAR-PD live byte-identity vs offline parquet.

C (parity 2026-05-07) — Tier 3 model alignment. The live engine's
``HARPDState.compute_features()`` must match
``vol_model/output/BTCUSDT_features_har_pd_15m.parquet`` to ≤1e-10
absolute tolerance. The offline path uses ``np.convolve``; the live
path uses a per-bar dot product. Both apply the same kernel weights to
the same strict-past values, so the only floating-point divergence
should be operation-order rounding (≤1e-15 typical, well under the
1e-10 gate).

If this test fails, the live engine MUST NOT be flipped to the
``_pruned_svi_har_pd_v1`` model — the divergence would silently
mis-feed LightGBM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.har_pd_live import (
    HAR_PD_RAW_FEATURES,
    K_BARS_DEFAULT,
    HARPDState,
)


# ── Inputs ──────────────────────────────────────────────────────────────────


REPO = Path(__file__).resolve().parent.parent
TSRV_PATH = REPO / "TSRV" / "output" / "BTCUSDT_tsrv_15m.parquet"
HAR_PD_PATH = REPO / "vol_model" / "output" / "BTCUSDT_features_har_pd_15m.parquet"


def _load_parquets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the offline TSRV input + the offline HAR-PD parquet."""
    if not TSRV_PATH.exists():
        pytest.skip(f"{TSRV_PATH} not on disk in this checkout")
    if not HAR_PD_PATH.exists():
        pytest.skip(f"{HAR_PD_PATH} not on disk in this checkout")
    tsrv = pd.read_parquet(TSRV_PATH)
    har_pd = pd.read_parquet(HAR_PD_PATH)
    # Both indexed on bar-end timestamps; trim to common slice.
    common = tsrv.index.intersection(har_pd.index)
    return tsrv.loc[common], har_pd.loc[common]


def test_byte_identity_first_500_bars_after_warmup():
    """For 500 bars starting at t=K_BARS_DEFAULT (96), live HAR-PD
    matches the offline parquet to ≤1e-10 absolute.

    Pattern: feed bars 0..K-1 into the buffer (warmup), then for each
    bar i in [K, K+500), compute live features BEFORE the bar's TSRV
    is appended. Compare against the offline parquet at index i.
    """
    tsrv, offline = _load_parquets()
    if len(tsrv) < K_BARS_DEFAULT + 500:
        pytest.skip(
            f"need {K_BARS_DEFAULT + 500} bars in TSRV, got {len(tsrv)}"
        )

    state = HARPDState(k_bars=K_BARS_DEFAULT)
    # Warmup: feed first K bars (these become "strict past" for bar t=K)
    for i in range(K_BARS_DEFAULT):
        row = tsrv.iloc[i]
        state.append(
            float(row["rs_up"]),
            float(row["rs_down"]),
            float(row["rv_ann"]),
        )

    max_abs_diff = 0.0
    bad_rows: list[tuple[int, str, float, float, float]] = []
    for i in range(K_BARS_DEFAULT, K_BARS_DEFAULT + 500):
        # Compute live features BEFORE appending bar i
        live = state.compute_features()
        offline_row = offline.iloc[i]
        for name in HAR_PD_RAW_FEATURES:
            o = float(offline_row[name])
            if not np.isfinite(o):
                continue  # offline NaN before its own warmup
            l = live[name]
            if not np.isfinite(l):
                bad_rows.append((i, name, l, o, float("nan")))
                continue
            diff = abs(l - o)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 1e-10:
                bad_rows.append((i, name, l, o, diff))
        # Append bar i so it becomes strict-past for bar i+1
        bar = tsrv.iloc[i]
        state.append(
            float(bar["rs_up"]),
            float(bar["rs_down"]),
            float(bar["rv_ann"]),
        )

    assert not bad_rows, (
        f"HAR-PD live diverges from offline at {len(bad_rows)} (row,feature) "
        f"pairs. First 5: {bad_rows[:5]}"
    )
    assert max_abs_diff < 1e-10, (
        f"max |diff| = {max_abs_diff:.3e} > 1e-10 — "
        "live HAR-PD math drifted from offline"
    )


def test_warmup_returns_nan_until_buffer_full():
    """Before ``k_bars`` entries, all 6 features are NaN (matches
    offline which leaves the first K rows NaN by construction).
    """
    state = HARPDState(k_bars=4)
    for i in range(3):
        feats = state.compute_features()
        assert all(np.isnan(v) for v in feats.values())
        state.append(0.01, 0.01, 0.01)
    # 4th observation makes the buffer warm
    state.append(0.01, 0.01, 0.01)
    feats = state.compute_features()
    assert all(np.isfinite(v) for v in feats.values())


def test_weight_orientation_matches_offline():
    """The largest kernel weight ``w_1`` must apply to the MOST RECENT
    strict-past bar (``buf[-1]``), not the oldest. This is the most
    common bug in HAR-PD live implementations and gets caught by the
    byte-identity test, but exposing it via a unit test makes the
    failure mode obvious.
    """
    state = HARPDState(k_bars=4)
    # Pure recent signal: zeros except buf[-1] (most recent)
    state.append(0.0, 0.0, 0.0)
    state.append(0.0, 0.0, 0.0)
    state.append(0.0, 0.0, 0.0)
    state.append(1.0, 0.0, 1.0)  # most recent bar
    feats = state.compute_features()
    # signed_flow at most-recent bar = sqrt(1.0) - sqrt(0.0) = 1.0.
    # R1(alpha=0.3) = w_1(0.3) * 1.0 + 0 + 0 + 0 = w_1(0.3)
    # Expected weight: alpha * (1-alpha)^0 = 0.3, normalized over 4 bars
    # The dominant weight at alpha=0.3 should land on the most recent
    # bar, so the result should be close to 0.3 (but not exact due to
    # normalisation across 4 bars). Floor: must be > 0.25 (the largest
    # any single mass should be).
    assert feats["har_pd_R1_a03"] > 0.25, (
        f"R1_a03={feats['har_pd_R1_a03']} — w_1 may be applied to "
        "wrong end of buffer (oldest instead of newest)"
    )


def test_prewarm_bulk_feed():
    """``prewarm`` shortcuts the 96-bar cold start at engine launch."""
    rows = [(float(i) * 0.01, float(i) * 0.005, float(i) * 0.02) for i in range(50)]
    state = HARPDState(k_bars=50)
    n = state.prewarm(rows)
    assert n == 50
    assert state.is_warm
    feats = state.compute_features()
    assert all(np.isfinite(v) for v in feats.values())
