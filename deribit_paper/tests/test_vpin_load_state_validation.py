"""D3 (parity-cycle round 2 — 2026-05-07): VPIN warm-start state load
validates that ``bucket_volume_btc`` and ``rolling_window`` match the
accumulator's constructor args. Pre-D3 mismatches were silently
swallowed, leaving the live engine running with imbalances on an
incompatible scale or window.
"""

from __future__ import annotations

import pytest

from deribit_paper.features.vpin_live import _LiveBucketAccumulator


def _make_state(
    *,
    bucket_volume_btc: float | None = 1000.0,
    rolling_window: int | None = 50,
    imbalances: list[float] | None = None,
) -> dict:
    state: dict = {
        "partial_vol": 0.0,
        "partial_buy": 0.0,
        "partial_sell": 0.0,
        "recent_imbalances": list(imbalances or []),
    }
    if bucket_volume_btc is not None:
        state["bucket_volume_btc"] = bucket_volume_btc
    if rolling_window is not None:
        state["rolling_window"] = rolling_window
    return state


def test_load_state_matching_args_succeeds():
    """Sanity: matching constructor args + state args → no raise."""
    acc = _LiveBucketAccumulator(bucket_volume_btc=1000.0, rolling_window=50)
    state = _make_state(imbalances=[0.1, 0.2, 0.3])
    acc.load_state(state)
    # 3 imbalances loaded
    assert len(acc._recent_imbalances) == 3


def test_load_state_bucket_volume_mismatch_raises():
    """D3: state was saved with V=2000, accumulator is V=1000 → ValueError."""
    acc = _LiveBucketAccumulator(bucket_volume_btc=1000.0, rolling_window=50)
    state = _make_state(bucket_volume_btc=2000.0)
    with pytest.raises(ValueError, match="bucket_volume_btc mismatch"):
        acc.load_state(state)


def test_load_state_rolling_window_mismatch_raises():
    """D3: state was saved with N=100, accumulator is N=50 → ValueError."""
    acc = _LiveBucketAccumulator(bucket_volume_btc=1000.0, rolling_window=50)
    state = _make_state(rolling_window=100)
    with pytest.raises(ValueError, match="rolling_window mismatch"):
        acc.load_state(state)


def test_load_state_omitted_args_back_compat():
    """Older sidecars that don't carry the args still load (fail-open compat)."""
    acc = _LiveBucketAccumulator(bucket_volume_btc=1000.0, rolling_window=50)
    state = _make_state(
        bucket_volume_btc=None, rolling_window=None,
        imbalances=[0.1, 0.2],
    )
    acc.load_state(state)  # no raise
    assert len(acc._recent_imbalances) == 2


def test_load_state_floating_tolerance_on_volume():
    """Tiny float drift on saved volume must not trigger ValueError."""
    acc = _LiveBucketAccumulator(bucket_volume_btc=1000.0, rolling_window=50)
    state = _make_state(bucket_volume_btc=1000.0 + 1e-12)
    acc.load_state(state)  # no raise
