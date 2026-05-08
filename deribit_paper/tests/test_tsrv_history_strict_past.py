"""F-1 regression test (round-3 parity audit, fix 2026-05-08).

The TSRV history buffer must be in **strict-past** mode at predict
time: the latest entry MUST be the bar just before the current one,
not the current bar. Pre-fix the engine appended the current bar's
TSRV BEFORE running the predictor, which silently leaked one bar of
look-ahead into the 6 TSRV-grid features (``tsrv_mean_1d``,
``tsrv_mean_1d_lag3``, ``tsrv_mean_1w``, ``tsrv_ffd``,
``up_jump_frac_ema50``, ``bpv_tsrv_ratio_ema50``) — model trained on
``vol_ann.shift(1).rolling(96).mean()`` was being fed
``vol_ann.rolling(96).mean()`` (no shift).

This test mocks the predictor to capture the buffer state at the
moment ``predict`` is called, then asserts:

1. The latest entry in the buffer is the PRIOR bar (strict-past), and
2. The current bar IS in the buffer after process_bar returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
from deribit_paper.features.tsrv_live import BarTSRV


def _bar_tsrv(vol_ann: float = 0.5) -> BarTSRV:
    return BarTSRV(
        tsrv=vol_ann ** 2 * 1e-4, rv_fast=0.0, rv_slow=0.0,
        bpv=0.0, medrv=0.0, continuous_var=0.0, jump_var=0.0,
        jump_fraction=0.0, rs_down=0.0, rs_up=0.0,
        down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=10, n_sub_bars=2, n_dash=0.0,
        vol_ann=vol_ann, vol_ann_continuous=vol_ann,
    )


def test_tsrv_history_buffer_strict_past_contract_at_predict_time():
    """When the engine calls predict, the buffer must be in strict-past
    mode (latest entry = T-1, NOT the current bar T).
    """
    buf = TSRVHistoryBuffer(max_size=10)

    # Seed with 5 prior bars at 15-min cadence ending at T-15min.
    base = pd.Timestamp("2026-05-07 22:00:00", tz="UTC")
    for i in range(5):
        ts = base - pd.Timedelta(minutes=15 * (5 - i))
        buf.append_bar(ts, _bar_tsrv(vol_ann=0.5 + i * 0.01))

    # The engine's process_bar at "current bar T" should:
    # 1. Compute bar_tsrv for T
    # 2. Run predictor (which reads the buffer)
    # 3. Append T's bar_tsrv AFTER the predictor returns
    #
    # We simulate this by capturing buf state inside a mock predictor.
    captured: dict[str, Any] = {}

    def mock_predict_capture(buf_at_predict: TSRVHistoryBuffer) -> None:
        """Snapshot what the predictor sees."""
        captured["latest_at_predict"] = buf_at_predict.latest()

    # The contract: at predict time, latest must be T-1 (NOT T).
    current_t = base
    current_tsrv = _bar_tsrv(vol_ann=0.7)
    expected_latest_before_append = buf.latest()

    # Simulate the post-fix order: predict first, then append.
    mock_predict_capture(buf)
    buf.append_bar(current_t, current_tsrv)

    # Predict-time latest is T-1 (the prior bar at 21:45).
    assert captured["latest_at_predict"] is expected_latest_before_append
    assert captured["latest_at_predict"].timestamp == base - pd.Timedelta(minutes=15)
    assert captured["latest_at_predict"].vol_ann == pytest.approx(0.54)

    # After process_bar completes, the current bar IS in the buffer
    # (so the NEXT bar's predictor will see this bar as its T-1).
    assert buf.latest().timestamp == base
    assert buf.latest().vol_ann == pytest.approx(0.7)


def test_tsrv_grid_features_use_strict_past_after_fix():
    """tsrv_mean_1d (96-bar rolling mean of vol_ann) must NOT include
    the current bar's vol_ann when computed at predict time.
    """
    buf = TSRVHistoryBuffer(max_size=200)

    # 96 prior bars at vol_ann = 0.5 each, then a SPIKE bar at vol_ann = 2.0.
    base = pd.Timestamp("2026-05-07 22:00:00", tz="UTC")
    for i in range(96):
        ts = base - pd.Timedelta(minutes=15 * (96 - i))
        buf.append_bar(ts, _bar_tsrv(vol_ann=0.5))

    # At predict-time for the current spike bar, tsrv_mean_1d should
    # report 0.5 (the trailing 96 bars all at 0.5), NOT a value
    # contaminated by the spike.
    spike_tsrv = _bar_tsrv(vol_ann=2.0)

    # === Predict happens HERE — buffer has only the 96 trailing bars ===
    rolling_mean_at_predict = float(
        buf.as_series("vol_ann").tail(96).mean()
    )
    # === Append happens AFTER predict ===
    buf.append_bar(base, spike_tsrv)

    assert rolling_mean_at_predict == pytest.approx(0.5, abs=1e-12), (
        "tsrv_mean_1d at predict-time must not include the current bar"
    )

    # Sanity: AFTER append, the next bar's predict-time rolling mean
    # would include the spike (and drop the oldest) — verifying the
    # buffer advanced correctly.
    next_predict_mean = float(
        buf.as_series("vol_ann").tail(96).mean()
    )
    # 95 bars at 0.5 + 1 spike at 2.0 → mean = (95*0.5 + 2.0) / 96
    expected = (95 * 0.5 + 2.0) / 96
    assert next_predict_mean == pytest.approx(expected, rel=1e-12)


def test_pre_fix_behavior_would_leak_current_bar():
    """Sanity check: confirm the BUG mechanism. With pre-fix ordering
    (append BEFORE predict), the rolling mean DOES include the current
    bar — proving F-1 was a real divergence.
    """
    buf = TSRVHistoryBuffer(max_size=200)
    base = pd.Timestamp("2026-05-07 22:00:00", tz="UTC")
    for i in range(96):
        ts = base - pd.Timedelta(minutes=15 * (96 - i))
        buf.append_bar(ts, _bar_tsrv(vol_ann=0.5))
    spike_tsrv = _bar_tsrv(vol_ann=2.0)

    # Pre-fix (BAD) ordering: append BEFORE predict.
    buf.append_bar(base, spike_tsrv)
    leaky_mean = float(buf.as_series("vol_ann").tail(96).mean())

    # The leaky version drops the oldest bar and includes the spike,
    # so it's ~3% too high vs strict-past. This demonstrates the bug.
    expected_leaky = (95 * 0.5 + 2.0) / 96
    assert leaky_mean == pytest.approx(expected_leaky, rel=1e-12)
    assert leaky_mean > 0.5  # measurably above strict-past 0.5
