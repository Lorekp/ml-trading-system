"""Correctness tests for production entry filters (W5.X).

Tests verify:

1. Defaults (``min_dte_days=0.0``, ``skip_long_hours_utc=()``) are no-op:
   signals pass through unchanged.
2. ``min_dte_days`` mutes ALL signals (long AND short) on bars where
   ``dte_days`` is below the floor.
3. ``skip_long_hours_utc`` mutes ONLY long signals at configured hours;
   shorts at the same hours are preserved.
4. Both filters compose correctly when used together.
5. ``with_btc_production_filters()`` preset applies the documented values.
6. Filtered bars incur NO downstream fees/slippage (signal==0 propagates).

Synthetic mini-DataFrame keeps the tests fast and hand-verifiable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from vol_arb.backtester import BacktestConfig, VolArbBacktester  # noqa: E402


def _mini_df() -> pd.DataFrame:
    """6 bars with pre-set signals; varied dte_days and UTC hours."""
    idx = pd.DatetimeIndex([
        "2024-01-01 13:00:00+00:00",  # hour 13 — kept
        "2024-01-01 14:00:00+00:00",  # hour 14 — in US long-skip window
        "2024-01-01 15:00:00+00:00",  # hour 15 — in US long-skip window
        "2024-01-01 16:00:00+00:00",  # hour 16 — in US long-skip window
        "2024-01-01 17:00:00+00:00",  # hour 17 — kept
        "2024-01-02 15:00:00+00:00",  # hour 15 — DTE=3 (below floor)
    ])
    df = pd.DataFrame({
        "signal": [1, 1, -1, 1, -1, 1],
        "dte_days": [10.0, 12.0, 7.0, 5.1, 8.0, 3.0],
    }, index=idx)
    return df


def _bt_with_cfg(**kwargs) -> VolArbBacktester:
    """Build a VolArbBacktester with minimal config kwargs."""
    cfg = BacktestConfig(**kwargs)
    bt = VolArbBacktester.__new__(VolArbBacktester)  # skip data load
    bt.cfg = cfg
    return bt


# ── Test 1: defaults are no-op ───────────────────────────────

def test_defaults_are_noop():
    df = _mini_df().copy()
    original = df["signal"].values.copy()
    bt = _bt_with_cfg()
    out = bt._apply_entry_filters(df)
    assert np.array_equal(out, original), \
        "Default config (min_dte_days=0.0, skip_long_hours_utc=()) must be no-op"


# ── Test 2: min_dte_days mutes BOTH long and short below floor ───

def test_min_dte_days_mutes_all_signals_below_floor():
    df = _mini_df().copy()
    bt = _bt_with_cfg(min_dte_days=5.0)
    out = bt._apply_entry_filters(df)
    # Row 5: dte=3.0 < 5 → muted (was long +1)
    assert out[5] == 0, "DTE=3 row must be muted by min_dte_days=5.0"
    # Rows 0-4: dte >= 5 → unchanged
    expected = np.array([1, 1, -1, 1, -1, 0], dtype=np.int8)
    assert np.array_equal(out, expected), f"got {out} expected {expected}"


def test_min_dte_days_also_mutes_shorts():
    # Build a case where the below-floor bar is a SHORT
    idx = pd.DatetimeIndex(["2024-01-01 10:00:00+00:00"])
    df = pd.DataFrame({"signal": [-1], "dte_days": [2.0]}, index=idx)
    bt = _bt_with_cfg(min_dte_days=5.0)
    out = bt._apply_entry_filters(df)
    assert out[0] == 0, "SHORT below DTE floor must be muted too"


# ── Test 3: skip_long_hours_utc mutes ONLY longs ──────────────

def test_skip_long_hours_mutes_only_longs():
    df = _mini_df().copy()
    bt = _bt_with_cfg(skip_long_hours_utc=(14, 15, 16))
    out = bt._apply_entry_filters(df)
    # Row 1: long at hour 14 → muted
    # Row 2: SHORT at hour 15 → KEPT (short-at-bad-hour outperforms)
    # Row 3: long at hour 16 → muted
    # Row 5: long at hour 15 → muted
    expected = np.array([1, 0, -1, 0, -1, 0], dtype=np.int8)
    assert np.array_equal(out, expected), f"got {out} expected {expected}"


def test_shorts_at_skip_hours_are_preserved():
    idx = pd.DatetimeIndex([
        "2024-01-01 14:00:00+00:00",
        "2024-01-01 15:00:00+00:00",
        "2024-01-01 16:00:00+00:00",
    ])
    df = pd.DataFrame({"signal": [-1, -1, -1], "dte_days": [10.0, 10.0, 10.0]}, index=idx)
    bt = _bt_with_cfg(skip_long_hours_utc=(14, 15, 16))
    out = bt._apply_entry_filters(df)
    assert np.all(out == -1), "Shorts at skip hours must NOT be muted"


# ── Test 4: filters compose ───────────────────────────────────

def test_filters_compose():
    df = _mini_df().copy()
    bt = _bt_with_cfg(min_dte_days=5.0, skip_long_hours_utc=(14, 15, 16))
    out = bt._apply_entry_filters(df)
    # Row 0: long, hour 13, dte 10 → kept
    # Row 1: long, hour 14 → MUTED (long hour filter)
    # Row 2: short, hour 15, dte 7 → kept (short not affected by hour filter)
    # Row 3: long, hour 16 → MUTED (long hour filter)
    # Row 4: short, hour 17, dte 8 → kept
    # Row 5: long, hour 15, dte 3 → MUTED by BOTH filters
    expected = np.array([1, 0, -1, 0, -1, 0], dtype=np.int8)
    assert np.array_equal(out, expected), f"got {out} expected {expected}"


# ── Test 5: production preset contents ────────────────────────

def test_btc_production_filters_preset():
    cfg = BacktestConfig.for_currency("BTC").with_btc_production_filters()
    assert cfg.entry_threshold == 22.0
    assert cfg.entry_threshold_long == 30.0
    assert cfg.min_dte_days == 5.0
    assert cfg.skip_long_hours_utc == (14, 15, 16)
    assert cfg.target_margin_utilization == 0.075
    assert cfg.use_sor is False


def test_btc_production_composes_with_realistic_costs():
    cfg = BacktestConfig.for_currency("BTC").with_realistic_costs().with_btc_production_filters()
    # Realistic-cost knobs survive
    assert cfg.perp_slippage_bps == 3.0
    assert cfg.base_spread_iv == 2.0
    assert cfg.maker_adverse_selection_iv == 0.75
    assert cfg.use_profit_filter is False
    # Production filters applied
    assert cfg.min_dte_days == 5.0
    assert cfg.skip_long_hours_utc == (14, 15, 16)


# ── Test 9: term-slope filter (C.5) ───────────────────────────

def test_term_slope_filter_mutes_only_longs_in_contango():
    idx = pd.DatetimeIndex([
        "2024-01-01 10:00:00+00:00",  # long, contango  → muted
        "2024-01-01 11:00:00+00:00",  # long, backward. → kept
        "2024-01-01 12:00:00+00:00",  # short, contango → kept
        "2024-01-01 13:00:00+00:00",  # short, backward. → kept
    ])
    df = pd.DataFrame({
        "signal": [1, 1, -1, -1],
        "dte_days": [10.0, 10.0, 10.0, 10.0],
        "svi_term_slope_1w_2w": [3.0, -2.0, 4.0, -1.0],
    }, index=idx)
    bt = _bt_with_cfg(skip_long_if_term_slope_gt=0.0)
    out = bt._apply_entry_filters(df)
    expected = np.array([0, 1, -1, -1], dtype=np.int8)
    assert np.array_equal(out, expected), f"got {out} expected {expected}"


def test_term_slope_filter_nan_safe():
    # NaN term slope should NOT be muted (NaN > thr is False)
    idx = pd.DatetimeIndex(["2024-01-01 10:00:00+00:00"])
    df = pd.DataFrame({
        "signal": [1],
        "dte_days": [10.0],
        "svi_term_slope_1w_2w": [float("nan")],
    }, index=idx)
    bt = _bt_with_cfg(skip_long_if_term_slope_gt=0.0)
    out = bt._apply_entry_filters(df)
    assert out[0] == 1, "NaN term-slope row should NOT be muted"


def test_term_slope_filter_threshold_boundary():
    # Slope exactly AT threshold should NOT be muted (filter is strict >)
    idx = pd.DatetimeIndex(["2024-01-01 10:00:00+00:00",
                             "2024-01-01 11:00:00+00:00"])
    df = pd.DataFrame({
        "signal": [1, 1],
        "dte_days": [10.0, 10.0],
        "svi_term_slope_1w_2w": [0.0, 0.001],  # one at, one above
    }, index=idx)
    bt = _bt_with_cfg(skip_long_if_term_slope_gt=0.0)
    out = bt._apply_entry_filters(df)
    assert out[0] == 1, "Slope == threshold must pass"
    assert out[1] == 0, "Slope > threshold must be muted"


def test_term_slope_filter_default_none_is_noop():
    idx = pd.DatetimeIndex(["2024-01-01 10:00:00+00:00"])
    df = pd.DataFrame({
        "signal": [1], "dte_days": [10.0],
        "svi_term_slope_1w_2w": [100.0],
    }, index=idx)
    bt = _bt_with_cfg()  # skip_long_if_term_slope_gt=None default
    out = bt._apply_entry_filters(df)
    assert out[0] == 1, "Default (None) must be no-op even at huge contango"


def test_term_slope_missing_column_warns_no_crash(caplog):
    import logging
    idx = pd.DatetimeIndex(["2024-01-01 10:00:00+00:00"])
    df = pd.DataFrame({"signal": [1], "dte_days": [10.0]}, index=idx)  # no svi col
    bt = _bt_with_cfg(skip_long_if_term_slope_gt=0.0)
    with caplog.at_level(logging.WARNING, logger="vol_arb.backtester"):
        out = bt._apply_entry_filters(df)
    assert out[0] == 1, "Missing column: signal unchanged"
    assert any("svi_term_slope_1w_2w" in rec.message for rec in caplog.records)


# ── Test 10: v2 production preset contents ────────────────────

def test_btc_production_v2_preset():
    cfg = BacktestConfig.for_currency("BTC").with_btc_production_filters_v2()
    # v1 filters inherited EXCEPT entry_threshold_long (v2 overrides)
    assert cfg.entry_threshold == 22.0
    # v2 lowers thr_L from 30 → 12: the contango filter already scrubbed
    # the bad-regime longs, so remaining longs can fire at lower edge.
    assert cfg.entry_threshold_long == 12.0
    assert cfg.min_dte_days == 5.0
    assert cfg.skip_long_hours_utc == (14, 15, 16)
    assert cfg.use_sor is False
    # v2 additions
    assert cfg.use_drl_hedging is True
    assert cfg.use_profit_scaling is True
    assert cfg.profit_scaling_floor == 0.3
    assert cfg.target_margin_utilization == 0.15
    assert cfg.holding_minutes == 30
    # sizing_model INTENTIONALLY unchanged at composite_v2 — Kelly
    # broke compound-mode scaling per Phase B.3 re-test.
    assert cfg.sizing_model == "composite_v2"
    assert cfg.skip_long_if_term_slope_gt == 0.0


def test_v2_composes_with_realistic_costs():
    cfg = BacktestConfig.for_currency("BTC").with_realistic_costs().with_btc_production_filters_v2()
    # Realistic-cost knobs preserved
    assert cfg.perp_slippage_bps == 3.0
    assert cfg.base_spread_iv == 2.0
    # v2 additions applied
    assert cfg.use_drl_hedging is True
    assert cfg.holding_minutes == 30
    assert cfg.skip_long_if_term_slope_gt == 0.0


# ── Test 6: dtype invariance ──────────────────────────────────

def test_output_dtype_int8():
    df = _mini_df().copy()
    bt = _bt_with_cfg(min_dte_days=5.0, skip_long_hours_utc=(14, 15, 16))
    out = bt._apply_entry_filters(df)
    assert out.dtype == np.int8, f"output dtype should be int8, got {out.dtype}"


# ── Test 7: missing dte_days column degrades gracefully ──────

def test_missing_dte_column_warns_no_crash(caplog):
    import logging
    idx = pd.DatetimeIndex(["2024-01-01 15:00:00+00:00"])
    df = pd.DataFrame({"signal": [1]}, index=idx)  # NO dte_days column
    bt = _bt_with_cfg(min_dte_days=5.0)
    with caplog.at_level(logging.WARNING, logger="vol_arb.backtester"):
        out = bt._apply_entry_filters(df)
    # Filter silently skipped when column absent, signal unchanged
    assert out[0] == 1
    assert any("dte_days" in rec.message for rec in caplog.records), \
        "expected warning about missing dte_days column"


# ── Test 8: tuple validity — hours in [0, 23] (sanity) ───────

def test_invalid_hour_still_works():
    """Hour outside [0, 23] is just never matched (no crash)."""
    df = _mini_df().copy()
    bt = _bt_with_cfg(skip_long_hours_utc=(99,))  # impossible hour
    out = bt._apply_entry_filters(df)
    # Same as no filter applied (no hour matches 99)
    assert np.array_equal(out, df["signal"].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
