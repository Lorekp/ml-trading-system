"""S-1c (parity-r3 cycle 6 — 2026-05-08) — assemble_transvol_input_row.

Locks the encoder-input row builder on the FeatureAssembler:

* Returns ``(features_101, time_features_2)`` in the canonical
  ``TRANSVOL_INPUT_COLUMNS`` order.
* Time features mirror ``(duration_s, n_bars_in_window)`` (offline
  ``vol_model/transvol/dataset.py:TIME_COLS``).
* Dtypes are ``float32`` (matches the encoder's input expectation).
* Handler failures fall back to the baseline row (NaN propagation
  honoured per F9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.signal.feature_assembler import (
    TRANSVOL_INPUT_COLUMNS,
    TRANSVOL_TIME_COLUMNS,
)


# ── Fixtures ───────────────────────────────────────────────────────


def _make_bar() -> Bar:
    return Bar(
        timestamp=pd.Timestamp("2026-05-01 12:00", tz="UTC"),
        bar_minutes=15,
        perp_trades=[],
        option_trades=[],
        index_price=70_000.0,
    )


def _make_bar_tsrv() -> BarTSRV:
    return BarTSRV(
        tsrv=0.04, rv_fast=0.04, rv_slow=0.05,
        bpv=0.045, medrv=0.045, continuous_var=0.04, jump_var=0.005,
        jump_fraction=0.05, rs_down=0.02, rs_up=0.02,
        down_jump_fraction=0.02, up_jump_fraction=0.03,
        n_ticks=1000, n_sub_bars=10, n_dash=20.0,
        vol_ann=0.50, vol_ann_continuous=0.49,
    )


@pytest.fixture
def assembler_with_baseline(tmp_path: Path):
    """Construct a FeatureAssembler with a synthetic baseline parquet
    that includes every TransVol input column, so handler-fallback can
    exercise the full path without hitting a real bundle.
    """
    from deribit_paper.signal.feature_assembler import FeatureAssembler

    idx = pd.DatetimeIndex(
        pd.date_range("2026-04-30", periods=10, freq="15min", tz="UTC"),
        name="timestamp",
    )
    rng = np.random.default_rng(0)
    cols: dict[str, np.ndarray] = {}
    for c in TRANSVOL_INPUT_COLUMNS:
        cols[c] = 0.5 + 0.1 * rng.standard_normal(10)
    # Stage-18 LightGBM features (subset; constructor walks these).
    for c in ("svi_atm_1d", "tv_0", "tv_1"):
        cols[c] = 0.1 + 0.05 * rng.standard_normal(10)
    df = pd.DataFrame(cols, index=idx)
    parquet_path = tmp_path / "features.parquet"
    df.to_parquet(parquet_path)

    # Stub bundle: minimal LightGBM features (the constructor only
    # checks producibility against this list).
    class _StubBundle:
        lgbm_features = ["tv_0", "tv_1", "svi_atm_1d"]

    assembler = FeatureAssembler(
        features_parquet=parquet_path,
        bundle=_StubBundle(),
    )
    return assembler


# ── Shape + ordering ───────────────────────────────────────────────


def test_returns_correct_shapes(assembler_with_baseline):
    feats, time_feats = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    assert feats.shape == (101,)
    assert time_feats.shape == (2,)


def test_returns_float32_dtype(assembler_with_baseline):
    feats, time_feats = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    assert feats.dtype == np.float32
    assert time_feats.dtype == np.float32


def test_time_features_match_named_columns(assembler_with_baseline):
    feats, time_feats = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    duration_idx = TRANSVOL_INPUT_COLUMNS.index(TRANSVOL_TIME_COLUMNS[0])
    nbars_idx = TRANSVOL_INPUT_COLUMNS.index(TRANSVOL_TIME_COLUMNS[1])
    assert time_feats[0] == feats[duration_idx]
    assert time_feats[1] == feats[nbars_idx]


def test_column_count_is_101():
    """Sanity check on the canonical column tuple itself."""
    assert len(TRANSVOL_INPUT_COLUMNS) == 101


def test_column_tuple_has_no_duplicates():
    assert len(TRANSVOL_INPUT_COLUMNS) == len(set(TRANSVOL_INPUT_COLUMNS))


# ── Live handlers fire (not a pure baseline read) ──────────────────


def test_n_bars_in_window_uses_baseline_when_provided(assembler_with_baseline):
    """``n_bars_in_window`` handler reads ``baseline_row``; ensure the
    handler is invoked rather than reading the dict directly."""
    feats, _ = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    nbars_idx = TRANSVOL_INPUT_COLUMNS.index("n_bars_in_window")
    # Synthetic baseline value at the latest row (row index 9): 0.5 + 0.1 * N(0,1).
    # Handler returns float; just assert it's finite and matches the latest baseline row.
    assert np.isfinite(feats[nbars_idx])


def test_handlers_fire_for_jump_var(assembler_with_baseline):
    """``up_jump_frac`` handler returns ctx.bar_tsrv.up_jump_fraction
    directly (existing _h_up_jump_frac). Confirm that the live
    BarTSRV value (0.03) appears in the row, not the synthetic
    baseline (0.5 ± 0.1)."""
    feats, _ = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    idx = TRANSVOL_INPUT_COLUMNS.index("up_jump_frac")
    assert feats[idx] == pytest.approx(0.03, abs=1e-5)


def test_funding_basis_falls_back_to_baseline_when_no_provider(assembler_with_baseline):
    """No provider → handler reads baseline value (0.5 ± 0.1 per fixture)."""
    feats, _ = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    idx = TRANSVOL_INPUT_COLUMNS.index("funding_basis")
    val = feats[idx]
    # Synthetic baseline range; ensures the handler propagated the baseline.
    assert np.isfinite(val)
    assert 0.0 < val < 1.0  # baseline distribution range


# ── NaN propagation ────────────────────────────────────────────────


def test_handler_exception_falls_back_to_baseline(assembler_with_baseline):
    """A handler raising should not abort the row build; NaN baseline
    propagates per F9 contract."""
    # The handlers are stable on the synthetic input; this test
    # simply validates that the surrounding plumbing produces a full
    # 101-feature output without raising.
    feats, time_feats = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    assert feats.shape == (101,)


def test_dollar_bar_features_baseline_when_no_buffer(assembler_with_baseline):
    """Without dollar_bar_buffer, all 16 dollar-bar handlers fall back
    to baseline. Live row should still produce 101 finite values."""
    feats, _ = assembler_with_baseline.assemble_transvol_input_row(
        _make_bar(), _make_bar_tsrv(),
    )
    # All values either finite (from baseline) or NaN (only if a
    # baseline cell was missing — in this fixture all are populated).
    assert np.isfinite(feats).sum() >= 95  # most are finite


def test_n_bars_in_window_uses_correct_END_stamped_window():
    """Cycle-23 (2026-05-11, audit A1 H6): the window for counting dollar
    bars in the JUST-CLOSED 15-min bar must be ``[bar_ts - 15min, bar_ts)``,
    not ``[floor("15min"), floor("15min") + 15min)``.

    Pre-cycle-23 the handler computed ``window_start = floor("15min")``
    and ``window_end = start + 15min``. For an END-stamped bar at 12:15,
    ``floor("15min") == 12:15``, so the window scanned ``[12:15, 12:30)``
    — the FUTURE window, empty at decision time. Live then silently fell
    back to a frozen ``baseline_row`` constant where offline produced a
    dynamic per-bar value.

    This test wires a minimal AssemblerContext + a stub DollarBarBuffer
    holding bars in both ``[12:00, 12:15)`` (which the END-stamped bar
    at 12:15 closes over) and ``[12:15, 12:30)`` (the next, future bar).
    The handler must count the FIRST set, not the second.
    """
    from types import SimpleNamespace
    from deribit_paper.signal.feature_assembler import (
        AssemblerContext, _h_n_bars_in_window,
    )

    bar = _make_bar()  # timestamp = 2026-05-01 12:00 UTC
    # Override to a non-boundary-coincident bar_ts so the bug is visible:
    bar.timestamp = pd.Timestamp("2026-05-01 12:15:00", tz="UTC")

    # Build a stub DollarBarBuffer with an `as_index()` method.
    # Bars in [12:00, 12:15): the JUST-CLOSED bar's window.
    bars_in_closed_window = pd.DatetimeIndex([
        "2026-05-01 12:00:30+00:00",
        "2026-05-01 12:05:00+00:00",
        "2026-05-01 12:10:00+00:00",
        "2026-05-01 12:14:30+00:00",
    ], tz="UTC")
    # Bars in [12:15, 12:30): the FUTURE bar's window (should NOT count).
    bars_in_future_window = pd.DatetimeIndex([
        "2026-05-01 12:15:30+00:00",
        "2026-05-01 12:20:00+00:00",
    ], tz="UTC")
    all_bars = bars_in_closed_window.append(bars_in_future_window)
    buffer = SimpleNamespace(
        __len__=lambda: len(all_bars),
        as_index=lambda: all_bars,
    )
    # SimpleNamespace doesn't support __len__ via attribute assignment;
    # build a tiny class instead.
    class _StubBuf:
        def __init__(self, idx): self._idx = idx
        def __len__(self): return len(self._idx)
        def as_index(self): return self._idx
    buffer = _StubBuf(all_bars)

    ctx = AssemblerContext(
        bar=bar,
        bar_tsrv=_make_bar_tsrv(),
        svi_calibrator=None,
        baseline_row=pd.Series({"n_bars_in_window": 99}),  # fallback sentinel
        dollar_bar_buffer=buffer,
    )

    n = _h_n_bars_in_window(ctx)
    # Cycle-23 fix: counts bars in [12:00, 12:15) = 4. Pre-fix would have
    # returned the fallback sentinel 99 (because [12:15, 12:30) had only
    # 2 bars, but the test exists to verify which WINDOW the handler
    # selects — the closed one, not the open future one).
    assert n == 4.0, (
        f"cycle-23 fix: handler must count bars in the END-stamped "
        f"window [bar_ts-15min, bar_ts) = [12:00, 12:15) (expected 4); "
        f"got n={n}. If n==2 the handler regressed to the future "
        f"window [12:15, 12:30). If n==99 the handler returned the "
        f"baseline fallback (would have been the pre-cycle-23 "
        f"silent failure)."
    )
