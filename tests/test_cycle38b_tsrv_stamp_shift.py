"""Cycle-38b (2026-05-17): pin the offline-trace TSRV stamp shift.

The replay-parity harness (cycle-38) surfaced a measurement bug: the
TSRV-derived columns of ``BTC_vol_arb_svi.parquet`` are OPEN-stamped
(row T carries the TSRV of ``[T, T+15min)``) while the live engine
END-stamps (bar T uses ``[T-15min, T)``). Proven decisively:
``compute_bar_tsrv`` over ``[14:00,14:15)`` reproduced the parquet's
14:00 row (n_ticks 28429, vol_ann 0.767) exactly, whereas
``[13:45,14:00)`` gave vol_ann 0.554.

The cycle-33/34/35 trace comparator joins ``live[T]`` <-> ``offline[T]``,
so without correction it compared two different 15-min windows on
every C1 field — a spurious TSRV divergence that masked the true
post-cycle-39 residual.

Cycle-38b: ``backtester._emit_feature_trace`` shifts the TSRV-derived
columns (``vol_ann``, ``jump_fraction``, ``tsrv``) +1 bar so the
offline trace row ``bar_close_ts=T`` carries the ``[T-15min, T)`` TSRV
that matches the live engine's end-stamped bar. ``index_price`` and
the iv/prediction columns are left unshifted (already end-stamped).

Tests:
1. The offline trace's ``vol_ann``/``jump_fraction`` at row T equal
   the input df's row ``T-15min`` value (shift(1)).
2. ``index_price_close`` is NOT shifted (stays end-stamped).
3. The backtester's own returned df is unmodified (shift is local).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame({
        "index_price": [63000.0, 63500.0, 64000.0, 64500.0],
        "vol_ann": [0.40, 0.50, 0.60, 0.70],
        "jump_fraction": [0.01, 0.02, 0.03, 0.04],
        "tsrv": [1e-6, 2e-6, 3e-6, 4e-6],
        "iv_used": [42.0, 43.0, 41.0, 44.0],
        "iv": [42.1, 43.1, 41.1, 44.1],
        "svi_iv_matched": [42.0, 43.0, 41.0, 44.0],
        "dvol_close": [40.0, 41.0, 39.0, 42.0],
        "rv_pred_pct": [44.0, 44.5, 43.0, 45.0],
        "y_pred": [-0.83, -0.81, -0.85, -0.80],
        "edge": [2.0, 1.5, 2.0, 1.0],
        "spread_iv": [1.5, 1.55, 1.7, 1.6],
        "signal": [0, 0, 0, 0],
    }, index=pd.DatetimeIndex(
        ["2026-05-14T13:45:00Z", "2026-05-14T14:00:00Z",
         "2026-05-14T14:15:00Z", "2026-05-14T14:30:00Z"],
        name="timestamp", tz="UTC",
    ))


def _emit(tmp_path: Path, df: pd.DataFrame) -> list[dict]:
    out = tmp_path / "trace.jsonl"
    cfg = BacktestConfig(feature_trace_path=str(out),
                         feature_trace_start=None, feature_trace_end=None)
    bt = VolArbBacktester.__new__(VolArbBacktester)
    bt.cfg = cfg
    bt._emit_feature_trace(df)
    return [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]


def test_tsrv_fields_shifted_one_bar(tmp_path: Path) -> None:
    """Offline trace row T carries the input df's row T-15min TSRV."""
    df = _synthetic_df()
    rows = _emit(tmp_path, df)
    by_ts = {r["bar_close_ts"][:19]: r for r in rows}

    # Row 14:00 must carry the 13:45 TSRV values (shift +1 bar).
    r1400 = by_ts["2026-05-14T14:00:00"]
    assert r1400["vol_ann"] == 0.40          # df 13:45 vol_ann
    assert r1400["jump_fraction"] == 0.01    # df 13:45 jump_fraction

    # Row 14:30 must carry the 14:15 TSRV values.
    r1430 = by_ts["2026-05-14T14:30:00"]
    assert r1430["vol_ann"] == 0.60          # df 14:15 vol_ann
    assert r1430["jump_fraction"] == 0.03

    # Row 13:45 (the first bar) has a NaN shifted TSRV -> field dropped.
    r1345 = by_ts["2026-05-14T13:45:00"]
    assert "vol_ann" not in r1345
    assert "jump_fraction" not in r1345


def test_index_price_not_shifted(tmp_path: Path) -> None:
    """``index_price`` is end-stamped already — must NOT be shifted."""
    df = _synthetic_df()
    rows = _emit(tmp_path, df)
    by_ts = {r["bar_close_ts"][:19]: r for r in rows}
    # Row 14:00 keeps its OWN index_price (63500), not 13:45's (63000).
    assert by_ts["2026-05-14T14:00:00"]["index_price_close"] == 63500.0


def test_emit_does_not_mutate_caller_df(tmp_path: Path) -> None:
    """The +1 shift is on a local copy — the caller's df is untouched."""
    df = _synthetic_df()
    before = df["vol_ann"].tolist()
    _emit(tmp_path, df)
    assert df["vol_ann"].tolist() == before, (
        "_emit_feature_trace must not mutate the backtester's df"
    )
