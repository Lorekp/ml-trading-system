"""Cycle-39 (2026-05-17): pin the TSRV sub-bar threshold SSOT.

Parity audit 2026-05-17 found live ``compute_bar_tsrv`` defaulted to a
50_000 sub-bar dollar threshold while the offline
``BTCUSDT_tsrv_15m.parquet`` was built at 250_000 — a 5x mismatch that
biased live ``jump_fraction`` +0.167 and ``vol_ann`` ~0.09 over a
377-bar window. Both paths share ``compute_tsrv_window``; only the
threshold differed.

Cycle-39 introduces ONE source of truth —
``TSRV.config.BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD`` — read by the
live engine and by ``run_tsrv_incremental``.

Tests:
1. The SSOT constant exists and equals 250_000.
2. ``tsrv_live.DEFAULT_DOLLAR_THRESHOLD_USD`` IS the SSOT (not a bare
   50_000) — source-regex sentinel against a regression.
3. The engine's ``compute_bar_tsrv`` call passes the SSOT explicitly.
4. ``compute_bar_tsrv`` with default threshold uses the SSOT value
   end-to-end (numerical: a synthetic trade window produces the same
   TSRV with the default as with an explicit 250_000).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_ssot_constant_value() -> None:
    """The canonical sub-bar threshold is 250_000 USD."""
    from TSRV.config import BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD
    assert BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD == 250_000.0


def test_tsrv_live_default_is_ssot() -> None:
    """``DEFAULT_DOLLAR_THRESHOLD_USD`` must equal the SSOT, not 50_000."""
    from TSRV.config import BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD
    from deribit_paper.features.tsrv_live import DEFAULT_DOLLAR_THRESHOLD_USD
    assert DEFAULT_DOLLAR_THRESHOLD_USD == BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD
    assert DEFAULT_DOLLAR_THRESHOLD_USD == 250_000.0


def test_tsrv_live_source_references_ssot() -> None:
    """Source sentinel: tsrv_live.py imports the SSOT, no bare 50_000 default."""
    src = Path("deribit_paper/features/tsrv_live.py").read_text(encoding="utf-8")
    assert "BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD" in src, (
        "tsrv_live.py must source the threshold from the SSOT constant"
    )
    assert "DEFAULT_DOLLAR_THRESHOLD_USD: float = 50_000" not in src, (
        "the pre-cycle-39 bare 50_000 default must not regress"
    )


def test_engine_passes_ssot_threshold() -> None:
    """Source sentinel: engine.process_bar passes the SSOT to compute_bar_tsrv."""
    src = Path("deribit_paper/engine.py").read_text(encoding="utf-8")
    assert "from TSRV.config import BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD" in src
    assert "dollar_threshold_usd=BTCUSDT_TSRV_15M_SUBBAR_THRESHOLD_USD" in src, (
        "engine compute_bar_tsrv call must name the SSOT threshold"
    )


def test_compute_bar_tsrv_default_matches_explicit_250k() -> None:
    """``compute_bar_tsrv`` default threshold == explicit 250_000 (end-to-end)."""
    from deribit_paper.features.tsrv_live import compute_bar_tsrv

    rng = np.random.default_rng(39)
    n = 4000
    base = pd.Timestamp("2026-05-14T12:00:00Z")
    ts = base + pd.to_timedelta(np.sort(rng.uniform(0, 900, n)), unit="s")
    price = 80_000.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    amount = rng.uniform(0.01, 0.5, n)
    df = pd.DataFrame({"timestamp": ts, "price": price, "amount": amount})

    default_bar = compute_bar_tsrv(df, bar_seconds=900.0)
    explicit_bar = compute_bar_tsrv(df, bar_seconds=900.0, dollar_threshold_usd=250_000.0)
    assert default_bar.tsrv == explicit_bar.tsrv
    assert default_bar.jump_fraction == explicit_bar.jump_fraction
    assert default_bar.vol_ann == explicit_bar.vol_ann

    # And the 50_000 (pre-cycle-39) value produces a DIFFERENT result —
    # proving the threshold actually moves jump_fraction.
    old_bar = compute_bar_tsrv(df, bar_seconds=900.0, dollar_threshold_usd=50_000.0)
    assert old_bar.jump_fraction != default_bar.jump_fraction, (
        "50k vs 250k must produce different jump_fraction (regression guard)"
    )
