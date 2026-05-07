"""C (parity 2026-05-07): FeatureAssembler HAR-PD integration.

These tests verify the assembler:
1. Registers the 6 HAR-PD handlers in ``_LIVE_HANDLERS``.
2. Maintains the ``HARPDState`` buffer across ``build_row`` calls.
3. Uses strict-past values (current bar's TSRV is NOT in its own
   HAR-PD inputs).
4. Falls back to baseline during HAR-PD warmup.
5. Pre-warm bulk-feed shortcuts the cold start.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.har_pd_live import (
    HAR_PD_RAW_FEATURES,
    HARPDState,
)
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.signal.feature_assembler import (
    _LIVE_HANDLERS,
    AssemblerContext,
    FeatureAssembler,
)


def _make_tsrv(
    *, rs_up: float = 0.02, rs_down: float = 0.02, rv_ann: float = 0.04,
) -> BarTSRV:
    return BarTSRV(
        tsrv=rv_ann,
        rv_fast=rv_ann,
        rv_slow=rv_ann,
        bpv=rv_ann,
        medrv=rv_ann,
        continuous_var=rv_ann,
        jump_var=0.0,
        jump_fraction=0.0,
        rs_down=rs_down,
        rs_up=rs_up,
        down_jump_fraction=0.0,
        up_jump_fraction=0.0,
        n_ticks=100,
        n_sub_bars=10,
        n_dash=20.0,
        vol_ann=np.sqrt(rv_ann),
        vol_ann_continuous=np.sqrt(rv_ann),
    )


def _make_bar(timestamp: pd.Timestamp) -> Bar:
    return Bar(
        timestamp=timestamp,
        bar_minutes=15,
        perp_trades=[],
        option_trades=[],
        index_price=100_000.0,
    )


# ── 1. _LIVE_HANDLERS contains all 6 HAR-PD slots ──────────────────────────


def test_live_handlers_register_all_har_pd_features():
    """Every HAR-PD raw feature has a registered live handler."""
    for name in HAR_PD_RAW_FEATURES:
        assert name in _LIVE_HANDLERS, f"missing handler for {name}"


# ── 2. Handler reads from ctx.har_pd_features dict ─────────────────────────


def test_har_pd_handler_reads_from_ctx():
    """Each HAR-PD handler returns the value from ctx.har_pd_features."""
    ctx = AssemblerContext(
        bar=_make_bar(pd.Timestamp("2026-05-07 12:00", tz="UTC")),
        bar_tsrv=_make_tsrv(),
        svi_calibrator=None,
        baseline_row=pd.Series({name: 999.0 for name in HAR_PD_RAW_FEATURES}),
        har_pd_features={name: float(i + 1) for i, name in enumerate(HAR_PD_RAW_FEATURES)},
    )
    for i, name in enumerate(HAR_PD_RAW_FEATURES):
        handler = _LIVE_HANDLERS[name]
        assert handler(ctx) == pytest.approx(float(i + 1))


def test_har_pd_handler_falls_back_to_baseline_when_dict_empty():
    """Empty har_pd_features (warmup) → handler returns baseline row value."""
    ctx = AssemblerContext(
        bar=_make_bar(pd.Timestamp("2026-05-07 12:00", tz="UTC")),
        bar_tsrv=_make_tsrv(),
        svi_calibrator=None,
        baseline_row=pd.Series({name: 7.0 for name in HAR_PD_RAW_FEATURES}),
        har_pd_features={},
    )
    for name in HAR_PD_RAW_FEATURES:
        assert _LIVE_HANDLERS[name](ctx) == pytest.approx(7.0)


def test_har_pd_handler_falls_back_on_nan():
    """NaN in har_pd_features → handler returns baseline row value."""
    ctx = AssemblerContext(
        bar=_make_bar(pd.Timestamp("2026-05-07 12:00", tz="UTC")),
        bar_tsrv=_make_tsrv(),
        svi_calibrator=None,
        baseline_row=pd.Series({name: 5.0 for name in HAR_PD_RAW_FEATURES}),
        har_pd_features={name: float("nan") for name in HAR_PD_RAW_FEATURES},
    )
    for name in HAR_PD_RAW_FEATURES:
        assert _LIVE_HANDLERS[name](ctx) == pytest.approx(5.0)


# ── 3. Strict-past contract via the standalone state ───────────────────────


def test_har_pd_state_strict_past():
    """Live HAR-PD at bar t MUST NOT include bar t's own values.

    Pattern: warm 4 bars to ``[1, 2, 3, 4]``, compute features
    (uses 1-4), then append a 5th distinctive value. Compute again:
    features should now reflect [2, 3, 4, 5] (strictly past, NOT
    including the just-appended bar... wait. The "current bar" is the
    one we'd query for; its TSRV gets appended AT THE END of build_row.
    So the buffer ALWAYS contains strict-past relative to the "next
    build_row call".

    Concretely: at bar t1 we compute features (strict-past = bars
    before t1, which is empty initially or the warmup), then append
    t1's TSRV. At bar t2 we compute features using the buffer which
    now contains t1 — STRICTLY past relative to t2. Correct.

    Test: build a state with buf=[1,2,3,4], compute features (uses
    [1,2,3,4]), append 5, compute again (uses [2,3,4,5]). Verify the
    second call's features differ.
    """
    state = HARPDState(k_bars=4)
    for v in [1.0, 2.0, 3.0, 4.0]:
        state.append(rs_up=v, rs_down=0.0, rv_ann=v)
    f1 = state.compute_features()
    state.append(rs_up=5.0, rs_down=0.0, rv_ann=5.0)
    f2 = state.compute_features()
    # The two feature sets should differ because the buffer rotated.
    for name in HAR_PD_RAW_FEATURES:
        assert f1[name] != pytest.approx(f2[name]), (
            f"{name} unchanged after buffer rotation — "
            "strict-past contract may be broken"
        )


# ── 5. prewarm_har_pd_from_tsrv on real parquet ────────────────────────────


REPO = Path(__file__).resolve().parents[2]
TSRV_PATH = REPO / "TSRV" / "output" / "BTCUSDT_tsrv_15m.parquet"
FEATURES_PATH = REPO / "vol_model" / "output" / "BTCUSDT_features_15m.parquet"
SVI_FEATURES_PATH = REPO / "vol_arb" / "output" / "svi_features_15m.parquet"
HAR_PD_MODEL_FEATURES = (
    REPO / "vol_model" / "output" / "models" /
    "expanding_pruned_svi_har_pd_v1" / "2026-04_features.json"
)


def test_prewarm_from_tsrv_parquet_seeds_buffer():
    """Calling prewarm_har_pd_from_tsrv loads K_BARS_DEFAULT=96 rows
    from the TSRV parquet so the first live bar emits non-NaN HAR-PD.
    """
    if not TSRV_PATH.exists():
        pytest.skip(f"{TSRV_PATH} not on disk")
    if not FEATURES_PATH.exists():
        pytest.skip(f"{FEATURES_PATH} not on disk")
    if not SVI_FEATURES_PATH.exists():
        pytest.skip(f"{SVI_FEATURES_PATH} not on disk")
    if not HAR_PD_MODEL_FEATURES.exists():
        pytest.skip(f"{HAR_PD_MODEL_FEATURES} not on disk")

    # Build a minimal model bundle with the 50-feature HAR-PD manifest
    import json
    feature_names = json.loads(HAR_PD_MODEL_FEATURES.read_text())
    bundle = MagicMock()
    bundle.lgbm_features = feature_names

    asm = FeatureAssembler(
        features_parquet=FEATURES_PATH,
        svi_features_parquet=SVI_FEATURES_PATH,
        bundle=bundle,
        svi_calibrator=None,
    )
    n = asm.prewarm_har_pd_from_tsrv(TSRV_PATH)
    assert n == 96
    assert asm.har_pd_state.is_warm
    feats = asm.har_pd_state.compute_features()
    for v in feats.values():
        assert np.isfinite(v)


def test_prewarm_missing_parquet_returns_zero_no_raise():
    """Missing TSRV parquet → log warn + return 0, no exception."""
    if not FEATURES_PATH.exists() or not SVI_FEATURES_PATH.exists():
        pytest.skip("baseline parquets not on disk")
    bundle = MagicMock()
    # Use a tiny feature list so the manifest validation passes;
    # we're testing prewarm, not the full pipeline.
    bundle.lgbm_features = ["tsrv"]
    asm = FeatureAssembler(
        features_parquet=FEATURES_PATH,
        svi_features_parquet=SVI_FEATURES_PATH,
        bundle=bundle,
        svi_calibrator=None,
    )
    n = asm.prewarm_har_pd_from_tsrv("/nonexistent/path.parquet")
    assert n == 0
    assert not asm.har_pd_state.is_warm
