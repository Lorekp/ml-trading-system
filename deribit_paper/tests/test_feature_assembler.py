"""Tests for the W3.1 manifest-driven FeatureAssembler.

Covers:
- Manifest validation: constructor raises for unknown features.
- Manifest-driven build: output is ordered + correctly sourced.
- TransVol embedding slot is picked up correctly.
- Live handler dispatch (tsrv_live overrides).
- ``svi_atm_1d`` live handler uses the calibrator, falls back to
  baseline on warmup, matches offline percent scale.
- Merged-parquet loading (base + SVI).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.signal.feature_assembler import (
    AssemblerContext,
    FeatureAssembler,
    FeatureRow,
    _LIVE_HANDLERS,
)


# ── Helpers ────────────────────────────────────────────────────────


@dataclass
class _StubBundle:
    lgbm_features: list[str]


@dataclass
class _StubBar:
    timestamp: pd.Timestamp
    n_perp_trades: int = 100
    perp_trades: list = None  # noqa: RUF012

    def __post_init__(self):
        if self.perp_trades is None:
            self.perp_trades = []


@dataclass
class _StubTSRV:
    tsrv: float = 0.123
    rv_fast: float = 0.100
    rv_slow: float = 0.110
    bpv: float = 0.115
    medrv: float = 0.118
    continuous_var: float = 0.012
    jump_var: float = 0.003
    jump_fraction: float = 0.25
    rs_down: float = 0.006
    rs_up: float = 0.006
    down_jump_fraction: float = 0.10
    up_jump_fraction: float = 0.15
    vol_ann_continuous: float = 55.0


class _StubSVICalibrator:
    def __init__(self, atm_1d: float = 54.321):
        self._atm_1d = atm_1d
        self.calls: list[float] = []

    def atm_vol_at_dte(self, dte_days: float) -> float:
        self.calls.append(dte_days)
        return self._atm_1d


def _write_stub_parquet(
    path: Path,
    *,
    columns: list[str],
    values: dict[str, float] | None = None,
    n_rows: int = 10,
) -> None:
    idx = pd.date_range("2026-01-01", periods=n_rows, freq="15min", tz="UTC")
    data: dict[str, np.ndarray] = {}
    for col in columns:
        if values and col in values:
            data[col] = np.full(n_rows, values[col], dtype=np.float64)
        else:
            data[col] = np.arange(n_rows, dtype=np.float64)
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    df.to_parquet(path)


@pytest.fixture
def minimal_parquets(tmp_path):
    """Create a tiny base-features parquet + SVI-features parquet."""
    base_path = tmp_path / "base_features.parquet"
    svi_path = tmp_path / "svi_features.parquet"
    _write_stub_parquet(
        base_path,
        columns=[
            "tsrv", "bpv", "medrv", "vol_ann",
            "vpin", "vpin_ema50",
            "hurst_100",
            "svi_atm_1d",   # stale duplicate — SVI parquet should win
        ],
        values={
            "tsrv": 0.05,
            "bpv": 0.05,
            "medrv": 0.05,
            "vol_ann": 50.0,
            "vpin": 0.2,
            "vpin_ema50": 0.2,
            "hurst_100": 0.5,
            "svi_atm_1d": 999.0,  # obviously stale
        },
    )
    _write_stub_parquet(
        svi_path,
        columns=["svi_atm_1d", "svi_atm_1w"],
        values={"svi_atm_1d": 58.0, "svi_atm_1w": 60.0},
    )
    return base_path, svi_path


# ── Manifest validation ────────────────────────────────────────────


def test_validation_raises_on_unknown_feature(minimal_parquets):
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(
        lgbm_features=[
            "tv_0", "tv_1",
            "tsrv", "bpv", "svi_atm_1d",
            "totally_not_a_real_feature",  # should cause init to raise
        ],
    )
    with pytest.raises(ValueError, match="cannot produce"):
        FeatureAssembler(
            features_parquet=base_path,
            svi_features_parquet=svi_path,
            bundle=bundle,
        )


def test_validation_passes_with_pure_transvol_only(minimal_parquets):
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(lgbm_features=[f"tv_{i}" for i in range(16)])
    # Should not raise — TransVol slots are always producible
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
    )
    assert assembler.baseline_timestamp is not None


def test_validation_passes_when_baseline_supplies_fallback(minimal_parquets):
    """Features not in the handler registry must fall back to baseline."""
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(
        lgbm_features=["vpin", "vpin_ema50", "hurst_100"],
    )
    FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
    )  # must not raise


# ── Merged-parquet loading ─────────────────────────────────────────


def test_svi_parquet_overrides_stale_base_svi_atm_1d(minimal_parquets):
    """When both parquets have svi_atm_1d, the SVI parquet wins."""
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(lgbm_features=["svi_atm_1d"])

    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
    )
    assert assembler._baseline_row["svi_atm_1d"] == pytest.approx(58.0)
    # The stale 999.0 from the base parquet must be gone
    assert assembler._baseline_row["svi_atm_1d"] != 999.0


# ── build_row: manifest-driven dispatch ────────────────────────────


def test_build_row_dispatches_live_handlers(minimal_parquets):
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(
        lgbm_features=["tsrv", "bpv", "medrv", "svi_atm_1d", "hurst_100"],
    )
    svi_cal = _StubSVICalibrator(atm_1d=54.321)
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
        svi_calibrator=svi_cal,
    )

    bar = _StubBar(timestamp=pd.Timestamp("2026-04-11 12:00", tz="UTC"))
    tsrv = _StubTSRV()
    row = assembler.build_row(bar, tsrv, transvol_embedding=None)

    # Live-handler-driven fields come from the stub TSRV, NOT the baseline
    assert row.as_dict["tsrv"] == pytest.approx(tsrv.tsrv)
    assert row.as_dict["bpv"] == pytest.approx(tsrv.bpv)
    assert row.as_dict["medrv"] == pytest.approx(tsrv.medrv)
    # svi_atm_1d comes from the live calibrator, not the baseline (58.0)
    assert row.as_dict["svi_atm_1d"] == pytest.approx(54.321)
    assert svi_cal.calls == [1.0]
    # hurst_100 has no live handler → baseline fallback
    assert row.as_dict["hurst_100"] == pytest.approx(0.5)


def test_build_row_applies_transvol_embedding(minimal_parquets):
    base_path, svi_path = minimal_parquets
    feature_names = [f"tv_{i}" for i in range(16)] + ["tsrv", "svi_atm_1d"]
    bundle = _StubBundle(lgbm_features=feature_names)
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
        svi_calibrator=_StubSVICalibrator(),
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-04-11 12:00", tz="UTC"))
    tsrv = _StubTSRV()
    embedding = np.arange(16, dtype=np.float64) * 0.1
    row = assembler.build_row(bar, tsrv, transvol_embedding=embedding)

    assert row.values.shape == (len(feature_names),)
    # TransVol slots must carry the embedding, in slot order
    for i in range(16):
        assert row.values[i] == pytest.approx(embedding[i])


def test_build_row_falls_back_when_svi_calibrator_missing(minimal_parquets):
    """Without a calibrator, ``svi_atm_1d`` must fall back to baseline."""
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(lgbm_features=["svi_atm_1d"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
        svi_calibrator=None,
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-04-11 12:00", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV(), transvol_embedding=None)
    # Baseline svi_atm_1d = 58.0 (from the SVI parquet)
    assert row.as_dict["svi_atm_1d"] == pytest.approx(58.0)


def test_build_row_falls_back_when_calibrator_returns_nan(minimal_parquets):
    """Live handler NaN → baseline fallback, not a zero."""
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(lgbm_features=["svi_atm_1d"])

    class _NaNCalibrator:
        def atm_vol_at_dte(self, dte_days: float) -> float:
            return float("nan")

    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
        svi_calibrator=_NaNCalibrator(),
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-04-11 12:00", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV(), transvol_embedding=None)
    # Falls back to the baseline value from the parquet
    assert row.as_dict["svi_atm_1d"] == pytest.approx(58.0)


def test_build_row_returns_none_on_nan_non_tv_feature(minimal_parquets):
    """R1: Non-TransVol NaN/inf features → build_row returns None (fail-fast)."""
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(lgbm_features=["vpin"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
    )
    # Inject an inf into the baseline row to simulate bad historical data
    assembler._baseline_row["vpin"] = float("inf")
    bar = _StubBar(timestamp=pd.Timestamp("2026-04-11 12:00", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV(), transvol_embedding=None)
    assert row is None  # R1: refuses to produce a broken feature row


def test_build_row_allows_tv_nan_with_zero_fill(minimal_parquets):
    """R1: TransVol NaN features are zero-filled (expected fallback path)."""
    base_path, svi_path = minimal_parquets
    bundle = _StubBundle(lgbm_features=["tv_0"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=svi_path,
        bundle=bundle,
    )
    # Inject NaN into a TransVol baseline slot
    assembler._baseline_row["tv_0"] = float("nan")
    bar = _StubBar(timestamp=pd.Timestamp("2026-04-11 12:00", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV(), transvol_embedding=None)
    assert row is not None
    assert np.isfinite(row.values).all()
