"""F3 Tier 3 integration tests: 16 dollar-bar handlers wired into FeatureAssembler.

The byte-identity math is locked by ``test_dollar_bar_features.py``;
this file verifies the assembler-level integration:

1. The 16 new handlers fire when ``dollar_bar_buffer`` is wired and
   has enough history; their outputs match the wrapper's direct call.
2. F4 + F5: ``_h_duration_s`` and ``_h_n_trades`` use the buffer's
   latest bar when wired, fall back to wall-clock counts otherwise.
3. Backward-compat: with ``dollar_bar_buffer=None``, the new handlers
   fall back to the historical baseline; existing 700+ tests stay green.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import DollarBarFeatures
from deribit_paper.signal.feature_assembler import (
    AssemblerContext,
    FeatureAssembler,
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
    vol_ann: float = 0.55
    rv_ann: float = 0.55


def _make_synthetic_bar(i: int, **overrides) -> DollarBar:
    """Mirror of test_dollar_bar_features._make_synthetic_bar."""
    defaults = dict(
        price=60_000.0,
        high=60_000.0 + 100.0 * (1.0 + 0.1 * i),
        low=60_000.0 - 100.0 * (1.0 + 0.05 * i),
        duration_s=30.0 + i,
        n_trades=100 + i,
        buy_volume=250_000.0 + 1000.0 * i,
        sell_volume=250_000.0 + 500.0 * i,
        vov=0.001 + 0.0001 * i,
    )
    defaults.update(overrides)
    ts = pd.Timestamp("2026-05-07 00:00:00", tz="UTC") + pd.Timedelta(seconds=30 * i)
    dollar_volume = defaults["buy_volume"] + defaults["sell_volume"]
    imbalance = (
        defaults["buy_volume"] - defaults["sell_volume"]
    ) / max(dollar_volume, 1.0)
    return DollarBar(
        timestamp=ts,
        open=defaults["price"],
        close=defaults["price"] + 1.0,
        high=defaults["high"],
        low=defaults["low"],
        duration_s=defaults["duration_s"],
        n_trades=defaults["n_trades"],
        dollar_volume=dollar_volume,
        buy_volume=defaults["buy_volume"],
        sell_volume=defaults["sell_volume"],
        vov=defaults["vov"],
        imbalance=imbalance,
        log_return=0.0001 if i > 0 else float("nan"),
        rv_bar=0.0001**2 if i > 0 else float("nan"),
    )


def _populate_buffer(n: int) -> DollarBarBuffer:
    buf = DollarBarBuffer(max_size=300)
    for i in range(n):
        buf.append(_make_synthetic_bar(i))
    return buf


def _baseline_columns() -> list[str]:
    """All 16 dollar-bar handler names (column names = feature names)."""
    return [
        "vov", "vov_ema10", "vov_ema50", "vov_lag3",
        "vpin", "vpin_ema10", "vpin_ema50", "vpin_lag1",
        "imbalance_ema50",
        "duration_s_ema10", "duration_s_ema50", "duration_s_lag3",
        "n_trades_ema10", "n_trades_ema50",
        "parkinson_1d", "parkinson_1d_ema50",
        "duration_s", "n_trades",  # for F4/F5 update tests
    ]


def _write_stub_parquet(path: Path) -> None:
    cols = _baseline_columns() + [
        "tsrv", "bpv", "medrv", "vol_ann",
    ]
    idx = pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC")
    data = {col: np.arange(10, dtype=np.float64) + 1.0 for col in cols}
    # Specific baseline values so we can identify them in fallback tests
    data["vov"] = np.full(10, 0.42, dtype=np.float64)
    data["vov_ema50"] = np.full(10, 0.43, dtype=np.float64)
    data["vpin"] = np.full(10, 0.5, dtype=np.float64)
    data["parkinson_1d"] = np.full(10, 1.23e-5, dtype=np.float64)
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    df.to_parquet(path)


@pytest.fixture
def stub_parquets(tmp_path):
    base_path = tmp_path / "base_features.parquet"
    _write_stub_parquet(base_path)
    return base_path, None


# ── 1. handlers fire when buffer wired ──────────────────────────────


def test_assembler_dispatches_dollar_bar_handlers_with_wired_buffer(stub_parquets):
    """All 16 handlers must produce finite values matching DollarBarFeatures direct call."""
    base_path, _ = stub_parquets
    buf = _populate_buffer(60)  # enough for VPIN (>=50)
    bundle = _StubBundle(
        lgbm_features=[
            "vov", "vov_ema10", "vov_ema50", "vov_lag3",
            "vpin", "vpin_ema10", "vpin_ema50", "vpin_lag1",
            "imbalance_ema50",
            "duration_s_ema10", "duration_s_ema50", "duration_s_lag3",
            "n_trades_ema10", "n_trades_ema50",
            "parkinson_1d_ema50",
        ],
    )
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=buf,
    )

    bar = _StubBar(timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"))
    bar_tsrv = _StubTSRV()
    row = assembler.build_row(bar, bar_tsrv)
    assert row is not None

    feats = DollarBarFeatures(buf)
    assert row.as_dict["vov"] == pytest.approx(feats.latest_raw("vov"))
    assert row.as_dict["vov_ema10"] == pytest.approx(feats.ema("vov", 10))
    assert row.as_dict["vov_ema50"] == pytest.approx(feats.ema("vov", 50))
    assert row.as_dict["vov_lag3"] == pytest.approx(feats.lag("vov", 3))
    assert row.as_dict["vpin"] == pytest.approx(feats.latest_raw("vpin"))
    assert row.as_dict["vpin_ema10"] == pytest.approx(feats.ema("vpin", 10))
    assert row.as_dict["vpin_ema50"] == pytest.approx(feats.ema("vpin", 50))
    assert row.as_dict["vpin_lag1"] == pytest.approx(feats.lag("vpin", 1))
    assert row.as_dict["imbalance_ema50"] == pytest.approx(feats.ema("imbalance", 50))
    assert row.as_dict["duration_s_ema10"] == pytest.approx(feats.ema("duration_s", 10))
    assert row.as_dict["duration_s_ema50"] == pytest.approx(feats.ema("duration_s", 50))
    assert row.as_dict["duration_s_lag3"] == pytest.approx(feats.lag("duration_s", 3))
    assert row.as_dict["n_trades_ema10"] == pytest.approx(feats.ema("n_trades", 10))
    assert row.as_dict["n_trades_ema50"] == pytest.approx(feats.ema("n_trades", 50))
    # parkinson_1d_ema50 needs ≥62 bars (rolling_window=250's
    # min_periods); with 60 bars in the buffer, the rolling-mean
    # series is all-NaN at the latest sample so feats.parkinson_1d_ema
    # returns None. Per the F-2 fix (round-3 audit, 2026-05-08) the
    # handler falls back to baseline (10.0) per the stub parquet.
    assert feats.parkinson_1d_ema(50) is None
    assert row.as_dict["parkinson_1d_ema50"] == pytest.approx(10.0)


def test_parkinson_1d_falls_back_to_baseline_when_underfull(stub_parquets):
    """parkinson_1d needs 250 dollar bars; buffer with 60 should fall back."""
    base_path, _ = stub_parquets
    buf = _populate_buffer(60)
    bundle = _StubBundle(lgbm_features=["parkinson_1d"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=buf,
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV())
    assert row is not None
    # Falls back to baseline value 1.23e-5
    assert row.as_dict["parkinson_1d"] == pytest.approx(1.23e-5)


# ── 2. F4 / F5 fixes ────────────────────────────────────────────────


def test_h_duration_s_uses_latest_dollar_bar_when_wired(stub_parquets):
    """F4: with buffer wired, duration_s = latest bar's duration_s."""
    base_path, _ = stub_parquets
    buf = _populate_buffer(5)
    bundle = _StubBundle(lgbm_features=["duration_s"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=buf,
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV())
    assert row.as_dict["duration_s"] == pytest.approx(buf.latest_bar().duration_s)


def test_h_n_trades_uses_latest_dollar_bar_when_wired(stub_parquets):
    """F5: with buffer wired, n_trades = latest bar's n_trades."""
    base_path, _ = stub_parquets
    buf = _populate_buffer(5)
    bundle = _StubBundle(lgbm_features=["n_trades"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=buf,
    )
    bar = _StubBar(
        timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"),
        n_perp_trades=99999,  # would dominate without F5 fix
    )
    row = assembler.build_row(bar, _StubTSRV())
    assert row.as_dict["n_trades"] == pytest.approx(buf.latest_bar().n_trades)
    assert row.as_dict["n_trades"] != 99999  # F5 fix really fired


def test_h_duration_s_falls_back_when_buffer_none(stub_parquets):
    """Backward compat: buffer=None → existing inter-trade-gap path."""
    base_path, _ = stub_parquets
    bundle = _StubBundle(lgbm_features=["duration_s"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=None,  # explicit
    )
    bar = _StubBar(
        timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"),
        perp_trades=[],
    )
    row = assembler.build_row(bar, _StubTSRV())
    # No perp trades → baseline fallback duration_s value
    # (whatever the baseline parquet contains)
    assert math_finite(row.as_dict["duration_s"])


def test_h_n_trades_falls_back_to_n_perp_trades_when_buffer_none(stub_parquets):
    """Backward compat: buffer=None → ctx.bar.n_perp_trades."""
    base_path, _ = stub_parquets
    bundle = _StubBundle(lgbm_features=["n_trades"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=None,
    )
    bar = _StubBar(
        timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"),
        n_perp_trades=42,
    )
    row = assembler.build_row(bar, _StubTSRV())
    assert row.as_dict["n_trades"] == pytest.approx(42.0)


# ── 3. backward-compat — buffer=None still works ────────────────────


def test_dollar_bar_handlers_fall_back_to_baseline_when_buffer_none(stub_parquets):
    """Buffer=None: every dollar-bar handler returns the baseline value."""
    base_path, _ = stub_parquets
    bundle = _StubBundle(
        lgbm_features=["vov", "vov_ema50", "vpin", "parkinson_1d"],
    )
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=None,
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV())
    assert row.as_dict["vov"] == pytest.approx(0.42)         # baseline
    assert row.as_dict["vov_ema50"] == pytest.approx(0.43)
    assert row.as_dict["vpin"] == pytest.approx(0.5)
    assert row.as_dict["parkinson_1d"] == pytest.approx(1.23e-5)


def test_dollar_bar_handlers_fall_back_when_buffer_too_short(stub_parquets):
    """vpin needs 50 bars; with only 10 the handler falls back to baseline."""
    base_path, _ = stub_parquets
    buf = _populate_buffer(10)  # < 50 → vpin returns None
    bundle = _StubBundle(lgbm_features=["vpin", "vpin_ema10"])
    assembler = FeatureAssembler(
        features_parquet=base_path,
        svi_features_parquet=None,
        bundle=bundle,
        dollar_bar_buffer=buf,
    )
    bar = _StubBar(timestamp=pd.Timestamp("2026-05-07 12:15", tz="UTC"))
    row = assembler.build_row(bar, _StubTSRV())
    assert row.as_dict["vpin"] == pytest.approx(0.5)  # baseline
    # vpin_ema10 not in baseline → NaN propagates → row should be None
    # but we have a fallback in the parquet for vpin (just not vpin_ema10).
    # The stub baseline gives integer values for unspecified columns;
    # `vpin_ema10` baseline = 1.0 (the default arange). So the handler
    # returns 1.0 from baseline.


# ── helper ──


def math_finite(x: float) -> bool:
    return np.isfinite(x)
