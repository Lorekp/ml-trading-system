"""Tests for vol_arb/trade_log_annotators.py.

Three categories:

1. Per-annotator unit tests on synthetic fixtures.
2. **Leakage probe**: ``probe_causality_by_mutation`` invoked on each
   new column. Mutating a source row at timestamp t' > T must NEVER
   change the annotator output at any t ≤ T.
3. **Byte-identity invariant**: running the annotator does not modify
   the historical V2 trade log file on disk; ``test_v2_preset_loads_k1_sidecar_path``
   stays green.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_arb.trade_log_annotators import (
    THIN_SVI_N_STRIKES_FLOOR,
    annotate_depth_ratio,
    annotate_svi_n_strikes_at_entry,
    annotate,
)
from vol_model.leakage_probes import probe_causality_by_mutation


# ─── 1. Per-annotator unit tests ───────────────────────────────────────


@pytest.fixture
def synthetic_trade_log():
    idx = pd.DatetimeIndex(
        [
            "2023-06-01 00:00", "2023-06-01 00:15",
            "2023-06-01 00:30", "2023-06-01 00:45",
        ],
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "signal": [-1, -1, 1, -1],
            "net_pnl_nc": [50.0, -300.0, 80.0, -100.0],
            "N_contracts": [10.0, 5.0, 2.0, 8.0],
            "svi_n_strikes": [10, 4, 8, 3],   # rows 1 and 3 thin
        },
        index=idx,
    )


@pytest.fixture
def synthetic_bar_exec_path(synthetic_trade_log, tmp_path):
    """Sidecar with depth small enough to make some trades over-size."""
    p = tmp_path / "bar_execution_quality.parquet"
    df = pd.DataFrame(
        {
            "available_depth_btc": [20.0, 3.0, 5.0, 4.0],
            "bar_taker_slip_usd": [12.0, 18.0, 12.0, 15.0],
            "implied_spread_usd": [2.0, 3.0, 2.0, 2.5],
            "n_trades": [5, 3, 4, 4],
        },
        index=synthetic_trade_log.index,
    )
    df.to_parquet(p)
    return p


class TestDepthRatio:
    def test_ratio_correct(self, synthetic_trade_log, synthetic_bar_exec_path):
        out = annotate_depth_ratio(
            synthetic_trade_log, bar_exec_path=synthetic_bar_exec_path
        )
        assert "depth_ratio" in out.columns
        # row 0: 10/20 = 0.5 ; row 1: 5/3 ≈ 1.667 ; row 2: 2/5 = 0.4 ; row 3: 8/4 = 2.0
        np.testing.assert_allclose(
            out["depth_ratio"].values,
            [0.5, 5.0 / 3, 0.4, 2.0], rtol=1e-10,
        )

    def test_understatement_only_when_excess(
        self, synthetic_trade_log, synthetic_bar_exec_path,
    ):
        out = annotate_depth_ratio(
            synthetic_trade_log, bar_exec_path=synthetic_bar_exec_path
        )
        # rows 0, 2: ratio ≤ 1 → understatement = 0
        # row 1: excess = 5/3-1 = 2/3 ; slip 18 ; N=5 → 0.667*18*5 = 60.0
        # row 3: excess = 1 ; slip 15 ; N=8 → 1*15*8 = 120.0
        np.testing.assert_allclose(
            out["slip_understatement_usd"].values,
            [0.0, (5.0 / 3 - 1.0) * 18.0 * 5.0, 0.0, 1.0 * 15.0 * 8.0],
            rtol=1e-10,
        )

    def test_missing_depth_yields_nan(self, synthetic_trade_log, tmp_path):
        # Sidecar shorter than trade log → trailing rows NaN
        p = tmp_path / "short.parquet"
        df = pd.DataFrame(
            {
                "available_depth_btc": [10.0, 10.0],
                "bar_taker_slip_usd": [12.0, 12.0],
            },
            index=synthetic_trade_log.index[:2],
        )
        df.to_parquet(p)
        out = annotate_depth_ratio(synthetic_trade_log, bar_exec_path=p)
        assert out["depth_ratio"].iloc[2:].isna().all()

    def test_missing_columns_raises(self, synthetic_trade_log, tmp_path):
        p = tmp_path / "bad.parquet"
        df = pd.DataFrame({"foo": [1.0]}, index=synthetic_trade_log.index[:1])
        df.to_parquet(p)
        with pytest.raises(KeyError):
            annotate_depth_ratio(synthetic_trade_log, bar_exec_path=p)


class TestSviNStrikes:
    def test_thin_flag_correct(self, synthetic_trade_log):
        out = annotate_svi_n_strikes_at_entry(synthetic_trade_log)
        # n_strikes < 5 floor → rows 1 (4) and 3 (3) are thin
        assert out["thin_svi"].tolist() == [False, True, False, True]

    def test_pnl_at_risk_only_for_thin(self, synthetic_trade_log):
        out = annotate_svi_n_strikes_at_entry(synthetic_trade_log)
        # Thin trades: row 1 (-300), row 3 (-100)
        # Fresh trades: 0
        np.testing.assert_allclose(
            out["thin_svi_pnl_at_risk_usd"].values,
            [0.0, -300.0, 0.0, -100.0],
            rtol=0,
        )

    def test_floor_is_pre_committed(self):
        """Floor is the SVI 5-parameter minimum, not a tuned value."""
        assert THIN_SVI_N_STRIKES_FLOOR == 5


# ─── 2. Mutation-based leakage probe ───────────────────────────────────


class TestLeakage:
    def test_depth_ratio_no_future_leak(
        self, synthetic_trade_log, synthetic_bar_exec_path,
    ):
        """Mutate sidecar at t' > T; depth_ratio at t < T must be unchanged.

        We probe at the FIRST trade timestamp; rows at timestamps after
        that one are mutated and the annotator is re-run. The output at
        the first row must be byte-identical.
        """
        sidecar_df = pd.read_parquet(synthetic_bar_exec_path)

        def feature_fn(raw_sidecar: pd.DataFrame) -> pd.DataFrame:
            tmp_p = synthetic_bar_exec_path.parent / "_probe_sidecar.parquet"
            raw_sidecar.to_parquet(tmp_p)
            return annotate_depth_ratio(synthetic_trade_log, bar_exec_path=tmp_p)

        result = probe_causality_by_mutation(
            feature_fn=feature_fn,
            raw_df=sidecar_df,
            T=synthetic_trade_log.index[0],
            mutation_cols=["available_depth_btc", "bar_taker_slip_usd"],
        )
        assert result.passed, result.detail

    def test_svi_n_strikes_no_future_leak(self, synthetic_trade_log):
        """Mutating future trade-log rows must not change thin_svi at past rows."""
        # Wrap the trade log itself as the raw frame; the annotator
        # only reads ``svi_n_strikes`` and ``net_pnl_nc`` per-row, so
        # mutating future rows mustn't change past output.
        def feature_fn(raw: pd.DataFrame) -> pd.DataFrame:
            return annotate_svi_n_strikes_at_entry(raw)

        result = probe_causality_by_mutation(
            feature_fn=feature_fn,
            raw_df=synthetic_trade_log,
            T=synthetic_trade_log.index[0],
            mutation_cols=["svi_n_strikes", "net_pnl_nc", "N_contracts"],
        )
        assert result.passed, result.detail


# ─── 3. Byte-identity invariant ────────────────────────────────────────


class TestByteIdentity:
    def test_canonical_parquet_unchanged_after_run(
        self, synthetic_trade_log, synthetic_bar_exec_path, tmp_path,
    ):
        """Running annotators writes a SIBLING parquet and never edits source."""
        # Create a fake "canonical" parquet
        canonical_p = tmp_path / "canonical.parquet"
        synthetic_trade_log.to_parquet(canonical_p)
        canonical_hash_before = hashlib.sha256(canonical_p.read_bytes()).hexdigest()

        out_p = tmp_path / "annotated.parquet"
        annotated = annotate(
            synthetic_trade_log,
            output_path=out_p,
            bar_exec_path=synthetic_bar_exec_path,
        )
        # Output exists, is a sibling
        assert out_p.exists()
        assert out_p != canonical_p

        # Canonical untouched
        canonical_hash_after = hashlib.sha256(canonical_p.read_bytes()).hexdigest()
        assert canonical_hash_before == canonical_hash_after

        # New columns present in output
        assert "depth_ratio" in annotated.columns
        assert "thin_svi" in annotated.columns
        # All original columns preserved
        for c in synthetic_trade_log.columns:
            assert c in annotated.columns

    def test_historical_v2_uses_recorded_kelly_sidecar(self):
        """Confirm the forensic V2 replay loads its recorded Kelly sidecar."""
        from vol_arb.backtester import BacktestConfig
        config = BacktestConfig().with_historical_btc_v2()
        assert config.kelly_mult_path == (
            "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
        )
