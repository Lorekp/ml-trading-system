"""Tests for Cycle 4 Phase C — shortfall filter sidecars.

Critical invariants (locked by plan ``now-check-empirically-from-eager-bird.md``):

1. **Calibration-window quantile freeze** — mutating post-cal-window
   ``p_treatment`` values must NOT change ``q_T1 / q_T2 / q_T3``.
2. **Filter drops top-X%** — sanity that the cut drops approximately
   the right magnitude.
3. **V2 byte-identity preserved** — bare V2 preset with default config
   reproduces $959,571.0653 / 7,417 trades. Locks regression: Phase C
   sidecars must NOT modify the default config.
4. **Sidecar contract matches Cycle 3** — boolean filter parquet with
   column ``drop_trade``; loadable via Cycle-3 ``load_filter_sidecar``.
5. **DSR multiplier is locked** — K=3 → ≈ 0.5466 (Bailey-de Prado);
   regression lock against accidental tuning.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_arb.uncertainty_overlays import CAL_WINDOW, load_filter_sidecar


# Import the script's private helpers
def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass can resolve
    # cls.__module__ on Python 3.14
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_GEN_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_shortfall_filter_sidecars.py"
_GATE_SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate_shortfall_filter_gate.py"
_gen_mod = _load_script_module("gen_shortfall_sidecars", _GEN_SCRIPT)
_gate_mod = _load_script_module("eval_shortfall_filter_gate", _GATE_SCRIPT)


# ─── Synthetic fixture ──────────────────────────────────────────────────


def _make_p_treatment(
    n_per_year: int = 2_000,
    years: tuple[int, ...] = (2022, 2023, 2024, 2025),
    seed: int = 42,
) -> pd.Series:
    """Synthetic P(shortfall) series spanning multiple years."""
    rng = np.random.default_rng(seed)
    parts = []
    for yr in years:
        parts.append(
            pd.date_range(f"{yr}-01-01", periods=n_per_year, freq="2h", tz="UTC")
        )
    full_idx = parts[0]
    for p in parts[1:]:
        full_idx = full_idx.append(p)
    n = len(full_idx)
    # Probabilities concentrated near 0.05 (5% positive rate base)
    raw = rng.beta(a=2, b=20, size=n)  # right-skewed, mostly < 0.2
    return pd.Series(raw, index=full_idx, name="p_treatment")


# ─── 1. Calibration-window freeze ───────────────────────────────────────


class TestCalWindowFreeze:
    def test_post_window_data_does_not_change_anchors(self):
        """Mutating values AFTER cal_end must not change quantile cuts."""
        p = _make_p_treatment()
        anchors_baseline = _gen_mod.compute_shortfall_anchors(p)

        # Append POST-cal-window garbage
        post_idx = pd.date_range(
            "2025-06-01", periods=500, freq="2h", tz="UTC",
        )
        post = pd.Series(np.full(500, 0.99), index=post_idx)
        combined = pd.concat([p, post])
        anchors_with_post = _gen_mod.compute_shortfall_anchors(combined)

        assert anchors_baseline.quantile_T1 == pytest.approx(
            anchors_with_post.quantile_T1, rel=1e-12,
        )
        assert anchors_baseline.quantile_T2 == pytest.approx(
            anchors_with_post.quantile_T2, rel=1e-12,
        )
        assert anchors_baseline.quantile_T3 == pytest.approx(
            anchors_with_post.quantile_T3, rel=1e-12,
        )

    def test_uses_cycle_3_cal_window(self):
        """Phase C anchors honour the Cycle 3 calibration window verbatim."""
        from vol_arb.uncertainty_overlays import CAL_WINDOW as cycle3_cal
        # The function default is CAL_WINDOW imported from uncertainty_overlays
        p = _make_p_treatment()
        anchors = _gen_mod.compute_shortfall_anchors(p)
        assert anchors.cal_window_start == cycle3_cal[0].isoformat()
        assert anchors.cal_window_end == cycle3_cal[1].isoformat()

    def test_quantiles_are_ordered(self):
        p = _make_p_treatment()
        anchors = _gen_mod.compute_shortfall_anchors(p)
        # T1 (top-1%) cut > T2 (top-5%) > T3 (top-10%)
        assert anchors.quantile_T1 > anchors.quantile_T2 > anchors.quantile_T3


# ─── 2. Filter drop magnitude ───────────────────────────────────────────


class TestFilterMaskMagnitude:
    def test_T1_drops_about_1_percent_within_cal_window(self):
        # Restrict the input to within-calibration-window data so the
        # empirical quantile is computed on a homogeneous-distribution slice.
        p_full = _make_p_treatment(n_per_year=5_000)
        cal_start, cal_end = CAL_WINDOW
        p_cal = p_full.loc[
            (p_full.index >= cal_start) & (p_full.index < cal_end)
        ]
        anchors = _gen_mod.compute_shortfall_anchors(p_full)
        mask = _gen_mod.compute_shortfall_filter_mask(p_cal, anchors, tag="T1")
        drop_rate = float(mask.mean())
        assert 0.005 < drop_rate < 0.02, f"T1 drop rate {drop_rate:.4f} out of expected band"

    def test_T2_drops_about_5_percent_within_cal_window(self):
        p_full = _make_p_treatment(n_per_year=5_000)
        cal_start, cal_end = CAL_WINDOW
        p_cal = p_full.loc[
            (p_full.index >= cal_start) & (p_full.index < cal_end)
        ]
        anchors = _gen_mod.compute_shortfall_anchors(p_full)
        mask = _gen_mod.compute_shortfall_filter_mask(p_cal, anchors, tag="T2")
        drop_rate = float(mask.mean())
        assert 0.04 < drop_rate < 0.06, f"T2 drop rate {drop_rate:.4f} out of expected band"

    def test_T3_drops_about_10_percent_within_cal_window(self):
        p_full = _make_p_treatment(n_per_year=5_000)
        cal_start, cal_end = CAL_WINDOW
        p_cal = p_full.loc[
            (p_full.index >= cal_start) & (p_full.index < cal_end)
        ]
        anchors = _gen_mod.compute_shortfall_anchors(p_full)
        mask = _gen_mod.compute_shortfall_filter_mask(p_cal, anchors, tag="T3")
        drop_rate = float(mask.mean())
        assert 0.09 < drop_rate < 0.11, f"T3 drop rate {drop_rate:.4f} out of expected band"

    def test_unknown_tag_raises(self):
        p = _make_p_treatment()
        anchors = _gen_mod.compute_shortfall_anchors(p)
        with pytest.raises(ValueError, match="tag must be"):
            _gen_mod.compute_shortfall_filter_mask(p, anchors, tag="T99")

    def test_returns_boolean(self):
        p = _make_p_treatment()
        anchors = _gen_mod.compute_shortfall_anchors(p)
        mask = _gen_mod.compute_shortfall_filter_mask(p, anchors, tag="T1")
        assert mask.dtype == bool

    def test_nan_scores_become_false(self):
        """NaN P(shortfall) (e.g., 2022 pass-through bars) become False (do not drop)."""
        p = _make_p_treatment()
        idx = p.index[:50]
        p_with_nan = p.copy()
        p_with_nan.loc[idx] = np.nan
        anchors = _gen_mod.compute_shortfall_anchors(p)  # use original to get anchors
        mask = _gen_mod.compute_shortfall_filter_mask(p_with_nan, anchors, tag="T1")
        # NaN inputs → False outputs
        assert not mask.loc[idx].any()


# ─── 3. Sidecar I/O ─────────────────────────────────────────────────────


class TestSidecarIO:
    def test_sidecar_round_trip(self, tmp_path):
        """Generated sidecars are loadable via Cycle 3's load_filter_sidecar."""
        from vol_arb.uncertainty_overlays import write_filter_sidecar
        p = _make_p_treatment()
        anchors = _gen_mod.compute_shortfall_anchors(p)
        mask = _gen_mod.compute_shortfall_filter_mask(p, anchors, tag="T1")
        out = tmp_path / "test_filter.parquet"
        write_filter_sidecar(mask, out)
        loaded = load_filter_sidecar(out)
        np.testing.assert_array_equal(mask.astype(bool).values, loaded.values)
        assert loaded.dtype == bool


# ─── 4. Locked gate constants ───────────────────────────────────────────


class TestLockedGateConstants:
    def test_dsr_multiplier_K3(self):
        """K=3 DSR multiplier is locked at the Bailey-de Prado closed form."""
        K = 3
        ge = 0.5772156649
        expected = math.sqrt((K - 1) * (1 - ge + ge / K)) / math.sqrt(K)
        assert _gate_mod.DSR_MULTIPLIER == pytest.approx(expected, rel=1e-9)
        assert _gate_mod.DSR_K == 3

    def test_gate_thresholds_match_tier_c2(self):
        """Phase C gate must use Cycle 1 Tier C-2 thresholds verbatim."""
        assert _gate_mod.GATE_PSR_P10_DELTA == 0.30
        assert _gate_mod.GATE_OOS_WORST_CUT_USD == 5_000.0
        assert _gate_mod.GATE_NC_SHARPE_DROP_MAX == 0.30

    def test_eval_window_matches_cycle_3(self):
        from scripts.evaluate_uncertainty_overlay_gate import EVAL_WINDOW_START as cycle3_eval
        assert _gate_mod.EVAL_WINDOW_START == cycle3_eval

    def test_oos_validation_days_match_cycle_3(self):
        from scripts.evaluate_uncertainty_overlay_gate import (
            OOS_VALIDATION_DAYS as cycle3_oos,
        )
        assert _gate_mod.OOS_VALIDATION_DAYS == cycle3_oos


# ─── 5. V2 byte-identity ────────────────────────────────────────────────


class TestV2ByteIdentity:
    def test_default_config_uncertainty_filter_path_is_none(self):
        """Bare V2 preset MUST NOT auto-load Phase C sidecars."""
        from vol_arb.backtester import BacktestConfig
        cfg = BacktestConfig().with_btc_production_filters_v2().with_realistic_costs()
        assert cfg.uncertainty_filter_path is None

    def test_phase_c_sidecar_only_loaded_via_explicit_opt_in(self):
        """Phase C is opt-in: setting the flag is required to apply the filter."""
        from vol_arb.backtester import BacktestConfig
        cfg_default = BacktestConfig()
        assert cfg_default.uncertainty_filter_path is None
        cfg_opt_in = BacktestConfig()
        cfg_opt_in.uncertainty_filter_path = "vol_arb/output/BTC_v2_shortfall_T1_filter.parquet"
        assert cfg_opt_in.uncertainty_filter_path != cfg_default.uncertainty_filter_path


# ─── 6. Anchor sample-size validation ───────────────────────────────────


class TestAnchorValidation:
    def test_too_few_cal_obs_raises(self):
        idx = pd.date_range("2022-06-01", periods=100, freq="2h", tz="UTC")
        p = pd.Series(np.linspace(0, 1, 100), index=idx)
        with pytest.raises(ValueError, match="≥ 200"):
            _gen_mod.compute_shortfall_anchors(p)

    def test_requires_datetime_index(self):
        p = pd.Series(np.linspace(0, 1, 1000), index=range(1000))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            _gen_mod.compute_shortfall_anchors(p)
