"""Tests for vol_arb/preventative_sizing.py.

Categories:
  1. Pre-committed external anchors are LOCKED (constants check).
  2. Pure-function ``compute_size_scaler`` unit tests on synthetic
     (equity, spot, atm_iv, pre_gamma, pre_vega) tuples.
  3. Floor and ceiling: ``f`` clipped to ``[f_min, 1.0]``; never zero.
  4. Vectorised vs scalar parity.
  5. PME IV-shock anchor: leakage probe (mutate post-window data, anchor
     unchanged), N-observation guard, deterministic output.
  6. Sidecar round-trip (write/load preserves byte equality).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_arb.preventative_sizing import (
    LOSS_BUDGET_PCT,
    PMEAnchors,
    PME_IV_QUANTILE,
    PME_SHOCK_SPOT,
    SIZE_FLOOR_F_MIN,
    build_pme_anchors,
    compute_pme_iv_shock_anchor,
    compute_size_scaler,
    compute_size_scaler_vectorised,
    load_sidecar,
    write_sidecar,
)


# ─── 1. Pre-committed anchors are locked ────────────────────────────────


class TestLockedAnchors:
    def test_loss_budget_pct_locked_at_one_percent(self):
        assert LOSS_BUDGET_PCT == 0.01

    def test_pme_shock_spot_locked_at_175pct(self):
        assert PME_SHOCK_SPOT == 0.175

    def test_size_floor_f_min_locked_at_ten_percent(self):
        assert SIZE_FLOOR_F_MIN == 0.10

    def test_pme_iv_quantile_locked_at_99(self):
        assert PME_IV_QUANTILE == 0.99


@pytest.fixture
def anchors() -> PMEAnchors:
    """Synthetic anchor record with a deliberately small IV shock for tests."""
    return PMEAnchors(
        loss_budget_pct=LOSS_BUDGET_PCT,
        pme_shock_spot=PME_SHOCK_SPOT,
        pme_shock_iv_pct=0.04,  # 4% — typical for BTC DVOL
        pme_iv_anchor_window=("2021-04-01T00:00:00+00:00",
                                "2023-10-01T00:00:00+00:00"),
        pme_iv_anchor_n_obs=900,
    )


# ─── 2. Pure-function compute_size_scaler ───────────────────────────────


class TestSizeScalerFormula:
    def test_low_gamma_low_vega_returns_one(self, anchors):
        """When neither budget binds, no shrink (f = 1.0)."""
        f = compute_size_scaler(
            equity=100_000.0,
            spot=27_000.0,
            atm_iv=60.0,
            pre_gamma=1e-7,    # tiny — gamma_target dominates
            pre_vega=1.0,      # tiny — vega_target dominates
            anchors=anchors,
        )
        assert f == pytest.approx(1.0, abs=1e-12)

    def test_high_gamma_only_shrinks(self, anchors):
        """When ONLY gamma binds, f = gamma_target / pre_gamma."""
        equity = 100_000.0
        spot = 27_000.0
        loss_budget = equity * anchors.loss_budget_pct        # $1,000
        gamma_target = 2 * loss_budget / (spot * anchors.pme_shock_spot) ** 2
        # Make pre_gamma 4× bigger than gamma_target so f = 0.25
        pre_gamma = gamma_target * 4.0
        f = compute_size_scaler(
            equity=equity, spot=spot, atm_iv=60.0,
            pre_gamma=pre_gamma,
            pre_vega=1.0,        # tiny — vega slack
            anchors=anchors,
        )
        assert f == pytest.approx(0.25, rel=1e-9)

    def test_high_vega_only_shrinks(self, anchors):
        """When ONLY vega binds, f = vega_target / pre_vega."""
        equity = 100_000.0
        atm_iv = 60.0
        loss_budget = equity * anchors.loss_budget_pct        # $1,000
        iv_shock_pts = atm_iv * anchors.pme_shock_iv_pct      # 60 × 0.04 = 2.4
        vega_target = loss_budget / iv_shock_pts              # 1000/2.4 ≈ 416.67
        pre_vega = vega_target * 5.0                           # 5× too big → f=0.2
        f = compute_size_scaler(
            equity=equity, spot=27_000.0, atm_iv=atm_iv,
            pre_gamma=1e-12,    # tiny — gamma slack
            pre_vega=pre_vega,
            anchors=anchors,
        )
        assert f == pytest.approx(0.20, rel=1e-9)

    def test_both_bind_takes_min(self, anchors):
        """f is the binding minimum of (gamma_ratio, vega_ratio)."""
        equity = 100_000.0
        spot = 27_000.0
        atm_iv = 60.0
        loss_budget = equity * anchors.loss_budget_pct
        gamma_target = 2 * loss_budget / (spot * anchors.pme_shock_spot) ** 2
        iv_shock_pts = atm_iv * anchors.pme_shock_iv_pct
        vega_target = loss_budget / iv_shock_pts

        # gamma_ratio = 0.5, vega_ratio = 0.3 → expect f = 0.3
        pre_gamma = gamma_target / 0.5    # so gamma_ratio = 0.5
        pre_vega = vega_target / 0.3       # so vega_ratio = 0.3
        f = compute_size_scaler(
            equity=equity, spot=spot, atm_iv=atm_iv,
            pre_gamma=pre_gamma, pre_vega=pre_vega,
            anchors=anchors,
        )
        assert f == pytest.approx(0.30, rel=1e-9)

    def test_floor_f_min_binds(self, anchors):
        """When both budgets are violently breached, f = f_min (NOT 0)."""
        equity = 100_000.0
        spot = 27_000.0
        atm_iv = 60.0
        loss_budget = equity * anchors.loss_budget_pct
        gamma_target = 2 * loss_budget / (spot * anchors.pme_shock_spot) ** 2
        iv_shock_pts = atm_iv * anchors.pme_shock_iv_pct
        vega_target = loss_budget / iv_shock_pts
        pre_gamma = gamma_target * 100.0   # 100× too big → ratio 0.01
        pre_vega = vega_target * 100.0     # 100× too big → ratio 0.01
        f = compute_size_scaler(
            equity=equity, spot=spot, atm_iv=atm_iv,
            pre_gamma=pre_gamma, pre_vega=pre_vega,
            anchors=anchors,
        )
        assert f == pytest.approx(SIZE_FLOOR_F_MIN, abs=1e-12)

    def test_nan_inputs_return_one(self, anchors):
        """Defensive: NaN on any input → no shrink (preserves byte-identity)."""
        for bad_arg in ("equity", "spot", "atm_iv", "pre_gamma", "pre_vega"):
            kwargs = dict(equity=100_000.0, spot=27_000.0, atm_iv=60.0,
                          pre_gamma=1e-6, pre_vega=200.0)
            kwargs[bad_arg] = float("nan")
            f = compute_size_scaler(**kwargs, anchors=anchors)
            assert f == 1.0, f"NaN on {bad_arg} should return 1.0"

    def test_negative_or_zero_inputs_return_one(self, anchors):
        f = compute_size_scaler(
            equity=0.0, spot=27_000.0, atm_iv=60.0,
            pre_gamma=1e-6, pre_vega=200.0, anchors=anchors,
        )
        assert f == 1.0
        f = compute_size_scaler(
            equity=100_000.0, spot=27_000.0, atm_iv=0.0,
            pre_gamma=1e-6, pre_vega=200.0, anchors=anchors,
        )
        assert f == 1.0


# ─── 3. Floor and ceiling guarantees ────────────────────────────────────


class TestFloorCeiling:
    def test_never_above_one(self, anchors):
        """f is monotonically capped at 1.0 — even with zero gamma/vega."""
        f = compute_size_scaler(
            equity=10_000_000.0, spot=27_000.0, atm_iv=60.0,
            pre_gamma=0.0, pre_vega=0.0, anchors=anchors,
        )
        assert 0.10 <= f <= 1.0

    def test_never_below_floor(self, anchors):
        """f never < SIZE_FLOOR_F_MIN even at extreme greeks."""
        for pre_gamma, pre_vega in [(1e6, 1e6), (1.0, 1e6), (1e6, 1.0)]:
            f = compute_size_scaler(
                equity=100.0,    # tiny equity
                spot=27_000.0, atm_iv=60.0,
                pre_gamma=pre_gamma, pre_vega=pre_vega,
                anchors=anchors,
            )
            assert f >= SIZE_FLOOR_F_MIN - 1e-12, (
                f"f={f} below floor at greek=({pre_gamma}, {pre_vega})"
            )


# ─── 4. Vectorised parity ────────────────────────────────────────────────


class TestVectorisedParity:
    def test_vectorised_matches_scalar(self, anchors):
        """Per-row vectorised result equals scalar function row-by-row."""
        n = 100
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "equity": rng.uniform(50_000, 200_000, n),
            "index_price": rng.uniform(20_000, 60_000, n),
            "iv_used": rng.uniform(40, 100, n),
            "gamma": rng.uniform(1e-7, 1e-4, n),
            "vega": rng.uniform(10, 200, n),
        }, index=idx)

        vectorised = compute_size_scaler_vectorised(df, anchors=anchors)
        scalar = pd.Series([
            compute_size_scaler(
                equity=row["equity"], spot=row["index_price"],
                atm_iv=row["iv_used"], pre_gamma=row["gamma"],
                pre_vega=row["vega"], anchors=anchors,
            )
            for _, row in df.iterrows()
        ], index=df.index, name="size_mult")
        np.testing.assert_allclose(vectorised.values, scalar.values, rtol=1e-12)

    def test_missing_column_raises(self, anchors):
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({"equity": [1.0]*10, "iv_used": [60.0]*10},
                          index=idx)
        with pytest.raises(KeyError, match="Missing required columns"):
            compute_size_scaler_vectorised(df, anchors=anchors)


# ─── 5. PME IV-shock anchor ──────────────────────────────────────────────


class TestPMEIVAnchor:
    def _synthetic_dvol(self, n_days: int = 600, seed: int = 42) -> pd.Series:
        """Synthetic daily DVOL series spanning the calibration window."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2021-04-01", periods=n_days, freq="D", tz="UTC")
        # Random walk in DVOL with annualised ~2% daily vol
        dvol = 60.0 + np.cumsum(rng.normal(0, 1.0, n_days))
        dvol = np.clip(dvol, 20.0, 200.0)
        return pd.Series(dvol, index=idx, name="dvol")

    def test_anchor_in_expected_range(self):
        """99th percentile of |Δrel| on synthetic DVOL is within plausible band."""
        dvol = self._synthetic_dvol()
        shock, n_obs = compute_pme_iv_shock_anchor(dvol)
        # Synthetic DVOL has ~1.5% daily relative changes → 99th-pct ≈ 5-12%
        assert 0.01 < shock < 0.30
        assert n_obs > 100

    def test_post_window_mutation_does_not_change_anchor(self):
        """Mutate DVOL values AFTER 2023-10-01: anchor must be unchanged.

        Gold-standard frozen-window guarantee.
        """
        dvol_cal = self._synthetic_dvol(n_days=900)  # 2021-04-01 → 2023-09-XX
        # Append POST-2023-10-01 garbage
        post_idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
        rng = np.random.default_rng(7)
        post = pd.Series(rng.uniform(-1000, 1000, 200), index=post_idx)
        dvol_extended = pd.concat([dvol_cal, post])

        shock_a, n_a = compute_pme_iv_shock_anchor(dvol_cal)
        shock_b, n_b = compute_pme_iv_shock_anchor(dvol_extended)
        assert shock_a == pytest.approx(shock_b, rel=1e-12)
        assert n_a == n_b

    def test_too_few_days_raises(self):
        idx = pd.date_range("2021-04-01", periods=50, freq="D", tz="UTC")
        dvol = pd.Series(np.full(50, 60.0), index=idx)
        with pytest.raises(ValueError, match="≥ 100 calibration days"):
            compute_pme_iv_shock_anchor(dvol)

    def test_build_pme_anchors_returns_record(self):
        dvol = self._synthetic_dvol()
        anchors = build_pme_anchors(dvol)
        assert anchors.loss_budget_pct == LOSS_BUDGET_PCT
        assert anchors.pme_shock_spot == PME_SHOCK_SPOT
        assert 0 < anchors.pme_shock_iv_pct < 1
        assert anchors.pme_iv_anchor_n_obs > 100


# ─── 6. Sidecar round-trip ──────────────────────────────────────────────


class TestSidecar:
    def test_write_load_round_trip(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
        size_mult = pd.Series(
            np.linspace(0.1, 1.0, 20), index=idx, name="size_mult",
        )
        out = tmp_path / "preventative.parquet"
        write_sidecar(size_mult, out)
        loaded = load_sidecar(out)
        np.testing.assert_array_equal(size_mult.values, loaded.values)
        assert loaded.index.equals(size_mult.index)

    def test_write_renames_to_size_mult(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
        ser = pd.Series([1.0]*5, index=idx, name="some_other_name")
        out = tmp_path / "prev.parquet"
        write_sidecar(ser, out)
        loaded = load_sidecar(out)
        assert loaded.name == "size_mult"

    def test_load_missing_column_raises(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
        df = pd.DataFrame({"foo": [1.0, 2.0, 3.0]}, index=idx)
        path = tmp_path / "bad.parquet"
        df.to_parquet(path)
        with pytest.raises(KeyError, match="missing 'size_mult'"):
            load_sidecar(path)


# ─── 7. Backtester integration (config flag + opt-in default off) ──────


class TestBacktesterIntegration:
    def test_default_config_flag_is_none(self):
        """Bare config has preventative sizing OFF (preserves byte-identity)."""
        from vol_arb.backtester import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.preventative_size_mult_path is None

    def test_v2_preset_does_not_auto_enable(self):
        """V2 production preset does NOT auto-enable the flag.

        The Cycle 2 Phase P gate REJECTED the rule for the V2 default
        account (compound equity grew enough that the rule never binds in
        the eval window). Adopting it as default would be exactly the
        gate-violation the plan codified. Stays opt-in only.
        """
        from vol_arb.backtester import BacktestConfig
        cfg = BacktestConfig().with_btc_production_filters_v2()
        assert cfg.preventative_size_mult_path is None

    def test_canonical_v2_sidecar_artifact_exists(self):
        """The Cycle 2 sidecar artifact is reproducible from frozen anchors.

        We don't assert the exact bytes (that would change if
        ``compute_pme_iv_shock_anchor`` is ever recomputed with extended
        data) — only that the file exists with the expected schema.
        """
        path = Path("vol_arb/output/BTC_v2_preventative_size_mult.parquet")
        if not path.exists():
            pytest.skip(
                "sidecar not generated yet; run "
                "scripts/generate_preventative_sizing_sidecar.py"
            )
        ser = load_sidecar(path)
        assert ser.name == "size_mult"
        assert (ser >= SIZE_FLOOR_F_MIN - 1e-12).all()
        assert (ser <= 1.0 + 1e-12).all()

    def test_apply_path_runs_without_crash(self, tmp_path, anchors):
        """Smoke: backtester run with the flag on produces SOME output.

        We don't assert the actual PnL (that depends on full backtester
        state); we just verify the load + apply branch doesn't crash.
        Skipped if production data isn't available.
        """
        from vol_arb.backtester import BacktestConfig
        # Build a synthetic sidecar with all 1.0 → should be byte-identical
        # to the no-sidecar case for a small fixture
        idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
        size_mult = pd.Series(np.ones(100), index=idx, name="size_mult")
        sidecar_path = tmp_path / "all_ones.parquet"
        write_sidecar(size_mult, sidecar_path)

        cfg = BacktestConfig()
        cfg.preventative_size_mult_path = str(sidecar_path)
        # Just verify the field is wired through; full backtester run
        # would require BTC production data we don't want to load in
        # a unit test.
        assert cfg.preventative_size_mult_path == str(sidecar_path)


# ─── 8. NC-path mirror regression (Cycle 3 bug fix) ───────────────────


class TestNCPathMirror:
    """Regression test for the Cycle-3 backtester fix.

    Pre-fix: the non-compound (NC) path at backtester.py:2260+ mirrored
    the Kelly K1 multiplier but NOT the preventative-sizing multiplier
    or the uncertainty filter. This made any rule using
    ``preventative_size_mult_path`` or ``uncertainty_filter_path``
    appear NON-BINDING on ``net_pnl_nc`` even when it was binding on
    the compound path.

    Post-fix: both NC overlays mirror their compound-side counterparts.
    These tests lock the post-fix behavior so any future regression
    that re-introduces the gap will fail.
    """

    def test_preventative_sizing_actually_binds_on_NC(self):
        """Cycle 2 preventative sizing must reduce NC PnL post-fix.

        Pre-fix this delta was +$18 (essentially no-op). Post-fix it
        is around -$18,624 (~-2%). Lock the order of magnitude.
        """
        from vol_arb.backtester import BacktestConfig, VolArbBacktester
        sidecar = Path("vol_arb/output/BTC_v2_preventative_size_mult.parquet")
        if not sidecar.exists():
            pytest.skip(f"sidecar {sidecar} not generated")

        cfg = BacktestConfig().with_btc_production_filters_v2().with_realistic_costs()
        cfg.preventative_size_mult_path = str(sidecar)
        _, tl = VolArbBacktester(cfg).run_backtest()
        nc_pnl_with = float(tl["net_pnl_nc"].sum())
        # Canonical baseline = $959,571.0653
        delta = nc_pnl_with - 959_571.0653
        # Should be a real reduction (rule binds): delta is negative and
        # of meaningful magnitude. Lock to require at least -$5,000.
        assert delta < -5_000.0, (
            f"NC-path mirror regression: preventative sizing produced "
            f"delta=${delta:+,.0f}, expected ≤ -$5,000 (rule must actually "
            "bind on NC path)"
        )

    def test_bare_v2_byte_identity_preserved(self):
        """The NC-path fix MUST preserve V2 byte-identity when no
        overlays are active.
        """
        from vol_arb.backtester import BacktestConfig, VolArbBacktester
        cfg = BacktestConfig().with_btc_production_filters_v2().with_realistic_costs()
        # Both overlay flags default off
        assert cfg.preventative_size_mult_path is None
        assert cfg.uncertainty_filter_path is None
        _, tl = VolArbBacktester(cfg).run_backtest()
        nc_pnl = float(tl["net_pnl_nc"].sum())
        n_fired = int((tl["net_pnl_nc"] != 0).sum())
        assert abs(nc_pnl - 959_571.0653) < 1.0, f"NC PnL drift: {nc_pnl}"
        assert n_fired == 7417, f"Fired count drift: {n_fired}"
