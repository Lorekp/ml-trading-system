"""Tests for vol_model/ou_logvol.py — bivariate OU calibration + simulator.

Test categories:
  1. Round-trip parameter recovery on synthetic OU paths.
  2. Determinism under fixed seed.
  3. Empirical correlation matches the ``rho`` parameter (Cholesky check).
  4. Calibration window leakage probe — mutating data POST-window must
     not change calibrated parameters.
  5. Frozen-JSON refuse-overwrite.
  6. Helper-loader sanity (clipping floor, log conversion).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_model.ou_logvol import (
    CALIBRATION_WINDOW,
    LOG_VOL_FLOOR_DECIMAL,
    OUParams2D,
    QUARANTINE_DAYS,
    calibrate_bivariate_ou,
    freeze_calibration,
    load_frozen_calibration,
    load_log_iv_minute,
    load_log_rv_minute,
    simulate_paths,
)


# ─── Helpers for synthetic data generation ──────────────────────────────


def _simulate_series(
    a: float, b: float, var_eps: float, n: int, seed: int,
) -> np.ndarray:
    """Single-series AR(1) simulator for use in tests."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=np.float64)
    x[0] = a / (1.0 - b)  # start at theta
    sigma = np.sqrt(var_eps)
    eps = rng.standard_normal(n) * sigma
    for t in range(1, n):
        x[t] = a + b * x[t - 1] + eps[t]
    return x


def _make_synthetic_logvol_pair(
    n_minutes: int = 200_000, seed: int = 42,
    rho_target: float = 0.6,
    a_iv: float = -0.001, b_iv: float = 0.999, var_eps_iv: float = 1e-5,
    a_rv: float = -0.002, b_rv: float = 0.998, var_eps_rv: float = 4e-5,
) -> tuple[pd.Series, pd.Series, dict]:
    """Build a synthetic bivariate AR(1) at 1-min frequency with known params.

    Index is at 1-min frequency starting 2021-04-01 UTC and entirely INSIDE
    the calibration window. Returns (log_iv, log_rv, true_params).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_minutes, 2))
    # Couple via Cholesky on residuals
    eps_iv = np.sqrt(var_eps_iv) * z[:, 0]
    eps_rv = (rho_target * np.sqrt(var_eps_rv) * z[:, 0]
              + np.sqrt(max(1.0 - rho_target ** 2, 0.0))
              * np.sqrt(var_eps_rv) * z[:, 1])
    iv = np.empty(n_minutes, dtype=np.float64)
    rv = np.empty(n_minutes, dtype=np.float64)
    iv[0] = a_iv / (1.0 - b_iv)
    rv[0] = a_rv / (1.0 - b_rv)
    for t in range(1, n_minutes):
        iv[t] = a_iv + b_iv * iv[t - 1] + eps_iv[t]
        rv[t] = a_rv + b_rv * rv[t - 1] + eps_rv[t]
    idx = pd.date_range(
        "2021-04-01 00:01:00+00:00", periods=n_minutes, freq="1min", tz="UTC",
    )
    log_iv = pd.Series(iv, index=idx, name="log_iv")
    log_rv = pd.Series(rv, index=idx, name="log_rv")
    true_params = {
        "a_iv": a_iv, "b_iv": b_iv, "var_eps_iv": var_eps_iv,
        "a_rv": a_rv, "b_rv": b_rv, "var_eps_rv": var_eps_rv,
        "rho": rho_target,
    }
    return log_iv, log_rv, true_params


# ─── 1. Parameter recovery ──────────────────────────────────────────────


class TestCalibration:
    def test_recovers_marginal_params(self):
        log_iv, log_rv, true_p = _make_synthetic_logvol_pair(
            n_minutes=250_000, seed=42,
        )
        params = calibrate_bivariate_ou(
            log_iv, log_rv,
            calibration_window=(
                pd.Timestamp("2021-04-01", tz="UTC"),
                pd.Timestamp("2023-10-01", tz="UTC"),
            ),
            quarantine_days=tuple(),
        )
        # b should be very close to true value (within a few standard errors)
        assert abs(params.b_iv - true_p["b_iv"]) < 1e-3, (
            f"b_iv: got {params.b_iv}, true {true_p['b_iv']}"
        )
        assert abs(params.b_rv - true_p["b_rv"]) < 1e-3
        # var_eps recovery (relative tolerance because std var is ~ 1/sqrt(N))
        assert abs(params.var_eps_iv - true_p["var_eps_iv"]) / true_p["var_eps_iv"] < 0.05
        assert abs(params.var_eps_rv - true_p["var_eps_rv"]) / true_p["var_eps_rv"] < 0.05
        # rho recovery
        assert abs(params.rho - true_p["rho"]) < 0.03, (
            f"rho: got {params.rho}, true {true_p['rho']}"
        )

    def test_continuous_params_consistent_with_discrete(self):
        log_iv, log_rv, _ = _make_synthetic_logvol_pair(n_minutes=100_000, seed=1)
        params = calibrate_bivariate_ou(
            log_iv, log_rv, quarantine_days=tuple(),
        )
        # κ = -ln(b) / Δt
        assert abs(params.kappa_iv - (-np.log(params.b_iv) / params.dt_minutes)) < 1e-12
        assert abs(params.kappa_rv - (-np.log(params.b_rv) / params.dt_minutes)) < 1e-12
        # θ = a / (1 - b)
        assert abs(params.theta_iv - params.a_iv / (1.0 - params.b_iv)) < 1e-12
        # σ² = var_eps · 2κ / (1 - b²)
        sigma_sq_iv = params.var_eps_iv * 2 * params.kappa_iv / (1 - params.b_iv ** 2)
        assert abs(params.sigma_iv ** 2 - sigma_sq_iv) < 1e-15

    def test_too_few_observations_raises(self):
        idx = pd.date_range("2021-06-01", periods=500, freq="1min", tz="UTC")
        log_iv = pd.Series(np.random.randn(500), index=idx, name="iv")
        log_rv = pd.Series(np.random.randn(500), index=idx, name="rv")
        with pytest.raises(ValueError, match="at least 10,000"):
            calibrate_bivariate_ou(log_iv, log_rv, quarantine_days=tuple())

    def test_quarantine_days_dropped(self):
        log_iv, log_rv, _ = _make_synthetic_logvol_pair(n_minutes=100_000, seed=2)
        # Inject NaN on the entire 2022-06-13 day (one of QUARANTINE_DAYS)
        # by mutating the source; the calibrator should drop that day.
        cal = calibrate_bivariate_ou(log_iv, log_rv, quarantine_days=tuple())
        # Now spike a quarantine day with extreme values; verify the fit is
        # unchanged when we re-calibrate with QUARANTINE_DAYS active.
        spiked_iv = log_iv.copy()
        spiked_rv = log_rv.copy()
        target_day = pd.Timestamp("2022-06-13", tz="UTC")
        spike_mask = spiked_iv.index.normalize() == target_day
        spiked_iv.loc[spike_mask] = 100.0  # absurd
        spiked_rv.loc[spike_mask] = 100.0
        cal_with_quar = calibrate_bivariate_ou(
            spiked_iv, spiked_rv, quarantine_days=QUARANTINE_DAYS,
        )
        # The two should be IDENTICAL (well, within float rounding) because
        # the quarantine list dropped the spike day in the second call,
        # and the first call had no quarantine but also no spike on 2022-06-13.
        # Use very tight tolerance — only float ordering should differ.
        assert abs(cal.b_iv - cal_with_quar.b_iv) < 1e-9
        assert abs(cal.b_rv - cal_with_quar.b_rv) < 1e-9


# ─── 2. Determinism + 3. correlation ─────────────────────────────────────


class TestSimulation:
    def _params(self) -> OUParams2D:
        return OUParams2D(
            kappa_iv=0.001, theta_iv=-0.5, sigma_iv=0.005,
            a_iv=-0.0005, b_iv=0.999, var_eps_iv=1e-5,
            kappa_rv=0.002, theta_rv=-0.3, sigma_rv=0.008,
            a_rv=-0.0006, b_rv=0.998, var_eps_rv=4e-5,
            rho=0.5,
            n_observations=100_000,
            calibration_window_start="2021-04-01T00:00:00+00:00",
            calibration_window_end="2023-10-01T00:00:00+00:00",
        )

    def test_deterministic_under_seed(self):
        p = self._params()
        a = simulate_paths(p, n_paths=10, n_minutes=100, x0_iv=-0.5,
                           x0_rv=-0.3, seed=42)
        b = simulate_paths(p, n_paths=10, n_minutes=100, x0_iv=-0.5,
                           x0_rv=-0.3, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_paths(self):
        p = self._params()
        a = simulate_paths(p, n_paths=5, n_minutes=50, x0_iv=-0.5,
                           x0_rv=-0.3, seed=42)
        b = simulate_paths(p, n_paths=5, n_minutes=50, x0_iv=-0.5,
                           x0_rv=-0.3, seed=43)
        # At least some path differs
        assert not np.allclose(a, b)

    def test_initial_state_correct(self):
        p = self._params()
        paths = simulate_paths(p, n_paths=4, n_minutes=10, x0_iv=-0.42,
                               x0_rv=-0.38, seed=7)
        assert np.allclose(paths[:, 0, 0], -0.42)
        assert np.allclose(paths[:, 0, 1], -0.38)
        assert paths.shape == (4, 10, 2)

    def test_empirical_residual_correlation_matches_rho(self):
        """Sample-residuals correlation across many paths should match rho."""
        p = self._params()
        # Many paths, short horizons → many independent residuals
        paths = simulate_paths(p, n_paths=2_000, n_minutes=200, x0_iv=p.theta_iv,
                               x0_rv=p.theta_rv, seed=99)
        # Compute innovations: ε_t = x_{t+1} - a - b·x_t
        iv = paths[:, :, 0]
        rv = paths[:, :, 1]
        eps_iv = iv[:, 1:] - p.a_iv - p.b_iv * iv[:, :-1]
        eps_rv = rv[:, 1:] - p.a_rv - p.b_rv * rv[:, :-1]
        # Flatten and compute correlation
        rho_emp = float(np.corrcoef(eps_iv.flatten(), eps_rv.flatten())[0, 1])
        assert abs(rho_emp - p.rho) < 0.02, f"rho_emp = {rho_emp}, target = {p.rho}"

    def test_zero_minutes_and_one_minute(self):
        p = self._params()
        with pytest.raises(ValueError):
            simulate_paths(p, n_paths=1, n_minutes=0,
                           x0_iv=0.0, x0_rv=0.0, seed=1)
        # n_minutes=1 should return just the initial state
        paths = simulate_paths(p, n_paths=2, n_minutes=1,
                               x0_iv=-0.1, x0_rv=-0.2, seed=1)
        assert paths.shape == (2, 1, 2)
        assert np.allclose(paths[:, 0, 0], -0.1)
        assert np.allclose(paths[:, 0, 1], -0.2)


# ─── 4. Calibration-window leakage probe ─────────────────────────────────


class TestLeakage:
    def test_post_window_mutation_does_not_change_params(self):
        """Mutate values AFTER 2023-10-01; calibrated params must not change.

        This is the gold-standard frozen-window guarantee. The calibrator
        restricts to the calibration window before fitting; post-window
        observations must be invisible to the fit.
        """
        log_iv, log_rv, _ = _make_synthetic_logvol_pair(
            n_minutes=200_000, seed=7,
        )
        # Append POST-2023-10-01 data (outside calibration window)
        post_n = 50_000
        post_idx = pd.date_range(
            "2024-01-01 00:00:00+00:00", periods=post_n, freq="1min", tz="UTC",
        )
        log_iv_extended = pd.concat([
            log_iv,
            pd.Series(np.full(post_n, -100.0), index=post_idx),  # absurd post-window
        ])
        log_rv_extended = pd.concat([
            log_rv,
            pd.Series(np.full(post_n, -100.0), index=post_idx),
        ])

        cal_baseline = calibrate_bivariate_ou(
            log_iv, log_rv, quarantine_days=tuple(),
        )
        cal_with_post = calibrate_bivariate_ou(
            log_iv_extended, log_rv_extended, quarantine_days=tuple(),
        )

        assert abs(cal_baseline.b_iv - cal_with_post.b_iv) < 1e-12
        assert abs(cal_baseline.b_rv - cal_with_post.b_rv) < 1e-12
        assert abs(cal_baseline.var_eps_iv - cal_with_post.var_eps_iv) < 1e-15
        assert abs(cal_baseline.var_eps_rv - cal_with_post.var_eps_rv) < 1e-15
        assert abs(cal_baseline.rho - cal_with_post.rho) < 1e-12
        assert cal_baseline.n_observations == cal_with_post.n_observations


# ─── 5. Frozen-JSON refuse-overwrite ─────────────────────────────────────


class TestFreeze:
    def _params(self) -> OUParams2D:
        return OUParams2D(
            kappa_iv=0.001, theta_iv=-0.5, sigma_iv=0.005,
            a_iv=-0.0005, b_iv=0.999, var_eps_iv=1e-5,
            kappa_rv=0.002, theta_rv=-0.3, sigma_rv=0.008,
            a_rv=-0.0006, b_rv=0.998, var_eps_rv=4e-5,
            rho=0.5,
            n_observations=100_000,
            calibration_window_start="2021-04-01T00:00:00+00:00",
            calibration_window_end="2023-10-01T00:00:00+00:00",
        )

    def test_writes_json(self, tmp_path):
        p = self._params()
        out = tmp_path / "calibration.json"
        freeze_calibration(p, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["b_iv"] == pytest.approx(p.b_iv, rel=1e-12)
        assert payload["rho"] == pytest.approx(p.rho, rel=1e-12)
        assert payload["calibration_window_start"] == p.calibration_window_start

    def test_refuses_overwrite_by_default(self, tmp_path):
        p = self._params()
        out = tmp_path / "calibration.json"
        freeze_calibration(p, out)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            freeze_calibration(p, out)

    def test_force_allows_overwrite(self, tmp_path):
        p1 = self._params()
        out = tmp_path / "calibration.json"
        freeze_calibration(p1, out)
        # Mutate one param and force-overwrite
        p2 = OUParams2D(**{**p1.__dict__, "rho": 0.7})
        freeze_calibration(p2, out, force=True)
        loaded = load_frozen_calibration(out)
        assert loaded.rho == pytest.approx(0.7, rel=1e-12)

    def test_load_round_trip(self, tmp_path):
        p = self._params()
        out = tmp_path / "calibration.json"
        freeze_calibration(p, out)
        loaded = load_frozen_calibration(out)
        for fname in p.__dataclass_fields__:
            assert getattr(p, fname) == getattr(loaded, fname), fname


# ─── 6. Helper loaders (clipping + log) ──────────────────────────────────


class TestLoaders:
    def test_log_iv_clipped_at_floor(self, tmp_path):
        # minute_atm_iv stores median_iv in vol points
        idx = pd.date_range("2023-01-01", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {"median_iv": [60.0, 0.0, 50.0, np.nan, 70.0]},  # 0 → floor
            index=idx,
        )
        path = tmp_path / "minute_iv.parquet"
        df.to_parquet(path)
        log_iv = load_log_iv_minute(path)
        assert log_iv.iloc[0] == pytest.approx(np.log(0.6), rel=1e-12)
        # 0 → floored to LOG_VOL_FLOOR_DECIMAL → log(...) finite
        assert np.isfinite(log_iv.iloc[1])
        assert log_iv.iloc[1] == pytest.approx(
            np.log(LOG_VOL_FLOOR_DECIMAL), rel=1e-12,
        )

    def test_log_rv_uses_specified_column(self, tmp_path):
        idx = pd.date_range("2023-01-01", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {"vol_ann": [0.5, 0.6, 0.7],
             "tsrv": [0.01, 0.01, 0.01]},
            index=idx,
        )
        path = tmp_path / "tsrv.parquet"
        df.to_parquet(path)
        log_rv = load_log_rv_minute(path, column="vol_ann")
        assert log_rv.iloc[0] == pytest.approx(np.log(0.5), rel=1e-12)
        # also confirm raises on missing column
        with pytest.raises(KeyError):
            load_log_rv_minute(path, column="nonexistent")
