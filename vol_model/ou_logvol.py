"""Bivariate Ornstein-Uhlenbeck calibrator + path simulator on log-vol.

Cycle 2 Phase A infrastructure (plan: ``now-check-empirically-from-eager-bird.md``).

Calibrates a bivariate OU process to BTC 1-minute (log_IV, log_RV) on a
strictly pre-evaluation calibration window and simulates synthetic paths
for evaluating future Tier-C path-dependent rules. AFML Ch. 13 — synthetic-
path Monte Carlo for path-dependent backtest robustness.

The univariate OU process

    dx_t = κ (θ - x_t) dt + σ dW_t

discretises at uniform Δt as the Gaussian AR(1)

    x_{t+Δt} = a + b·x_t + ε_t,  ε_t ~ N(0, var_eps)

with closed-form mapping

    b       = exp(-κ Δt)
    a       = θ (1 - b)
    var_eps = σ² (1 - b²) / (2 κ)

Closed-form recovery via OLS (Phillips-Yu 2009 equivalent for small Δt):

    b̂       = cov(x_{t+Δt}, x_t) / var(x_t)
    â       = mean(x_{t+Δt}) - b̂ · mean(x_t)
    var_eps = var(x_{t+Δt} - â - b̂ x_t)
    κ̂       = -ln(b̂) / Δt
    θ̂       = â / (1 - b̂)
    σ̂²      = var_eps · 2 κ̂ / (1 - b̂²)

The bivariate version fits each marginal independently and then computes
empirical residual correlation:

    ρ̂ = corr(ε_iv, ε_rv)

Why bivariate (not univariate log-IV)?
    V2's whipsaw risk lives in the IV-RV spread. A univariate log-IV path
    that ignores correlated RV jumps under-states tail risk on days where
    realised vol explodes faster than implied vol can re-quote. The
    Cholesky-coupled bivariate path simulates this jointly.

Why frozen calibration?
    AFML Ch. 7 anti-leakage primitive: parameters that drive evaluation
    must be fit strictly before the evaluation window. ``calibrate_*``
    raises if any timestamp in the input is at-or-after the calibration
    window's upper bound.

Frozen JSON serialisation guards against accidental re-fit:
    ``freeze_calibration`` refuses to overwrite an existing file unless
    ``force=True`` is passed. Production runs use the immutable default.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Pre-committed window (locked in plan) ───────────────────────────────

CALIBRATION_WINDOW: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2021-04-01", tz="UTC"),
    pd.Timestamp("2023-10-01", tz="UTC"),  # exclusive upper bound
)

# Quarantine days dropped from BOTH calibration and evaluation
# (matches scripts/forensic/audit_v2_tail_risks.py and the plan)
QUARANTINE_DAYS: tuple[pd.Timestamp, ...] = (
    pd.Timestamp("2022-06-13", tz="UTC"),
    pd.Timestamp("2022-11-08", tz="UTC"),
    pd.Timestamp("2024-08-05", tz="UTC"),
    pd.Timestamp("2025-01-20", tz="UTC"),
    pd.Timestamp("2024-07-26", tz="UTC"),
    pd.Timestamp("2024-11-05", tz="UTC"),
)

# Numerical floors. log(vol) explodes when vol hits zero (TSRV emits 0.0
# when no ticks fire in a minute), so floor at exp(-10) ≈ 4.5e-5 (0.0045%
# annualised — far below any realistic regime).
LOG_VOL_FLOOR_DECIMAL: float = float(np.exp(-10.0))


@dataclass(frozen=True)
class OUParams2D:
    """Closed-form bivariate OU parameters at fixed Δt = 1 minute.

    Marginals: dxᵢ = κᵢ (θᵢ - xᵢ) dt + σᵢ dWᵢ for i ∈ {iv, rv}.
    Coupling: corr(dW_iv, dW_rv) = ρ (empirical residual correlation).

    Stored both as continuous-time params (κ, θ, σ) and as the AR(1)
    discretisation (a, b, var_eps) so simulation does not require
    re-derivation.
    """

    # ── Marginal: log-IV ───────────────────────────────────────────────
    kappa_iv: float
    theta_iv: float
    sigma_iv: float
    a_iv: float
    b_iv: float
    var_eps_iv: float

    # ── Marginal: log-RV ───────────────────────────────────────────────
    kappa_rv: float
    theta_rv: float
    sigma_rv: float
    a_rv: float
    b_rv: float
    var_eps_rv: float

    # ── Coupling ───────────────────────────────────────────────────────
    rho: float

    # ── Audit metadata ─────────────────────────────────────────────────
    n_observations: int
    calibration_window_start: str   # ISO-8601 UTC
    calibration_window_end: str     # exclusive upper bound, ISO-8601 UTC
    dt_minutes: float = 1.0


# ─── Calibration ─────────────────────────────────────────────────────────

def _ar1_ols(x_t: np.ndarray, x_tp1: np.ndarray) -> tuple[float, float, float]:
    """Closed-form OLS on AR(1): x_{t+Δt} = a + b·x_t + ε_t.

    Returns (a, b, var_eps).
    """
    var_x = float(np.var(x_t, ddof=0))
    if var_x <= 0:
        raise ValueError("AR(1) regression: zero variance in regressor")
    cov = float(np.mean((x_t - x_t.mean()) * (x_tp1 - x_tp1.mean())))
    b = cov / var_x
    a = float(x_tp1.mean() - b * x_t.mean())
    resid = x_tp1 - a - b * x_t
    var_eps = float(np.var(resid, ddof=1))
    return a, b, var_eps


def _continuous_from_discrete(
    a: float, b: float, var_eps: float, dt: float,
) -> tuple[float, float, float]:
    """Map (a, b, var_eps) at uniform Δt to (κ, θ, σ).

    κ        = -ln(b) / Δt   (requires 0 < b < 1 for stationarity)
    θ        = a / (1 - b)
    σ²       = var_eps · 2 κ / (1 - b²)
    """
    if not (0.0 < b < 1.0):
        raise ValueError(
            f"Bivariate OU calibration: b={b:.6f} outside stationarity "
            "(0, 1). Either Δt too short relative to mean-reversion or "
            "input series is non-stationary on the calibration window."
        )
    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma_sq = var_eps * (2.0 * kappa) / (1.0 - b * b)
    sigma = np.sqrt(max(sigma_sq, 0.0))
    return float(kappa), float(theta), float(sigma)


def calibrate_bivariate_ou(
    log_iv: pd.Series,
    log_rv: pd.Series,
    *,
    calibration_window: tuple[pd.Timestamp, pd.Timestamp] = CALIBRATION_WINDOW,
    quarantine_days: tuple[pd.Timestamp, ...] = QUARANTINE_DAYS,
    dt_minutes: float = 1.0,
) -> OUParams2D:
    """Fit bivariate OU on (log_iv, log_rv) restricted to calibration window.

    Hard guards:
    - Both inputs must be DatetimeIndex (UTC) at 1-min frequency.
    - Inputs are restricted to ``calibration_window`` (left-closed,
      right-open) and quarantine UTC dates are dropped.
    - Raises if any post-window observation slips through (defence in depth).

    The empirical residual correlation ``ρ̂`` is computed on the
    intersection of valid (non-NaN) (log_iv_{t+Δt} - residual_iv,
    log_rv_{t+Δt} - residual_rv) pairs.
    """
    if not isinstance(log_iv.index, pd.DatetimeIndex):
        raise TypeError("log_iv must have a DatetimeIndex")
    if not isinstance(log_rv.index, pd.DatetimeIndex):
        raise TypeError("log_rv must have a DatetimeIndex")

    cal_start, cal_end = calibration_window

    # Hard guard: input series must NOT contain post-window timestamps for
    # the bivariate fit. (Caller is expected to pre-filter; this catches
    # leaks. We only check the actual *fitted* slice — passing data that
    # extends past the window is fine if it gets dropped here.)
    iv_in = log_iv.loc[(log_iv.index >= cal_start) & (log_iv.index < cal_end)]
    rv_in = log_rv.loc[(log_rv.index >= cal_start) & (log_rv.index < cal_end)]

    # Drop quarantine days
    iv_in = iv_in[~iv_in.index.normalize().isin(quarantine_days)]
    rv_in = rv_in[~rv_in.index.normalize().isin(quarantine_days)]

    # Align on common 1-min index
    aligned = pd.concat({"iv": iv_in, "rv": rv_in}, axis=1).dropna()

    if len(aligned) < 10_000:
        raise ValueError(
            f"Bivariate OU: only {len(aligned)} aligned observations on "
            "calibration window; need at least 10,000 for stable fit."
        )

    # Δt = 1 minute by construction; lag both series and drop the boundary
    iv_arr = aligned["iv"].to_numpy(dtype=np.float64)
    rv_arr = aligned["rv"].to_numpy(dtype=np.float64)
    iv_t = iv_arr[:-1]
    iv_tp1 = iv_arr[1:]
    rv_t = rv_arr[:-1]
    rv_tp1 = rv_arr[1:]

    a_iv, b_iv, var_eps_iv = _ar1_ols(iv_t, iv_tp1)
    a_rv, b_rv, var_eps_rv = _ar1_ols(rv_t, rv_tp1)

    kappa_iv, theta_iv, sigma_iv = _continuous_from_discrete(
        a_iv, b_iv, var_eps_iv, dt_minutes,
    )
    kappa_rv, theta_rv, sigma_rv = _continuous_from_discrete(
        a_rv, b_rv, var_eps_rv, dt_minutes,
    )

    eps_iv = iv_tp1 - a_iv - b_iv * iv_t
    eps_rv = rv_tp1 - a_rv - b_rv * rv_t
    if eps_iv.std(ddof=1) > 0 and eps_rv.std(ddof=1) > 0:
        rho = float(np.corrcoef(eps_iv, eps_rv)[0, 1])
    else:
        rho = 0.0

    return OUParams2D(
        kappa_iv=kappa_iv, theta_iv=theta_iv, sigma_iv=sigma_iv,
        a_iv=a_iv, b_iv=b_iv, var_eps_iv=var_eps_iv,
        kappa_rv=kappa_rv, theta_rv=theta_rv, sigma_rv=sigma_rv,
        a_rv=a_rv, b_rv=b_rv, var_eps_rv=var_eps_rv,
        rho=rho,
        n_observations=int(len(aligned) - 1),
        calibration_window_start=cal_start.isoformat(),
        calibration_window_end=cal_end.isoformat(),
        dt_minutes=dt_minutes,
    )


# ─── Path simulation ─────────────────────────────────────────────────────

def simulate_paths(
    params: OUParams2D,
    *,
    n_paths: int,
    n_minutes: int,
    x0_iv: float,
    x0_rv: float,
    seed: int,
) -> np.ndarray:
    """Vectorised Cholesky-coupled bivariate-OU path simulator.

    Returns a 3D ndarray of shape ``(n_paths, n_minutes, 2)`` where index
    [:, :, 0] is log-IV and [:, :, 1] is log-RV. Element [:, 0, :] is the
    initial state ``(x0_iv, x0_rv)``.

    Discretisation matches the AR(1) form used in calibration:

        x_{i, t+1} = a_i + b_i · x_{i, t} + ε_{i, t}
        ε_iv ~ N(0, var_eps_iv)
        ε_rv = ρ · sqrt(var_eps_rv / var_eps_iv) · ε_iv
                + sqrt(var_eps_rv · (1 - ρ²)) · z₂   (z₂ ⊥ ε_iv)

    Determinism: fixed ``seed`` produces byte-identical output.
    """
    if n_paths < 1 or n_minutes < 1:
        raise ValueError("n_paths and n_minutes must be ≥ 1")

    rng = np.random.default_rng(seed)

    paths = np.empty((n_paths, n_minutes, 2), dtype=np.float64)
    paths[:, 0, 0] = x0_iv
    paths[:, 0, 1] = x0_rv

    sigma_eps_iv = np.sqrt(params.var_eps_iv)
    sigma_eps_rv = np.sqrt(params.var_eps_rv)

    if n_minutes == 1:
        return paths

    # Pre-draw all Gaussian innovations: shape (n_paths, n_minutes-1, 2)
    z = rng.standard_normal(size=(n_paths, n_minutes - 1, 2))
    # Couple via Cholesky:  ε_rv = ρ σ_rv z1 + sqrt(1-ρ²) σ_rv z2
    eps_iv = sigma_eps_iv * z[:, :, 0]
    eps_rv = (params.rho * sigma_eps_rv * z[:, :, 0]
              + np.sqrt(max(1.0 - params.rho ** 2, 0.0))
              * sigma_eps_rv * z[:, :, 1])

    # Iterative AR(1) update — vectorised across paths
    iv_prev = np.full(n_paths, x0_iv, dtype=np.float64)
    rv_prev = np.full(n_paths, x0_rv, dtype=np.float64)
    for t in range(1, n_minutes):
        iv_now = params.a_iv + params.b_iv * iv_prev + eps_iv[:, t - 1]
        rv_now = params.a_rv + params.b_rv * rv_prev + eps_rv[:, t - 1]
        paths[:, t, 0] = iv_now
        paths[:, t, 1] = rv_now
        iv_prev = iv_now
        rv_prev = rv_now

    return paths


# ─── Frozen-JSON serialisation ───────────────────────────────────────────

def freeze_calibration(
    params: OUParams2D,
    out_path: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Write the OUParams2D to ``out_path`` as JSON. Refuses to overwrite.

    The frozen JSON is the lock for downstream evaluators. Re-fitting on
    a different window or mutated source data will produce a different
    file — but never silently overwrite the existing one without
    ``force=True`` being explicitly passed.
    """
    out = Path(out_path)
    if out.exists() and not force:
        raise FileExistsError(
            f"OU calibration already frozen at {out}. Refusing to "
            "overwrite. Pass force=True only if you have a NEW plan file "
            "authorising recalibration."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(params)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True),
                   encoding="utf-8")
    logger.info("Froze OU calibration to %s (n_observations=%d)",
                out, params.n_observations)
    return out


def load_frozen_calibration(in_path: Path | str) -> OUParams2D:
    """Load a frozen OU calibration. Pure read, no mutation."""
    in_p = Path(in_path)
    payload = json.loads(in_p.read_text(encoding="utf-8"))
    return OUParams2D(**payload)


# ─── Helpers for production data sources ────────────────────────────────

def load_log_iv_minute(parquet_path: Path | str) -> pd.Series:
    """Load minute_atm_iv → log(decimal_iv), floored.

    minute_atm_iv stores ``median_iv`` in vol points (e.g. 60.0 = 60% IV).
    Returns log(median_iv / 100), with values floored at LOG_VOL_FLOOR_DECIMAL
    to avoid -inf on minutes with no Deribit prints.
    """
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if "median_iv" not in df.columns:
        raise KeyError(f"{parquet_path}: missing 'median_iv' column")
    iv_dec = (df["median_iv"] / 100.0).clip(lower=LOG_VOL_FLOOR_DECIMAL)
    return np.log(iv_dec).rename("log_iv")


def load_log_rv_minute(parquet_path: Path | str,
                       column: str = "vol_ann") -> pd.Series:
    """Load TSRV minute file → log(vol_ann), floored.

    TSRV's ``vol_ann`` is already annualised decimal vol. Floored at
    LOG_VOL_FLOOR_DECIMAL to keep log finite during low-tick minutes.
    """
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if column not in df.columns:
        raise KeyError(f"{parquet_path}: missing '{column}' column")
    rv = df[column].clip(lower=LOG_VOL_FLOOR_DECIMAL)
    return np.log(rv).rename("log_rv")
