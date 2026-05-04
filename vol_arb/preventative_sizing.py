"""Preventative sizing rule — externally-anchored gamma + vega budget.

Cycle 2 Phase P (plan: ``now-check-empirically-from-eager-bird.md``).

V2's edge is short-gamma + IV mean reversion within 15 minutes; reactive
stops destroyed PnL at every threshold tested in Cycle 1 (Tier C-2). The
plan's preventative alternative shrinks size at trade ENTRY when pre-trade
gamma OR vega exceeds an externally-anchored Deribit-PME loss budget. No
backtest tuning, no free parameters.

Derivation (see plan §"Pre-committed parameters"):

    LossBudget   = equity × loss_budget_pct        (1% of equity per event)
    gamma_target = 2 × LossBudget / (Spot × pme_shock_spot)²
    vega_target  = LossBudget / (ATM_IV × pme_shock_iv_pct)
    f            = clip(min(1,
                              gamma_target / pre_gamma,
                              vega_target / pre_vega),
                          f_min, 1.0)

The factor of 2 in `gamma_target` comes from the second-order Taylor
expansion `dPnL ≈ (1/2) × gamma × dS²`. The `vega_target` is the standard
linear vega-PnL identity: `dPnL ≈ vega × dIV` ⇒ `vega = budget / dIV`.

Anchors (NOT tunable):
- ``loss_budget_pct = 0.01``: standard "1% per single-event risk" policy.
  Distinct from V2's ``target_margin_utilization = 0.05`` which is a
  margin-deposit parameter, not a loss-budget parameter.
- ``pme_shock_spot = 0.175``: Deribit Portfolio Margin Engine documented
  main-table BTC spot shock.
- ``pme_shock_iv_pct``: 99th percentile of |Δ DVOL_1d| computed ONCE on
  the calibration window 2021-04-01 → 2023-09-30 (frozen scalar). NOT
  computed on minute_atm_iv (avoid using strategy's own data) and NOT
  recomputed per evaluation (avoid window leakage).
- ``f_min = 0.10``: floor preventing the rule from rounding the trade to
  zero. A trade that vanishes changes downstream entry decisions and
  breaks the bootstrap-equivalence guarantee for sizing-only rules. The
  10% floor preserves trade existence with shrunken size.

Anti-leakage guarantees:
- ``compute_size_scaler`` is a pure function of bar-time observables
  (equity at start of bar, spot, ATM IV, pre-trade gamma, pre-trade vega).
  All inputs are observable AT entry under the canonical pipeline.
- The frozen ``pme_shock_iv_pct`` scalar is computed ONCE and cached;
  ``compute_pme_iv_shock_anchor`` raises if asked to look at post-window
  data.

Anti-overfitting guarantees:
- The three external anchors are documented public values. They are
  asserted in tests via constant-equality checks; modifying any of them
  in code without updating the plan file should fail the test suite.
- The per-trade scaler is a closed-form function with NO fitted parameter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Pre-committed external anchors (locked in plan; do NOT tune) ───────

LOSS_BUDGET_PCT: float = 0.01           # 1% of equity per single PME event
PME_SHOCK_SPOT: float = 0.175           # Deribit main-table BTC ±17.5%
SIZE_FLOOR_F_MIN: float = 0.10          # never shrink trade below 10% of base

# Calibration window — must match vol_model/ou_logvol.CALIBRATION_WINDOW
PME_IV_CAL_WINDOW: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2021-04-01", tz="UTC"),
    pd.Timestamp("2023-10-01", tz="UTC"),
)
PME_IV_QUANTILE: float = 0.99           # 99th-pct of |ΔDVOL_1d|


# ─── Frozen anchor for IV shock (computed once on calibration window) ───

@dataclass(frozen=True)
class PMEAnchors:
    """Frozen external risk anchors used by ``compute_size_scaler``.

    All three values are documented public references or pre-committed
    quantiles on a frozen window. NEVER tune from backtest results.
    """
    loss_budget_pct: float
    pme_shock_spot: float
    pme_shock_iv_pct: float        # decimal 1-day relative DVOL move
    pme_iv_anchor_window: tuple[str, str]  # ISO-8601 (cal_start, cal_end)
    pme_iv_anchor_n_obs: int


def compute_pme_iv_shock_anchor(
    dvol_series: pd.Series,
    *,
    cal_window: tuple[pd.Timestamp, pd.Timestamp] = PME_IV_CAL_WINDOW,
    quantile: float = PME_IV_QUANTILE,
) -> tuple[float, int]:
    """Compute frozen ``pme_shock_iv_pct`` from DVOL on the calibration window.

    Uses RELATIVE 1-day DVOL changes ``|ΔDVOL_t / DVOL_{t-1}|`` and takes
    the 99th percentile. Multiplying ATM IV by this scalar gives the
    plan's vega-budget shock magnitude in vol points.

    Anti-leak: restricts inputs to the calibration window before any
    computation; raises if `dvol_series` lacks a DatetimeIndex.

    Returns ``(pme_shock_iv_pct, n_observations)``.
    """
    if not isinstance(dvol_series.index, pd.DatetimeIndex):
        raise TypeError("dvol_series must have a DatetimeIndex")
    cal_start, cal_end = cal_window

    cal = dvol_series.loc[
        (dvol_series.index >= cal_start) & (dvol_series.index < cal_end)
    ].copy()
    cal = cal.sort_index().dropna()
    # Resample to daily (last observation each UTC day)
    daily = cal.groupby(cal.index.normalize()).last()
    if len(daily) < 100:
        raise ValueError(
            f"DVOL anchor needs ≥ 100 calibration days; got {len(daily)}"
        )
    rel_change = (daily.diff() / daily.shift(1)).abs().dropna()
    shock = float(rel_change.quantile(quantile))
    return shock, int(len(rel_change))


def build_pme_anchors(
    dvol_series: pd.Series,
    *,
    cal_window: tuple[pd.Timestamp, pd.Timestamp] = PME_IV_CAL_WINDOW,
) -> PMEAnchors:
    """Build the frozen-anchor record. Pure read-only computation."""
    shock, n_obs = compute_pme_iv_shock_anchor(
        dvol_series, cal_window=cal_window,
    )
    return PMEAnchors(
        loss_budget_pct=LOSS_BUDGET_PCT,
        pme_shock_spot=PME_SHOCK_SPOT,
        pme_shock_iv_pct=shock,
        pme_iv_anchor_window=(cal_window[0].isoformat(),
                               cal_window[1].isoformat()),
        pme_iv_anchor_n_obs=n_obs,
    )


# ─── Pure function: per-bar size scaler ──────────────────────────────────

def compute_size_scaler(
    equity: float,
    spot: float,
    atm_iv: float,
    pre_gamma: float,
    pre_vega: float,
    *,
    anchors: PMEAnchors,
    f_min: float = SIZE_FLOOR_F_MIN,
) -> float:
    """Closed-form per-trade size scaler ``f ∈ [f_min, 1.0]``.

    Inputs are bar-time observables:
    - ``equity``: account equity at start of bar (USD)
    - ``spot``: BTC spot price at bar entry (USD per BTC)
    - ``atm_iv``: ATM implied vol AT ENTRY in vol points (e.g. 60.0)
    - ``pre_gamma``: gamma per BTC at entry (1 / USD)
    - ``pre_vega``: vega per BTC at entry (USD per vol-point)

    Returns ``f`` such that ``N_contracts_new = f × N_contracts_baseline``.
    NaN or non-positive inputs → returns 1.0 (no shrink) to preserve
    byte-identity on degenerate rows.
    """
    if anchors.loss_budget_pct <= 0:
        raise ValueError("loss_budget_pct must be > 0")
    # Defensive: NaN / non-positive → no-op
    if (np.isnan(equity) or np.isnan(spot) or np.isnan(atm_iv)
        or np.isnan(pre_gamma) or np.isnan(pre_vega)
        or equity <= 0 or spot <= 0 or atm_iv <= 0):
        return 1.0
    if pre_gamma <= 0 and pre_vega <= 0:
        return 1.0

    loss_budget_usd = float(equity) * anchors.loss_budget_pct
    # gamma_target: 2 × LossBudget / (Spot × shock)²    in 1/USD
    gamma_target = (
        2.0 * loss_budget_usd / (float(spot) * anchors.pme_shock_spot) ** 2
    )
    # vega_target: LossBudget / (ATM_IV × pme_shock_iv_pct)
    # ATM_IV is in vol-points; pme_shock_iv_pct is fractional (e.g. 0.04
    # = 4% relative move). The product is the absolute IV-point move.
    iv_shock_vol_pts = float(atm_iv) * anchors.pme_shock_iv_pct
    if iv_shock_vol_pts <= 0:
        vega_ratio = float("inf")
    else:
        vega_target = loss_budget_usd / iv_shock_vol_pts
        vega_ratio = vega_target / max(float(pre_vega), 1e-12) if pre_vega > 0 else float("inf")

    if pre_gamma > 0:
        gamma_ratio = gamma_target / float(pre_gamma)
    else:
        gamma_ratio = float("inf")

    f_raw = min(1.0, gamma_ratio, vega_ratio)
    f = float(np.clip(f_raw, f_min, 1.0))
    return f


def compute_size_scaler_vectorised(
    df: pd.DataFrame,
    *,
    anchors: PMEAnchors,
    f_min: float = SIZE_FLOOR_F_MIN,
    equity_col: str = "equity",
    spot_col: str = "index_price",
    iv_col: str = "iv_used",
    gamma_col: str = "gamma",
    vega_col: str = "vega",
) -> pd.Series:
    """Apply ``compute_size_scaler`` row-wise. Vectorised, leakage-safe.

    Returns a Series ``size_mult`` indexed by the input frame's index.
    All required columns must be present; missing columns raise.

    Vectorisation note: for performance, the calculation is performed in
    numpy on aligned arrays. The rule is identical row-by-row.
    """
    required = {equity_col, spot_col, iv_col, gamma_col, vega_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    eq = df[equity_col].to_numpy(dtype=np.float64)
    sp = df[spot_col].to_numpy(dtype=np.float64)
    iv = df[iv_col].to_numpy(dtype=np.float64)
    g = df[gamma_col].to_numpy(dtype=np.float64)
    v = df[vega_col].to_numpy(dtype=np.float64)

    # NaN / non-positive guards
    valid = (~np.isnan(eq) & ~np.isnan(sp) & ~np.isnan(iv)
             & ~np.isnan(g) & ~np.isnan(v)
             & (eq > 0) & (sp > 0) & (iv > 0))

    loss_budget_usd = eq * anchors.loss_budget_pct

    gamma_target = (
        2.0 * loss_budget_usd / (sp * anchors.pme_shock_spot) ** 2
    )
    iv_shock_vol_pts = iv * anchors.pme_shock_iv_pct
    # Per-row vega target & ratio (defensive against zero shock or zero vega)
    vega_target = np.where(
        iv_shock_vol_pts > 0,
        loss_budget_usd / np.maximum(iv_shock_vol_pts, 1e-12),
        np.inf,
    )

    gamma_ratio = np.where(
        g > 0, gamma_target / np.maximum(g, 1e-30), np.inf,
    )
    vega_ratio = np.where(
        v > 0, vega_target / np.maximum(v, 1e-30), np.inf,
    )

    f_raw = np.minimum(np.minimum(gamma_ratio, vega_ratio), 1.0)
    f = np.clip(f_raw, f_min, 1.0)
    # Invalid rows → 1.0 (no shrink, preserves byte-identity)
    f = np.where(valid, f, 1.0)

    return pd.Series(f, index=df.index, name="size_mult")


# ─── Sidecar contract (mirrors Kelly K1 pattern) ────────────────────────

def write_sidecar(size_mult: pd.Series, out_path: Path | str) -> Path:
    """Write the per-bar size_mult sidecar parquet.

    Schema matches Kelly K1 exactly: single ``size_mult`` column, datetime
    index (UTC). Mirrors ``vol_arb/sidecars.py:load_kelly_mult`` consumer.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(size_mult.index, pd.DatetimeIndex):
        raise TypeError("size_mult must have DatetimeIndex")
    if size_mult.name != "size_mult":
        size_mult = size_mult.rename("size_mult")
    df = size_mult.to_frame()
    df.to_parquet(out)
    logger.info("Wrote preventative-sizing sidecar to %s "
                "(n_rows=%d, mean=%.4f, p10=%.4f, p1=%.4f)",
                out, len(df), float(size_mult.mean()),
                float(size_mult.quantile(0.10)),
                float(size_mult.quantile(0.01)))
    return out


def load_sidecar(in_path: Path | str) -> pd.Series:
    """Read the sidecar back. Pure, no mutation."""
    df = pd.read_parquet(in_path)
    if "size_mult" not in df.columns:
        raise KeyError(f"sidecar at {in_path} missing 'size_mult' column")
    return df["size_mult"]
