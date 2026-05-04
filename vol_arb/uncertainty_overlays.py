"""5 pre-committed trade overlays driven by epistemic uncertainty.

Cycle 3 Phase P (plan: ``now-check-empirically-from-eager-bird.md``).

Mirrors the DFL Head Stage 28 pattern: 5 pre-committed strategies, ONE
backtest run, gated. All filter quantile cuts and scaler reference scalars
are FROZEN on the calibration window 2021-04-01 → 2023-09-30 — same
discipline as Cycle 2 regime-tercile cuts.

Three filters (drop trades whose σ²_epi at entry exceeds a frozen
calibration-window quantile):
  - F1: quantile = 0.99 (drop top-1%)
  - F2: quantile = 0.95 (drop top-5%)
  - F3: quantile = 0.90 (drop top-10%)

Two scalers (shrink position size at entry by Bayesian-shrinkage factor):
  - S1: f = 1 / (1 + σ²_epi / median_cal(σ²_epi))   (no free param)
  - S2: f = σ²_train / (σ²_train + σ²_epi)           (Bayesian-Kelly,
        Murphy 2012 §3 closed-form, σ²_train frozen on cal window)

Both scalers floored at f_min = 0.10 to preserve the no-zero-tradeable
guarantee that bootstrap-equivalence relies on.

Anti-leakage:
- Quantile cuts and σ²_train computed ONCE on the calibration window;
  applied frozen to the evaluation window.
- σ²_epi at bar T comes from the deep-ensemble's OOF predictions which
  are themselves leakage-safe by ``PurgedKFold`` (with embargo) inside
  ``vol_model.uncertainty.train_har_pd_ensemble``.

Anti-overfitting:
- 5 strategies, locked in plan; do NOT add a 6th.
- F1/F2/F3 quantiles locked at {0.99, 0.95, 0.90}; do NOT sweep.
- S1, S2 functional forms have NO free parameters.
- f_min = 0.10 matches preventative-sizing floor (Cycle 2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Pre-committed parameters (locked in plan) ─────────────────────────

FILTER_QUANTILES: dict[str, float] = {
    "F1": 0.99,   # drop top-1%
    "F2": 0.95,   # drop top-5%
    "F3": 0.90,   # drop top-10%
}

SIZE_FLOOR_F_MIN: float = 0.10
EPSILON_DENOM: float = 1e-12

CAL_WINDOW: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2021-04-01", tz="UTC"),
    pd.Timestamp("2023-10-01", tz="UTC"),
)


# ─── Anchors computed on calibration window ───────────────────────────

@dataclass(frozen=True)
class UncertaintyAnchors:
    """Frozen anchors used by all 5 overlays.

    All values computed ONCE on the calibration window and propagated
    to overlay generators. NEVER recompute on eval-window data.
    """
    quantile_F1: float    # σ²_epi cut at q=0.99 on cal window
    quantile_F2: float    # σ²_epi cut at q=0.95 on cal window
    quantile_F3: float    # σ²_epi cut at q=0.90 on cal window
    median_cal: float     # median(σ²_epi) on cal window — denominator for S1
    mean_cal: float       # mean(σ²_epi) on cal window — σ²_train for S2
    cal_n_obs: int
    cal_window_start: str
    cal_window_end: str


def compute_anchors(
    sigma_epi_squared: pd.Series,
    *,
    cal_window: tuple[pd.Timestamp, pd.Timestamp] = CAL_WINDOW,
) -> UncertaintyAnchors:
    """Compute frozen anchors from σ²_epi on calibration window only.

    Anti-leak: explicitly restricts to ``cal_window``; raises if input
    lacks DatetimeIndex.
    """
    if not isinstance(sigma_epi_squared.index, pd.DatetimeIndex):
        raise TypeError("sigma_epi_squared must have a DatetimeIndex")
    cal_start, cal_end = cal_window
    cal = sigma_epi_squared.loc[
        (sigma_epi_squared.index >= cal_start)
        & (sigma_epi_squared.index < cal_end)
    ].dropna()
    if len(cal) < 1000:
        raise ValueError(
            f"calibration slice has {len(cal)} bars; need ≥ 1000"
        )
    return UncertaintyAnchors(
        quantile_F1=float(cal.quantile(FILTER_QUANTILES["F1"])),
        quantile_F2=float(cal.quantile(FILTER_QUANTILES["F2"])),
        quantile_F3=float(cal.quantile(FILTER_QUANTILES["F3"])),
        median_cal=float(cal.median()),
        mean_cal=float(cal.mean()),
        cal_n_obs=int(len(cal)),
        cal_window_start=cal_start.isoformat(),
        cal_window_end=cal_end.isoformat(),
    )


# ─── Filter overlays (F1 / F2 / F3) ────────────────────────────────────

def compute_filter_mask(
    sigma_epi_squared: pd.Series,
    anchors: UncertaintyAnchors,
    *,
    tag: str,
) -> pd.Series:
    """Boolean mask: True where σ²_epi exceeds the anchor's frozen quantile.

    True ⇒ DROP this bar's trade. Applied at entry; downstream backtester
    forces ``signal=0`` on True rows. Uses calibration-window cut applied
    uniformly to the full input (cal AND eval bars).

    Anti-leak: cut is FROZEN at anchor-build time; mutating eval-window
    σ²_epi after anchor construction does NOT change the cut.
    """
    if tag not in FILTER_QUANTILES:
        raise ValueError(f"tag must be one of {list(FILTER_QUANTILES)}, got {tag}")
    cut = {
        "F1": anchors.quantile_F1,
        "F2": anchors.quantile_F2,
        "F3": anchors.quantile_F3,
    }[tag]
    mask = sigma_epi_squared > cut
    return mask.fillna(False).rename(f"filter_{tag.lower()}")


# ─── Scaler overlays (S1 / S2) ─────────────────────────────────────────

def compute_scaler_S1(
    sigma_epi_squared: pd.Series,
    anchors: UncertaintyAnchors,
    *,
    f_min: float = SIZE_FLOOR_F_MIN,
) -> pd.Series:
    """S1: f = 1 / (1 + σ²_epi / median_cal(σ²_epi)), clipped to [f_min, 1].

    Properties:
    - σ²_epi = 0 → f = 1.0 (no shrink at zero uncertainty)
    - σ²_epi = median_cal → f = 0.5 (50% shrink at typical cal uncertainty)
    - σ²_epi → ∞ → f → 0 (clipped at f_min)
    - Monotonically decreasing in σ²_epi
    - NO free parameters (median is frozen)
    """
    denom = max(anchors.median_cal, EPSILON_DENOM)
    f_raw = 1.0 / (1.0 + sigma_epi_squared.fillna(0.0) / denom)
    f = f_raw.clip(lower=f_min, upper=1.0)
    return f.rename("size_mult")


def compute_scaler_S2(
    sigma_epi_squared: pd.Series,
    anchors: UncertaintyAnchors,
    *,
    f_min: float = SIZE_FLOOR_F_MIN,
) -> pd.Series:
    """S2: f = σ²_train / (σ²_train + σ²_epi), Bayesian-Kelly Murphy §3.

    σ²_train is the calibration-window mean σ²_epi (frozen scalar; the
    model's "typical" uncertainty under in-distribution data). The
    closed-form Bayesian posterior precision shrinkage:
    - σ²_epi = 0 → f = 1.0
    - σ²_epi = σ²_train → f = 0.5
    - σ²_epi → ∞ → f → 0 (clipped at f_min)
    - NO free parameters.
    """
    sigma_train = max(anchors.mean_cal, EPSILON_DENOM)
    f_raw = sigma_train / (sigma_train + sigma_epi_squared.fillna(0.0))
    f = f_raw.clip(lower=f_min, upper=1.0)
    return f.rename("size_mult")


# ─── Sidecar I/O ───────────────────────────────────────────────────────

def write_filter_sidecar(mask: pd.Series, out_path: Path | str) -> Path:
    """Write a boolean filter sidecar (True = drop trade)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(mask.index, pd.DatetimeIndex):
        raise TypeError("mask must have DatetimeIndex")
    df = mask.astype(bool).to_frame(name="drop_trade")
    df.to_parquet(out)
    logger.info("Wrote filter sidecar to %s (n=%d, drop_count=%d, %.2f%%)",
                out, len(df), int(df["drop_trade"].sum()),
                100.0 * float(df["drop_trade"].mean()))
    return out


def write_scaler_sidecar(size_mult: pd.Series, out_path: Path | str) -> Path:
    """Write a per-bar size_mult sidecar (matches Kelly K1 / preventative
    sizing contract)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(size_mult.index, pd.DatetimeIndex):
        raise TypeError("size_mult must have DatetimeIndex")
    if size_mult.name != "size_mult":
        size_mult = size_mult.rename("size_mult")
    df = size_mult.to_frame()
    df.to_parquet(out)
    logger.info("Wrote scaler sidecar to %s (n=%d, mean=%.4f, p10=%.4f, p1=%.4f)",
                out, len(df), float(size_mult.mean()),
                float(size_mult.quantile(0.10)),
                float(size_mult.quantile(0.01)))
    return out


def load_filter_sidecar(in_path: Path | str) -> pd.Series:
    df = pd.read_parquet(in_path)
    if "drop_trade" not in df.columns:
        raise KeyError(f"{in_path}: missing 'drop_trade' column")
    return df["drop_trade"].astype(bool)


def load_scaler_sidecar(in_path: Path | str) -> pd.Series:
    df = pd.read_parquet(in_path)
    if "size_mult" not in df.columns:
        raise KeyError(f"{in_path}: missing 'size_mult' column")
    return df["size_mult"]
