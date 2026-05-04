"""Synthetic-path + block-bootstrap evaluator for V2 candidate rules.

Cycle 2 Phase A infrastructure (plan: ``now-check-empirically-from-eager-bird.md``).

Two leakage-safe evaluation surfaces for a candidate sizing/path rule:

1. **Block-bootstrap**: resamples HISTORICAL daily NC PnL in 5-day blocks
   (preserves serial correlation; matches ``08_bootstrap_sharpe.py``
   methodology). Distributional metric: 10/50/90 percentile of
   (candidate - baseline) Sharpe / PnL / MaxDD over n_reps resamples.

2. **OU synthetic-path**: per fired trade, simulates ``n_paths`` alternative
   exit-IV trajectories starting from the trade's actual entry IV using the
   frozen ``OUParams2D``. Computes the dominant vega PnL term under each
   alternative path. Useful for path-dependent rules (stops, dynamic exits)
   and as a sensitivity check on the IV path for entry-only rules.

Anti-leakage guarantees:
- Bootstrap reads only historical daily PnL — no model, no future data.
- Synthetic mode uses entry-time IV (observable at entry by construction)
  as the OU starting state; the alternative path is generated forward
  from entry, so no future-IV information leaks into the evaluation.
- Frozen ``OUParams2D`` is calibrated strictly pre-evaluation-window
  (``vol_model.ou_logvol``); the evaluator does not re-fit.

Anti-overfitting guarantees:
- Selection metric is **10th-percentile** across reps/paths, not mean.
- Block-bootstrap block-size is pre-committed (5 days; matches plan).
- Adoption gate is loaded from caller; not negotiable in this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_model.ou_logvol import OUParams2D, simulate_paths

logger = logging.getLogger(__name__)

_DAYS_PER_YEAR: float = 365.25


# ─── Pre-committed evaluation parameters (locked in plan) ───────────────

DEFAULT_N_BOOTSTRAP_REPS: int = 1_000
DEFAULT_BLOCK_SIZE_DAYS: int = 5
DEFAULT_N_OU_PATHS: int = 1_000
DEFAULT_HOLDING_MINUTES: int = 15


@dataclass(frozen=True)
class DistributionSummary:
    """10/50/90 percentile + mean/std of a metric across reps/paths."""

    metric_name: str
    p10: float
    p50: float
    p90: float
    mean: float
    std: float
    n_reps: int


# ─── Block-bootstrap on daily PnL ────────────────────────────────────────

def _annualised_sharpe(daily_pnl: np.ndarray) -> float:
    """Sharpe = mean / std × sqrt(365.25); zero std → 0.0."""
    if len(daily_pnl) < 2:
        return 0.0
    mu = float(np.mean(daily_pnl))
    sd = float(np.std(daily_pnl, ddof=1))
    if sd <= 0.0:
        return 0.0
    return mu / sd * np.sqrt(_DAYS_PER_YEAR)


def _maxdd(daily_pnl: np.ndarray) -> float:
    """Cumulative-PnL drawdown; returns the most-negative trough (≤ 0)."""
    if len(daily_pnl) < 2:
        return 0.0
    cum = np.cumsum(daily_pnl)
    peaks = np.maximum.accumulate(cum)
    dd = cum - peaks
    return float(dd.min())


def _psr(daily_pnl: np.ndarray, sharpe_threshold: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2014).

    PSR = Φ((SR̂ - SR*) · sqrt(N - 1) / sqrt(1 - γ₃·SR̂ + γ₄/4·SR̂²))

    where SR̂ is observed Sharpe (not annualised — daily here),
    γ₃ is skewness, γ₄ is excess kurtosis. Larger PSR ⇔ stronger
    evidence that the true Sharpe exceeds ``sharpe_threshold``.
    """
    from scipy.special import ndtr
    n = len(daily_pnl)
    if n < 4:
        return 0.0
    mu = float(np.mean(daily_pnl))
    sd = float(np.std(daily_pnl, ddof=1))
    if sd <= 0:
        return 0.0
    sr_hat = mu / sd  # daily, NOT annualised
    g3 = float(_skewness(daily_pnl))
    g4 = float(_excess_kurtosis(daily_pnl))
    denom = np.sqrt(max(1.0 - g3 * sr_hat + (g4 / 4.0) * sr_hat ** 2, 1e-12))
    z = (sr_hat - sharpe_threshold) * np.sqrt(max(n - 1, 1)) / denom
    return float(ndtr(z))


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(((x - mu) ** 3).mean() / sd ** 3)


def _excess_kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(((x - mu) ** 4).mean() / sd ** 4) - 3.0


def block_bootstrap_deltas(
    baseline_daily: pd.Series,
    candidate_daily: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    block_size_days: int = DEFAULT_BLOCK_SIZE_DAYS,
    seed: int = 42,
) -> dict[str, DistributionSummary]:
    """Block-bootstrap distribution of (candidate − baseline) deltas.

    Both inputs must be aligned daily NC PnL series (date-indexed).
    The same resampled day-index is applied to both series, so the
    bootstrap measures the rule's incremental contribution holding
    realisation noise constant.

    Returns dict with DistributionSummary for each of:
    - sharpe_delta (annualised)
    - pnl_delta (cumulative NC)
    - maxdd_delta
    - psr_delta (PSR(candidate) - PSR(baseline) at SR* = 0)
    """
    if not isinstance(baseline_daily.index, pd.DatetimeIndex):
        raise TypeError("baseline_daily must have a DatetimeIndex")
    if not isinstance(candidate_daily.index, pd.DatetimeIndex):
        raise TypeError("candidate_daily must have a DatetimeIndex")

    common_idx = baseline_daily.index.intersection(candidate_daily.index)
    if len(common_idx) < 30:
        raise ValueError(
            f"Need at least 30 aligned days for bootstrap; got {len(common_idx)}"
        )
    baseline_a = baseline_daily.reindex(common_idx).to_numpy(dtype=np.float64)
    candidate_a = candidate_daily.reindex(common_idx).to_numpy(dtype=np.float64)

    n_days = len(common_idx)
    rng = np.random.default_rng(seed)
    n_blocks_needed = int(np.ceil(n_days / block_size_days))

    sharpe_deltas = np.empty(n_reps, dtype=np.float64)
    pnl_deltas = np.empty(n_reps, dtype=np.float64)
    maxdd_deltas = np.empty(n_reps, dtype=np.float64)
    psr_deltas = np.empty(n_reps, dtype=np.float64)

    for r in range(n_reps):
        starts = rng.integers(
            0, n_days - block_size_days + 1, size=n_blocks_needed,
        )
        idx_chunks = [
            np.arange(s, s + block_size_days) for s in starts
        ]
        sample_idx = np.concatenate(idx_chunks)[:n_days]
        b_sample = baseline_a[sample_idx]
        c_sample = candidate_a[sample_idx]

        sharpe_deltas[r] = _annualised_sharpe(c_sample) - _annualised_sharpe(b_sample)
        pnl_deltas[r] = float(c_sample.sum() - b_sample.sum())
        maxdd_deltas[r] = _maxdd(c_sample) - _maxdd(b_sample)
        psr_deltas[r] = _psr(c_sample) - _psr(b_sample)

    def summarise(name: str, arr: np.ndarray) -> DistributionSummary:
        return DistributionSummary(
            metric_name=name,
            p10=float(np.percentile(arr, 10)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)),
            n_reps=n_reps,
        )

    return {
        "sharpe_delta": summarise("sharpe_delta", sharpe_deltas),
        "pnl_delta": summarise("pnl_delta", pnl_deltas),
        "maxdd_delta": summarise("maxdd_delta", maxdd_deltas),
        "psr_delta": summarise("psr_delta", psr_deltas),
    }


# ─── OU synthetic-path mode ──────────────────────────────────────────────

def _to_decimal_iv(log_iv_path: np.ndarray) -> np.ndarray:
    """Convert log-decimal-IV path to vol-points (e.g. 60.0 = 60%)."""
    return np.exp(log_iv_path) * 100.0


def simulate_alternative_trade_pnl(
    trade_log: pd.DataFrame,
    ou_params: OUParams2D,
    *,
    n_paths: int = DEFAULT_N_OU_PATHS,
    holding_minutes: int = DEFAULT_HOLDING_MINUTES,
    seed: int = 42,
    n_contracts_override: pd.Series | None = None,
) -> np.ndarray:
    """For each fired trade, simulate alternative vega-PnL across OU paths.

    Returns a (n_paths, n_fired_trades) ndarray of alternative net vega
    PnL in USD. Per trade i and path k:

        alt_pnl[k, i] = signal[i] × vega[i] × N_contracts[i]
                          × (iv_exit_path_k[i] - iv_entry[i])

    Sign convention: a LONG straddle (signal = +1) earns PnL when IV rises;
    a SHORT (signal = -1) earns PnL when IV falls. Matches the canonical
    PnL identity used by the backtester (the audit script's "adverse"
    formula tracks loss magnitude, which is the negative of this).

    where ``iv_entry[i]`` is the trade's actual ``iv_used`` (observable
    at entry by construction) and ``iv_exit_path_k[i]`` is the simulated
    OU path's IV at minute ``holding_minutes``.

    The bivariate OU simulator is invoked once per fired trade with
    ``x0_iv = log(iv_entry / 100)`` (the bar-OPEN log-decimal-IV) and
    ``x0_rv = ou_params.theta_rv`` (long-run RV mean — RV path is not
    consumed for this PnL term but is required by the simulator API).

    Anti-leakage: ``iv_used`` at trade i is observable at entry; the
    simulator runs forward from entry so no future-IV is consumed.

    Anti-overfit: the OU parameters are pre-frozen on the calibration
    window; this function does not re-fit.

    If ``n_contracts_override`` is provided, it must be a Series indexed by
    trade timestamp (subset of trade_log.index) — used to evaluate a
    candidate sizing rule's per-trade scaler ``f × N_contracts``.
    """
    fired = trade_log[trade_log["net_pnl_nc"] != 0].copy()
    if len(fired) == 0:
        return np.empty((n_paths, 0), dtype=np.float64)
    required = {"signal", "vega", "N_contracts", "iv_used"}
    missing = required - set(fired.columns)
    if missing:
        raise KeyError(f"trade_log missing columns: {sorted(missing)}")

    if n_contracts_override is not None:
        # Align override to fired index, fall back to canonical N_contracts
        n_arr = n_contracts_override.reindex(fired.index).fillna(
            fired["N_contracts"]
        ).to_numpy(dtype=np.float64)
    else:
        n_arr = fired["N_contracts"].to_numpy(dtype=np.float64)

    signal_arr = fired["signal"].to_numpy(dtype=np.float64)
    vega_arr = fired["vega"].to_numpy(dtype=np.float64)
    iv_entry_pts = fired["iv_used"].to_numpy(dtype=np.float64)

    # x0 in log-decimal space: log(iv_entry / 100)
    x0_log_iv = np.log(np.clip(iv_entry_pts / 100.0, 1e-6, None))

    n_trades = len(fired)
    rng = np.random.default_rng(seed)

    # Vectorise across trades: simulate one big tensor (n_paths, n_trades+1, 2)
    # using block-rng draws. To keep memory tractable, batch trades.
    batch = 1_000  # trades per simulator call
    out = np.empty((n_paths, n_trades), dtype=np.float64)

    for start in range(0, n_trades, batch):
        end = min(start + batch, n_trades)
        seg_n = end - start
        # We want (n_paths, holding_minutes+1, 2) per trade with x0 differing
        # per trade. Simulate each trade in a small loop to stay simple/correct.
        for j, t_idx in enumerate(range(start, end)):
            # Fresh seed per trade so paths are independent across trades
            sub_seed = int(rng.integers(0, 2**31 - 1))
            paths = simulate_paths(
                ou_params,
                n_paths=n_paths,
                n_minutes=holding_minutes + 1,
                x0_iv=float(x0_log_iv[t_idx]),
                x0_rv=ou_params.theta_rv,
                seed=sub_seed,
            )
            iv_exit_pts = _to_decimal_iv(paths[:, holding_minutes, 0])
            d_iv = iv_exit_pts - iv_entry_pts[t_idx]  # vol-points
            out[:, t_idx] = (
                signal_arr[t_idx] * vega_arr[t_idx] * n_arr[t_idx] * d_iv
            )

    return out


def aggregate_path_to_daily(
    trade_log: pd.DataFrame,
    alt_pnl: np.ndarray,
) -> np.ndarray:
    """Aggregate per-trade alternative PnL to per-day across paths.

    Returns ndarray ``(n_paths, n_unique_days)``. The day-index is the
    unique normalised dates of the fired trades, in chronological order.
    """
    fired = trade_log[trade_log["net_pnl_nc"] != 0]
    if len(fired) != alt_pnl.shape[1]:
        raise ValueError(
            f"alt_pnl has {alt_pnl.shape[1]} trade-columns but "
            f"trade_log has {len(fired)} fired trades"
        )
    days = fired.index.normalize()
    unique_days = pd.DatetimeIndex(sorted(set(days)))
    day_to_col = {d: i for i, d in enumerate(unique_days)}
    n_paths = alt_pnl.shape[0]
    n_days = len(unique_days)
    out = np.zeros((n_paths, n_days), dtype=np.float64)
    cols = np.array([day_to_col[d] for d in days], dtype=np.intp)
    for k in range(n_paths):
        np.add.at(out[k], cols, alt_pnl[k])
    return out


def synthetic_path_metrics(
    daily_per_path: np.ndarray,
    metric: str = "sharpe",
) -> DistributionSummary:
    """Distribution of a metric across the OU paths.

    ``daily_per_path`` is shape ``(n_paths, n_days)``. Returns a
    DistributionSummary of the per-path metric.
    """
    n_paths = daily_per_path.shape[0]
    metrics = np.empty(n_paths, dtype=np.float64)
    if metric == "sharpe":
        for k in range(n_paths):
            metrics[k] = _annualised_sharpe(daily_per_path[k])
        name = "synthetic_sharpe"
    elif metric == "total_pnl":
        metrics = daily_per_path.sum(axis=1)
        name = "synthetic_total_pnl"
    elif metric == "maxdd":
        for k in range(n_paths):
            metrics[k] = _maxdd(daily_per_path[k])
        name = "synthetic_maxdd"
    else:
        raise ValueError(f"unknown metric: {metric}")
    return DistributionSummary(
        metric_name=name,
        p10=float(np.percentile(metrics, 10)),
        p50=float(np.percentile(metrics, 50)),
        p90=float(np.percentile(metrics, 90)),
        mean=float(metrics.mean()),
        std=float(metrics.std(ddof=1)),
        n_reps=n_paths,
    )
