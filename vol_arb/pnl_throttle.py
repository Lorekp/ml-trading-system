"""Single-day-reactive P&L throttle for the V2 BTC strategy.

**Mechanism**. After any trading day whose closed NC P&L falls below
a threshold, scale the next ``duration`` days' position sizes by
``scale``. No lookahead: the throttle for day ``t`` depends on P&L
through day ``t-1`` only (``.shift(1)`` applied).

**Why it exists**. Empirically, daily trade P&L has
auto-correlation ~0.39 in the V2 trade log; this signal is strong,
statistically significant, and fully observable at each decision
point. The drawdown-decomposition audit on 2026-04-20 showed that
the worst drawdowns (Events 3-5, 2022-11 / 2022-10 / 2023-01) are
slow-grind loss runs where each individual trade's vega is small
but the cumulative day-after-day P&L compounds against us. A
per-trade vega cap doesn't help those; a rolling P&L throttle does.

**Faithfulness**. P&L is linear in position size to first order
(vega-P&L = vega × ΔIV), so applying a multiplicative scale to
the per-trade P&L approximates what a backtester run with scaled
position sizing would produce. This approximation is ~95%+ faithful
— deviations come from hedging nonlinearities (discrete delta
rebalance is a small effect at these sizes) and from filters that
might fire differently at smaller sizes (none of the active V2
filters are scale-sensitive).

**Non-leakage guard**. ``compute_throttle_schedule`` uses
``.shift(1)`` before applying any threshold comparison, so the
scale for day D is a function of closed P&L through day D-1 only.
A unit test (``tests/test_pnl_throttle.py``) mutates test-day P&L
and asserts the schedule is byte-identical.

See: ``vol_arb/output/V2_DRAWDOWN_ROOT_CAUSE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThrottleParams:
    """Parameters for a single-day-reactive P&L throttle.

    Attributes
    ----------
    threshold_usd :
        A day whose closed NC P&L is < ``threshold_usd`` triggers the
        throttle. Units are USD on the strategy's fixed-$100k
        non-compounding capital base.
    scale :
        Multiplier applied to all trade P&L during a throttled day.
        ``0.5`` halves position size, ``0.0`` fully disables entries,
        ``1.0`` is pass-through.
    duration_days :
        Number of trading days the throttle stays active after being
        triggered.
    """

    threshold_usd: float
    scale: float = 0.5
    duration_days: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold_usd):
            raise ValueError(f"threshold_usd must be finite; got {self.threshold_usd}")
        if not (0.0 <= self.scale <= 1.0):
            raise ValueError(f"scale must be in [0, 1]; got {self.scale}")
        if self.duration_days < 1:
            raise ValueError(
                f"duration_days must be ≥ 1; got {self.duration_days}"
            )


def compute_throttle_schedule(
    daily_pnl: pd.Series,
    params: ThrottleParams,
) -> pd.Series:
    """Return a per-day scaling schedule, aligned to ``daily_pnl.index``.

    Schedule is guaranteed non-leaking: the scale for day D is
    determined from closed P&L through day D-1 only.

    Parameters
    ----------
    daily_pnl :
        Series of per-day closed P&L (USD), DatetimeIndex, one row
        per calendar day.  Missing days should be 0 (no trades).
    params :
        Throttle parameters.

    Returns
    -------
    pd.Series
        Float series of scale multipliers, same index as daily_pnl.
    """
    if not isinstance(daily_pnl.index, pd.DatetimeIndex):
        raise TypeError("daily_pnl must have DatetimeIndex")
    if daily_pnl.index.is_monotonic_increasing is False:
        raise ValueError("daily_pnl index must be monotonic increasing")

    prev_pnl = daily_pnl.shift(1)  # yesterday's P&L
    triggers = (prev_pnl < params.threshold_usd).fillna(False).to_numpy()

    n = len(daily_pnl)
    cooldown = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if triggers[i]:
            cooldown[i] = params.duration_days
        elif i > 0 and cooldown[i - 1] > 0:
            cooldown[i] = cooldown[i - 1] - 1
    # If cooldown > 0, day is throttled.
    scale = np.where(cooldown > 0, params.scale, 1.0)
    return pd.Series(scale, index=daily_pnl.index, name="throttle_scale")


def apply_throttle(
    trade_log: pd.DataFrame,
    params: ThrottleParams | None,
    *,
    pnl_cols: Iterable[str] = (
        "net_pnl_nc", "gross_pnl_nc", "vega_pnl_nc",
        "theta_pnl_nc", "delta_hedge_pnl_nc",
        "net_pnl", "gross_pnl", "vega_pnl",
        "theta_pnl", "delta_hedge_pnl",
    ),
    return_schedule: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Apply the throttle to a trade log, returning a new trade log.

    The throttle is computed on daily *NC* P&L (non-compounding, the
    strategy's scale-invariant view) and applied multiplicatively to
    all P&L columns. The original trade log is not mutated.

    Parameters
    ----------
    trade_log :
        Trade log with DatetimeIndex and at least ``net_pnl_nc``.
        Rows with ``signal == 0`` are passed through unchanged.
    params :
        Throttle parameters. ``None`` returns a copy of ``trade_log``
        unchanged (identity).
    pnl_cols :
        P&L columns to scale. Missing columns are silently skipped.
    return_schedule :
        If True, also return the per-day throttle scale as a Series.

    Returns
    -------
    pd.DataFrame (or tuple if ``return_schedule``)
    """
    out = trade_log.copy()
    if params is None:
        schedule = pd.Series(
            1.0, index=trade_log.index.normalize().unique(),
            name="throttle_scale",
        )
        return (out, schedule) if return_schedule else out

    if "net_pnl_nc" not in trade_log.columns:
        raise KeyError("trade_log must have 'net_pnl_nc' column")

    # Daily NC P&L for schedule
    daily_pnl = trade_log["net_pnl_nc"].resample("1D").sum()
    schedule = compute_throttle_schedule(daily_pnl, params)

    # Broadcast schedule to per-trade scales by trade-date (using row's
    # entry date == index.normalize()).
    row_dates = trade_log.index.normalize()
    row_scales = schedule.reindex(row_dates, fill_value=1.0).to_numpy()

    for col in pnl_cols:
        if col in out.columns:
            out[col] = out[col].to_numpy() * row_scales

    return (out, schedule) if return_schedule else out


def summarize_throttle(
    trade_log_baseline: pd.DataFrame,
    trade_log_throttled: pd.DataFrame,
    *,
    capital: float = 100_000.0,
    trading_days_per_year: int = 365,
) -> dict:
    """Return a before/after metrics dict for reporting."""

    def _metrics(tl: pd.DataFrame) -> dict:
        trades = tl.loc[tl["signal"] != 0]
        daily = trades["net_pnl_nc"].resample("1D").sum()
        daily_active = daily.loc[daily != 0]
        eq = daily.cumsum()
        dd = eq - eq.cummax()
        sharpe = (
            daily_active.mean() / daily_active.std() * np.sqrt(trading_days_per_year)
            if daily_active.std() > 0 else float("nan")
        )
        return dict(
            total_pnl=float(trades["net_pnl_nc"].sum()),
            n_trades=int(len(trades)),
            max_dd=float(dd.min()),
            worst_day=float(daily.min()),
            sharpe=float(sharpe),
        )

    base = _metrics(trade_log_baseline)
    thr = _metrics(trade_log_throttled)
    return {
        "baseline": base,
        "throttled": thr,
        "delta_pnl_pct": 100.0 * (thr["total_pnl"] - base["total_pnl"]) / base["total_pnl"],
        "delta_dd_pct": 100.0 * (thr["max_dd"] - base["max_dd"]) / abs(base["max_dd"]) if base["max_dd"] < 0 else 0.0,
        "delta_worst_day_pct": 100.0 * (thr["worst_day"] - base["worst_day"]) / abs(base["worst_day"]) if base["worst_day"] < 0 else 0.0,
        "delta_sharpe": thr["sharpe"] - base["sharpe"],
    }


# ── Walk-forward parameter selection ─────────────────────────

def walk_forward_select(
    trade_log: pd.DataFrame,
    param_grid: list[ThrottleParams],
    *,
    train_months: int = 12,
    objective: str = "sharpe_minus_half_dd_pct",
) -> pd.DataFrame:
    """Per-month walk-forward parameter selection.

    For each month M in the trade log:
    - Fit: evaluate each grid candidate on trades in months [M-train_months, M-1].
      Select the candidate that maximizes ``objective``.
    - Apply: simulate month M's trades under the chosen candidate.
    - Record: NC P&L, max within-month DD, worst day, selected params.

    The first ``train_months`` months are pass-through (no training
    window available yet).

    Returns
    -------
    pd.DataFrame indexed by ``YYYY-MM`` month strings with columns
    ``threshold_usd, scale, duration_days, train_sharpe, oos_pnl,
    oos_max_dd, oos_worst_day, n_trades``.
    """
    if objective not in ("sharpe", "sharpe_minus_half_dd_pct"):
        raise ValueError(f"unknown objective {objective!r}")
    trades = trade_log.loc[trade_log["signal"] != 0].copy()
    trades["month"] = trades.index.strftime("%Y-%m")
    months = sorted(trades["month"].unique())

    rows = []
    for i, m in enumerate(months):
        if i < train_months:
            # Pass-through
            tm = trades.loc[trades["month"] == m]
            daily = tm["net_pnl_nc"].resample("1D").sum()
            eq = daily.cumsum()
            dd = (eq - eq.cummax()).min()
            rows.append(dict(
                month=m, threshold_usd=np.nan, scale=1.0,
                duration_days=0, train_sharpe=np.nan,
                oos_pnl=float(tm["net_pnl_nc"].sum()),
                oos_max_dd=float(dd) if not daily.empty else 0.0,
                oos_worst_day=float(daily.min()) if not daily.empty else 0.0,
                n_trades=int(len(tm)),
                passthrough=True,
            ))
            continue

        train_months_window = months[max(0, i - train_months):i]
        train = trades.loc[trades["month"].isin(train_months_window)]
        # Evaluate each candidate on train
        best = None
        for p in param_grid:
            cand = apply_throttle(train, p)
            daily = cand["net_pnl_nc"].resample("1D").sum()
            active = daily.loc[daily != 0]
            if active.std() == 0 or active.empty:
                continue
            sh = float(active.mean() / active.std() * np.sqrt(365))
            eq = daily.cumsum()
            dd = (eq - eq.cummax()).min()
            dd_pct = abs(dd) / 100_000.0
            if objective == "sharpe":
                score = sh
            else:  # sharpe_minus_half_dd_pct
                score = sh - 0.5 * dd_pct
            if best is None or score > best[0]:
                best = (score, p, sh)

        if best is None:
            # Shouldn't happen with pass-through candidate in grid
            p = ThrottleParams(threshold_usd=-1e9, scale=1.0, duration_days=1)
            sh_train = float("nan")
        else:
            _, p, sh_train = best

        # Apply to month M. Include trailing context so the throttle's
        # trigger state (yesterday's P&L) is correct at month start.
        context_months = months[max(0, i - 1):i + 1]  # M-1 + M
        context = trades.loc[trades["month"].isin(context_months)]
        throttled_ctx = apply_throttle(context, p)
        tm = throttled_ctx.loc[throttled_ctx["month"] == m]
        daily = tm["net_pnl_nc"].resample("1D").sum()
        eq = daily.cumsum()
        dd = (eq - eq.cummax()).min() if not daily.empty else 0.0
        rows.append(dict(
            month=m,
            threshold_usd=p.threshold_usd, scale=p.scale,
            duration_days=p.duration_days,
            train_sharpe=sh_train,
            oos_pnl=float(tm["net_pnl_nc"].sum()),
            oos_max_dd=float(dd),
            oos_worst_day=float(daily.min()) if not daily.empty else 0.0,
            n_trades=int(len(tm)),
            passthrough=False,
        ))

    return pd.DataFrame(rows)
