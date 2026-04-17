"""Full statistics computation for backtest results.

Extracted from experiments/multi_approach_comparison.py (lines 244-344).
Computes Sharpe, Sortino, Calmar, CAGR, max DD, win rate, profit factor,
expectancy, PSR, skewness, kurtosis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def compute_full_stats(
    daily_pnl: pd.Series,
    starting_capital: float,
    n_trades: int,
    trades_per_day: float,
    final_equity: float | None = None,
) -> dict[str, Any]:
    """Compute all standard quant metrics from a daily PnL series.

    Args:
        daily_pnl: Daily PnL in dollar terms (sum of trade PnLs per day).
        starting_capital: Starting equity (used to normalise returns).
        n_trades: Total number of executed trades.
        trades_per_day: Average trades per calendar day.

    Returns:
        Dictionary of metrics.  Contains ``"error"`` key if fewer than 5
        days of data.
    """
    daily_ret = daily_pnl / starting_capital
    n_days = len(daily_ret)

    if n_days < 5:
        return {"error": "insufficient_days", "n_days": n_days}

    # Core
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    sharpe = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0.0

    downside = daily_ret[daily_ret < 0].std()
    sortino = (
        mean_r / downside * np.sqrt(365.25)
        if (downside is not None and downside > 0)
        else 0.0
    )

    equity = (1 + daily_ret).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    calmar = (mean_r * 365.25) / max_dd if max_dd > 0 else 0.0

    total_ret = equity.iloc[-1] - 1 if len(equity) > 0 else 0.0
    years = n_days / 365.25
    cagr = (
        (1 + total_ret) ** (1 / years) - 1
        if years > 0 and total_ret > -1
        else 0.0
    )

    # Win / Loss (daily)
    win_days = (daily_ret > 0).sum()
    n_days_total = n_days
    win_rate_days = win_days / n_days_total if n_days_total > 0 else 0.0

    winning = daily_ret[daily_ret > 0]
    losing = daily_ret[daily_ret < 0]
    avg_win = winning.mean() * 10000 if len(winning) > 0 else 0.0  # bps
    avg_loss = losing.mean() * 10000 if len(losing) > 0 else 0.0  # bps
    profit_factor = (
        winning.sum() / abs(losing.sum()) if losing.sum() != 0 else float("inf")
    )
    expectancy = mean_r * 10000  # bps per day

    # Risk
    ann_std = std_r * np.sqrt(365.25)
    ann_var = ann_std ** 2

    # Drawdown recovery (days from max DD to new high)
    dd_recovery = -1
    if len(drawdown) > 0:
        dd_idx = drawdown.idxmin()
        post_dd = equity.loc[dd_idx:]
        recovery_idx = post_dd[post_dd >= running_max.loc[dd_idx]].index
        if len(recovery_idx) > 0:
            try:
                dd_recovery = (recovery_idx[0] - dd_idx).days
            except (AttributeError, TypeError):
                dd_recovery = -1

    # Statistical
    skew = float(daily_ret.skew()) if len(daily_ret) > 3 else 0.0
    kurt = (
        float(daily_ret.kurtosis() + 3) if len(daily_ret) > 4 else 3.0
    )  # regular (not excess) kurtosis

    # PSR (Probabilistic Sharpe Ratio)
    sr_unann = sharpe / np.sqrt(365.25)
    denom_inner = (
        1 - skew * sr_unann + ((kurt - 1) / 4) * sr_unann ** 2
    ) / max(n_days - 1, 1)
    denom = np.sqrt(max(denom_inner, 0))
    psr = norm.cdf(sr_unann / denom) * 100 if denom > 0 else 50.0

    if final_equity is None:
        final_equity = (1 + total_ret) * starting_capital
    if not np.isfinite(total_ret) and np.isfinite(final_equity) and starting_capital > 0:
        total_ret = (final_equity / starting_capital) - 1

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "cagr_pct": round(cagr * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_days": round(win_rate_days, 4),
        "loss_rate_days": round(1 - win_rate_days, 4),
        "profit_factor": round(min(profit_factor, 99.0), 3),
        "avg_win_bps": round(avg_win, 1),
        "avg_loss_bps": round(avg_loss, 1),
        "expectancy_bps": round(expectancy, 2),
        "ann_std": round(ann_std, 4),
        "ann_var": round(ann_var, 6),
        "dd_recovery_days": dd_recovery,
        "psr_pct": round(psr, 1),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 1),
        "n_days": n_days,
        "n_trades": n_trades,
        "trades_per_day": round(trades_per_day, 1),
        "final_equity": round(final_equity, 2),
    }


def compute_rolling_sharpe(
    daily_pnl: pd.Series,
    starting_capital: float,
    window_days: int = 20,
) -> float | None:
    """Compute rolling Sharpe ratio for the most recent *window_days*.

    Args:
        daily_pnl: Daily PnL in dollar terms.
        starting_capital: Used to normalise PnL to returns.
        window_days: Number of calendar days in the lookback window.

    Returns:
        Annualised Sharpe for the window, or ``None`` if fewer than
        *window_days* data points are available.
    """
    if len(daily_pnl) < window_days:
        return None
    recent = daily_pnl.iloc[-window_days:]
    daily_ret = recent / starting_capital
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    if std_r <= 0:
        return 0.0
    return round(float(mean_r / std_r * np.sqrt(365.25)), 3)
