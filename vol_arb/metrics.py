"""Professional-grade performance metrics for the vol-arb backtester.

Computes the full toolkit a quant fund would want to see:
  - Return metrics: CAGR, arithmetic/geometric mean
  - Risk metrics: vol, downside vol, VaR, CVaR, skew, kurtosis
  - Risk-adjusted: Sharpe, Sortino, Calmar, Omega, tail ratio, Martin
  - Drawdown: max, avg, duration, recovery time, time underwater
  - Trade stats: win rate, profit factor, expectancy, Kelly, streaks
  - Concentration: top-N contribution, Gini, Herfindahl
  - Stability: rolling Sharpe, R² of equity curve, monthly consistency
  - Significance: t-stat, p-value, DSR (deflated Sharpe)
  - Regime: per-year breakdown

Inputs are the trade log DataFrame produced by VolArbBacktester.run_backtest().
All metrics assume 15-min bars on a 24/7 crypto market (35,064 bars/year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BARS_PER_YEAR = 365.25 * 24 * 4  # 35,064 (15-min bars in a 24/7 year)
DAYS_PER_YEAR = 365.25


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────

def _equity_curve(trade_log: pd.DataFrame, initial_equity: float) -> pd.Series:
    """Return the compounded equity curve from the trade log."""
    if "equity" in trade_log.columns:
        return trade_log["equity"]
    # Fallback: compound from net_pnl
    return initial_equity + trade_log["net_pnl"].cumsum()


def _daily_pnl(trade_log: pd.DataFrame) -> pd.Series:
    """Aggregate net_pnl to daily bars (for daily-frequency stats)."""
    pnl = trade_log["net_pnl"]
    return pnl.groupby(pnl.index.normalize()).sum()


def _daily_returns(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> pd.Series:
    """Daily arithmetic returns.

    Two modes, selected by the explicit ``is_nc`` keyword:

    - **Compound (default, ``is_nc=False``)**: returns = ``equity.pct_change()``.
      This is correct for a true compounded equity curve where each
      day's return is reinvested into the next day's sizing.

    - **Non-compound (``is_nc=True``)**: returns = ``daily_pnl / initial_equity``.
      The non-compound "equity" column in the trade log is actually
      ``initial_equity + cumsum(net_pnl_nc)`` — a running offset, not a
      true equity curve.  Taking ``pct_change()`` on it divides by a
      drifting denominator (early days ÷$100K, late days ÷$689K) which
      produces neither compound nor true non-compound returns.  The
      correct definition is ``daily_pnl / fixed_reference``.

    ``is_nc`` is a REQUIRED (keyword-only) hint.  Callers that render
    the non-compound view must pass ``is_nc=True`` explicitly; callers
    on the compound view can use the default.  This removes the earlier
    dependency on ``trade_log.attrs``, which pandas documents as fragile
    (boolean indexing, concat, and many other ops drop attrs).
    """
    if is_nc:
        daily_pnl = _daily_pnl(trade_log)
        return (daily_pnl / initial_equity).dropna()
    eq = _equity_curve(trade_log, initial_equity)
    daily_eq = eq.groupby(eq.index.normalize()).last()
    return daily_eq.pct_change().dropna()


def _drawdown_series(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Return (daily_eq, drawdown_series) appropriate for the view mode.

    **Compound**: drawdown = ``(daily_eq - running_max) / running_max`` —
    the classic percent-from-peak of a compounding equity curve.

    **Non-compound**: drawdown = ``(daily_eq - running_max) / initial_equity``
    — peak-to-trough loss as a fraction of the fixed initial notional.
    The NC "equity" column is additive (``initial + cumsum(pnl_nc)``),
    so dividing by the running max would mix units (additive numerator,
    additive denominator-with-drift); the honest unit is percent of the
    fixed reference.

    Both series share the same event-detection logic downstream (a
    drawdown event is any contiguous run where the series is strictly
    below zero), so ``drawdown_metrics`` can consume either uniformly.
    """
    eq = _equity_curve(trade_log, initial_equity)
    daily_eq = eq.groupby(eq.index.normalize()).last()
    running_max = daily_eq.cummax()
    if is_nc:
        drawdowns = (daily_eq - running_max) / initial_equity
    else:
        drawdowns = (daily_eq - running_max) / running_max
    return daily_eq, drawdowns


# ────────────────────────────────────────────────────────────
# RETURN METRICS
# ────────────────────────────────────────────────────────────

def return_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> dict:
    """Return-level metrics.

    **Compound** reports CAGR as usual.  **Non-compound** sets
    ``cagr_pct = None`` because the NC "equity" column is additive —
    ``(final / initial) ** (1/years)`` on a shifted additive curve is
    not a compound growth rate in any meaningful sense.  The honest
    return summary for the NC view is ``ann_return_arith_pct``
    (arithmetic average of ``daily_pnl / initial_equity`` annualized),
    which is mathematically equivalent to ``total_pnl / initial_equity
    / years``.
    """
    eq = _equity_curve(trade_log, initial_equity)
    final_equity = float(eq.iloc[-1])
    total_pnl = final_equity - initial_equity
    total_return = final_equity / initial_equity - 1.0

    span_days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400.0
    years = span_days / DAYS_PER_YEAR

    if is_nc:
        cagr_pct: float | None = None
    else:
        cagr = (
            (final_equity / initial_equity) ** (1.0 / max(years, 1e-9)) - 1.0
            if final_equity > 0
            else -1.0
        )
        cagr_pct = cagr * 100.0

    # Arithmetic mean annualized from daily returns
    r_daily = _daily_returns(trade_log, initial_equity, is_nc=is_nc)
    mean_daily = float(r_daily.mean()) if len(r_daily) else 0.0
    ann_return_arith = mean_daily * DAYS_PER_YEAR

    return {
        "total_pnl": total_pnl,
        "final_equity": final_equity,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr_pct,
        "ann_return_arith_pct": ann_return_arith * 100.0,
        "years": years,
    }


# ────────────────────────────────────────────────────────────
# RISK METRICS
# ────────────────────────────────────────────────────────────

def risk_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> dict:
    r_daily = _daily_returns(trade_log, initial_equity, is_nc=is_nc)
    if len(r_daily) < 2:
        return {}

    vol_daily = float(r_daily.std(ddof=1))
    ann_vol = vol_daily * np.sqrt(DAYS_PER_YEAR)

    downside = r_daily[r_daily < 0]
    down_vol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    ann_down_vol = down_vol * np.sqrt(DAYS_PER_YEAR)

    # VaR / CVaR (Expected Shortfall) at 1% and 5%
    var_95 = float(r_daily.quantile(0.05))
    var_99 = float(r_daily.quantile(0.01))
    cvar_95 = float(r_daily[r_daily <= var_95].mean()) if (r_daily <= var_95).any() else 0.0
    cvar_99 = float(r_daily[r_daily <= var_99].mean()) if (r_daily <= var_99).any() else 0.0

    # Distribution shape
    skew = float(stats.skew(r_daily))
    kurt = float(stats.kurtosis(r_daily))  # excess kurtosis (normal = 0)

    # Tail ratio: 95th percentile abs / 5th percentile abs
    p95 = float(r_daily.quantile(0.95))
    tail_ratio = abs(p95) / abs(var_95) if var_95 < 0 else np.inf

    return {
        "ann_volatility_pct": ann_vol * 100.0,
        "ann_downside_vol_pct": ann_down_vol * 100.0,
        "daily_var_95_pct": var_95 * 100.0,
        "daily_var_99_pct": var_99 * 100.0,
        "daily_cvar_95_pct": cvar_95 * 100.0,
        "daily_cvar_99_pct": cvar_99 * 100.0,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "tail_ratio": tail_ratio,
    }


# ────────────────────────────────────────────────────────────
# RISK-ADJUSTED METRICS
# ────────────────────────────────────────────────────────────

def risk_adjusted_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> dict:
    """Sharpe / Sortino / Calmar / Omega / Martin / Ulcer.

    Calmar and Martin use the **arithmetic** annualized return in NC
    mode (CAGR is undefined on an additive curve — see
    :func:`return_metrics`).  Ulcer is the RMS of the view-appropriate
    drawdown series from :func:`_drawdown_series`, so it is expressed
    in the same units (percent of peak in compound, percent of initial
    equity in NC).
    """
    r_daily = _daily_returns(trade_log, initial_equity, is_nc=is_nc)
    if len(r_daily) < 2:
        return {}

    mean_daily = float(r_daily.mean())
    std_daily = float(r_daily.std(ddof=1))
    sharpe = (mean_daily / std_daily) * np.sqrt(DAYS_PER_YEAR) if std_daily > 0 else 0.0

    downside = r_daily[r_daily < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_daily / down_std) * np.sqrt(DAYS_PER_YEAR) if down_std > 0 else 0.0

    # Calmar uses CAGR in compound mode, arithmetic annualized return in NC.
    ret_stats = return_metrics(trade_log, initial_equity, is_nc=is_nc)
    dd_stats = drawdown_metrics(trade_log, initial_equity, is_nc=is_nc)
    max_dd = abs(dd_stats.get("max_drawdown_pct", 0.0)) / 100.0
    if is_nc:
        numerator_pct = ret_stats["ann_return_arith_pct"]
    else:
        numerator_pct = ret_stats["cagr_pct"]
    calmar = (numerator_pct / 100.0) / max_dd if max_dd > 0 else 0.0

    # Omega ratio: prob-weighted gains / prob-weighted losses at threshold 0
    gains = r_daily[r_daily > 0].sum()
    losses = abs(r_daily[r_daily < 0].sum())
    omega = float(gains / losses) if losses > 0 else np.inf

    # Martin: numerator / Ulcer Index (RMS of view-appropriate drawdowns)
    _, dd_series = _drawdown_series(trade_log, initial_equity, is_nc=is_nc)
    ulcer = float(np.sqrt(np.mean(dd_series ** 2)))
    martin = (numerator_pct / 100.0) / ulcer if ulcer > 0 else 0.0

    # Gain/loss ratio
    avg_gain = float(r_daily[r_daily > 0].mean()) if (r_daily > 0).any() else 0.0
    avg_loss = float(r_daily[r_daily < 0].mean()) if (r_daily < 0).any() else 0.0
    gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else np.inf

    return {
        "sharpe_daily": sharpe,
        "sortino_daily": sortino,
        "calmar": calmar,
        "omega": omega,
        "martin": martin,
        "gain_loss_ratio": gain_loss_ratio,
        "ulcer_index": ulcer * 100.0,  # as percentage
    }


# ────────────────────────────────────────────────────────────
# DRAWDOWN METRICS
# ────────────────────────────────────────────────────────────

def drawdown_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> dict:
    """Drawdown distribution.

    The drawdown series is view-appropriate (percent-of-peak for
    compound, percent-of-initial-equity for non-compound).  Event
    detection (contiguous below-zero runs) is identical in both modes.
    """
    daily_eq, drawdowns = _drawdown_series(
        trade_log, initial_equity, is_nc=is_nc,
    )

    max_dd = float(drawdowns.min())

    # Drawdown events: contiguous periods where equity < running max
    in_dd = drawdowns < -1e-9
    # Find start/end of each drawdown event
    starts, durations, depths, recoveries = [], [], [], []
    i = 0
    n = len(drawdowns)
    while i < n:
        if in_dd.iloc[i]:
            start_i = i
            peak_dd = drawdowns.iloc[i]
            while i < n and in_dd.iloc[i]:
                peak_dd = min(peak_dd, drawdowns.iloc[i])
                i += 1
            end_i = i - 1
            starts.append(daily_eq.index[start_i])
            durations.append(end_i - start_i + 1)
            depths.append(float(peak_dd))
            # Recovery time: days from peak to recovery (same as duration for now)
            recoveries.append(end_i - start_i + 1)
        else:
            i += 1

    avg_dd = float(np.mean(depths)) if depths else 0.0
    max_dd_duration = int(max(durations)) if durations else 0
    avg_dd_duration = float(np.mean(durations)) if durations else 0.0
    pct_time_in_dd = float(in_dd.mean()) if len(in_dd) else 0.0
    n_drawdown_events = len(depths)

    return {
        "max_drawdown_pct": max_dd * 100.0,
        "avg_drawdown_pct": avg_dd * 100.0,
        "max_dd_duration_days": max_dd_duration,
        "avg_dd_duration_days": avg_dd_duration,
        "pct_time_in_drawdown": pct_time_in_dd * 100.0,
        "n_drawdown_events": n_drawdown_events,
    }


def top_drawdown_episodes(
    trade_log: pd.DataFrame,
    initial_equity: float = 100_000,
    *,
    is_nc: bool = False,
    k: int = 10,
) -> list[dict]:
    """Return the K worst drawdown EPISODES with per-event detail.

    Complement to :func:`drawdown_metrics`, which computes aggregate
    statistics (max, avg, count) and drops the per-episode information
    the root-cause investigator actually needs.  This helper walks the
    same view-appropriate drawdown series and returns a list sorted by
    ``depth_pct`` ascending (deepest first — i.e., the most negative
    number first, which is the worst episode).

    Each episode dict contains:

    - ``start_ts``       : first day strictly below the running-max peak
    - ``trough_ts``      : day of minimum equity within the episode
    - ``recovery_ts``    : first day equity reaches a new high
      (``None`` if the series ends while still under water)
    - ``depth_pct``      : most-negative point of the episode, in the
      same units as :func:`drawdown_metrics` (percent-of-peak for
      compound, percent-of-initial-equity for NC); already signed so
      ``depth_pct < 0`` and deeper episodes are more negative
    - ``duration_days``  : number of days the equity stayed below peak
    - ``trades_count``   : count of rows with ``signal != 0`` whose
      timestamps fall in ``[start_ts, recovery_ts)`` — or in
      ``[start_ts, last_ts + 1d)`` when the episode is still open at
      series end

    Parameters
    ----------
    trade_log, initial_equity, is_nc
        Same semantics as :func:`drawdown_metrics`.
    k
        Maximum episodes to return; deepest first.  If fewer events
        exist, returns all of them (possibly an empty list).
    """
    daily_eq, drawdowns = _drawdown_series(
        trade_log, initial_equity, is_nc=is_nc,
    )
    in_dd = drawdowns < -1e-9
    n = len(drawdowns)

    # Timestamps of ACTIVE trades — used for per-episode trade counts.
    # If there's no ``signal`` column (e.g., synthetic test fixture that
    # omits it), fall back to every row.
    if "signal" in trade_log.columns:
        active_ts = trade_log.index[trade_log["signal"] != 0]
    else:
        active_ts = trade_log.index
    active_ts = pd.DatetimeIndex(active_ts).sort_values()

    episodes: list[dict] = []
    i = 0
    while i < n:
        if not in_dd.iloc[i]:
            i += 1
            continue
        start_i = i
        trough_i = i
        trough_val = float(drawdowns.iloc[i])
        while i < n and in_dd.iloc[i]:
            dd_val = float(drawdowns.iloc[i])
            if dd_val < trough_val:
                trough_val = dd_val
                trough_i = i
            i += 1
        end_i = i - 1

        start_ts = daily_eq.index[start_i]
        trough_ts = daily_eq.index[trough_i]
        if end_i + 1 < n:
            recovery_ts = daily_eq.index[end_i + 1]
            window_end = recovery_ts
        else:
            recovery_ts = None
            window_end = daily_eq.index[end_i] + pd.Timedelta(days=1)

        trades_count = int(
            ((active_ts >= start_ts) & (active_ts < window_end)).sum()
        )

        episodes.append({
            "start_ts": start_ts,
            "trough_ts": trough_ts,
            "recovery_ts": recovery_ts,
            "depth_pct": trough_val * 100.0,
            "duration_days": int(end_i - start_i + 1),
            "trades_count": trades_count,
        })

    # Ascending sort on depth_pct — most negative (deepest DD) first.
    episodes.sort(key=lambda d: d["depth_pct"])
    return episodes[:k]


# ────────────────────────────────────────────────────────────
# TRADE STATISTICS
# ────────────────────────────────────────────────────────────

def trade_stats(trade_log: pd.DataFrame) -> dict:
    active = trade_log[trade_log["signal"] != 0].copy()
    if active.empty:
        return {"n_trades": 0}

    pnl = active["net_pnl"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n = len(pnl)
    n_wins = len(wins)
    n_losses = len(losses)

    win_rate = n_wins / n if n > 0 else 0.0
    gross_profit = float(wins.sum()) if n_wins else 0.0
    gross_loss = float(abs(losses.sum())) if n_losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_win = float(wins.mean()) if n_wins else 0.0
    avg_loss = float(losses.mean()) if n_losses else 0.0
    avg_trade = float(pnl.mean())
    median_trade = float(np.median(pnl))

    # Expectancy: avg R per trade
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

    # Kelly fraction (simplified: f* = W - (1-W)/R where R = avg_win/|avg_loss|)
    if avg_loss < 0:
        R = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / R if R > 0 else 0.0
    else:
        kelly = 0.0

    # Streak analysis
    signs = np.sign(pnl).astype(int)
    max_win_streak = 0
    max_loss_streak = 0
    cur_win, cur_loss = 0, 0
    for s in signs:
        if s > 0:
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        elif s < 0:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
        else:
            cur_win, cur_loss = 0, 0

    # PnL distribution percentiles (in USD)
    percentiles = {f"pnl_p{p}": float(np.percentile(pnl, p)) for p in [1, 5, 25, 50, 75, 95, 99]}

    return {
        "n_trades": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate_pct": win_rate * 100.0,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade,
        "median_trade_pnl": median_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "expectancy": expectancy,
        "kelly_fraction": kelly,
        "max_consecutive_wins": max_win_streak,
        "max_consecutive_losses": max_loss_streak,
        **percentiles,
    }


# ────────────────────────────────────────────────────────────
# CONCENTRATION / ROBUSTNESS
# ────────────────────────────────────────────────────────────

def concentration_metrics(trade_log: pd.DataFrame) -> dict:
    active = trade_log[trade_log["signal"] != 0]
    if active.empty:
        return {}

    # Daily PnL
    daily = active["net_pnl"].groupby(active.index.normalize()).sum().sort_values(ascending=False)
    total = float(daily.sum())
    if total == 0:
        return {}

    n_days = len(daily)

    def pct_from_top(frac: float) -> float:
        k = max(1, int(np.ceil(n_days * frac)))
        top_k_sum = float(daily.head(k).sum())
        return 100.0 * top_k_sum / total

    top_1pct = pct_from_top(0.01)
    top_5pct = pct_from_top(0.05)
    top_10pct = pct_from_top(0.10)

    # Gini coefficient of ABSOLUTE daily PnL magnitudes (inequality of contribution)
    abs_daily = np.sort(np.abs(daily.values))
    n = len(abs_daily)
    if abs_daily.sum() > 0:
        gini = (2 * np.sum(np.arange(1, n + 1) * abs_daily) / (n * abs_daily.sum())) - (n + 1) / n
    else:
        gini = 0.0

    # Herfindahl index on positive daily PnL (concentration of winning days)
    pos_daily = daily[daily > 0]
    if len(pos_daily) and pos_daily.sum() > 0:
        shares = pos_daily / pos_daily.sum()
        herfindahl = float((shares ** 2).sum())
    else:
        herfindahl = 0.0

    # Best vs worst day
    best_day = float(daily.iloc[0]) if n_days else 0.0
    worst_day = float(daily.iloc[-1]) if n_days else 0.0

    return {
        "n_trading_days": n_days,
        "top_1pct_days_contribution_pct": top_1pct,
        "top_5pct_days_contribution_pct": top_5pct,
        "top_10pct_days_contribution_pct": top_10pct,
        "gini_of_abs_daily_pnl": float(gini),
        "herfindahl_of_winning_days": herfindahl,
        "best_day_pnl": best_day,
        "worst_day_pnl": worst_day,
    }


# ────────────────────────────────────────────────────────────
# STABILITY / REGIME
# ────────────────────────────────────────────────────────────

def stability_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> dict:
    """Equity curve stability + rolling Sharpe distribution.

    R² is computed against a linear time trend of ``log(daily_eq)`` in
    compound mode (exponential growth becomes linear under log) and
    against the raw daily equity curve in non-compound mode (the
    additive curve is already linear-ish, so the log transform would
    distort the fit).
    """
    eq = _equity_curve(trade_log, initial_equity)
    if len(eq) < 10:
        return {}

    daily_eq = eq.groupby(eq.index.normalize()).last()
    x = np.arange(len(daily_eq))
    if is_nc:
        y = daily_eq.values.astype(float)
    else:
        y = np.log(daily_eq.clip(lower=1e-9)).values
    if np.std(y) > 0:
        _slope, _intercept, r_value, _, _ = stats.linregress(x, y)
        r_squared = float(r_value ** 2)
    else:
        r_squared = 0.0

    # Rolling 30-day Sharpe — measure of stability
    r_daily = _daily_returns(trade_log, initial_equity, is_nc=is_nc)
    if len(r_daily) >= 30:
        rolling = r_daily.rolling(30)
        rolling_sharpe = (rolling.mean() / rolling.std(ddof=1)) * np.sqrt(DAYS_PER_YEAR)
        rolling_sharpe = rolling_sharpe.dropna()
        median_rolling = float(rolling_sharpe.median()) if len(rolling_sharpe) else 0.0
        min_rolling = float(rolling_sharpe.min()) if len(rolling_sharpe) else 0.0
        pct_positive = float((rolling_sharpe > 0).mean()) * 100.0 if len(rolling_sharpe) else 0.0
    else:
        median_rolling = 0.0
        min_rolling = 0.0
        pct_positive = 0.0

    # Monthly consistency: fraction of profitable months
    monthly = r_daily.groupby(pd.Grouper(freq="ME")).sum()
    pct_profitable_months = float((monthly > 0).mean()) * 100.0 if len(monthly) else 0.0

    return {
        "equity_curve_r_squared": r_squared,
        "rolling_30d_sharpe_median": median_rolling,
        "rolling_30d_sharpe_min": min_rolling,
        "pct_positive_rolling_30d": pct_positive,
        "pct_profitable_months": pct_profitable_months,
        "n_months": len(monthly),
    }


# ────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE
# ────────────────────────────────────────────────────────────

def significance_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float,
    *,
    is_nc: bool = False,
) -> dict:
    r_daily = _daily_returns(trade_log, initial_equity, is_nc=is_nc)
    if len(r_daily) < 10:
        return {}

    # One-sample t-test: is mean daily return > 0?
    t_stat, p_value = stats.ttest_1samp(r_daily, 0.0)
    # Two-sided → one-sided p (we want mean > 0)
    one_sided_p = p_value / 2 if t_stat > 0 else 1 - p_value / 2

    # Deflated Sharpe (adjusts for skew/kurtosis) — Bailey & Lopez de Prado 2014
    mean_daily = float(r_daily.mean())
    std_daily = float(r_daily.std(ddof=1))
    sharpe_daily = mean_daily / std_daily if std_daily > 0 else 0.0
    skew = float(stats.skew(r_daily))
    kurt = float(stats.kurtosis(r_daily))  # excess kurtosis
    n = len(r_daily)
    # Standard error of Sharpe with non-normal returns
    if n > 1 and std_daily > 0:
        sharpe_se = np.sqrt((1 - skew * sharpe_daily + ((kurt) / 4) * sharpe_daily ** 2) / (n - 1))
        sharpe_t = sharpe_daily / sharpe_se if sharpe_se > 0 else 0.0
    else:
        sharpe_se = 0.0
        sharpe_t = 0.0

    # Ljung-Box for return autocorrelation (is there structure left over?)
    try:
        from scipy.stats import jarque_bera
        jb_stat, jb_p = jarque_bera(r_daily)
    except Exception:
        jb_stat, jb_p = 0.0, 1.0

    return {
        "t_stat_daily_returns": float(t_stat),
        "p_value_one_sided": float(one_sided_p),
        "sharpe_standard_error": float(sharpe_se * np.sqrt(DAYS_PER_YEAR)),
        "sharpe_t_stat": float(sharpe_t),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p_value": float(jb_p),
    }


# ────────────────────────────────────────────────────────────
# REGIME BREAKDOWN
# ────────────────────────────────────────────────────────────

def regime_metrics(trade_log: pd.DataFrame, initial_equity: float) -> dict:
    """Per-year PnL, Sharpe, trade count."""
    active = trade_log[trade_log["signal"] != 0].copy()
    if active.empty:
        return {"yearly": {}}

    active["year"] = active.index.year
    yearly = {}
    for year, grp in active.groupby("year"):
        pnl = grp["net_pnl"]
        r = grp["r_t"] if "r_t" in grp.columns else pnl / initial_equity
        mean_r = float(r.mean())
        std_r = float(r.std(ddof=1)) if len(r) > 1 else 0.0
        sharpe = (mean_r / std_r) * np.sqrt(BARS_PER_YEAR) if std_r > 0 else 0.0
        yearly[int(year)] = {
            "n_trades": int(len(pnl)),
            "total_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()),
            "win_rate_pct": float((pnl > 0).mean() * 100),
            "sharpe": sharpe,
        }
    return {"yearly": yearly}


# ────────────────────────────────────────────────────────────
# AGGREGATED TOOLKIT
# ────────────────────────────────────────────────────────────

def summarize_trade_log(
    trade_log: pd.DataFrame,
    initial_equity: float = 100_000,
    label: str = "",
    *,
    is_nc: bool = False,
) -> dict:
    """Compact side-by-side comparison stats for a single candidate config.

    Returns the minimum fields worth reporting in a two- or three-config
    comparison table — the pattern used extensively during the v2 sweep.
    ``n_trades`` is the FIRST key by design: callers SHOULD render it as
    the first column of any comparison table. A $500K PnL with N=50 and
    one with N=5,000 are fundamentally different strategies — hiding N
    hides that distinction.

    Accepts either the full trade log (rows with ``signal == 0`` are
    filtered internally) or an already-filtered active-only DataFrame.

    Pass ``is_nc=True`` when the input is a non-compound view (built via
    :func:`build_nc_view`) or when you want NC-aware MDD / Calmar.  When
    False (default), compound-path columns (``net_pnl``, ``equity``) are
    used; when True, columns are assumed already renamed (``net_pnl``
    sourced from ``net_pnl_nc``, etc.).

    Intended for ad-hoc Python comparison scripts:

        >>> from vol_arb.metrics import summarize_trade_log
        >>> for label, cfg in configs.items():
        ...     _, tl = VolArbBacktester(cfg).run_backtest()
        ...     print(label, summarize_trade_log(tl, label=label))
    """
    active = trade_log[trade_log.get("signal", 1) != 0] \
        if "signal" in trade_log.columns else trade_log
    n = len(active)
    if n == 0:
        return {"label": label, "n_trades": 0}

    # Prefer NC columns for the core per-trade stats — compound-path
    # ``net_pnl`` accumulates equity-scaled quantities so its cumsum MDD
    # is not a meaningful drawdown metric (equity re-bases each trade).
    # ``build_nc_view`` renames _nc → standard; in that case ``is_nc=True``
    # tells us to read standard names directly. On a raw compound trade
    # log, ``net_pnl_nc`` is the NC sibling we want for comparison.
    if is_nc:
        pnl_col = "net_pnl"
    elif "net_pnl_nc" in active.columns:
        pnl_col = "net_pnl_nc"
    else:
        pnl_col = "net_pnl"
    r = active[pnl_col].values
    ts = pd.to_datetime(active.index)
    years = max((ts.max() - ts.min()).days / 365.25, 1e-9)

    # NC MDD from cumulative PnL (scale-invariant interpretation on $100K)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    mdd_abs = float(-(cum - peak).min())

    neg = r[r < 0]
    downside = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 1e-9

    daily = pd.Series(r, index=ts).resample("D").sum()
    daily = daily[daily != 0]
    monthly = pd.Series(r, index=ts).resample("ME").sum()

    yearly = pd.Series(r, index=ts).groupby(ts.year).sum()

    pos_mask = r > 0
    pos_sum = float(r[pos_mask].sum())
    neg_sum = float(-r[~pos_mask & (r != 0)].sum())
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")

    ann_pnl = float(r.sum()) / years

    out: dict = {
        # trade count FIRST — intentional ordering
        "n_trades": int(n),
        "label": label,
        # PnL
        "nc_pnl": float(r.sum()),
        "ann_nc_pnl": ann_pnl,
        "ann_pct_on_initial": ann_pnl / initial_equity * 100.0,
        "per_trade": float(r.mean()),
        # Risk
        "mdd_nc_abs": mdd_abs,
        "mdd_nc_pct": mdd_abs / initial_equity * 100.0,
        # Risk-adjusted
        "trade_sharpe": (
            float(r.mean() / r.std() * np.sqrt(len(r) / years))
            if r.std() > 0 else 0.0
        ),
        "daily_sharpe": (
            float(daily.mean() / daily.std() * np.sqrt(252))
            if daily.std() > 0 else 0.0
        ),
        "sortino": float(r.mean() / downside * np.sqrt(len(r) / years)),
        "calmar": ann_pnl / mdd_abs if mdd_abs > 0 else float("inf"),
        # Trade-stats
        "profit_factor": pf,
        "win_rate": float(pos_mask.mean() * 100.0),
        "monthly_pos_frac": float((monthly > 0).mean() * 100.0),
        # Stability
        "min_year_pnl": float(yearly.min()) if len(yearly) > 0 else 0.0,
        "all_years_positive": bool((yearly > 0).all()) if len(yearly) > 0 else False,
        "worst_trade": float(r.min()),
        "worst_day": float(daily.min()) if len(daily) > 0 else 0.0,
        "worst_month": float(monthly.min()) if len(monthly) > 0 else 0.0,
    }

    # Compound view — only if equity column is populated by a compound run
    if "equity" in active.columns and not is_nc:
        eq = active["equity"].values
        if len(eq) > 0:
            eq_peak = np.maximum.accumulate(eq)
            eq_dd = (eq - eq_peak) / np.where(eq_peak == 0, 1, eq_peak)
            out["compound_pnl"] = float(eq[-1] - initial_equity)
            out["compound_mdd_pct"] = float(-eq_dd.min() * 100.0)
            out["compound_end_multiple"] = float(eq[-1] / initial_equity) if initial_equity > 0 else 0.0

    return out


def full_metrics(
    trade_log: pd.DataFrame,
    initial_equity: float = 100_000,
    *,
    is_nc: bool = False,
) -> dict:
    """Run the entire metrics toolkit and return a flat dict.

    Pass ``is_nc=True`` when consuming a non-compound view built with
    :func:`build_nc_view`.  Downstream metrics that need view-aware
    formulas (returns, drawdowns, Calmar, Martin, stability R²) receive
    the flag via keyword threading.  ``trade_stats``,
    ``concentration_metrics``, and ``regime_metrics`` are view-
    agnostic — they read ``net_pnl`` / ``r_t`` columns directly, which
    :func:`build_nc_view` has already renamed on the NC path.
    """
    out: dict = {}
    out.update(return_metrics(trade_log, initial_equity, is_nc=is_nc))
    out.update(risk_metrics(trade_log, initial_equity, is_nc=is_nc))
    out.update(risk_adjusted_metrics(trade_log, initial_equity, is_nc=is_nc))
    out.update(drawdown_metrics(trade_log, initial_equity, is_nc=is_nc))
    out.update(trade_stats(trade_log))
    out.update(concentration_metrics(trade_log))
    out.update(stability_metrics(trade_log, initial_equity, is_nc=is_nc))
    out.update(significance_metrics(trade_log, initial_equity, is_nc=is_nc))
    out.update(regime_metrics(trade_log, initial_equity))
    return out


# ────────────────────────────────────────────────────────────
# NON-COMPOUNDING VIEW
# ────────────────────────────────────────────────────────────

# Columns that the backtester writes in both compound and _nc versions.
# The view helper maps _nc → standard names so the entire metrics toolkit
# can be reused on the non-compounding path with zero code duplication.
_NC_COLUMN_MAP = {
    "net_pnl_nc": "net_pnl",
    "equity_nc": "equity",
    "r_t_nc": "r_t",
    "N_contracts_nc": "N_contracts",
    "slippage_nc": "slippage",
    "vega_pnl_nc": "vega_pnl",
    "theta_pnl_nc": "theta_pnl",
    "delta_hedge_pnl_nc": "delta_hedge_pnl",
    "hedge_costs_nc": "hedge_costs",
    "option_fees_nc": "option_fees",
    "gross_pnl_nc": "gross_pnl",
    "maker_filled_nc": "maker_filled",
}


def build_nc_view(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Return a view of ``trade_log`` in non-compound coordinates.

    The vol_arb backtester stores compound quantities under the standard
    column names (``net_pnl``, ``equity``, ``r_t``, ...) and non-compound
    quantities under ``*_nc`` suffixed names.  This helper returns a DataFrame
    where the ``_nc`` columns are RENAMED to the standard names and the
    compound columns are DROPPED — making it a drop-in replacement for
    ``full_metrics`` / ``print_extended_tearsheet`` / any function that
    expects the standard column names.

    Raises
    ------
    ValueError
        If the trade log has no non-compound columns (backtester was run
        with ``report_non_compounding=False``).
    """
    if "net_pnl_nc" not in trade_log.columns:
        raise ValueError(
            "trade_log has no non-compounding columns — "
            "run the backtester with report_non_compounding=True",
        )
    rename_map = {k: v for k, v in _NC_COLUMN_MAP.items() if k in trade_log.columns}
    cols_to_drop = [v for v in rename_map.values() if v in trade_log.columns]
    view = trade_log.drop(columns=cols_to_drop).rename(columns=rename_map)
    # NOTE: callers that run the metrics toolkit on this view must pass
    # ``is_nc=True`` explicitly (e.g., ``full_metrics(view, ..., is_nc=True)``).
    # We deliberately do NOT tag the view via ``.attrs`` — pandas drops
    # attrs silently across many operations (boolean indexing, concat,
    # many arithmetic ops), which produced subtle bugs where NC metrics
    # were computed with compound formulas.  Explicit-is-better-than-
    # implicit wins here.
    return view


def full_metrics_both(
    trade_log: pd.DataFrame, initial_equity: float = 100_000,
) -> dict:
    """Run :func:`full_metrics` on both the compound and non-compound views.

    Returns a dict with keys ``"compound"`` and (if the trade log has _nc
    columns) ``"non_compound"``, each containing the full metrics dict.
    ``is_nc`` is passed explicitly on each call — no hidden state.
    """
    out: dict = {
        "compound": full_metrics(trade_log, initial_equity, is_nc=False),
    }
    if "net_pnl_nc" in trade_log.columns:
        nc_view = build_nc_view(trade_log)
        out["non_compound"] = full_metrics(
            nc_view, initial_equity, is_nc=True,
        )
    return out


# ────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ────────────────────────────────────────────────────────────

def print_extended_tearsheet(
    trade_log: pd.DataFrame,
    initial_equity: float = 100_000,
    *,
    is_nc: bool = False,
) -> None:
    """Print the full pro-grade tearsheet.

    Pass ``is_nc=True`` when rendering a non-compound view (e.g., one
    returned by :func:`build_nc_view`).  In NC mode the CAGR row shows
    ``N/A`` because CAGR is undefined on an additive curve — the
    arithmetic annualized return is the honest substitute and is
    always rendered.
    """
    m = full_metrics(trade_log, initial_equity, is_nc=is_nc)

    def _pct(value, precision: int = 2, width: int = 20) -> str:
        """Format a float percent cell; render None as N/A."""
        if value is None:
            return f"{'N/A':>{width}}"
        return f"{value:>{width},.{precision}f}%"

    W = 72
    print("=" * W)
    header = "EXTENDED TEARSHEET (NON-COMPOUND)" if is_nc else "EXTENDED TEARSHEET"
    print(f"  {header}")
    print("=" * W)

    # --- Returns ---
    print("\n[ RETURNS ]")
    print(f"  {'Total PnL':<36}${m['total_pnl']:>20,.0f}")
    print(f"  {'Final Equity':<36}${m['final_equity']:>20,.0f}")
    print(f"  {'Total Return':<36}{m['total_return_pct']:>20,.2f}%")
    print(f"  {'CAGR':<36}{_pct(m['cagr_pct'])}")
    print(f"  {'Annualized Return (Arithmetic)':<36}{m['ann_return_arith_pct']:>20,.2f}%")
    print(f"  {'Period (years)':<36}{m['years']:>20,.2f}")

    # --- Risk ---
    print("\n[ RISK ]")
    print(f"  {'Annualized Volatility':<36}{m.get('ann_volatility_pct', 0):>20,.2f}%")
    print(f"  {'Annualized Downside Vol':<36}{m.get('ann_downside_vol_pct', 0):>20,.2f}%")
    print(f"  {'Daily VaR (95%)':<36}{m.get('daily_var_95_pct', 0):>20,.3f}%")
    print(f"  {'Daily VaR (99%)':<36}{m.get('daily_var_99_pct', 0):>20,.3f}%")
    print(f"  {'Daily CVaR (95%)':<36}{m.get('daily_cvar_95_pct', 0):>20,.3f}%")
    print(f"  {'Daily CVaR (99%)':<36}{m.get('daily_cvar_99_pct', 0):>20,.3f}%")
    print(f"  {'Skewness':<36}{m.get('skewness', 0):>20,.3f}")
    print(f"  {'Excess Kurtosis':<36}{m.get('excess_kurtosis', 0):>20,.3f}")
    print(f"  {'Tail Ratio (95/5)':<36}{m.get('tail_ratio', 0):>20,.3f}")

    # --- Risk-adjusted ---
    print("\n[ RISK-ADJUSTED ]")
    print(f"  {'Sharpe (daily)':<36}{m.get('sharpe_daily', 0):>20,.3f}")
    print(f"  {'Sortino (daily)':<36}{m.get('sortino_daily', 0):>20,.3f}")
    print(f"  {'Calmar (CAGR / MaxDD)':<36}{m.get('calmar', 0):>20,.3f}")
    print(f"  {'Omega (threshold 0)':<36}{m.get('omega', 0):>20,.3f}")
    print(f"  {'Martin (CAGR / Ulcer)':<36}{m.get('martin', 0):>20,.3f}")
    print(f"  {'Gain/Loss Ratio':<36}{m.get('gain_loss_ratio', 0):>20,.3f}")
    print(f"  {'Ulcer Index':<36}{m.get('ulcer_index', 0):>20,.3f}%")

    # --- Drawdown ---
    print("\n[ DRAWDOWN ]")
    print(f"  {'Max Drawdown':<36}{m.get('max_drawdown_pct', 0):>20,.2f}%")
    print(f"  {'Avg Drawdown':<36}{m.get('avg_drawdown_pct', 0):>20,.2f}%")
    print(f"  {'Max DD Duration (days)':<36}{m.get('max_dd_duration_days', 0):>20,.0f}")
    print(f"  {'Avg DD Duration (days)':<36}{m.get('avg_dd_duration_days', 0):>20,.1f}")
    print(f"  {'% Time in Drawdown':<36}{m.get('pct_time_in_drawdown', 0):>20,.1f}%")
    print(f"  {'# Drawdown Events':<36}{m.get('n_drawdown_events', 0):>20,d}")

    # --- Trades ---
    print("\n[ TRADE STATISTICS ]")
    print(f"  {'Total Trades':<36}{m.get('n_trades', 0):>20,d}")
    print(f"  {'Winners':<36}{m.get('n_wins', 0):>20,d}")
    print(f"  {'Losers':<36}{m.get('n_losses', 0):>20,d}")
    print(f"  {'Win Rate':<36}{m.get('win_rate_pct', 0):>20,.2f}%")
    print(f"  {'Profit Factor':<36}{m.get('profit_factor', 0):>20,.3f}")
    print(f"  {'Avg Trade PnL':<36}${m.get('avg_trade_pnl', 0):>19,.2f}")
    print(f"  {'Median Trade PnL':<36}${m.get('median_trade_pnl', 0):>19,.2f}")
    print(f"  {'Avg Win':<36}${m.get('avg_win', 0):>19,.2f}")
    print(f"  {'Avg Loss':<36}${m.get('avg_loss', 0):>19,.2f}")
    print(f"  {'Largest Win':<36}${m.get('largest_win', 0):>19,.0f}")
    print(f"  {'Largest Loss':<36}${m.get('largest_loss', 0):>19,.0f}")
    print(f"  {'Expectancy':<36}${m.get('expectancy', 0):>19,.3f}")
    print(f"  {'Kelly Fraction':<36}{m.get('kelly_fraction', 0):>20,.4f}")
    print(f"  {'Max Consecutive Wins':<36}{m.get('max_consecutive_wins', 0):>20,d}")
    print(f"  {'Max Consecutive Losses':<36}{m.get('max_consecutive_losses', 0):>20,d}")
    print(f"  {'PnL 1%ile':<36}${m.get('pnl_p1', 0):>19,.2f}")
    print(f"  {'PnL 5%ile':<36}${m.get('pnl_p5', 0):>19,.2f}")
    print(f"  {'PnL 95%ile':<36}${m.get('pnl_p95', 0):>19,.2f}")
    print(f"  {'PnL 99%ile':<36}${m.get('pnl_p99', 0):>19,.2f}")

    # --- Concentration ---
    print("\n[ CONCENTRATION / ROBUSTNESS ]")
    print(f"  {'# Trading Days':<36}{m.get('n_trading_days', 0):>20,d}")
    print(f"  {'Top 1% Days Contribution':<36}{m.get('top_1pct_days_contribution_pct', 0):>20,.1f}%")
    print(f"  {'Top 5% Days Contribution':<36}{m.get('top_5pct_days_contribution_pct', 0):>20,.1f}%")
    print(f"  {'Top 10% Days Contribution':<36}{m.get('top_10pct_days_contribution_pct', 0):>20,.1f}%")
    print(f"  {'Gini (|daily PnL|)':<36}{m.get('gini_of_abs_daily_pnl', 0):>20,.3f}")
    print(f"  {'Herfindahl (winning days)':<36}{m.get('herfindahl_of_winning_days', 0):>20,.4f}")
    print(f"  {'Best Day PnL':<36}${m.get('best_day_pnl', 0):>19,.0f}")
    print(f"  {'Worst Day PnL':<36}${m.get('worst_day_pnl', 0):>19,.0f}")

    # --- Stability ---
    print("\n[ STABILITY ]")
    stability_label = (
        "Equity Curve R² (raw vs time)"
        if is_nc
        else "Equity Curve R² (log vs time)"
    )
    print(f"  {stability_label:<36}{m.get('equity_curve_r_squared', 0):>20,.4f}")
    print(f"  {'Rolling 30d Sharpe (median)':<36}{m.get('rolling_30d_sharpe_median', 0):>20,.2f}")
    print(f"  {'Rolling 30d Sharpe (min)':<36}{m.get('rolling_30d_sharpe_min', 0):>20,.2f}")
    print(f"  {'% of Rolling 30d Windows +':<36}{m.get('pct_positive_rolling_30d', 0):>20,.1f}%")
    print(f"  {'% Profitable Months':<36}{m.get('pct_profitable_months', 0):>20,.1f}%")
    print(f"  {'# Months':<36}{m.get('n_months', 0):>20,d}")

    # --- Significance ---
    print("\n[ STATISTICAL SIGNIFICANCE ]")
    print(f"  {'t-stat (daily returns vs 0)':<36}{m.get('t_stat_daily_returns', 0):>20,.3f}")
    print(f"  {'p-value (one-sided)':<36}{m.get('p_value_one_sided', 1):>20.2e}")
    print(f"  {'Sharpe Std Error (ann)':<36}{m.get('sharpe_standard_error', 0):>20,.3f}")
    print(f"  {'Sharpe t-stat':<36}{m.get('sharpe_t_stat', 0):>20,.3f}")
    print(f"  {'Jarque-Bera normality stat':<36}{m.get('jarque_bera_stat', 0):>20,.2f}")
    print(f"  {'Jarque-Bera p-value':<36}{m.get('jarque_bera_p_value', 1):>20.2e}")

    # --- Regime ---
    yearly = m.get("yearly", {})
    if yearly:
        print("\n[ REGIME BREAKDOWN (Yearly) ]")
        print(f"  {'Year':>6s}  {'Trades':>8s}  {'Total PnL':>15s}  {'Avg PnL':>12s}  {'Win%':>6s}  {'Sharpe':>8s}")
        print("  " + "-" * 68)
        for year in sorted(yearly.keys()):
            y = yearly[year]
            print(
                f"  {year:>6d}  {y['n_trades']:>8,d}  ${y['total_pnl']:>14,.0f}"
                f"  ${y['avg_pnl']:>11,.2f}  {y['win_rate_pct']:>5.1f}%  {y['sharpe']:>8,.2f}"
            )

    print("=" * W + "\n")
