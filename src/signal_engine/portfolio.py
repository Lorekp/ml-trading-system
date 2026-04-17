"""Portfolio simulation with shared capital pool.

Extracted from experiments/expanded_portfolio_test.py (lines 762-897).
Probability-based sizing, per-asset and total exposure limits,
dynamic deleveraging, and drawdown halt.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from signal_engine.config import PortfolioConfig

logger = logging.getLogger(__name__)


def _edge_from_probability(probability: float, signal_strength: float) -> float:
    if np.isfinite(probability):
        return float(np.clip(abs(2.0 * probability - 1.0), 0.0, 1.0))
    return float(np.clip(abs(signal_strength), 0.0, 1.0))


def simulate_portfolio(
    signals_df: pd.DataFrame,
    config: PortfolioConfig,
) -> dict[str, Any] | None:
    """Simulate a multi-asset portfolio on pre-computed signals.

    Processes signals in chronological order, sizing each trade based on
    signal strength, enforcing per-asset and aggregate exposure limits,
    and halting when drawdown breaches ``config.max_dd``.

    Args:
        signals_df: Combined signals from all assets.  Must contain columns:
            ``timestamp, direction, fwd_return, signal_strength``.
            Optional column: ``asset``.
        config: Portfolio risk parameters.

    Returns:
        Dictionary with arrays ``equity, pnl, dd, sizes, skipped,
        timestamps``, scalars ``halted, final_equity, n_trades``.
        ``None`` if fewer than 10 signals.
    """
    if signals_df is None or len(signals_df) < 10:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = (
        sub["asset"].values if "asset" in sub.columns else np.full(n, "SINGLE")
    )
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = (
        sub["signal_strength"].values.astype(np.float64)
        if "signal_strength" in sub.columns
        else None
    )
    probabilities = (
        sub["probability"].values.astype(np.float64)
        if "probability" in sub.columns
        else np.full(n, np.nan, dtype=np.float64)
    )
    regime_mults = (
        sub["regime_exposure_mult"].values.astype(np.float64)
        if "regime_exposure_mult" in sub.columns
        else np.ones(n, dtype=np.float64)
    )
    sizing_mults = (
        sub["sizing_mult"].values.astype(np.float64)
        if "sizing_mult" in sub.columns
        else np.ones(n, dtype=np.float64)
    )
    hold_bars_arr = (
        sub["hold_bars"].values.astype(np.int64)
        if "hold_bars" in sub.columns
        else np.full(n, 4, dtype=np.int64)
    )
    strategies_arr = (
        sub["strategy"].values if "strategy" in sub.columns
        else np.full(n, "default")
    )

    equity_arr = np.empty(n)
    pnl_arr = np.zeros(n)
    dd_arr = np.empty(n)
    size_arr = np.zeros(n)
    skipped = np.ones(n, dtype=bool)

    equity = config.starting_capital
    peak_equity = equity
    halted = False

    # Active positions: list of (expiry_timestamp, size, asset, strategy)
    active_positions: list[tuple] = []

    # Build unique timestamp list for expiry tracking
    unique_ts = pd.Series(timestamps).unique()
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}

    use_prob = signal_strengths is not None

    for i in range(n):
        # Expire old positions
        current_ts = timestamps[i]
        current_ts_idx = ts_to_idx.get(current_ts, 0)
        active_positions = [
            (exp_ts_idx, sz, a, st) for exp_ts_idx, sz, a, st in active_positions
            if exp_ts_idx > current_ts_idx
        ]

        # Compute current exposure from active positions
        current_exposure = sum(sz for _, sz, _, _ in active_positions)
        asset_exposure: dict[str, float] = {}
        strategy_exposure: dict[str, float] = {}
        for _, sz, a, st in active_positions:
            asset_exposure[a] = asset_exposure.get(a, 0.0) + sz
            strategy_exposure[st] = strategy_exposure.get(st, 0.0) + sz

        if halted:
            equity_arr[i] = equity
            dd_arr[i] = (
                (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            )
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

        if dd >= config.max_dd:
            halted = True
            equity_arr[i] = equity
            dd_arr[i] = dd
            continue

        # Dynamic deleveraging
        if dd > config.delever_start:
            dd_ratio = (dd - config.delever_start) / (
                config.max_dd - config.delever_start
            )
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        # Position sizing
        if use_prob:
            sig_abs = abs(signal_strengths[i])  # type: ignore[index]
            if config.sizing_mode == "probability_edge":
                sig_abs = _edge_from_probability(probabilities[i], sig_abs)
            if config.use_regime_exposure_multiplier:
                sig_abs *= float(
                    np.clip(regime_mults[i], config.regime_exposure_floor, 1.0)
                )
            raw_size = equity * config.base_sizing_pct * sig_abs * size_mult * sizing_mults[i]
            raw_size = min(
                raw_size,
                equity * config.base_sizing_pct * config.max_signal_mult * size_mult,
            )
        else:
            raw_size = equity * config.base_sizing_pct * size_mult

        # Per-asset exposure limit
        asset = assets_arr[i]
        asset_exp = asset_exposure.get(asset, 0.0)

        if asset_exp + raw_size > equity * config.max_per_asset:
            raw_size = max(equity * config.max_per_asset - asset_exp, 0.0)

        # Per-strategy exposure limit
        strategy = strategies_arr[i]
        max_per_strategy = getattr(config, "max_per_strategy", 1.0)
        strat_exp = strategy_exposure.get(strategy, 0.0)
        if max_per_strategy < 1.0 and strat_exp + raw_size > equity * max_per_strategy:
            raw_size = max(equity * max_per_strategy - strat_exp, 0.0)

        # Aggregate exposure limit
        if current_exposure + raw_size > equity * config.max_exposure:
            raw_size = max(equity * config.max_exposure - current_exposure, 0.0)

        if raw_size < 1.0:
            equity_arr[i] = equity
            dd_arr[i] = dd
            continue

        # Execute trade
        trade_return = directions[i] * fwd_returns[i] - config.slippage
        pnl = raw_size * trade_return

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        # Track position: expires hold_bars unique timestamps from now
        expiry_ts_idx = current_ts_idx + int(hold_bars_arr[i])
        active_positions.append((expiry_ts_idx, raw_size, asset, strategy))

        equity_arr[i] = equity
        pnl_arr[i] = pnl
        dd_arr[i] = (
            (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        )
        size_arr[i] = raw_size
        skipped[i] = False

    result = {
        "equity": equity_arr,
        "pnl": pnl_arr,
        "dd": dd_arr,
        "sizes": size_arr,
        "skipped": skipped,
        "timestamps": timestamps,
        "halted": halted,
        "final_equity": equity,
        "n_trades": int((~skipped).sum()),
    }
    logger.info(
        "Portfolio sim: %d trades, final equity %.2f, halted=%s",
        result["n_trades"],
        equity,
        halted,
    )
    return result


def compute_daily_pnl_series(sim: dict[str, Any]) -> pd.Series | None:
    """Extract a daily PnL series from simulation output.

    Args:
        sim: Result dict from :func:`simulate_portfolio`.

    Returns:
        Daily PnL ``pd.Series`` (non-zero days only), or ``None`` if
        fewer than 5 active trading days.
    """
    if sim is None or sim["n_trades"] < 10:
        return None

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
    timestamps = sim["timestamps"][active]

    ts_series = pd.Series(pnls, index=pd.to_datetime(timestamps))
    daily = ts_series.resample("1D").sum()

    # Filter to only days with activity
    t0 = pd.Timestamp(timestamps[0])
    daily = daily[daily.index >= t0]
    # Include all days (including zero-PnL) for correct Sharpe annualization
    return daily if len(daily) >= 5 else None
