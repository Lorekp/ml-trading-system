"""Path-dependent portfolio simulation with variable hold durations.

Same exposure tracking and position management as portfolio.py but each
position has an individually-computed hold duration from the exit engine.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from signal_engine.config import ExitConfig, PortfolioConfig
from signal_engine.exits import compute_path_exits

logger = logging.getLogger(__name__)

_NAT_I64 = np.datetime64("NaT").view(np.int64)


def _edge_from_probability(probability: float, signal_strength: float) -> float:
    if np.isfinite(probability):
        return float(np.clip(abs(2.0 * probability - 1.0), 0.0, 1.0))
    return float(np.clip(abs(signal_strength), 0.0, 1.0))


def _net_trade_return(
    trade_return: float,
    direction: float,
    already_signed: bool,
    slippage: float,
) -> float:
    if already_signed:
        return float(trade_return)
    return float(direction * trade_return - slippage)


def _define_numba_kernel():
    """Define and return the Numba-JIT simulation kernel (compiled once on first call)."""

    @numba.njit(cache=True)
    def _edge_prob_jit(probability, signal_strength):
        if np.isfinite(probability):
            val = abs(2.0 * probability - 1.0)
            return min(max(val, 0.0), 1.0)
        val = abs(signal_strength)
        return min(max(val, 0.0), 1.0)

    @numba.njit(cache=True)
    def _net_return_jit(trade_return, direction, already_signed, slippage):
        if already_signed:
            return trade_return
        return direction * trade_return - slippage

    @numba.njit(cache=True)
    def _sim_core(
        timeline_i64,
        entry_unique_ts_i64,
        entry_starts,
        entry_ends,
        timestamps_i64,
        exit_timestamps_i64,
        has_exit_timestamps,
        asset_ids,
        n_assets,
        directions,
        fwd_returns,
        de_risk_returns,
        final_exit_returns,
        bars_held,
        signal_strengths,
        has_signal_strengths,
        probabilities,
        regime_mults,
        de_risk_timestamps_i64,
        remaining_fractions,
        return_already_signed,
        starting_capital,
        slippage,
        base_sizing_pct,
        max_signal_mult,
        max_per_asset,
        max_exposure,
        delever_start,
        max_dd_param,
        sizing_mode_prob_edge,
        use_regime,
        regime_floor,
        max_events,
    ):
        MAX_POS = 512
        # Position parallel arrays
        p_exit_ts = np.empty(MAX_POS, dtype=np.int64)
        p_entry_size = np.empty(MAX_POS, dtype=np.float64)
        p_current_size = np.empty(MAX_POS, dtype=np.float64)
        p_asset_id = np.empty(MAX_POS, dtype=np.int32)
        p_direction = np.empty(MAX_POS, dtype=np.float64)
        p_de_risk_ts = np.empty(MAX_POS, dtype=np.int64)
        p_de_risk_return = np.empty(MAX_POS, dtype=np.float64)
        p_exit_return = np.empty(MAX_POS, dtype=np.float64)
        p_remaining = np.empty(MAX_POS, dtype=np.float64)
        p_ret_signed = np.empty(MAX_POS, dtype=numba.boolean)
        p_de_risk_applied = np.empty(MAX_POS, dtype=numba.boolean)
        n_pos = 0

        equity = starting_capital
        peak_equity = equity
        halted = False
        n_trades = 0

        # Output arrays (pre-allocated to max possible size)
        out_ts = np.empty(max_events, dtype=np.int64)
        out_equity = np.empty(max_events, dtype=np.float64)
        out_pnl = np.empty(max_events, dtype=np.float64)
        out_dd = np.empty(max_events, dtype=np.float64)
        out_sizes = np.empty(max_events, dtype=np.float64)
        out_skipped = np.empty(max_events, dtype=numba.boolean)
        n_events = 0

        asset_exposure = np.zeros(n_assets, dtype=np.float64)

        for ti in range(len(timeline_i64)):
            current_ts = timeline_i64[ti]
            # searchsorted for current_ts_idx
            current_ts_idx = np.searchsorted(entry_unique_ts_i64, current_ts)

            pnl = 0.0
            # Process active positions
            new_n_pos = 0
            for pi in range(n_pos):
                dr_ts = p_de_risk_ts[pi]
                # NaT check: NaT.view(int64) == _NAT_I64
                if dr_ts != _NAT_I64 and not p_de_risk_applied[pi] and dr_ts <= current_ts:
                    close_frac = 1.0 - p_remaining[pi]
                    if close_frac > 0.0 and p_entry_size[pi] > 0.0:
                        pnl += (
                            p_entry_size[pi]
                            * close_frac
                            * _net_return_jit(
                                p_de_risk_return[pi],
                                p_direction[pi],
                                p_ret_signed[pi],
                                slippage,
                            )
                        )
                    p_current_size[pi] = p_entry_size[pi] * p_remaining[pi]
                    p_de_risk_applied[pi] = True

                ex_ts = p_exit_ts[pi]
                if ex_ts != _NAT_I64 and ex_ts <= current_ts:
                    if p_current_size[pi] > 0.0:
                        pnl += p_current_size[pi] * _net_return_jit(
                            p_exit_return[pi],
                            p_direction[pi],
                            p_ret_signed[pi],
                            slippage,
                        )
                    p_current_size[pi] = 0.0

                if p_current_size[pi] > 0.0 and (ex_ts == _NAT_I64 or ex_ts > current_ts):
                    # Keep position — compact in place
                    if new_n_pos != pi:
                        p_exit_ts[new_n_pos] = p_exit_ts[pi]
                        p_entry_size[new_n_pos] = p_entry_size[pi]
                        p_current_size[new_n_pos] = p_current_size[pi]
                        p_asset_id[new_n_pos] = p_asset_id[pi]
                        p_direction[new_n_pos] = p_direction[pi]
                        p_de_risk_ts[new_n_pos] = p_de_risk_ts[pi]
                        p_de_risk_return[new_n_pos] = p_de_risk_return[pi]
                        p_exit_return[new_n_pos] = p_exit_return[pi]
                        p_remaining[new_n_pos] = p_remaining[pi]
                        p_ret_signed[new_n_pos] = p_ret_signed[pi]
                        p_de_risk_applied[new_n_pos] = p_de_risk_applied[pi]
                    new_n_pos += 1
            n_pos = new_n_pos

            if pnl != 0.0:
                equity += pnl
                if equity > peak_equity:
                    peak_equity = equity

            # Compute exposure
            current_exposure = 0.0
            for ai in range(n_assets):
                asset_exposure[ai] = 0.0
            for pi in range(n_pos):
                current_exposure += p_current_size[pi]
                asset_exposure[p_asset_id[pi]] += p_current_size[pi]

            dd = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 0.0
            if dd >= max_dd_param:
                halted = True

            opened_size = 0.0
            skipped_all = True
            any_activity = pnl != 0.0

            # Find entries for this timestamp
            eidx = np.searchsorted(entry_unique_ts_i64, current_ts)
            if eidx < len(entry_starts) and entry_unique_ts_i64[eidx] == current_ts:
                estart = entry_starts[eidx]
                eend = entry_ends[eidx]
            else:
                estart = 0
                eend = 0

            for i in range(estart, eend):
                any_activity = True
                if halted:
                    continue

                if dd > delever_start:
                    dd_ratio = (dd - delever_start) / (max_dd_param - delever_start)
                    size_mult = max(1.0 - dd_ratio, 0.0)
                else:
                    size_mult = 1.0

                if has_signal_strengths:
                    sig_abs = abs(signal_strengths[i])
                    if sizing_mode_prob_edge:
                        sig_abs = _edge_prob_jit(probabilities[i], sig_abs)
                    if use_regime:
                        rm = regime_mults[i]
                        rm = min(max(rm, regime_floor), 1.0)
                        sig_abs *= rm
                    raw_size = equity * base_sizing_pct * sig_abs * size_mult
                    raw_size = min(raw_size, equity * base_sizing_pct * max_signal_mult * size_mult)
                else:
                    raw_size = equity * base_sizing_pct * size_mult

                aid = asset_ids[i]
                a_exp = asset_exposure[aid]
                if a_exp + raw_size > equity * max_per_asset:
                    raw_size = max(equity * max_per_asset - a_exp, 0.0)
                if current_exposure + raw_size > equity * max_exposure:
                    raw_size = max(equity * max_exposure - current_exposure, 0.0)
                if raw_size < 1.0:
                    continue

                if has_exit_timestamps:
                    expiry_ts = exit_timestamps_i64[i]
                else:
                    hold = max(bars_held[i], 1)
                    eidx2 = min(current_ts_idx + hold, len(entry_unique_ts_i64) - 1)
                    expiry_ts = entry_unique_ts_i64[eidx2]

                rf = remaining_fractions[i]
                rf = min(max(rf, 0.0), 1.0)

                if n_pos < MAX_POS:
                    p_exit_ts[n_pos] = expiry_ts
                    p_entry_size[n_pos] = raw_size
                    p_current_size[n_pos] = raw_size
                    p_asset_id[n_pos] = aid
                    p_direction[n_pos] = directions[i]
                    p_de_risk_ts[n_pos] = de_risk_timestamps_i64[i]
                    p_de_risk_return[n_pos] = de_risk_returns[i]
                    p_exit_return[n_pos] = final_exit_returns[i]
                    p_remaining[n_pos] = rf
                    p_ret_signed[n_pos] = return_already_signed[i]
                    p_de_risk_applied[n_pos] = False
                    n_pos += 1

                current_exposure += raw_size
                asset_exposure[aid] += raw_size
                opened_size += raw_size
                skipped_all = False
                n_trades += 1

            if any_activity and n_events < max_events:
                out_ts[n_events] = current_ts
                out_equity[n_events] = equity
                out_pnl[n_events] = pnl
                out_dd[n_events] = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 0.0
                out_sizes[n_events] = opened_size
                out_skipped[n_events] = skipped_all and pnl == 0.0
                n_events += 1

            if halted and n_pos == 0:
                break

        return (
            out_ts[:n_events],
            out_equity[:n_events],
            out_pnl[:n_events],
            out_dd[:n_events],
            out_sizes[:n_events],
            out_skipped[:n_events],
            halted,
            equity,
            n_trades,
        )

    return _sim_core


_numba_kernel = None


def _get_numba_kernel():
    global _numba_kernel
    if _numba_kernel is None:
        _numba_kernel = _define_numba_kernel()
    return _numba_kernel


def simulate_portfolio_from_arrays(
    timestamps: np.ndarray,
    exit_timestamps: np.ndarray | None,
    assets_arr: np.ndarray,
    directions: np.ndarray,
    fwd_returns: np.ndarray,
    de_risk_returns: np.ndarray,
    final_exit_returns: np.ndarray,
    bars_held: np.ndarray,
    signal_strengths: np.ndarray | None,
    probabilities: np.ndarray,
    regime_mults: np.ndarray,
    de_risk_timestamps: np.ndarray,
    remaining_fractions: np.ndarray,
    return_already_signed: np.ndarray,
    config: PortfolioConfig,
) -> dict[str, Any] | None:
    """Core simulation on pre-sorted numpy arrays. Uses Numba JIT when available."""
    n = len(timestamps)
    if n < 10:
        return None

    # Build timeline
    timeline_parts = [timestamps]
    if exit_timestamps is not None:
        valid_exit = exit_timestamps[~np.isnat(exit_timestamps)]
        if len(valid_exit) > 0:
            timeline_parts.append(valid_exit)
    valid_de_risk = de_risk_timestamps[~np.isnat(de_risk_timestamps)]
    if len(valid_de_risk) > 0:
        timeline_parts.append(valid_de_risk)
    timeline = np.unique(np.concatenate(timeline_parts))

    entry_unique_ts, _entry_starts = np.unique(timestamps, return_index=True)
    _entry_ends = np.append(_entry_starts[1:], n).astype(np.int64)
    _entry_starts = _entry_starts.astype(np.int64)

    # Convert string assets to integer IDs for Numba
    unique_assets = np.unique(assets_arr)
    asset_to_id = {a: i for i, a in enumerate(unique_assets)}
    asset_ids = np.array([asset_to_id.get(a, 0) for a in assets_arr], dtype=np.int32)
    n_assets = len(unique_assets)

    # Prepare int64 views of datetime64 arrays for Numba
    ts_i64 = timestamps.view(np.int64)
    timeline_i64 = timeline.view(np.int64)
    entry_unique_ts_i64 = entry_unique_ts.view(np.int64)
    exit_ts_i64 = exit_timestamps.view(np.int64) if exit_timestamps is not None else np.empty(0, dtype=np.int64)
    de_risk_ts_i64 = de_risk_timestamps.view(np.int64)

    sig_str = signal_strengths if signal_strengths is not None else np.empty(0, dtype=np.float64)
    max_events = len(timeline)

    if HAS_NUMBA:
        kernel = _get_numba_kernel()
        result_tuple = kernel(
            timeline_i64,
            entry_unique_ts_i64,
            _entry_starts,
            _entry_ends,
            ts_i64,
            exit_ts_i64,
            exit_timestamps is not None,
            asset_ids,
            np.int32(n_assets),
            directions.astype(np.float64),
            fwd_returns.astype(np.float64),
            de_risk_returns.astype(np.float64),
            final_exit_returns.astype(np.float64),
            bars_held.astype(np.int32),
            sig_str.astype(np.float64),
            signal_strengths is not None,
            probabilities.astype(np.float64),
            regime_mults.astype(np.float64),
            de_risk_ts_i64,
            remaining_fractions.astype(np.float64),
            return_already_signed.astype(np.bool_),
            float(config.starting_capital),
            float(config.slippage),
            float(config.base_sizing_pct),
            float(config.max_signal_mult),
            float(config.max_per_asset),
            float(config.max_exposure),
            float(config.delever_start),
            float(config.max_dd),
            config.sizing_mode == "probability_edge",
            bool(config.use_regime_exposure_multiplier),
            float(config.regime_exposure_floor),
            np.int64(max_events),
        )
        (out_ts_i64, out_equity, out_pnl, out_dd, out_sizes,
         out_skipped, halted, final_equity, n_trades_val) = result_tuple
        result = {
            "equity": out_equity,
            "pnl": out_pnl,
            "dd": out_dd,
            "sizes": out_sizes,
            "skipped": out_skipped,
            "timestamps": out_ts_i64.view("datetime64[ns]"),
            "halted": bool(halted),
            "final_equity": float(final_equity),
            "n_trades": int(n_trades_val),
        }
        return result

    # Fallback: pure Python (identical logic)
    return _simulate_portfolio_from_arrays_python(
        timestamps, exit_timestamps, assets_arr, directions, fwd_returns,
        de_risk_returns, final_exit_returns, bars_held, signal_strengths,
        probabilities, regime_mults, de_risk_timestamps, remaining_fractions,
        return_already_signed, config, timeline, entry_unique_ts,
        _entry_starts, _entry_ends,
    )


def _simulate_portfolio_from_arrays_python(
    timestamps, exit_timestamps, assets_arr, directions, fwd_returns,
    de_risk_returns, final_exit_returns, bars_held, signal_strengths,
    probabilities, regime_mults, de_risk_timestamps, remaining_fractions,
    return_already_signed, config, timeline, entry_unique_ts,
    _entry_starts, _entry_ends,
):
    """Pure Python fallback for simulate_portfolio_from_arrays."""
    n = len(timestamps)
    _entry_ts_to_range = {
        entry_unique_ts[j]: (int(_entry_starts[j]), int(_entry_ends[j]))
        for j in range(len(entry_unique_ts))
    }

    use_prob = signal_strengths is not None
    active_positions: list[dict[str, Any]] = []
    event_timestamps: list = []
    event_equity: list[float] = []
    event_pnl: list[float] = []
    event_dd: list[float] = []
    event_sizes: list[float] = []
    event_skipped: list[bool] = []
    n_trades = 0
    equity = config.starting_capital
    peak_equity = equity
    halted = False
    slippage = config.slippage
    base_sizing_pct = config.base_sizing_pct
    max_signal_mult = config.max_signal_mult
    max_per_asset = config.max_per_asset
    max_exposure_val = config.max_exposure
    delever_start = config.delever_start
    max_dd_val = config.max_dd
    sizing_mode = config.sizing_mode
    use_regime = config.use_regime_exposure_multiplier
    regime_floor = config.regime_exposure_floor

    for current_ts in timeline:
        current_ts_idx = int(np.searchsorted(entry_unique_ts, current_ts))
        pnl = 0.0
        next_active = []
        for pos in active_positions:
            pos = dict(pos)
            dr_ts = pos["de_risk_ts"]
            if dr_ts == dr_ts and not pos["de_risk_applied"] and dr_ts <= current_ts:
                cf = 1.0 - pos["remaining_fraction"]
                if cf > 0.0 and pos["entry_size"] > 0.0:
                    pnl += pos["entry_size"] * cf * _net_trade_return(
                        pos["de_risk_return"], pos["direction"],
                        pos["return_already_signed"], slippage)
                pos["current_size"] = pos["entry_size"] * pos["remaining_fraction"]
                pos["de_risk_applied"] = True
            ex_ts = pos["exit_ts"]
            if ex_ts == ex_ts and ex_ts <= current_ts:
                if pos["current_size"] > 0.0:
                    pnl += pos["current_size"] * _net_trade_return(
                        pos["exit_return"], pos["direction"],
                        pos["return_already_signed"], slippage)
                pos["current_size"] = 0.0
            if pos["current_size"] > 0.0 and (ex_ts != ex_ts or ex_ts > current_ts):
                next_active.append(pos)
        active_positions = next_active
        if pnl != 0.0:
            equity += pnl
            if equity > peak_equity:
                peak_equity = equity
        current_exposure = sum(p["current_size"] for p in active_positions)
        asset_exposure: dict[str, float] = {}
        for p in active_positions:
            asset_exposure[p["asset"]] = asset_exposure.get(p["asset"], 0.0) + p["current_size"]
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= max_dd_val:
            halted = True
        opened_size = 0.0
        skipped_all = True
        any_activity = pnl != 0.0
        _erange = _entry_ts_to_range.get(current_ts)
        for i in range(_erange[0], _erange[1]) if _erange else ():
            any_activity = True
            if halted:
                continue
            sm = max(1.0 - (dd - delever_start) / (max_dd_val - delever_start), 0.0) if dd > delever_start else 1.0
            if use_prob:
                sa = abs(signal_strengths[i])
                if sizing_mode == "probability_edge":
                    sa = _edge_from_probability(probabilities[i], sa)
                if use_regime:
                    sa *= float(np.clip(regime_mults[i], regime_floor, 1.0))
                raw_size = equity * base_sizing_pct * sa * sm
                raw_size = min(raw_size, equity * base_sizing_pct * max_signal_mult * sm)
            else:
                raw_size = equity * base_sizing_pct * sm
            asset = assets_arr[i]
            a_exp = asset_exposure.get(asset, 0.0)
            if a_exp + raw_size > equity * max_per_asset:
                raw_size = max(equity * max_per_asset - a_exp, 0.0)
            if current_exposure + raw_size > equity * max_exposure_val:
                raw_size = max(equity * max_exposure_val - current_exposure, 0.0)
            if raw_size < 1.0:
                continue
            if exit_timestamps is not None:
                expiry_ts = exit_timestamps[i]
            else:
                hold = max(int(bars_held[i]), 1)
                eidx = min(current_ts_idx + hold, len(entry_unique_ts) - 1)
                expiry_ts = entry_unique_ts[eidx]
            rf = float(np.clip(remaining_fractions[i], 0.0, 1.0))
            active_positions.append({
                "exit_ts": expiry_ts, "entry_size": raw_size, "current_size": raw_size,
                "asset": asset, "direction": float(directions[i]),
                "de_risk_ts": de_risk_timestamps[i], "de_risk_return": float(de_risk_returns[i]),
                "exit_return": float(final_exit_returns[i]), "remaining_fraction": rf,
                "return_already_signed": bool(return_already_signed[i]), "de_risk_applied": False,
            })
            current_exposure += raw_size
            asset_exposure[asset] = asset_exposure.get(asset, 0.0) + raw_size
            opened_size += raw_size
            skipped_all = False
            n_trades += 1
        if any_activity:
            event_timestamps.append(current_ts)
            event_equity.append(equity)
            event_pnl.append(pnl)
            event_dd.append((peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0)
            event_sizes.append(opened_size)
            event_skipped.append(skipped_all and pnl == 0.0)
        if halted and len(active_positions) == 0:
            break

    return {
        "equity": np.asarray(event_equity, dtype=np.float64),
        "pnl": np.asarray(event_pnl, dtype=np.float64),
        "dd": np.asarray(event_dd, dtype=np.float64),
        "sizes": np.asarray(event_sizes, dtype=np.float64),
        "skipped": np.asarray(event_skipped, dtype=bool),
        "timestamps": np.asarray(event_timestamps, dtype="datetime64[ns]"),
        "halted": halted,
        "final_equity": equity,
        "n_trades": int(n_trades),
    }


def simulate_portfolio_precomputed_exits(
    exited_signals_df: pd.DataFrame,
    config: PortfolioConfig,
) -> dict[str, Any] | None:
    """Simulate a portfolio with pre-computed returns and exit timestamps."""
    if exited_signals_df is None or len(exited_signals_df) < 10:
        return None

    sub = exited_signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = pd.to_datetime(sub["timestamp"]).to_numpy(dtype="datetime64[ns]")
    exit_timestamps = (
        pd.to_datetime(sub["exit_timestamp"]).to_numpy(dtype="datetime64[ns]")
        if "exit_timestamp" in sub.columns
        else None
    )
    assets_arr = (
        sub["asset"].values if "asset" in sub.columns else np.full(n, "SINGLE")
    )
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    de_risk_returns = (
        sub["de_risk_return"].values.astype(np.float64)
        if "de_risk_return" in sub.columns
        else fwd_returns
    )
    final_exit_returns = (
        sub["expiry_return"].values.astype(np.float64)
        if "expiry_return" in sub.columns
        else fwd_returns
    )
    bars_held = (
        sub["bars_held"].values.astype(np.int32)
        if "bars_held" in sub.columns
        else np.ones(n, dtype=np.int32)
    )
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
    de_risk_timestamps_arr = (
        pd.to_datetime(sub["de_risk_timestamp"]).to_numpy(dtype="datetime64[ns]")
        if "de_risk_timestamp" in sub.columns
        else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    )
    remaining_fractions = (
        sub["remaining_fraction"].to_numpy(dtype=np.float64)
        if "remaining_fraction" in sub.columns
        else np.zeros(n, dtype=np.float64)
    )
    return_already_signed = (
        sub["return_already_signed"].astype(bool).to_numpy()
        if "return_already_signed" in sub.columns
        else np.zeros(n, dtype=bool)
    )

    return simulate_portfolio_from_arrays(
        timestamps,
        exit_timestamps,
        assets_arr,
        directions,
        fwd_returns,
        de_risk_returns,
        final_exit_returns,
        bars_held,
        signal_strengths,
        probabilities,
        regime_mults,
        de_risk_timestamps_arr,
        remaining_fractions,
        return_already_signed,
        config,
    )


def simulate_portfolio_path(
    signals_df: pd.DataFrame,
    ohlcv_dict: dict[str, pd.DataFrame],
    config: PortfolioConfig,
    exit_config: ExitConfig,
) -> dict[str, Any] | None:
    """Simulate a multi-asset portfolio with path-dependent exits.

    1. Computes actual exit returns by walking each signal's price path.
    2. Runs portfolio simulation with per-position hold durations.

    Returns the same dict structure as ``portfolio.simulate_portfolio``.
    """
    if signals_df is None or len(signals_df) < 10:
        return None

    # Step 1: compute path-dependent exits (replaces fwd_return, adds bars_held)
    exited = compute_path_exits(signals_df, ohlcv_dict, exit_config)
    if len(exited) < 10:
        return None

    return simulate_portfolio_precomputed_exits(exited, config)
