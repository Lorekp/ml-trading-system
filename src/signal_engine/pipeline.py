"""Full backtest pipeline orchestrator.

Wires data loading, feature computation, signal generation, filtering,
portfolio simulation, and statistics into a single ``run_backtest`` call.

Supports parallel processing via ``max_workers`` config parameter.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

import numpy as np

from signal_engine.config import SystemConfig, VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.filters import apply_confidence_filter, apply_regime_filter
from signal_engine.model import WalkForwardModel
from signal_engine.portfolio import compute_daily_pnl_series, simulate_portfolio
from signal_engine.stats import compute_full_stats

logger = logging.getLogger(__name__)


def apply_vol_overlay(
    signals: pd.DataFrame,
    btc_close_daily: pd.Series,
    config: VolOverlayConfig,
) -> pd.DataFrame:
    """Scale signal_strength by BTC volatility regime.

    Uses backward-looking rolling vol percentile — no data leakage.

    Args:
        signals: Combined signals DataFrame with 'timestamp' and 'signal_strength'.
        btc_close_daily: BTC daily close prices (tz-naive DatetimeIndex).
        config: Vol overlay parameters.

    Returns:
        Signals with adjusted signal_strength. Original DataFrame is not mutated.
    """
    btc_ret = btc_close_daily.pct_change()
    btc_vol = btc_ret.rolling(config.vol_window).std() * np.sqrt(365.25)
    btc_vol_pct = btc_vol.rolling(config.pct_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    # Shift by 1 day so intra-day signals only see yesterday's completed
    # daily vol — avoids using same-day close which isn't final yet.
    btc_vol_pct = btc_vol_pct.shift(1)

    ts = pd.to_datetime(signals["timestamp"])
    ts_day = ts.dt.normalize()
    if ts_day.dt.tz is not None:
        ts_day = ts_day.dt.tz_localize(None)

    vol_pct = btc_vol_pct.reindex(ts_day.values).values

    mult = np.where(
        vol_pct < config.low_pct, config.low_mult,
        np.where(vol_pct >= config.high_pct, config.high_mult, config.mid_mult),
    )
    # NaN vol (early period before enough history) → no adjustment
    mult = np.where(np.isfinite(vol_pct), mult, 1.0)

    out = signals.copy()
    out["sizing_mult"] = mult

    n_low = (vol_pct < config.low_pct).sum()
    n_high = (vol_pct >= config.high_pct).sum()
    logger.info(
        "Vol overlay: %d signals scaled (low=%d, high=%d)",
        len(signals), n_low, n_high,
    )
    return out


def apply_per_asset_vol_overlay(
    signals: pd.DataFrame,
    vol_series: dict[str, pd.Series],
    target_vol: float,
    floor: float,
    cap: float,
) -> pd.DataFrame:
    """Per-asset inverse-vol sizing overlay — risk parity across the universe.

    Multiplies the existing ``sizing_mult`` column by
    ``clip(target_vol / asset_daily_vol, floor, cap)``. Mirrors the live
    ``S8Sym15Strategy.compute_vol_mult`` and the experiments backtester's
    ``apply_vol()``.

    FAILS SAFE: an asset with no vol series, an out-of-range point-in-time
    lookup, or a non-finite vol is scaled by ``floor`` (the smallest size) —
    never left at full notional. Equal-dollar-notional sizing across the
    ~27x-vol-dispersion live universe is what blew up DENT/PIPPIN in live
    paper trading. The ``idx-1`` searchsorted lookup keeps the vol
    point-in-time (no lookahead). Original DataFrame is not mutated.
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    col = out.columns.get_loc("sizing_mult")
    n_failsafe = 0
    for i in range(len(out)):
        a = out.iloc[i].get("asset", "")
        vs = vol_series.get(a)
        mult = floor  # fail-safe: unknown vol -> smallest size, never 1.0
        if vs is not None and len(vs):
            idx = vs.index.searchsorted(ts.iloc[i])
            if 0 < idx <= len(vs):
                v = vs.iloc[idx - 1]
                if pd.notna(v) and v > 0:
                    mult = float(np.clip(target_vol / v, floor, cap))
        if mult == floor:
            n_failsafe += 1
        out.iat[i, col] *= mult
    logger.info(
        "Per-asset vol overlay: %d signals scaled (%d fail-safe to floor)",
        len(out), n_failsafe,
    )
    return out


def _process_single_asset(
    asset_name: str,
    symbol: str,
    config: SystemConfig,
) -> dict[str, Any]:
    """Process one asset end-to-end (load → features → signals → filter).

    Designed to run in a separate process. Creates its own WalkForwardModel
    instance since sklearn models aren't safely shareable across processes.

    Returns a dict with signal DataFrame (as dict for pickling) and metadata.
    """
    import warnings
    warnings.filterwarnings("ignore")

    data_dirs = [Path(d) for d in config.data.data_dirs]
    timeframe = config.data.timeframe
    regime_timeframe = config.data.regime_timeframe

    t0 = time.time()

    try:
        ohlcv_1m = load_binance_1m(symbol, data_dirs)
    except FileNotFoundError as exc:
        return {
            "asset": asset_name,
            "error": str(exc),
            "signals": None,
            "n_raw": 0, "n_conf": 0, "n_regime": 0, "n_bars": 0,
            "elapsed_s": 0,
        }

    ohlcv_tf = resample_ohlcv(ohlcv_1m, timeframe)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, regime_timeframe)

    features = compute_features(ohlcv_tf, config.features)

    wf_model = WalkForwardModel(config)
    raw_signals = wf_model.generate_signals_batch(ohlcv_tf, features, asset_name)

    n_raw = len(raw_signals)

    # Per-asset confidence filter
    if n_raw > 0:
        conf_filtered = apply_confidence_filter(raw_signals, config.confidence_filter)
    else:
        conf_filtered = pd.DataFrame()
    n_conf = len(conf_filtered)

    # Per-asset regime filter
    if n_conf > 0 and config.regime_filter.enabled:
        regime_filtered = apply_regime_filter(conf_filtered, ohlcv_4h, config.regime_filter)
    elif n_conf > 0:
        regime_filtered = conf_filtered
    else:
        regime_filtered = pd.DataFrame()
    n_regime = len(regime_filtered)

    elapsed = round(time.time() - t0, 1)

    return {
        "asset": asset_name,
        "error": None,
        "signals": regime_filtered.to_dict("list") if n_regime > 0 else None,
        "n_raw": n_raw,
        "n_conf": n_conf,
        "n_regime": n_regime,
        "n_bars": len(ohlcv_tf),
        "elapsed_s": elapsed,
    }


def run_backtest(config: SystemConfig, max_workers: int = 1) -> dict[str, Any]:
    """Run the full backtest pipeline end-to-end.

    1. For each asset (optionally in parallel): load 1m data -> resample
       -> compute features -> walk-forward signal generation
       -> confidence filter (PER-ASSET) -> regime filter (PER-ASSET).
    2. Combine all filtered asset signals.
    3. Portfolio simulation.
    4. Daily PnL series.
    5. Full statistics.

    Args:
        config: Fully-populated ``SystemConfig``.
        max_workers: Number of parallel processes for per-asset work.
            1 = sequential (default, safe). >1 = parallel via ProcessPoolExecutor.

    Returns:
        Dictionary with stats, per_asset info, signal counts, sim, daily_pnl.
    """
    t_start = time.time()

    all_filtered_signals: list[pd.DataFrame] = []
    per_asset_info: dict[str, dict[str, Any]] = {}
    n_total_raw = 0
    n_after_confidence = 0
    n_after_regime = 0

    assets = list(config.assets.items())

    # ------------------------------------------------------------------
    # Phase 1: per-asset signal generation + filtering
    # ------------------------------------------------------------------
    if max_workers > 1 and len(assets) > 1:
        logger.info(
            "Processing %d assets with %d workers...", len(assets), max_workers
        )
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_single_asset, name, sym, config): name
                for name, sym in assets
            }
            for future in as_completed(futures):
                result = future.result()
                asset_name = result["asset"]

                if result["error"]:
                    logger.warning("Skipping %s: %s", asset_name, result["error"])
                    continue

                n_total_raw += result["n_raw"]
                n_after_confidence += result["n_conf"]
                n_after_regime += result["n_regime"]

                per_asset_info[asset_name] = {
                    "n_raw_signals": result["n_raw"],
                    "n_confidence_filtered": result["n_conf"],
                    "n_regime_filtered": result["n_regime"],
                    "n_bars": result["n_bars"],
                    "elapsed_s": result["elapsed_s"],
                }

                if result["signals"] is not None:
                    df = pd.DataFrame(result["signals"])
                    all_filtered_signals.append(df)

                logger.info(
                    "%s: raw=%d, conf=%d, regime=%d in %.1fs",
                    asset_name,
                    result["n_raw"], result["n_conf"], result["n_regime"],
                    result["elapsed_s"],
                )
    else:
        # Sequential (original behavior)
        wf_model = WalkForwardModel(config)
        data_dirs = [Path(d) for d in config.data.data_dirs]
        timeframe = config.data.timeframe
        regime_timeframe = config.data.regime_timeframe

        for asset_name, symbol in assets:
            logger.info("Processing %s (%s)...", asset_name, symbol)
            asset_start = time.time()

            try:
                ohlcv_1m = load_binance_1m(symbol, data_dirs)
            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", asset_name, exc)
                continue

            ohlcv_tf = resample_ohlcv(ohlcv_1m, timeframe)
            ohlcv_4h = resample_ohlcv(ohlcv_1m, regime_timeframe)

            features = compute_features(ohlcv_tf, config.features)
            raw_signals = wf_model.generate_signals_batch(
                ohlcv_tf, features, asset_name
            )

            n_raw = len(raw_signals)
            n_total_raw += n_raw

            if n_raw > 0:
                conf_filtered = apply_confidence_filter(
                    raw_signals, config.confidence_filter
                )
            else:
                conf_filtered = pd.DataFrame()
            n_conf = len(conf_filtered)
            n_after_confidence += n_conf

            if n_conf > 0 and config.regime_filter.enabled:
                regime_filtered = apply_regime_filter(
                    conf_filtered, ohlcv_4h, config.regime_filter
                )
            elif n_conf > 0:
                regime_filtered = conf_filtered
            else:
                regime_filtered = pd.DataFrame()
            n_regime = len(regime_filtered)
            n_after_regime += n_regime

            elapsed = time.time() - asset_start
            per_asset_info[asset_name] = {
                "n_raw_signals": n_raw,
                "n_confidence_filtered": n_conf,
                "n_regime_filtered": n_regime,
                "n_bars": len(ohlcv_tf),
                "elapsed_s": round(elapsed, 1),
            }

            if n_regime > 0:
                all_filtered_signals.append(regime_filtered)

            logger.info(
                "%s: raw=%d, conf=%d, regime=%d in %.1fs",
                asset_name, n_raw, n_conf, n_regime, elapsed,
            )

    # ------------------------------------------------------------------
    # Phase 2: combine filtered signals
    # ------------------------------------------------------------------
    if not all_filtered_signals:
        logger.warning("No signals survived filtering for any asset")
        return {
            "stats": {"error": "no_signals"},
            "per_asset": per_asset_info,
            "n_total_raw": n_total_raw,
            "n_after_confidence": n_after_confidence,
            "n_after_regime": n_after_regime,
            "sim": None,
            "daily_pnl": None,
            "elapsed_s": round(time.time() - t_start, 1),
        }

    filtered = pd.concat(all_filtered_signals, ignore_index=True)
    logger.info(
        "Combined: %d raw -> %d confidence -> %d regime",
        n_total_raw, n_after_confidence, n_after_regime,
    )

    # ------------------------------------------------------------------
    # Phase 2b: Per-asset allocation weighting (WALK-FORWARD)
    # Supports: "kelly", "nco", or "equal"
    # ------------------------------------------------------------------
    alloc_method = config.portfolio.allocation_method
    if config.portfolio.kelly_sizing or alloc_method in ("kelly", "nco"):
        import numpy as np
        from signal_engine.allocation import compute_kelly_weights, compute_nco_weights

        if alloc_method == "equal":
            logger.info("Equal-weight allocation (no per-asset weighting)")
        else:
            filtered = filtered.sort_values("timestamp").reset_index(drop=True)
            timestamps = pd.to_datetime(filtered["timestamp"])
            months = sorted(timestamps.dt.to_period("M").unique())

            adjusted = []
            for month in months:
                month_mask = timestamps.dt.to_period("M") == month
                month_signals = filtered[month_mask]
                past_mask = timestamps.dt.to_period("M") < month
                past_signals = filtered[past_mask]

                if len(past_signals) < 50:
                    adjusted.append(month_signals)
                    continue

                if alloc_method == "nco":
                    # NCO needs correlation matrix from past daily returns
                    past_ts = pd.to_datetime(past_signals["timestamp"])
                    assets_in_month = sorted(month_signals["asset"].unique())
                    # NEW (B-027): apply exponential time decay to past_signals.
                    # Recent year weighted more than year 1; prevents NCO from
                    # over-emphasizing stale assets in regime change.
                    # Decay: weight = exp(-months_ago / 12)
                    _months_ago = ((month - past_ts.dt.to_period("M"))
                                   .apply(lambda x: x.n)).astype(float)
                    _decay = np.exp(-_months_ago / 12.0)
                    # Build per-asset daily return series from past signals
                    daily_rets = {}
                    for a in assets_in_month:
                        a_mask = past_signals["asset"] == a
                        a_past = past_signals[a_mask]
                        if len(a_past) < 10:
                            continue
                        ts_a = pd.to_datetime(a_past["timestamp"])
                        rets = pd.Series(
                            a_past["direction"].values * a_past["fwd_return"].values
                            * _decay[a_mask].values,
                            index=ts_a,
                        )
                        daily_rets[a] = rets.resample("D").sum()
                    if len(daily_rets) < 3:
                        adjusted.append(month_signals)
                        continue
                    ret_df = pd.DataFrame(daily_rets).fillna(0)
                    corr = ret_df.corr().values
                    names = list(ret_df.columns)
                    exp_r = np.array([
                        float((past_signals[past_signals["asset"] == a]["direction"]
                               * past_signals[past_signals["asset"] == a]["fwd_return"]
                               * _decay[past_signals["asset"] == a].values).mean())
                        if a in past_signals["asset"].values else 0.0
                        for a in names
                    ])
                    weights = compute_nco_weights(
                        corr, exp_r, names,
                        n_clusters=config.portfolio.nco_clusters,
                    )
                else:
                    # NEW (B-027): apply time decay to Kelly's past_signals too.
                    _kp = past_signals.copy()
                    _kp["fwd_return"] = _kp["fwd_return"] * _decay.values
                    weights = compute_kelly_weights(_kp)

                month_copy = month_signals.copy()
                month_copy["signal_strength"] = month_copy.apply(
                    lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0),
                    axis=1,
                )
                adjusted.append(month_copy)

            filtered = pd.concat(adjusted, ignore_index=True)
            logger.info(
                "Walk-forward %s weighting applied across %d months",
                alloc_method.upper(), len(months),
            )

    # ------------------------------------------------------------------
    # Phase 2c: BTC vol regime sizing overlay (optional)
    # ------------------------------------------------------------------
    if config.vol_overlay.enabled:
        btc_symbol = config.assets.get("BTC", "BTCUSDT")
        try:
            data_dirs = [Path(d) for d in config.data.data_dirs]
            btc_1m = load_binance_1m(btc_symbol, data_dirs)
            btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
            if btc_daily.index.tz is not None:
                btc_daily.index = btc_daily.index.tz_localize(None)
            filtered = apply_vol_overlay(filtered, btc_daily, config.vol_overlay)
        except Exception as exc:
            logger.warning("Vol overlay skipped: %s", exc)

    # ------------------------------------------------------------------
    # Phase 2d: per-asset inverse-vol sizing overlay (risk parity)
    # ------------------------------------------------------------------
    if config.portfolio.per_asset_vol_scaling:
        data_dirs = [Path(d) for d in config.data.data_dirs]
        vol_series: dict[str, pd.Series] = {}
        for asset in filtered["asset"].unique():
            symbol = config.assets.get(asset, f"{asset}USDT")
            try:
                daily = resample_ohlcv(
                    load_binance_1m(symbol, data_dirs), "1D")["close"]
                if daily.index.tz is not None:
                    daily.index = daily.index.tz_localize(None)
                vs = daily.pct_change().rolling(20).std()
                if vs.notna().any():
                    vol_series[asset] = vs
            except Exception as exc:
                logger.warning("per-asset vol skipped for %s: %s", asset, exc)
        filtered = apply_per_asset_vol_overlay(
            filtered, vol_series,
            config.portfolio.per_asset_vol_target,
            config.portfolio.per_asset_vol_floor,
            config.portfolio.per_asset_vol_cap,
        )

    # ------------------------------------------------------------------
    # Phase 3: portfolio simulation
    # ------------------------------------------------------------------
    sim = simulate_portfolio(filtered, config.portfolio)

    daily_pnl = compute_daily_pnl_series(sim) if sim else None

    if daily_pnl is not None and sim is not None:
        n_trades = sim["n_trades"]
        n_active_days = len(daily_pnl)
        trades_per_day = n_trades / max(n_active_days, 1)
        stats = compute_full_stats(
            daily_pnl, config.portfolio.starting_capital, n_trades, trades_per_day
        )
    else:
        stats = {"error": "no_trades"}

    elapsed = round(time.time() - t_start, 1)
    logger.info("Backtest complete in %.1fs", elapsed)

    return {
        "stats": stats,
        "per_asset": per_asset_info,
        "n_total_raw": n_total_raw,
        "n_after_confidence": n_after_confidence,
        "n_after_regime": n_after_regime,
        "sim": sim,
        "daily_pnl": daily_pnl,
        "elapsed_s": elapsed,
    }
