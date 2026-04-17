"""
Experiment Runner — Orchestrates baseline vs ML model comparisons.

Runs all models on identical data splits with identical cost assumptions.
Produces comparison tables and GO/NO-GO decisions.

This is the core of Milestone 3: honest, fair evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import load_symbol, list_available_symbols
from models.baselines import AlwaysFlat, BuyAndHold, MomentumBaseline, SARIMABaseline, evaluate_baseline
from afml.evaluation import rank_ic, directional_accuracy, net_sharpe, deflated_sharpe_ratio

logger = logging.getLogger(__name__)


# Cost models per plan v2
COST_NORMAL = 0.0015    # 0.15% round-trip (taker + slippage)
COST_CRASH = 0.0035     # 0.35% round-trip (crash event conditions)

TIMEFRAMES = ["5m", "15m", "1H", "4H"]  # Skip 1m (too expensive for SARIMA)
EVAL_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run_baseline_experiment(
    data_dir: str | Path,
    output_dir: str | Path = "results/baselines",
    timeframes: list[str] | None = None,
    assets: list[str] | None = None,
    cache_dir: str | Path | None = "data/cache",
) -> pd.DataFrame:
    """
    Run all baseline models on specified assets and timeframes.

    Args:
        data_dir: Root directory with per-symbol Binance CSV dirs.
        output_dir: Where to save results.
        timeframes: List of timeframes to test.
        assets: List of symbols to test.
        cache_dir: Cache directory for resampled data.

    Returns:
        DataFrame with one row per (model, asset, timeframe) combination.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES
    if assets is None:
        available = list_available_symbols(data_dir)
        assets = [a for a in EVAL_ASSETS if a in available]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for asset in assets:
        for tf in timeframes:
            logger.info(f"=== {asset} {tf} ===")

            try:
                ohlcv = load_symbol(asset, data_dir, tf, cache_dir=cache_dir)
            except Exception as e:
                logger.warning(f"Failed to load {asset} {tf}: {e}")
                continue

            close = ohlcv["close"]
            returns = close.pct_change().dropna()

            if len(returns) < 200:
                logger.warning(f"{asset} {tf}: only {len(returns)} bars, skipping")
                continue

            # Split: use last 20% as test
            split = int(len(returns) * 0.8)
            test_returns = returns.iloc[split:]
            test_close = close.iloc[split - 1:]  # include one extra for initial context

            # --- Run baselines ---
            baselines = {
                "always_flat": AlwaysFlat(),
                "buy_hold": BuyAndHold(),
                "momentum_7": MomentumBaseline(lookback=7),
                "momentum_14": MomentumBaseline(lookback=14),
                "momentum_42": MomentumBaseline(lookback=42),
            }

            for name, model in baselines.items():
                pred = model.predict(test_close)
                # Align with test returns
                common = pred.dropna().index.intersection(test_returns.index)
                if len(common) < 50:
                    continue

                result = evaluate_baseline(
                    pred.loc[common], test_returns.loc[common], COST_NORMAL,
                )
                result.update({
                    "model": name,
                    "asset": asset,
                    "timeframe": tf,
                    "n_bars_total": len(ohlcv),
                    "n_bars_test": len(test_returns),
                })
                results.append(result)

            # --- SARIMA (fast mode for screening) ---
            logger.info(f"  Running SARIMA fast on {asset} {tf}...")
            try:
                sarima = SARIMABaseline(
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 1, _get_seasonal_period(tf)),
                )
                t0 = time.time()
                pred = sarima.predict_fast(test_close, eval_points=min(100, len(test_close) // 2))
                elapsed = time.time() - t0

                common = pred.dropna().index.intersection(test_returns.index)
                if len(common) >= 20:
                    result = evaluate_baseline(
                        pred.loc[common], test_returns.loc[common], COST_NORMAL,
                    )
                    result.update({
                        "model": "sarima_fast",
                        "asset": asset,
                        "timeframe": tf,
                        "n_bars_total": len(ohlcv),
                        "n_bars_test": len(test_returns),
                        "eval_time_sec": elapsed,
                    })
                    results.append(result)
            except Exception as e:
                logger.warning(f"  SARIMA failed on {asset} {tf}: {e}")

    # Build results table
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "baseline_results.csv", index=False)

    # Pretty print summary
    if len(df) > 0:
        summary = df.groupby(["model", "timeframe"]).agg({
            "accuracy": "mean",
            "net_sharpe": "mean",
            "gross_sharpe": "mean",
        }).round(4)
        summary.to_csv(output_dir / "baseline_summary.csv")
        logger.info(f"\n{summary}")

    return df


def run_kronos_experiment(
    data_dir: str | Path,
    output_dir: str | Path = "results/kronos",
    timeframes: list[str] | None = None,
    assets: list[str] | None = None,
    n_eval: int = 50,
    cache_dir: str | Path | None = "data/cache",
) -> pd.DataFrame:
    """
    Run Kronos zero-shot on specified assets and timeframes.

    Args:
        data_dir: Root data directory.
        output_dir: Where to save results.
        timeframes: Timeframes to test.
        assets: Symbols to test.
        n_eval: Number of evaluation points per asset per timeframe.
        cache_dir: Cache directory.

    Returns:
        DataFrame with Kronos results.
    """
    from models.kronos_wrapper import quick_kronos_test

    if timeframes is None:
        timeframes = TIMEFRAMES
    if assets is None:
        available = list_available_symbols(data_dir)
        assets = [a for a in EVAL_ASSETS if a in available]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for asset in assets:
        for tf in timeframes:
            logger.info(f"=== Kronos {asset} {tf} ===")
            try:
                ohlcv = load_symbol(asset, data_dir, tf, cache_dir=cache_dir)
            except Exception as e:
                logger.warning(f"Failed to load {asset} {tf}: {e}")
                continue

            if len(ohlcv) < 250:
                logger.warning(f"{asset} {tf}: too few bars ({len(ohlcv)})")
                continue

            # Use last 20% for evaluation
            split = int(len(ohlcv) * 0.8)
            test_ohlcv = ohlcv.iloc[split - 180:]  # 180 lookback + test

            t0 = time.time()
            result = quick_kronos_test(test_ohlcv, n_eval=n_eval)
            elapsed = time.time() - t0

            if "error" in result:
                logger.warning(f"  Kronos failed: {result['error']}")
                continue

            results.append({
                "model": result["model"],
                "asset": asset,
                "timeframe": tf,
                "accuracy": result["accuracy"],
                "n_eval": result["n_eval"],
                "eval_time_sec": elapsed,
            })

            logger.info(f"  Accuracy: {result['accuracy']:.3f} ({result['n_eval']} points, {elapsed:.1f}s)")

    df = pd.DataFrame(results)
    if len(df) > 0:
        df.to_csv(output_dir / "kronos_results.csv", index=False)

        summary = df.groupby(["model", "timeframe"]).agg({
            "accuracy": "mean",
            "n_eval": "sum",
        }).round(4)
        summary.to_csv(output_dir / "kronos_summary.csv")
        logger.info(f"\n{summary}")

    return df


def _get_seasonal_period(timeframe: str) -> int:
    """Get SARIMA seasonal period for a timeframe."""
    periods = {
        "1m": 60,    # 1 hour
        "5m": 12,    # 1 hour
        "15m": 4,    # 1 hour
        "1H": 24,    # 1 day
        "4H": 6,     # 1 day
        "1D": 7,     # 1 week
    }
    return periods.get(timeframe, 6)
