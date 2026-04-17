"""Close model integration for the combined portfolio backtest.

Orchestrates: OHLCV loading → close decision examples → walk-forward
training → threshold selection → adjusted fwd_return signals.

All building blocks are in close_model.py — this module wires them
together for the multi-strategy backtest pipeline.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_engine.close_model import (
    build_close_decision_examples,
    close_feature_columns,
    select_close_predictions,
    walk_forward_close_prediction_cache,
)
from signal_engine.config import CloseModelConfig, FeatureConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features

logger = logging.getLogger(__name__)

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
CACHE_DIR = Path("data/cache/portfolio_close")


def _build_asset_examples(
    asset_name: str,
    symbol: str,
    signals: pd.DataFrame,
    strategy: str,
    feature_config: FeatureConfig,
    close_config: CloseModelConfig,
) -> pd.DataFrame | None:
    """Build close decision examples for one (asset, strategy) pair."""
    import warnings
    warnings.filterwarnings("ignore")

    asset_sigs = signals[signals["asset"] == asset_name].copy()
    if len(asset_sigs) < 5:
        return None

    # Determine timeframe params
    if strategy == "logreg_15m":
        tf = "15min"
        max_hold = 60
        min_hold = 15
    elif strategy == "logreg_4h":
        tf = "4h"
        max_hold = 480
        min_hold = 240
    else:
        return None

    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
        ohlcv_tf = resample_ohlcv(ohlcv_1m, tf)
        ohlcv_1h = resample_ohlcv(ohlcv_1m, "1h")

        features_tf = compute_features(ohlcv_tf, feature_config)
        features_1h = compute_features(ohlcv_1h, feature_config)

        examples = build_close_decision_examples(
            signals_df=asset_sigs,
            entry_ohlcv=ohlcv_tf,
            decision_ohlcv=ohlcv_tf,
            decision_features=features_tf,
            feature_cols=feature_config.feature_list,
            decision_timeframe=tf,
            max_hold_minutes=max_hold,
            min_hold_minutes=min_hold,
            slippage=0.0001,
            label_mode=close_config.label_mode,
            pt_mult=close_config.pt_mult,
            sl_mult=close_config.sl_mult,
            min_improvement_bps=close_config.min_improvement_bps,
            regime_feature_sets={"1h": features_1h},
        )

        if len(examples) > 0:
            # Prefix trade_id with asset for cross-asset uniqueness
            if "trade_id" in examples.columns:
                examples["trade_id"] = asset_name + ":" + examples["trade_id"].astype(str)
            return examples

    except Exception as e:
        logger.warning("Close examples failed for %s/%s: %s", asset_name, strategy, e)

    return None


def compute_close_adjusted_signals(
    signals: pd.DataFrame,
    strategy: str,
    assets: dict[str, str],
    feature_config: FeatureConfig,
    close_config: CloseModelConfig,
    portfolio_config: PortfolioConfig,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Generate close-model-adjusted signals for one LogReg strategy.

    Pipeline:
    1. Build close decision examples per asset (parallel)
    2. Combine into single DataFrame
    3. Walk-forward train/predict (monthly, cached)
    4. Select threshold and materialize adjusted returns

    Returns:
        DataFrame with adjusted fwd_return, exit_timestamp, etc.
        Columns include return_already_signed=True.
    """
    t0 = time.time()

    # Check cache
    cache_path = Path(f"results/signals_cache/{strategy}_close_adjusted.parquet")
    if cache_path.exists():
        logger.info("Loading cached close-adjusted signals: %s", cache_path)
        return pd.read_parquet(cache_path)

    strat_signals = signals[signals["strategy"] == strategy].copy()
    if len(strat_signals) < 50:
        logger.warning("Too few signals for %s (%d), skipping close model", strategy, len(strat_signals))
        return strat_signals

    bar_minutes = 15 if strategy == "logreg_15m" else 240

    # Phase 1B: Build examples per asset (parallel)
    logger.info("[1B] Building close examples for %s (%d assets)...", strategy, len(assets))
    all_examples = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _build_asset_examples, name, sym, strat_signals,
                strategy, feature_config, close_config,
            ): name
            for name, sym in assets.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    all_examples.append(result)
                    logger.info("  %s: %d examples", name, len(result))
            except Exception as e:
                logger.warning("  %s: failed (%s)", name, e)

    if not all_examples:
        logger.warning("No close examples generated for %s", strategy)
        return strat_signals

    # Phase 2A: Combine
    combined = pd.concat(all_examples, ignore_index=True)
    logger.info("[2A] Combined: %d examples for %s", len(combined), strategy)

    # Phase 2B: Walk-forward train/predict
    logger.info("[2B] Walk-forward close model training for %s...", strategy)
    feat_cols = close_feature_columns(
        feature_config.feature_list, regime_timeframes=["1h"]
    )

    predictions, validation, diagnostics = walk_forward_close_prediction_cache(
        decision_examples=combined,
        feature_cols=feat_cols,
        lookback_months=close_config.lookbacks[0] if close_config.lookbacks else 10,
        close_cfg=close_config,
        test_start="2021-01",
        signal_bar_minutes=bar_minutes,
        min_train_rows=200,
        portfolio_cfg=portfolio_config,
    )

    if len(predictions) == 0:
        logger.warning("No predictions from close model for %s", strategy)
        return strat_signals

    logger.info("[2B] Predictions: %d rows for %s", len(predictions), strategy)

    # Phase 2C: Threshold selection
    logger.info("[2C] Selecting threshold for %s...", strategy)
    selected, diag = select_close_predictions(
        predictions, validation, close_config,
        signal_bar_minutes=bar_minutes,
        portfolio_cfg=portfolio_config,
        prediction_diagnostics=diagnostics,
    )

    if len(selected) == 0:
        logger.warning("No selected trades from close model for %s", strategy)
        return strat_signals

    # Add strategy tag
    selected["strategy"] = strategy

    # Determine hold_bars from actual bars_held
    if "bars_held" in selected.columns:
        selected["hold_bars"] = selected["bars_held"]
    else:
        selected["hold_bars"] = 4 if strategy == "logreg_15m" else 2

    elapsed = time.time() - t0
    logger.info(
        "[DONE] %s close model: %d -> %d signals in %.0fs "
        "(early_exit=%.1f%%, avg_hold=%.1f bars)",
        strategy, len(strat_signals), len(selected), elapsed,
        diag.get("diagnostics", {}).get("positive_rate", 0) * 100,
        selected["hold_bars"].mean() if "hold_bars" in selected.columns else -1,
    )

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(cache_path, index=False)

    return selected
