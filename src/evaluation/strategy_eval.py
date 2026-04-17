"""
Strategy Battery — run all strategies on all assets, produce comparison table.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.loader import load_symbol
from models.vol_sizer import VolatilitySizer
from strategies.vol_scaled_sizing import VolScaledSizing
from strategies.vol_squeeze import VolSqueeze
from strategies.vol_breakout import VolBreakout
from strategies.vol_regime_momentum import VolRegimeMomentum
from strategies.cross_asset_vol import CrossAssetVol

logger = logging.getLogger(__name__)


def _run_single(strategy, ohlcv, predicted_vol, symbol, tf, **kwargs):
    """Run one strategy on one asset. Thread-safe."""
    try:
        result = strategy.backtest(ohlcv, predicted_vol, **kwargs)
        result["symbol"] = symbol
        result["timeframe"] = tf
        return result
    except Exception as e:
        return {"strategy": strategy.name, "symbol": symbol, "timeframe": tf, "error": str(e)}


def run_strategy_battery(
    data_dir: str = "data/raw/binance-1m",
    cache_dir: str = "data/cache",
    symbols: list[str] | None = None,
    timeframe: str = "15m",
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Run all strategies on all assets and produce a comparison table.

    Returns DataFrame with one row per (strategy, symbol) combination.
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Load data and train vol models
    data = {}
    vol_preds = {}
    for sym in symbols:
        try:
            ohlcv = load_symbol(sym, data_dir, timeframe, cache_dir=cache_dir)
            sizer = VolatilitySizer()
            sizer.train(ohlcv, forward_window=10)
            # Get predicted vol for full series
            feats = sizer.compute_features(ohlcv)
            preds = pd.Series(
                sizer.model.predict(feats.dropna()),
                index=feats.dropna().index,
            )
            data[sym] = ohlcv
            vol_preds[sym] = preds
        except Exception as e:
            logger.warning(f"Failed to load {sym}: {e}")

    if not data:
        return pd.DataFrame()

    # Define strategies
    strategies = [
        VolScaledSizing(crash_threshold=-0.005),     # Strategy 5 (15m threshold)
        VolSqueeze(expansion_ratio=1.5),              # Strategy 3
        VolBreakout(zscore_threshold=2.0),            # Strategy 2
        VolRegimeMomentum(momentum_lookback=14),      # Strategy 1
    ]

    # Run all combinations in parallel
    results = []
    jobs = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for sym in data:
            for strat in strategies:
                fut = pool.submit(
                    _run_single, strat, data[sym], vol_preds[sym], sym, timeframe,
                )
                jobs.append(fut)

            # Cross-asset: use BTC as reference for non-BTC assets
            if "BTCUSDT" in data and sym != "BTCUSDT":
                cross = CrossAssetVol()
                fut = pool.submit(
                    lambda s=sym: cross.backtest(data[s], vol_preds[s]),
                )
                # Can't easily pass extra kwargs through backtest... skip for now
                # Cross-asset needs special handling

        for fut in as_completed(jobs):
            results.append(fut.result())

    df = pd.DataFrame(results)

    # Sort by net_sharpe descending
    if "net_sharpe" in df.columns:
        df = df.sort_values("net_sharpe", ascending=False)

    return df


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table."""
    if len(df) == 0:
        print("No results")
        return

    cols = ["strategy", "symbol", "win_rate", "p_value", "net_sharpe", "n_trades_dir", "max_drawdown", "hhi"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    logging.basicConfig(level=logging.WARNING)
    df = run_strategy_battery()
    print("\n=== STRATEGY COMPARISON ===\n")
    print_comparison(df)
    df.to_csv("results/strategy_comparison.csv", index=False)
