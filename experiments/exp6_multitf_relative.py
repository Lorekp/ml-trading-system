"""EXP-6: Multi-TF Relative Features — PROPER Re-Test.

Previous test added raw copies of 15m features at 1H/4H. That was unfair
because a linear model can't learn cross-timeframe interactions from raw copies.

This re-test uses RELATIVE features:
- RSI divergence: rsi_14_4h - rsi_14_15m (bullish when 4H RSI > 15m RSI)
- Price vs higher TF trend: close_15m / ema20_4h - 1 (distance from 4H trend)
- Vol ratio: atr_15m / atr_4h (relative volatility across timeframes)
- Momentum alignment: sign(ret_1_15m) * sign(ret_1_4h) (agreement)
"""

import sys
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import time
from pathlib import Path

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.technical import rsi, atr


def compute_relative_mtf_features(ohlcv_15m, ohlcv_4h, feature_config):
    """Compute relative multi-timeframe features.

    All features are RELATIVE (differences/ratios between timeframes),
    not raw copies. This gives the linear model actionable cross-TF info.

    CRITICAL: 4H features use shift(1) to avoid look-ahead.
    """
    close_15m = ohlcv_15m["close"]

    # 4H indicators (shift(1) to use PREVIOUS completed bar)
    rsi_4h = rsi(ohlcv_4h["close"], 14).shift(1)
    atr_4h = atr(ohlcv_4h["high"], ohlcv_4h["low"], ohlcv_4h["close"], 14).shift(1)
    ema20_4h = ohlcv_4h["close"].ewm(span=20, adjust=False).mean().shift(1)
    ret_1_4h = ohlcv_4h["close"].pct_change(1).shift(1)

    # Align 4H to 15m timestamps (forward-fill)
    rsi_4h_aligned = rsi_4h.reindex(ohlcv_15m.index, method="ffill")
    atr_4h_aligned = atr_4h.reindex(ohlcv_15m.index, method="ffill")
    ema20_4h_aligned = ema20_4h.reindex(ohlcv_15m.index, method="ffill")
    ret_1_4h_aligned = ret_1_4h.reindex(ohlcv_15m.index, method="ffill")

    # 15m indicators
    rsi_15m = rsi(close_15m, 14)
    atr_15m = atr(ohlcv_15m["high"], ohlcv_15m["low"], close_15m, 14)
    ret_1_15m = close_15m.pct_change(1)

    # Relative features
    f = pd.DataFrame(index=ohlcv_15m.index)
    f["rsi_divergence"] = (rsi_4h_aligned - rsi_15m).astype(np.float32)
    f["price_vs_4h_trend"] = ((close_15m / ema20_4h_aligned) - 1).astype(np.float32)
    f["vol_ratio_15m_4h"] = (atr_15m / atr_4h_aligned.replace(0, np.nan)).astype(np.float32)
    f["momentum_alignment"] = (np.sign(ret_1_15m) * np.sign(ret_1_4h_aligned)).astype(np.float32)

    return f


def main():
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(d) for d in cfg.data.data_dirs]
    assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
              "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

    print("=" * 70)
    print("EXP-6: Multi-TF RELATIVE Features — Fair Re-Test")
    print("  4 relative features: RSI divergence, price vs 4H trend,")
    print("  vol ratio, momentum alignment. All shift(1) on 4H.")
    print("=" * 70, flush=True)

    configs_to_test = [
        ("14 features (baseline)", False),
        ("14 + 4 relative MTF (18 features)", True),
    ]

    for label, add_mtf in configs_to_test:
        print(f"\n--- {label} ---", flush=True)
        all_signals = []
        t0 = time.time()

        for asset_name, symbol in assets.items():
            print(f"  {asset_name}...", end="", flush=True)
            ohlcv_1m = load_binance_1m(symbol, data_dirs)
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")

            features = compute_features(ohlcv_15m, cfg.features)

            if add_mtf:
                mtf = compute_relative_mtf_features(ohlcv_15m, ohlcv_4h, cfg.features)
                features = pd.concat([features, mtf], axis=1)
                # Update feature list for this run
                import copy
                cfg_mod = copy.deepcopy(cfg)
                cfg_mod.features.feature_list = cfg.features.feature_list + list(mtf.columns)
            else:
                cfg_mod = cfg

            wf = WalkForwardModel(cfg_mod)
            raw = wf.generate_signals_batch(ohlcv_15m, features, asset_name)

            if len(raw) > 0:
                filtered = apply_confidence_filter(raw, cfg_mod.confidence_filter)
                all_signals.append(filtered)
                print(f" {len(filtered)} signals", flush=True)
            else:
                print(" no signals", flush=True)

        elapsed = time.time() - t0

        if all_signals:
            combined = pd.concat(all_signals).sort_values("timestamp")
            sim = simulate_portfolio(combined, cfg.portfolio)
            if sim and sim["n_trades"] >= 10:
                dpnl = compute_daily_pnl_series(sim)
                if dpnl is not None:
                    stats = compute_full_stats(
                        dpnl, cfg.portfolio.starting_capital,
                        sim["n_trades"], sim["n_trades"] / len(dpnl),
                    )
                    print(f"\n  RESULTS ({label}):")
                    print(f"    Sharpe:  {stats['sharpe']:.3f}")
                    print(f"    Return:  {stats['total_return_pct']:.2f}%")
                    print(f"    MaxDD:   {stats['max_dd_pct']:.2f}%")
                    print(f"    Trades:  {sim['n_trades']}")
                    print(f"    Time:    {elapsed:.0f}s")


if __name__ == "__main__":
    main()
