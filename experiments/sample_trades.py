"""Extract 100 sample trades with surrounding OHLCV context.

Runs the streaming backtest, picks 100 trades spread across the period,
and saves each trade's surrounding OHLCV bars (20 before, fwd_bars after)
to a CSV for manual inspection.
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.streaming_backtest import StreamingBacktester, Signal

# Reuse strategy from ablation
from ablation_binary_search import AblationStrategy

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

LOOKBACK_MONTHS = 8
FWD_BARS = 4
CONTEXT_BEFORE = 20
CONTEXT_AFTER = FWD_BARS + 4


def main():
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading data...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

    print("Computing features...")
    base_features = compute_features(ohlcv_15m, cfg.features)
    regime_parts = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        aligned = rtf_feat.reindex(ohlcv_15m.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts.append(aligned)
    all_features = pd.concat([base_features] + regime_parts, axis=1)

    close = ohlcv_15m["close"].to_numpy(dtype=np.float64)
    vol = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)
    timestamps = ohlcv_15m.index

    start_ts = pd.Timestamp("2022-01-01", tz="UTC")
    if timestamps.tz is None:
        start_ts = start_ts.tz_localize(None)
    start_bar = int(timestamps.searchsorted(start_ts))

    print("Running backtest...")
    engine = StreamingBacktester(
        fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1, slippage=0.0001,
        retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
        confidence_mode="none",
    )
    result = engine.run(AblationStrategy(), all_features, close, vol, timestamps,
                        start_bar=start_bar, progress_every=0)

    trades = result.trades
    print(f"  {len(trades)} total trades, Sharpe={result.sharpe:.2f}")

    # Pick 100 evenly spaced trades
    n = len(trades)
    indices = np.linspace(0, n - 1, 100, dtype=int)
    sample = [trades[i] for i in indices]

    # Build trade summary
    rows = []
    for i, t in enumerate(sample):
        rows.append({
            "trade_id": i,
            "entry_ts": t.entry_ts,
            "exit_ts": t.exit_ts,
            "entry_bar": t.entry_bar,
            "exit_bar": t.exit_bar,
            "direction": "LONG" if t.direction > 0 else "SHORT",
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "fwd_return_bps": t.fwd_return * 10000,
            "pnl_bps": t.pnl * 10000,
            "win": t.pnl > 0,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv("experiments/sample_trades_summary.csv", index=False)
    print(f"\nSaved trade summary to experiments/sample_trades_summary.csv")

    # Build OHLCV context for each trade
    context_rows = []
    for i, t in enumerate(sample):
        bar_start = max(0, t.entry_bar - CONTEXT_BEFORE)
        bar_end = min(len(ohlcv_15m) - 1, t.entry_bar + CONTEXT_AFTER)
        for b in range(bar_start, bar_end + 1):
            rel = b - t.entry_bar  # negative = before entry, 0 = entry, positive = after
            marker = ""
            if b == t.entry_bar:
                marker = "ENTRY"
            elif b == t.exit_bar:
                marker = "EXIT"
            context_rows.append({
                "trade_id": i,
                "bar_idx": b,
                "rel_bar": rel,
                "marker": marker,
                "timestamp": timestamps[b],
                "open": ohlcv_15m.iloc[b]["open"],
                "high": ohlcv_15m.iloc[b]["high"],
                "low": ohlcv_15m.iloc[b]["low"],
                "close": ohlcv_15m.iloc[b]["close"],
                "volume": ohlcv_15m.iloc[b]["volume"],
                "direction": "LONG" if t.direction > 0 else "SHORT",
                "pnl_bps": t.pnl * 10000,
            })

    context = pd.DataFrame(context_rows)
    context.to_csv("experiments/sample_trades_context.csv", index=False)
    print(f"Saved OHLCV context to experiments/sample_trades_context.csv")

    # Print a few examples
    print(f"\n{'='*80}")
    print("SAMPLE TRADES (first 10)")
    print(f"{'='*80}")
    for i in range(min(10, len(sample))):
        t = sample[i]
        direction = "LONG" if t.direction > 0 else "SHORT"
        win = "WIN" if t.pnl > 0 else "LOSS"
        print(f"\n  Trade {i}: {direction} at {t.entry_ts} -> {t.exit_ts}")
        print(f"    Entry={t.entry_price:.2f}  Exit={t.exit_price:.2f}  "
              f"Return={t.fwd_return*10000:.1f}bps  PnL={t.pnl*10000:.1f}bps  {win}")

        # Show surrounding bars
        bar_start = max(0, t.entry_bar - 5)
        bar_end = min(len(ohlcv_15m) - 1, t.entry_bar + FWD_BARS + 2)
        print(f"    {'Bar':>5s} {'Time':>20s} {'Open':>10s} {'High':>10s} {'Low':>10s} "
              f"{'Close':>10s} {'Vol':>12s}")
        for b in range(bar_start, bar_end + 1):
            marker = " <-- ENTRY" if b == t.entry_bar else (" <-- EXIT" if b == t.exit_bar else "")
            ts = str(timestamps[b])[:19]
            row = ohlcv_15m.iloc[b]
            print(f"    {b:>5d} {ts:>20s} {row['open']:>10.2f} {row['high']:>10.2f} "
                  f"{row['low']:>10.2f} {row['close']:>10.2f} {row['volume']:>12.1f}{marker}")

    # Stats on the sample
    wins = summary["win"].sum()
    print(f"\n{'='*80}")
    print(f"SAMPLE STATS: {wins}/100 wins ({wins}%), "
          f"mean PnL={summary['pnl_bps'].mean():.1f}bps, "
          f"long={summary['direction'].eq('LONG').sum()}%, short={summary['direction'].eq('SHORT').sum()}%")


if __name__ == "__main__":
    main()
