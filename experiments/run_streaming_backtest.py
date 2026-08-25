"""Run leak-proof streaming backtest — parallelized.

Pre-computes causal features once, then runs 5 strategies in parallel.
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def run_single_strategy(args):
    """Run one strategy in a subprocess."""
    name, features_path, close_path, vol_path, ts_path, start_bar, use_features, vol_q, conf_p, strategy_type = args

    import sys
    sys.path.insert(0, "src")
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from signal_engine.streaming_backtest import StreamingBacktester
    from signal_engine.strategies import WalkForwardXGBStrategy, AlwaysLongStrategy, RandomStrategy

    features = pd.read_parquet(features_path)
    close = np.load(close_path)
    vol = np.load(vol_path)
    timestamps = pd.DatetimeIndex(np.load(ts_path, allow_pickle=True))

    if strategy_type == "xgb":
        strategy = WalkForwardXGBStrategy(
            fwd_bars=4,
            vol_filter_quantile=vol_q,
            confidence_percentile=conf_p,
            use_features=use_features,
        )
    elif strategy_type == "always_long":
        strategy = AlwaysLongStrategy()
    elif strategy_type == "random":
        strategy = RandomStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")

    engine = StreamingBacktester(
        fwd_bars=4, embargo_bars=5,
        retrain_frequency_bars=30 * 24 * 4,
        slippage=0.0001,
    )
    result = engine.run(strategy, features, close, vol, timestamps,
                        start_bar=start_bar, progress_every=20000)
    return name, result.sharpe, result.win_rate, result.mean_return_bps, result.n_trades, result.n_days


def main():
    t0 = time.time()
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features

    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading BTC 15min data...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    print(f"  {len(ohlcv_15m)} bars")

    print("Pre-computing features (causal — no lookahead)...")
    ft0 = time.time()
    features = compute_features(ohlcv_15m, cfg.features)
    print(f"  {features.shape[1]} features in {time.time()-ft0:.1f}s")

    close = ohlcv_15m["close"].to_numpy(dtype=np.float64)
    vol = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)
    timestamps = ohlcv_15m.index

    start_ts = pd.Timestamp("2022-01-01", tz="UTC")
    if timestamps.tz is None:
        start_ts = start_ts.tz_localize(None)
    start_bar = int(timestamps.searchsorted(start_ts))
    print(f"  Backtest from bar {start_bar} ({timestamps[start_bar]})")

    # Save to temp files for subprocess access
    tmp = Path("experiments/cache/_streaming_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    features.to_parquet(tmp / "features.parquet")
    np.save(tmp / "close.npy", close)
    np.save(tmp / "vol.npy", vol)
    np.save(tmp / "timestamps.npy", timestamps.values)

    fp = str(tmp / "features.parquet")
    cp = str(tmp / "close.npy")
    vp = str(tmp / "vol.npy")
    tp = str(tmp / "timestamps.npy")

    tasks = [
        # (name, feat_path, close_path, vol_path, ts_path, start_bar, use_features, vol_q, conf_p, type)
        ("XGB Real Features",     fp, cp, vp, tp, start_bar, True,  0.50, 70.0, "xgb"),
        ("XGB Constant Features", fp, cp, vp, tp, start_bar, False, 0.50, 70.0, "xgb"),
        ("XGB No Vol Filter",     fp, cp, vp, tp, start_bar, True,  0.00, 70.0, "xgb"),
        ("XGB No Conf Filter",    fp, cp, vp, tp, start_bar, True,  0.50,  0.0, "xgb"),
        ("Always Long",           fp, cp, vp, tp, start_bar, True,  0.00,  0.0, "always_long"),
        ("Random",                fp, cp, vp, tp, start_bar, True,  0.00,  0.0, "random"),
    ]

    workers = min(len(tasks), max(1, os.cpu_count() or 1))
    print(f"\nRunning {len(tasks)} strategies with {workers} workers...\n")

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, sharpe, win, bps, trades, days in pool.map(run_single_strategy, tasks):
            results[name] = (sharpe, win, bps, trades, days)
            print(f"  {name:<25s} Sharpe={sharpe:>7.3f}  Win={win*100:>5.1f}%  Mean={bps:>6.1f}bps  Trades={trades:>6d}  Days={days}", flush=True)

    print()
    print("=" * 75)
    print("STREAMING BACKTEST COMPARISON (BTC, flat $5 bets, leak-proof)")
    print("=" * 75)
    print(f"  {'Strategy':<25s} {'Sharpe':>8s} {'Win%':>7s} {'bps':>7s} {'Trades':>8s} {'Days':>6s}")
    print("  " + "-" * 63)
    for name in [t[0] for t in tasks]:
        s, w, b, tr, d = results[name]
        print(f"  {name:<25s} {s:>8.3f} {w*100:>6.1f}% {b:>6.1f} {tr:>8d} {d:>6d}")

    real = results.get("XGB Real Features", (0,0,0,0,0))
    const = results.get("XGB Constant Features", (0,0,0,0,0))
    always = results.get("Always Long", (0,0,0,0,0))
    rand = results.get("Random", (0,0,0,0,0))

    print()
    print("INTERPRETATION:")
    print(f"  Feature contribution:    {real[0] - const[0]:+.3f} Sharpe")
    print(f"  Model over always-long:  {real[0] - always[0]:+.3f} Sharpe")
    print(f"  Model over random:       {real[0] - rand[0]:+.3f} Sharpe")
    print(f"  Constant over random:    {const[0] - rand[0]:+.3f} Sharpe (base rate exploit)")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
