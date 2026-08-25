"""No-feature test: replace all features with constants.

Also tests with and without the vol filter to isolate its contribution.
"""

import sys
sys.path.insert(0, "src")

import os
os.environ["ML_CLOSE_SKIP_CV"] = "1"

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LB = 8
SEARCH_START = "2022-01"
SEARCH_END = "2024-12"
HOLDOUT_START = "2025-01"


def portfolio_sharpe(df, pcfg):
    """Correct Sharpe via portfolio simulation (daily PnL, no overlap inflation)."""
    if df is None or len(df) < 20:
        return 0.0
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    sim = simulate_portfolio(df, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return 0.0
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return 0.0
    dr = dpnl / pcfg.starting_capital
    s = dr.std()
    return float(dr.mean() / s * np.sqrt(365.25)) if s > 0 else 0.0


def period_filter(df, start, end=None):
    months = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m")
    mask = months >= start
    if end:
        mask &= months <= end
    return df.loc[mask].reset_index(drop=True)


def run_variant(args):
    label, use_constant_features, vol_filter_quantile = args

    import sys
    sys.path.insert(0, "src")
    import os
    os.environ["ML_CLOSE_SKIP_CV"] = "1"

    import numpy as np
    from pathlib import Path
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features

    from signal_engine.research import close_walkforward as _mod
    generate_asset_signals = _mod.generate_asset_signals

    cfg = load_config(Path("configs/default.yaml"))

    # Override vol filter quantile
    cfg.features.vol_filter_quantile = vol_filter_quantile

    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    features_15m = compute_features(ohlcv_15m, cfg.features)

    regime_features = {}
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf) if rtf != "15min" else ohlcv_15m
        regime_features[rtf] = compute_features(rtf_ohlcv, cfg.features)

    if use_constant_features:
        # Replace all features with constant 0.0 — model has NO information
        for col in features_15m.columns:
            features_15m[col] = 0.0

    _mod.LOOKBACKS = [LB]
    signals = generate_asset_signals(f"BTC_{label}", ohlcv_15m, features_15m, regime_features, cfg)
    df = signals.get(LB, pd.DataFrame())

    search = period_filter(df, SEARCH_START, SEARCH_END) if len(df) > 0 else pd.DataFrame()
    holdout = period_filter(df, HOLDOUT_START) if len(df) > 0 else pd.DataFrame()

    from signal_engine.config import PortfolioConfig
    pcfg = PortfolioConfig(**cfg.portfolio.model_dump())
    sr_s = portfolio_sharpe(search, pcfg)
    sr_h = portfolio_sharpe(holdout, pcfg)

    # Compute base rate of filtered bars
    long_frac = float((search["direction"] == 1).mean()) if len(search) > 0 else 0
    win_rate = float((search["direction"].values * search["fwd_return"].values > 0).mean()) if len(search) > 0 else 0
    mean_ret = float((search["direction"].values * search["fwd_return"].values).mean() * 10000) if len(search) > 0 else 0

    return label, sr_s, sr_h, len(search), len(holdout), long_frac, win_rate, mean_ret


def main():
    t0 = time.time()

    tasks = [
        # (label, constant_features, vol_filter_quantile)
        ("Real features + vol filter (baseline)", False, 0.50),
        ("Constant features + vol filter", True, 0.50),
        ("Real features + NO vol filter", False, 0.0),
        ("Constant features + NO vol filter", True, 0.0),
        ("Real features + strict vol filter (75th)", False, 0.75),
        ("Constant features + strict vol filter (75th)", True, 0.75),
    ]

    print(f"Running {len(tasks)} variants with parallel workers...")
    print()

    workers = min(len(tasks), max(1, os.cpu_count() or 1))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_variant, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            label, sr_s, sr_h, n_s, n_h, long_frac, win_rate, mean_ret = future.result()
            results.append((label, sr_s, sr_h, n_s, n_h, long_frac, win_rate, mean_ret))
            print(f"  [{i}/{len(tasks)}] {label}: done", flush=True)

    # Sort by original task order
    task_order = {t[0]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: task_order.get(r[0], 99))

    print()
    print("=" * 95)
    print("RESULTS")
    print("=" * 95)
    print(f"  {'Variant':<45s} {'SR(s)':>7s} {'SR(h)':>7s} {'n(s)':>7s} {'Long%':>6s} {'Win%':>6s} {'bps':>6s}")
    print("  " + "-" * 89)
    for label, sr_s, sr_h, n_s, n_h, long_frac, win_rate, mean_ret in results:
        print(f"  {label:<45s} {sr_s:>7.1f} {sr_h:>7.1f} {n_s:>7d} {long_frac*100:>5.1f}% {win_rate*100:>5.1f}% {mean_ret:>5.0f}")

    print()
    print("INTERPRETATION:")
    r = {label: (sr_s, sr_h, long_frac, win_rate) for label, sr_s, sr_h, _, _, long_frac, win_rate, _ in results}
    baseline = r.get("Real features + vol filter (baseline)", (0, 0, 0, 0))
    const_vol = r.get("Constant features + vol filter", (0, 0, 0, 0))
    real_novol = r.get("Real features + NO vol filter", (0, 0, 0, 0))
    const_novol = r.get("Constant features + NO vol filter", (0, 0, 0, 0))

    print(f"  Vol filter contribution:   {baseline[0] - real_novol[0]:+.1f} Sharpe (search)")
    print(f"  Feature contribution:      {baseline[0] - const_vol[0]:+.1f} Sharpe (search)")
    print(f"  Pure base rate (no feat, no vol): Sharpe {const_novol[0]:.1f}")
    print(f"  Vol filter win rate bias:  {const_vol[3]*100:.1f}% (with) vs {const_novol[3]*100:.1f}% (without)")
    print(f"\n  Completed in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
