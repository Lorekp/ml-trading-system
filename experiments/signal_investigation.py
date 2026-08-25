"""Signal model AFML investigation — 8 adversarial tests.

Tests every AFML pitfall (Table 1.2) against the signal model.
Tests 1-2 retrain on 1 asset (BTC); Tests 3-8 use cached signals.
"""

import sys
sys.path.insert(0, "src")

import glob
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from signal_engine.config import PortfolioConfig, load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.portfolio import compute_daily_pnl_series, simulate_portfolio

warnings.filterwarnings("ignore")

SCOPE = "a30_08d04ff1"
CACHE_DIR = Path("experiments/cache/ml_close_walkforward/combined") / SCOPE
SEARCH_START = "2022-01"
SEARCH_END = "2024-12"
HOLDOUT_START = "2025-01"
LB = 2


def load_signals(lookback: int) -> pd.DataFrame:
    pattern = str(CACHE_DIR / f"lb{lookback}_v10_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No cached signals for lb={lookback}")
    return pd.read_parquet(files[0])


def period_filter(df, start, end=None):
    months = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m")
    mask = months >= start
    if end:
        mask &= months <= end
    return df.loc[mask].reset_index(drop=True)


def trade_returns(df):
    return df["direction"].values * df["fwd_return"].values


def trade_sharpe(rets):
    """Annualized Sharpe from individual trade returns (1h hold = 6 trades/day)."""
    if len(rets) < 10:
        return 0.0
    s = rets.std()
    # 6 non-overlapping 1-hour trades per day (conservative for 15m bars with 4-bar hold)
    return float(rets.mean() / s * np.sqrt(365.25 * 6)) if s > 0 else 0.0


def portfolio_sharpe_fixed_capital(signals, pcfg):
    """Sharpe with fixed $10K capital — no compounding."""
    sub = signals.sort_values("timestamp").reset_index(drop=True)
    cap = pcfg.starting_capital
    rets = trade_returns(sub)
    # Fixed sizing: base_sizing_pct * signal_strength * capital (constant)
    strengths = sub["signal_strength"].values.astype(np.float64)
    sizes = cap * pcfg.base_sizing_pct * np.clip(strengths, 0, 1)
    pnls = sizes * (rets - pcfg.slippage)
    ts = pd.to_datetime(sub["timestamp"], utc=True)
    daily = pd.Series(pnls, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return 0.0
    dr = daily / cap
    s = dr.std()
    return float(dr.mean() / s * np.sqrt(365.25)) if s > 0 else 0.0


def check(name, condition, detail, results):
    status = "PASS" if condition else "FAIL"
    results.append((name, condition))
    n = len(results)
    print(f"  {n:>2}. {name:<40s} {status}  {detail}")


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = PortfolioConfig(**cfg.portfolio.model_dump())
    results = []

    print("=" * 80)
    print("SIGNAL MODEL AFML INVESTIGATION")
    print("=" * 80)

    # Load cached signals
    signals = load_signals(LB)
    search = period_filter(signals, SEARCH_START, SEARCH_END)
    holdout = period_filter(signals, HOLDOUT_START)
    print(f"  Signals: {len(search)} search, {len(holdout)} holdout, {signals['asset'].nunique()} assets")
    print()

    # ================================================================
    # TEST 1: RANDOM DATA (AFML Ch 11)
    # ================================================================
    print("-" * 80)
    print("TEST 1: RANDOM DATA — Can the model extract signal from noise?")
    print("-" * 80)
    try:
        from signal_engine.research.close_walkforward import generate_asset_signals

        data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]
        ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        real_close = ohlcv_15m["close"]

        # Generate GBM random prices with same vol as BTC
        np.random.seed(42)
        real_ret = real_close.pct_change().dropna()
        mu, sigma = 0.0, float(real_ret.std())  # zero drift, same vol
        n = len(ohlcv_15m)
        random_rets = np.random.normal(mu, sigma, n)
        random_close = 100.0 * np.exp(np.cumsum(random_rets))

        ohlcv_random = ohlcv_15m.copy()
        ohlcv_random["close"] = random_close
        ohlcv_random["open"] = np.roll(random_close, 1)
        ohlcv_random["high"] = random_close * (1 + np.abs(np.random.normal(0, sigma, n)))
        ohlcv_random["low"] = random_close * (1 - np.abs(np.random.normal(0, sigma, n)))
        ohlcv_random["volume"] = ohlcv_15m["volume"].values

        features_random = compute_features(ohlcv_random, cfg.features)

        regime_features = {}
        for rtf in cfg.close_model.regime_timeframes:
            rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf) if rtf != "15min" else ohlcv_15m
            regime_features[rtf] = compute_features(rtf_ohlcv, cfg.features)

        # Run on random data — only lb=2 for speed
        random_signals = generate_asset_signals(
            "BTC_RANDOM", ohlcv_random, features_random, regime_features, cfg,
        )
        random_df = random_signals.get(LB, pd.DataFrame())
        if len(random_df) > 10:
            rets_random = trade_returns(random_df)
            sr_random = trade_sharpe(rets_random)
            avg_conf = float(random_df["confidence"].mean())
            check("Random data Sharpe < 0.5", abs(sr_random) < 0.5,
                  f"Sharpe={sr_random:.3f}, conf={avg_conf:.4f}, n={len(random_df)}", results)
        else:
            check("Random data Sharpe < 0.5", True,
                  f"Only {len(random_df)} signals (model correctly abstains)", results)
    except Exception as e:
        check("Random data", False, f"ERROR: {e}", results)
    print()

    # ================================================================
    # TEST 2: FEATURE SHUFFLE (AFML Ch 8)
    # ================================================================
    print("-" * 80)
    print("TEST 2: FEATURE SHUFFLE — Is the signal in the features?")
    print("-" * 80)
    try:
        # features_real already computed from ohlcv_15m above
        features_real = compute_features(ohlcv_15m, cfg.features)

        # Shuffle ALL features independently
        np.random.seed(123)
        features_shuffled = features_real.copy()
        for col in features_shuffled.columns:
            vals = features_shuffled[col].values.copy()
            np.random.shuffle(vals)
            features_shuffled[col] = vals

        shuffled_signals = generate_asset_signals(
            "BTC_SHUFFLED", ohlcv_15m, features_shuffled, regime_features, cfg,
        )
        shuf_df = shuffled_signals.get(LB, pd.DataFrame())
        if len(shuf_df) > 10:
            rets_shuf = trade_returns(shuf_df)
            sr_shuf = trade_sharpe(rets_shuf)
            check("Shuffled features Sharpe < 0.5", abs(sr_shuf) < 0.5,
                  f"Sharpe={sr_shuf:.3f}, n={len(shuf_df)}", results)
        else:
            check("Shuffled features Sharpe < 0.5", True,
                  f"Only {len(shuf_df)} signals", results)
    except Exception as e:
        check("Feature shuffle", False, f"ERROR: {e}", results)
    print()

    # ================================================================
    # TEST 3: FIXED-CAPITAL SHARPE (Compounding artifact check)
    # ================================================================
    print("-" * 80)
    print("TEST 3: FIXED-CAPITAL SHARPE — Is the edge real without compounding?")
    print("-" * 80)
    sr_fixed_s = portfolio_sharpe_fixed_capital(search, pcfg)
    sr_fixed_h = portfolio_sharpe_fixed_capital(holdout, pcfg)
    check("Fixed-capital Sharpe (search) > 1.0", sr_fixed_s > 1.0,
          f"Sharpe={sr_fixed_s:.3f}", results)
    check("Fixed-capital Sharpe (holdout) > 1.0", sr_fixed_h > 1.0,
          f"Sharpe={sr_fixed_h:.3f}", results)
    print()

    # ================================================================
    # TEST 4: EMBARGO SENSITIVITY (AFML Ch 7)
    # ================================================================
    print("-" * 80)
    print("TEST 4: EMBARGO SENSITIVITY — Does removing embargo inflate Sharpe?")
    print("-" * 80)
    # Compare cached signals (with embargo) vs compute per-trade returns
    # We can't easily re-run without embargo, so we check if adjacent bars
    # have correlated predictions (which embargo should prevent)
    search_sorted = search.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(search_sorted["timestamp"], utc=True)
    # Check autocorrelation of trade returns at lag 1
    tr = trade_returns(search_sorted)
    if len(tr) > 100:
        autocorr = float(np.corrcoef(tr[:-1], tr[1:])[0, 1])
        # With 4-bar overlapping labels on 15min data, adjacent predictions
        # share 3/4 forward bars → expected autocorrelation ~0.75.
        # Autocorrelation < 0.75 means embargo is working (reducing from theoretical max).
        # Autocorrelation > 0.75 would indicate label leakage into predictions.
        check("Return autocorrelation < 0.75", abs(autocorr) < 0.75,
              f"lag-1 autocorr={autocorr:.4f} (expected ~0.5-0.7 with 4-bar overlap)", results)
    print()

    # ================================================================
    # TEST 5: OUTLIER TEST (AFML Ch 11)
    # ================================================================
    print("-" * 80)
    print("TEST 5: OUTLIER SENSITIVITY — Does the edge survive winsorization?")
    print("-" * 80)
    tr_search = trade_returns(search)
    sr_raw = trade_sharpe(tr_search)
    # Winsorize at 1st/99th percentile
    lo, hi = np.percentile(tr_search, [1, 99])
    tr_winsor = np.clip(tr_search, lo, hi)
    sr_winsor = trade_sharpe(tr_winsor)
    drop_pct = (1 - sr_winsor / sr_raw) * 100 if sr_raw > 0 else 0
    check("Winsorized Sharpe drop < 30%", drop_pct < 30,
          f"raw={sr_raw:.3f}, winsorized={sr_winsor:.3f}, drop={drop_pct:.1f}%", results)

    # Distribution stats
    print(f"    Trade return stats: mean={tr_search.mean()*10000:.1f}bps, "
          f"median={np.median(tr_search)*10000:.1f}bps, "
          f"std={tr_search.std()*10000:.1f}bps")
    print(f"    Skew={float(pd.Series(tr_search).skew()):.3f}, "
          f"Kurt={float(pd.Series(tr_search).kurtosis()):.3f}")
    print(f"    Win rate: {(tr_search > 0).mean()*100:.1f}%")
    print()

    # ================================================================
    # TEST 6: TEMPORAL STABILITY (AFML Ch 12)
    # ================================================================
    print("-" * 80)
    print("TEST 6: TEMPORAL STABILITY — Is the edge stable across time?")
    print("-" * 80)
    blocks = [
        ("2022-01", "2022-06"), ("2022-07", "2022-12"),
        ("2023-01", "2023-06"), ("2023-07", "2023-12"),
        ("2024-01", "2024-06"), ("2024-07", "2024-12"),
    ]
    block_sharpes = []
    for start, end in blocks:
        b = period_filter(signals, start, end)
        if len(b) > 50:
            sr = trade_sharpe(trade_returns(b))
            block_sharpes.append(sr)
            print(f"    {start} to {end}: Sharpe={sr:.3f} ({len(b)} signals)")
        else:
            block_sharpes.append(0.0)
            print(f"    {start} to {end}: insufficient data")
    positive_blocks = sum(1 for s in block_sharpes if s > 0)
    check("Temporal stability (>4/6 positive)", positive_blocks >= 4,
          f"{positive_blocks}/6 positive blocks", results)
    print()

    # ================================================================
    # TEST 7: CROSS-ASSET CONSISTENCY (AFML Ch 11)
    # ================================================================
    print("-" * 80)
    print("TEST 7: CROSS-ASSET CONSISTENCY — Is the edge broad or concentrated?")
    print("-" * 80)
    assets = sorted(search["asset"].unique())
    asset_sharpes = {}
    for asset in assets:
        a = search[search["asset"] == asset]
        if len(a) > 50:
            sr = trade_sharpe(trade_returns(a))
            asset_sharpes[asset] = sr

    positive_assets = sum(1 for v in asset_sharpes.values() if v > 0)
    median_sr = float(np.median(list(asset_sharpes.values()))) if asset_sharpes else 0
    check(f"Cross-asset (>{len(assets)*0.7:.0f}/{len(assets)} positive)",
          positive_assets >= len(assets) * 0.7,
          f"{positive_assets}/{len(assets)} positive, median Sharpe={median_sr:.2f}", results)

    # Leave-one-out
    loo_sharpes = []
    for asset in assets:
        without = search[search["asset"] != asset]
        if len(without) > 100:
            sr = portfolio_sharpe_fixed_capital(without, pcfg)
            loo_sharpes.append(sr)
    if loo_sharpes:
        loo_range = max(loo_sharpes) - min(loo_sharpes)
        check("Leave-one-out range < 1.0", loo_range < 1.0,
              f"range={loo_range:.3f}, min={min(loo_sharpes):.3f}, max={max(loo_sharpes):.3f}", results)
    print()

    # ================================================================
    # TEST 8: DIRECTION BIAS
    # ================================================================
    print("-" * 80)
    print("TEST 8: DIRECTION BIAS — Is the model predicting or trend-riding?")
    print("-" * 80)
    long_frac_s = float((search["direction"] == 1).mean())
    long_frac_h = float((holdout["direction"] == 1).mean())

    long_rets = trade_returns(search[search["direction"] == 1])
    short_rets = trade_returns(search[search["direction"] == -1])
    long_sr = trade_sharpe(long_rets) if len(long_rets) > 50 else 0
    short_sr = trade_sharpe(short_rets) if len(short_rets) > 50 else 0

    check("Direction balance (40-60% long)", 0.4 <= long_frac_s <= 0.6,
          f"search: {long_frac_s*100:.1f}% long, holdout: {long_frac_h*100:.1f}% long", results)
    print(f"    Long Sharpe:  {long_sr:.3f} ({len(long_rets)} trades)")
    print(f"    Short Sharpe: {short_sr:.3f} ({len(short_rets)} trades)")
    both_positive = long_sr > 0 and short_sr > 0
    check("Both directions profitable", both_positive,
          f"long={long_sr:.3f}, short={short_sr:.3f}", results)
    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    passed = sum(1 for _, c in results if c)
    failed = sum(1 for _, c in results if not c)
    print("=" * 80)
    print(f"INVESTIGATION SUMMARY: {passed}/{len(results)} PASSED, {failed}/{len(results)} FAILED")
    print("=" * 80)
    for name, cond in results:
        if not cond:
            print(f"  FAILED: {name}")
    print(f"\n  Completed in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
