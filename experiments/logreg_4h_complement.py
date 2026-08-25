"""4H LogReg Complement Strategy — Walk-Forward Validation

Hypothesis: LogReg on 4H bars with 1-bar (4-hour) hold captures different
timescale dynamics than the 15m/1h production model. Trades at 1/16th
frequency. Different holding period should be structurally uncorrelated.

Methodology:
1. Resample 1m data to 4H bars (instead of 15m)
2. Same features, same walk-forward, same LogReg L1
3. fwd_bars=1 (one 4H bar = 4 hours hold)
4. Compare: Sharpe, $/cal, AUC, daily return correlation with 15m

Kill criteria:
- SR < 1.5 OOS
- Daily return correlation with 15m LogReg > 0.5
- AUC < 0.51 (no signal)
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from signal_engine.config import (
    SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig,
    RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig,
)
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_confidence_filter, apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats

warnings.filterwarnings("ignore")

# ============================================================
# Asset Universe (same as production T1)
# ============================================================
ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SOL": "SOLUSDT", "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "FIL": "FILUSDT",
    "FTM": "FTMUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT",
    "SXP": "SXPUSDT", "ZEC": "ZECUSDT", "XLM": "XLMUSDT",
    "FET": "FETUSDT", "WLD": "WLDUSDT", "BNX": "BNXUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
}

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

# Base features (same as production)
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]


def make_config(timeframe: str, fwd_bars: int) -> SystemConfig:
    """Build a SystemConfig for a given timeframe and forward horizon."""
    return SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=fwd_bars,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(
            feature_list=FEATURES,
            winsor_lo=0.01, winsor_hi=0.99,
            vol_filter_quantile=0.50,
            vol_zscore_window=30, swing_lookback=20,
        ),
        portfolio=PortfolioConfig(
            starting_capital=10000.0,
            max_dd=0.15, delever_start=0.075,
            slippage=0.0001, max_exposure=0.80,
            max_per_asset=0.06, base_sizing_pct=0.02,
            max_signal_mult=2.0,
            sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True,
            regime_exposure_floor=0.50,
        ),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(
            data_dirs=[str(d) for d in DATA_DIRS],
            timeframe=timeframe,
            regime_timeframe="4h",
        ),
        assets=ASSETS,
    )


def _process_asset(asset_name: str, symbol: str, config: SystemConfig) -> dict:
    """Process one asset: load -> features -> signals -> filter."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    timeframe = config.data.timeframe

    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
    except FileNotFoundError as e:
        return {"asset": asset_name, "error": str(e), "signals": None,
                "n_raw": 0, "n_conf": 0, "n_regime": 0}

    ohlcv_tf = resample_ohlcv(ohlcv_1m, timeframe)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")

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
        "n_raw": n_raw, "n_conf": n_conf, "n_regime": n_regime,
        "elapsed_s": elapsed,
    }


def run_pipeline(config: SystemConfig, label: str, max_workers: int = 8) -> dict:
    """Run full pipeline for a given config and return stats + daily PnL."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Timeframe: {config.data.timeframe}, fwd_bars: {config.model.fwd_bars}")
    print(f"{'='*80}")

    assets = list(config.assets.items())
    all_signals = []
    per_asset = {}
    n_raw_total = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_asset, name, sym, config): name
            for name, sym in assets
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            name = result["asset"]

            if result["error"]:
                print(f"  [{i:2d}/{len(assets)}] {name}: SKIPPED ({result['error'][:50]})")
                continue

            n_raw_total += result["n_raw"]
            per_asset[name] = {
                "n_raw": result["n_raw"],
                "n_conf": result["n_conf"],
                "n_regime": result["n_regime"],
            }

            if result["signals"] is not None:
                df = pd.DataFrame(result["signals"])
                all_signals.append(df)

            print(f"  [{i:2d}/{len(assets)}] {name}: raw={result['n_raw']}, "
                  f"conf={result['n_conf']}, regime={result['n_regime']} "
                  f"({result['elapsed_s']}s)")

    if not all_signals:
        print("  NO SIGNALS SURVIVED FILTERING")
        return {"stats": {"error": "no_signals"}, "daily_pnl": None,
                "n_raw": n_raw_total, "per_asset": per_asset}

    filtered = pd.concat(all_signals, ignore_index=True)
    print(f"\n  Combined: {n_raw_total} raw -> {len(filtered)} filtered")

    # Portfolio simulation
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

    return {"stats": stats, "daily_pnl": daily_pnl,
            "n_raw": n_raw_total, "per_asset": per_asset,
            "sim": sim, "signals": filtered}


def compute_correlation(pnl_a: pd.Series, pnl_b: pd.Series) -> float:
    """Compute daily PnL correlation between two strategies."""
    # Align on common dates
    combined = pd.DataFrame({"A": pnl_a, "B": pnl_b}).fillna(0)
    # Only use dates where at least one strategy traded
    mask = (combined["A"] != 0) | (combined["B"] != 0)
    combined = combined[mask]
    if len(combined) < 30:
        return float("nan")
    return float(combined["A"].corr(combined["B"]))


def main():
    t_start = time.time()
    n_workers = min(os.cpu_count() or 4, 11)

    print("=" * 80)
    print("  4H LogReg COMPLEMENT STRATEGY — Walk-Forward Validation")
    print(f"  {len(ASSETS)} assets, {n_workers} workers")
    print("=" * 80)

    # ================================================================
    # Run BOTH timeframes
    # ================================================================

    # A: Baseline 15m (production reference)
    config_15m = make_config("15min", fwd_bars=4)
    result_15m = run_pipeline(config_15m, "A: Baseline 15m (fwd=4, ~1h hold)",
                              max_workers=n_workers)

    # B: 4H with 1-bar forward (4h hold)
    config_4h = make_config("4h", fwd_bars=1)
    result_4h = run_pipeline(config_4h, "B: 4H LogReg (fwd=1, ~4h hold)",
                             max_workers=n_workers)

    # C: 4H with 2-bar forward (8h hold)
    config_4h_2 = make_config("4h", fwd_bars=2)
    result_4h_2 = run_pipeline(config_4h_2, "C: 4H LogReg (fwd=2, ~8h hold)",
                               max_workers=n_workers)

    # ================================================================
    # Results Comparison
    # ================================================================
    print(f"\n{'='*100}")
    print("RESULTS COMPARISON")
    print(f"{'='*100}")

    configs = [
        ("A: 15m baseline", result_15m),
        ("B: 4H fwd=1", result_4h),
        ("C: 4H fwd=2", result_4h_2),
    ]

    header = f"{'Config':<25} {'SR':>8} {'$/cal':>10} {'DD%':>8} {'Trades':>10} {'Signals':>10}"
    print(header)
    print("-" * len(header))

    for label, res in configs:
        s = res["stats"]
        if "error" in s:
            print(f"{label:<25} {'ERROR':>8} {s['error']}")
            continue
        sr = s.get("sharpe", 0)
        pnl_cal = s.get("pnl_per_cal_day", 0)
        dd = s.get("max_dd_pct", 0)
        trades = s.get("n_trades", 0)
        n_sig = res["n_raw"]
        print(f"{label:<25} {sr:>8.3f} ${pnl_cal:>9.2f} {dd:>7.1f}% {trades:>10,} {n_sig:>10,}")

    # ================================================================
    # Correlation Analysis (THE KEY METRIC)
    # ================================================================
    print(f"\n{'='*100}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*100}")

    pnl_15m = result_15m.get("daily_pnl")
    pnl_4h_1 = result_4h.get("daily_pnl")
    pnl_4h_2 = result_4h_2.get("daily_pnl")

    if pnl_15m is not None and pnl_4h_1 is not None:
        corr_b = compute_correlation(pnl_15m, pnl_4h_1)
        print(f"  Correlation(15m, 4H fwd=1): {corr_b:.4f}"
              f"  {'PASS (<0.5)' if corr_b < 0.5 else 'FAIL (>0.5) - TOO CORRELATED'}")

    if pnl_15m is not None and pnl_4h_2 is not None:
        corr_c = compute_correlation(pnl_15m, pnl_4h_2)
        print(f"  Correlation(15m, 4H fwd=2): {corr_c:.4f}"
              f"  {'PASS (<0.5)' if corr_c < 0.5 else 'FAIL (>0.5) - TOO CORRELATED'}")

    if pnl_4h_1 is not None and pnl_4h_2 is not None:
        corr_bc = compute_correlation(pnl_4h_1, pnl_4h_2)
        print(f"  Correlation(4H fwd=1, 4H fwd=2): {corr_bc:.4f}")

    # ================================================================
    # Combined Portfolio Analysis
    # ================================================================
    print(f"\n{'='*100}")
    print("COMBINED PORTFOLIO (50/50 allocation)")
    print(f"{'='*100}")

    if pnl_15m is not None and pnl_4h_1 is not None:
        combined = pd.DataFrame({
            "15m": pnl_15m,
            "4h_1": pnl_4h_1,
        }).fillna(0)
        combined_pnl = combined["15m"] * 0.5 + combined["4h_1"] * 0.5
        combined_nonzero = combined_pnl[combined_pnl != 0]

        if len(combined_nonzero) >= 10:
            combined_stats = compute_full_stats(
                combined_nonzero, 10000.0, 0, 0
            )
            sr_combined = combined_stats.get("sharpe", 0)
            dd_combined = combined_stats.get("max_dd_pct", 0)
            pnl_combined = combined_stats.get("pnl_per_cal_day", 0)
            print(f"  Combined SR:    {sr_combined:.3f}")
            print(f"  Combined $/cal: ${pnl_combined:.2f}")
            print(f"  Combined DD%:   {dd_combined:.1f}%")

            sr_15m = result_15m["stats"].get("sharpe", 0)
            print(f"\n  Diversification benefit: {sr_combined:.3f} vs {sr_15m:.3f} (15m alone)")
            if sr_combined > sr_15m:
                print(f"  +{sr_combined - sr_15m:.3f} SR from diversification")
            else:
                print(f"  {sr_combined - sr_15m:.3f} SR (no diversification benefit)")

    # ================================================================
    # Monthly PnL Breakdown
    # ================================================================
    print(f"\n{'='*100}")
    print("MONTHLY PnL (OOS)")
    print(f"{'='*100}")

    monthly_data = {}
    for label, res in configs:
        pnl = res.get("daily_pnl")
        if pnl is not None:
            monthly = pnl.resample("MS").sum()
            monthly_data[label] = monthly

    if monthly_data:
        all_months = sorted(set().union(*(m.index for m in monthly_data.values())))
        header = f"{'Month':<12}"
        for label in monthly_data:
            header += f" {label:>18}"
        print(header)

        for month in all_months:
            row = f"{month.strftime('%Y-%m'):<12}"
            for label, monthly in monthly_data.items():
                val = monthly.get(month, 0)
                row += f" ${val:>16.0f}"
            print(row)

    # ================================================================
    # Decision
    # ================================================================
    print(f"\n{'='*100}")
    print("DECISION")
    print(f"{'='*100}")

    for label, res in [("B: 4H fwd=1", result_4h), ("C: 4H fwd=2", result_4h_2)]:
        s = res["stats"]
        if "error" in s:
            print(f"  {label}: KILL (no trades)")
            continue

        sr = s.get("sharpe", 0)
        pnl = res.get("daily_pnl")
        corr = compute_correlation(pnl_15m, pnl) if pnl_15m is not None and pnl is not None else float("nan")

        verdict = "KILL"
        reasons = []
        if sr < 1.5:
            reasons.append(f"SR={sr:.2f} < 1.5")
        if corr > 0.5:
            reasons.append(f"corr={corr:.3f} > 0.5")
        if not reasons:
            verdict = "PASS"
            reasons.append(f"SR={sr:.2f}, corr={corr:.3f}")

        print(f"  {label}: {verdict} ({', '.join(reasons)})")

    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
