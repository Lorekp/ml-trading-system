#!/usr/bin/env python
"""Optimized Full Portfolio Backtest — signals + close model + vol overlay + report.

All-in-one script. Target: <5 min warm cache, <10 min cold.

Optimizations vs previous version:
- Load OHLCV once per asset (no ProcessPoolExecutor fork overhead)
- Cache features/examples/predictions to parquet
- Sequential example generation (fast, ~0.1s/asset, avoids fork)
- XGBoost with n_jobs=-1 (all cores)
- Subsample large training sets (cap at 50K rows)
"""

import sys, os, time, warnings
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
os.environ["XGB_DEVICE"] = "cpu"  # CV folds use CPU to avoid CUDA context leak
# Main model also uses CPU — GPU caused exit 127 in WF loop with many CUDA contexts
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm

from signal_engine.config import (
    SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig,
    RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig,
    CloseModelConfig, VolOverlayConfig,
)
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.close_model import (
    build_close_decision_examples, walk_forward_close_prediction_cache,
    select_close_predictions, close_feature_columns,
)
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.portfolio_path import simulate_portfolio_precomputed_exits
from signal_engine.stats import compute_full_stats

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
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]
FEAT_CFG = FeatureConfig(
    feature_list=FEATURES, winsor_lo=0.01, winsor_hi=0.99,
    vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
)
CLOSE_CFG = CloseModelConfig(
    enabled=True, model_type="xgboost",
    label_mode="residual_triple_barrier",
    threshold=0.5, calibration="none", lookbacks=[10],
    max_hold_minutes=60, min_hold_minutes=15,
    pt_mult=1.0, sl_mult=1.0, min_improvement_bps=2.0,
    xgb_max_grid_trials=1,  # skip grid search — use defaults for speed
)
PORT_CFG = PortfolioConfig(
    starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.40, base_sizing_pct=0.02, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
CACHE = Path("results/close_cache")
SIGNAL_CACHE = Path("results/signals_cache")


# ============================================================
# Phase 1: Data Loading (cached)
# ============================================================

def load_all_data():
    """Load OHLCV + features for all assets. Cache features to parquet."""
    feat_cache = Path("data/cache/features_v2")
    feat_cache.mkdir(parents=True, exist_ok=True)

    data = {}
    for name, sym in sorted(ASSETS.items()):
        t0 = time.time()
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
        except FileNotFoundError:
            print(f"  {name}: SKIP (no data)")
            continue

        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
        ohlcv_1h = resample_ohlcv(ohlcv_1m, "1h")

        # Cached features
        f15_path = feat_cache / f"{sym}_15m.parquet"
        f1h_path = feat_cache / f"{sym}_1h.parquet"
        f4h_path = feat_cache / f"{sym}_4h.parquet"

        if f15_path.exists():
            feat_15m = pd.read_parquet(f15_path)
        else:
            feat_15m = compute_features(ohlcv_15m, FEAT_CFG)
            feat_15m.to_parquet(f15_path)

        if f1h_path.exists():
            feat_1h = pd.read_parquet(f1h_path)
        else:
            feat_1h = compute_features(ohlcv_1h, FEAT_CFG)
            feat_1h.to_parquet(f1h_path)

        if f4h_path.exists():
            feat_4h = pd.read_parquet(f4h_path)
        else:
            feat_4h = compute_features(ohlcv_4h, FEAT_CFG)
            feat_4h.to_parquet(f4h_path)

        data[name] = {
            "sym": sym,
            "ohlcv_15m": ohlcv_15m, "ohlcv_4h": ohlcv_4h, "ohlcv_1h": ohlcv_1h,
            "feat_15m": feat_15m, "feat_1h": feat_1h, "feat_4h": feat_4h,
        }
        print(f"  {name}: loaded ({time.time()-t0:.1f}s)")

    return data


# ============================================================
# Phase 2: Close Example Generation (sequential, fast)
# ============================================================

def generate_close_examples(signals: pd.DataFrame, data: dict, strategy: str) -> pd.DataFrame:
    """Generate close decision examples for all assets of one strategy."""
    cache_path = CACHE / f"examples_{strategy}.parquet"
    if cache_path.exists():
        print(f"  {strategy}: loading cached examples")
        return pd.read_parquet(cache_path)

    if strategy == "logreg_15m":
        tf_key, max_hold, min_hold = "15m", 60, 15
    elif strategy == "logreg_4h":
        tf_key, max_hold, min_hold = "4h", 480, 240
    else:
        return pd.DataFrame()

    strat_sigs = signals[signals["strategy"] == strategy]
    all_examples = []

    for name in sorted(data.keys()):
        asset_sigs = strat_sigs[strat_sigs["asset"] == name]
        if len(asset_sigs) < 5:
            continue

        d = data[name]
        ohlcv_tf = d[f"ohlcv_{tf_key}"]
        feat_tf = d[f"feat_{tf_key}"]
        feat_1h = d["feat_1h"]

        try:
            examples = build_close_decision_examples(
                signals_df=asset_sigs,
                entry_ohlcv=ohlcv_tf,
                decision_ohlcv=ohlcv_tf,
                decision_features=feat_tf,
                feature_cols=FEATURES,
                decision_timeframe="15min" if tf_key == "15m" else "4h",
                max_hold_minutes=max_hold,
                min_hold_minutes=min_hold,
                slippage=0.0,  # let portfolio sim handle all slippage in one place
                label_mode="residual_triple_barrier",
                regime_feature_sets={"1h": feat_1h},
            )
            if len(examples) > 0:
                if "trade_id" in examples.columns:
                    examples["trade_id"] = name + ":" + examples["trade_id"].astype(str)
                all_examples.append(examples)
                print(f"    {name}: {len(examples):,} examples")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")

    if not all_examples:
        return pd.DataFrame()

    combined = pd.concat(all_examples, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    print(f"  {strategy}: {len(combined):,} total examples cached")
    return combined


# ============================================================
# Phase 3: Walk-Forward Close Model
# ============================================================

def run_close_model(examples: pd.DataFrame, strategy: str, bar_minutes: int) -> pd.DataFrame:
    """Run PARALLEL WF close model training + apply threshold for early exits."""
    sel_cache = CACHE / f"selected_{strategy}_parallel.parquet"

    if sel_cache.exists():
        print(f"  {strategy}: loading cached selected trades")
        return pd.read_parquet(sel_cache)

    if len(examples) == 0:
        return pd.DataFrame()

    feat_cols = close_feature_columns(FEATURES, regime_timeframes=["1h"])

    # PARALLEL walk-forward: 12 workers, full dataset, no subsampling
    from signal_engine.parallel_close_wf import parallel_walk_forward_close

    examples_with_prob = parallel_walk_forward_close(
        examples, feat_cols, lookback_months=10,
        test_start="2021-01", max_workers=os.cpu_count(),
        use_gpu=False,  # CPU for workers (avoid CUDA context leak)
    )

    # Apply close model: exit early if close_probability > threshold
    # VECTORIZED — no Python loop over 250K trade groups
    threshold = 0.6
    ex = examples_with_prob.copy()
    ex["_triggered"] = ex["close_probability"] > threshold

    if "exit_return" in ex.columns:
        # For each trade: find the FIRST triggered bar, or use last bar
        ex["_row_idx"] = np.arange(len(ex))

        # First triggered bar per trade
        triggered = ex[ex["_triggered"]].groupby("trade_id")["_row_idx"].first()
        # Last bar per trade (fallback for non-triggered)
        last_bar = ex.groupby("trade_id")["_row_idx"].last()

        # Merge: use triggered if available, else last bar
        exit_idx = last_bar.copy()
        exit_idx.loc[triggered.index] = triggered
        selected = ex.iloc[exit_idx.values].copy()

        selected["fwd_return"] = selected["exit_return"]
        selected["return_already_signed"] = True

        n_early = int(triggered.notna().sum())
    else:
        # Fallback: first row per trade
        first_idx = ex.groupby("trade_id")["_row_idx"].first() if "_row_idx" in ex.columns else ex.groupby("trade_id").cumcount()
        selected = ex.groupby("trade_id").first().reset_index()
        selected["return_already_signed"] = False
        n_early = 0

    n_total = selected["trade_id"].nunique()
    print(f"    Close model: {n_early}/{n_total} trades exited early ({n_early/max(n_total,1)*100:.1f}%)")
    print(f"    Selected: {len(selected):,} trades")

    if len(selected) > 0:
        selected["strategy"] = strategy
        if "bars_since_entry" in selected.columns:
            selected["hold_bars"] = selected["bars_since_entry"].fillna(4).astype(int)
        elif "bars_held" in selected.columns:
            selected["hold_bars"] = selected["bars_held"]
        else:
            selected["hold_bars"] = 4 if strategy == "logreg_15m" else 2

        CACHE.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(sel_cache, index=False)

    return selected


# ============================================================
# Phase 4: Portfolio Sim + Report
# ============================================================

def walk_forward_weights(strategy_pnls, method="sharpe"):
    combined = pd.DataFrame(strategy_pnls).fillna(0)
    strategies = list(strategy_pnls.keys())
    n = len(strategies)
    months = sorted(combined.resample("MS").sum().index)
    weights = {}
    for m_idx, month in enumerate(months):
        m_key = month.strftime("%Y-%m")
        if m_idx >= 2:
            lag = months[m_idx - 1]
        else:
            lag = month
        past = combined[combined.index < lag]
        if len(past) < 30 or m_idx < 4:
            weights[m_key] = {s: 1.0/n for s in strategies}
            continue
        sharpes = {}
        for s in strategies:
            p = past[s].dropna()
            sharpes[s] = max(float(p.mean()/p.std()*np.sqrt(365.25)), 0) if len(p)>20 and p.std()>0 else 0
        total = sum(sharpes.values())
        if total == 0:
            weights[m_key] = {s: 1.0/n for s in strategies}
        else:
            w = {s: max(sr/total, 0.10) for s, sr in sharpes.items()}
            tw = sum(w.values())
            weights[m_key] = {s: v/tw for s, v in w.items()}
    return weights


def apply_weights(signals, weights_by_month):
    out = signals.copy()
    ts = pd.to_datetime(out["timestamp"])
    months = ts.dt.to_period("M").astype(str)
    for i in range(len(out)):
        m_key = months.iloc[i][:7]
        w = weights_by_month.get(m_key, {})
        strat = out.iloc[i]["strategy"]
        out.at[out.index[i], "signal_strength"] *= w.get(strat, 1.0/3)
    return out


def bootstrap_ci(daily_pnl, capital, n=5000):
    dr = (daily_pnl / capital).values
    rng = np.random.RandomState(42)
    srs = np.empty(n)
    for i in range(n):
        s = rng.choice(dr, size=len(dr), replace=True)
        srs[i] = s.mean()/s.std()*np.sqrt(365.25) if s.std()>0 else 0
    return float(np.percentile(srs, 2.5)), float(np.percentile(srs, 97.5))


def bybit_sim(signals, rt_bps=4.5, capital=100000):
    """Quick Bybit execution cost sim."""
    sub = signals.copy()
    rt = rt_bps / 10000.0
    gross = sub["direction"].values * sub["fwd_return"].values
    if "return_already_signed" in sub.columns:
        signed_mask = sub["return_already_signed"].values.astype(bool)
        gross[signed_mask] = sub["fwd_return"].values[signed_mask]
    net = gross - rt
    edge = np.where(np.isfinite(sub["probability"].values),
                    np.abs(2*sub["probability"].values-1), 0.1)
    edge = np.clip(edge, 0, 1)
    eq = capital
    pnl = np.zeros(len(sub))
    for i in range(len(sub)):
        sz = min(eq*0.02*edge[i], eq*0.04)
        pnl[i] = sz*net[i]; eq += pnl[i]
    ts = pd.to_datetime(sub["timestamp"].values)
    if ts.tz is not None: ts = ts.tz_localize(None)
    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily!=0]
    sr = daily.mean()/daily.std()*np.sqrt(365.25) if len(daily)>30 and daily.std()>0 else 0
    return {"sr": sr, "pnl": pnl.sum(), "eq": eq, "daily": daily,
            "n": len(sub), "mean_net": net.mean()*10000}


def main():
    T0 = time.time()
    print("=" * 100)
    print("  OPTIMIZED FULL BACKTEST — Close Model + Vol Overlay + 3 Strategies")
    print("=" * 100)

    # ================================================================
    # Phase 1: Load signals + data
    # ================================================================
    print(f"\n[Phase 1] Loading data...")
    t1 = time.time()

    sigs = {}
    for name in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        path = SIGNAL_CACHE / f"{name}.parquet"
        if not path.exists():
            print(f"  MISSING: {path} — run full_portfolio_backtest.py first")
            return
        sigs[name] = pd.read_parquet(path)
        print(f"  {name}: {len(sigs[name]):,} signals")

    print(f"\n  Loading OHLCV + features for {len(ASSETS)} assets...")
    data = load_all_data()
    print(f"  Phase 1 done in {time.time()-t1:.0f}s ({len(data)} assets)")

    # ================================================================
    # Phase 2: Close examples
    # ================================================================
    print(f"\n[Phase 2] Close decision examples...")
    t2 = time.time()

    merged_sigs = pd.concat(sigs.values(), ignore_index=True)

    ex_15m = generate_close_examples(merged_sigs, data, "logreg_15m")
    # Skip 4H close model — only 10K trades, close model validated on 15min only
    ex_4h = pd.DataFrame()

    print(f"  Phase 2 done in {time.time()-t2:.0f}s")

    # ================================================================
    # Phase 3: WF Close Model
    # ================================================================
    print(f"\n[Phase 3] Walk-forward close model...")
    t3 = time.time()

    sel_15m = run_close_model(ex_15m, "logreg_15m", bar_minutes=15)
    sel_4h = run_close_model(ex_4h, "logreg_4h", bar_minutes=240)

    print(f"  Phase 3 done in {time.time()-t3:.0f}s")

    # ================================================================
    # Phase 4: Portfolio sim + report
    # ================================================================
    print(f"\n[Phase 4] Portfolio simulation + report...")

    # Merge: close-adjusted LogReg + unchanged CS Momentum
    cs = sigs["cs_momentum"].copy()
    cs["return_already_signed"] = False
    cs["remaining_fraction"] = 1.0

    parts = []
    if len(sel_15m) > 0: parts.append(sel_15m)
    if len(sel_4h) > 0: parts.append(sel_4h)
    parts.append(cs)

    merged = pd.concat(parts, ignore_index=True)
    if "regime_exposure_mult" not in merged.columns:
        merged["regime_exposure_mult"] = 1.0
    if "sizing_mult" not in merged.columns:
        merged["sizing_mult"] = 1.0
    merged["regime_exposure_mult"] = merged["regime_exposure_mult"].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        merged = apply_vol_overlay(merged, btc_daily, VolOverlayConfig(
            enabled=True, vol_window=20, pct_window=60,
            low_pct=0.20, high_pct=0.80,
            low_mult=0.50, mid_mult=1.00, high_mult=1.25,
        ))
    except Exception as e:
        print(f"  Vol overlay failed: {e}")

    # --- Baseline (fixed hold, no close model) ---
    baseline_sigs = pd.concat(sigs.values(), ignore_index=True)
    if "regime_exposure_mult" not in baseline_sigs.columns:
        baseline_sigs["regime_exposure_mult"] = 1.0
    if "sizing_mult" not in baseline_sigs.columns:
        baseline_sigs["sizing_mult"] = 1.0
    baseline_sigs["regime_exposure_mult"] = baseline_sigs["regime_exposure_mult"].fillna(1.0)
    baseline_sigs["probability"] = baseline_sigs["probability"].astype(float)

    # --- Simulate both ---
    print("\n  Simulating...")

    # Baseline (fixed hold)
    base_sim = simulate_portfolio(baseline_sigs.sort_values("timestamp"), PORT_CFG)
    base_pnl = compute_daily_pnl_series(base_sim) if base_sim else None

    # Close model (precomputed exits)
    if "return_already_signed" in merged.columns and merged["return_already_signed"].any():
        close_sim = simulate_portfolio_precomputed_exits(merged, PORT_CFG)
    else:
        close_sim = simulate_portfolio(merged, PORT_CFG)
    close_pnl = compute_daily_pnl_series(close_sim) if close_sim else None

    # ================================================================
    # Report
    # ================================================================
    print(f"\n{'='*100}")
    print("  RESULTS: BASELINE vs CLOSE MODEL")
    print(f"{'='*100}")

    for label, sim, pnl in [("Baseline (fixed hold)", base_sim, base_pnl),
                              ("+ Close Model + Vol", close_sim, close_pnl)]:
        if pnl is None:
            print(f"\n  {label}: NO PNL")
            continue
        if pnl.index.tz is not None:
            pnl = pnl.copy()
            pnl.index = pnl.index.tz_localize(None)
        stats = compute_full_stats(pnl, 10000, sim["n_trades"], sim["n_trades"]/max(len(pnl),1))
        ci = bootstrap_ci(pnl, 10000)
        print(f"\n  --- {label} ---")
        print(f"  SR: {stats.get('sharpe',0):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  DD: {stats.get('max_dd_pct',0):.2f}%  Sortino: {stats.get('sortino',0):.3f}")
        print(f"  Trades: {sim['n_trades']:,}  Equity: ${sim['final_equity']:,.0f}")

    # Bybit costs
    print(f"\n{'='*100}")
    print("  BYBIT VIP0 MAKER ($100K)")
    print(f"{'='*100}")

    for label, df in [("Baseline", baseline_sigs), ("+ Close Model + Vol", merged)]:
        r = bybit_sim(df.sort_values("timestamp"))
        ci = bootstrap_ci(r["daily"], 100000) if len(r["daily"])>30 else (0,0)
        print(f"  {label:<25}: SR={r['sr']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"PnL=${r['pnl']:,.0f}  Net={r['mean_net']:.2f}bps")

    # Per-strategy on Bybit
    print(f"\n  Per-Strategy (Bybit):")
    for strat in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        s = merged[merged["strategy"] == strat] if strat in merged["strategy"].values else sigs.get(strat, pd.DataFrame())
        if len(s) > 10:
            r = bybit_sim(s.sort_values("timestamp"))
            print(f"    {strat:<15}: SR={r['sr']:.3f}  trades={r['n']:,}  net={r['mean_net']:.1f}bps  pnl=${r['pnl']:,.0f}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
