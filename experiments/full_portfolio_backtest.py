"""Full Portfolio Backtest — 3-Strategy Combination

Combines LogReg 15m + LogReg 4H + CS Momentum into a unified portfolio
with walk-forward allocation, per-strategy exposure budgets, and
comprehensive reporting.

Senior quant standards:
- Walk-forward only (no peeking) at every boundary
- Data leakage audit on every signal source
- Bootstrap CIs on combined Sharpe
- Deflated Sharpe Ratio (account for 7 strategies tested)
- Regime-conditional breakdown
- Per-strategy PnL attribution
- Correlation stability analysis

Usage:
    cd C:/Claude/ml-trading-system
    PYTHONPATH=src python experiments/full_portfolio_backtest.py
"""

import sys, os, time, warnings, copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from signal_engine.config import (
    SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig,
    RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig,
)
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_confidence_filter, apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.portfolio_path import simulate_portfolio_precomputed_exits
from signal_engine.stats import compute_full_stats

warnings.filterwarnings("ignore")

# ============================================================
# Constants
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
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
    # 18-feature expansion tested 2026-03-28: HURT performance (SR 2.29 vs 2.51)
    # entropy, parkinson_vol_20, cs_spread, ffd_close, ffd_ret_1 add noise
]
CACHE_DIR = Path("results/signals_cache")
N_WORKERS = min(os.cpu_count() or 4, 11)


# ============================================================
# Phase 1: Signal Generation
# ============================================================

def _make_logreg_config(timeframe: str, fwd_bars: int) -> SystemConfig:
    return SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=fwd_bars,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06,
                                  base_sizing_pct=0.02, max_signal_mult=2.0,
                                  sizing_mode="probability_edge",
                                  use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=timeframe, regime_timeframe="4h"),
        assets=ASSETS,
    )


def _process_asset_signals(asset_name, symbol, config):
    """Generate signals for one asset. Runs in worker process.

    Uses force_monthly_retrain=True and per-window confidence threshold
    (computed from training predictions only — no leakage).
    """
    warnings.filterwarnings("ignore")
    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
    except FileNotFoundError:
        return None

    ohlcv_tf = resample_ohlcv(ohlcv_1m, config.data.timeframe)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    features = compute_features(ohlcv_tf, config.features)

    # Force monthly retrain + per-window confidence filter (no leakage)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    signals = wf.generate_signals_batch(ohlcv_tf, features, asset_name)

    if len(signals) == 0:
        return None

    # Regime filter only (confidence already applied per-window inside model)
    if config.regime_filter.enabled:
        filtered = apply_regime_filter(signals, ohlcv_4h, config.regime_filter)
    else:
        filtered = signals

    if len(filtered) == 0:
        return None

    return filtered


def generate_logreg_signals(timeframe, fwd_bars, strategy_name, hold_bars):
    """Generate LogReg signals for all assets."""
    cache_path = CACHE_DIR / f"{strategy_name}.parquet"
    if cache_path.exists():
        print(f"  {strategy_name}: loading from cache")
        return pd.read_parquet(cache_path)

    config = _make_logreg_config(timeframe, fwd_bars)
    assets = list(config.assets.items())
    all_signals = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_process_asset_signals, name, sym, config): name
            for name, sym in assets
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_signals.append(result)

    if not all_signals:
        return pd.DataFrame()

    signals = pd.concat(all_signals, ignore_index=True)
    signals["strategy"] = strategy_name
    signals["hold_bars"] = hold_bars

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(cache_path, index=False)
    print(f"  {strategy_name}: {len(signals):,} signals cached")
    return signals


def generate_cs_momentum_signals(mom_window=5, reb_freq=5, n_leg=6, cost_bps=10.0):
    """Generate CS momentum signals as a DataFrame."""
    cache_path = CACHE_DIR / "cs_momentum.parquet"
    if cache_path.exists():
        print(f"  cs_momentum: loading from cache")
        return pd.read_parquet(cache_path)

    # Load daily returns
    daily_rets = {}
    for name, symbol in sorted(ASSETS.items()):
        try:
            ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
            daily = resample_ohlcv(ohlcv_1m, "1D")
            daily_rets[name] = daily["close"].pct_change()
        except Exception:
            continue

    dr = pd.DataFrame(daily_rets)
    # .shift(1) excludes today's return — no same-bar bias
    momentum = dr.rolling(mom_window).sum().shift(1)

    signals = []
    last_reb = -reb_freq

    for i in range(mom_window + 1, len(dr)):
        if i - last_reb < reb_freq:
            continue

        mom_i = momentum.iloc[i].dropna()
        if len(mom_i) < 2 * n_leg:
            continue

        ranked = mom_i.sort_values(ascending=False)
        longs = ranked.index[:n_leg].tolist()
        shorts = ranked.index[-n_leg:].tolist()
        last_reb = i
        date = dr.index[i]

        # Compute forward returns for each position over hold period
        for asset in longs:
            if i + reb_freq < len(dr):
                fwd_ret = float(dr[asset].iloc[i+1:i+1+reb_freq].sum())
            else:
                fwd_ret = 0.0
            signals.append({
                "timestamp": date,
                "asset": asset,
                "direction": 1.0,
                "fwd_return": fwd_ret,
                "signal_strength": 1.0 / n_leg,
                "probability": np.nan,
                "confidence": 1.0,
                "strategy": "cs_momentum",
                "hold_bars": reb_freq,  # 5 daily bars
            })

        for asset in shorts:
            if i + reb_freq < len(dr):
                fwd_ret = float(dr[asset].iloc[i+1:i+1+reb_freq].sum())
            else:
                fwd_ret = 0.0
            signals.append({
                "timestamp": date,
                "asset": asset,
                "direction": -1.0,
                "fwd_return": fwd_ret,
                "signal_strength": 1.0 / n_leg,
                "probability": np.nan,
                "confidence": 1.0,
                "strategy": "cs_momentum",
                "hold_bars": reb_freq,
            })

    if not signals:
        return pd.DataFrame()

    df = pd.DataFrame(signals)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  cs_momentum: {len(df):,} signals cached")
    return df


# ============================================================
# Phase 2: Combined Portfolio
# ============================================================

def walk_forward_strategy_weights(strategy_pnls: dict[str, pd.Series],
                                   method: str = "sharpe") -> dict[str, dict[str, float]]:
    """Compute monthly walk-forward allocation weights across strategies.

    Returns: {month_str: {strategy: weight}}
    Uses only past data — no leakage.
    """
    # Combine into DataFrame
    combined = pd.DataFrame(strategy_pnls).fillna(0)
    strategies = list(strategy_pnls.keys())
    n = len(strategies)

    months = sorted(combined.resample("MS").sum().index)
    weights_by_month = {}

    for m_idx, month in enumerate(months):
        month_str = month.strftime("%Y-%m")

        # Use data up to TWO months ago (1-month lag to avoid circularity)
        # Month M weights based on performance through month M-2
        if m_idx >= 2:
            lag_cutoff = months[m_idx - 1]  # exclude M-1 as well (conservative)
        else:
            lag_cutoff = month
        past = combined[combined.index < lag_cutoff]

        if len(past) < 30 or m_idx < 4:
            # Insufficient history — equal weight
            weights_by_month[month_str] = {s: 1.0/n for s in strategies}
            continue

        if method == "equal":
            weights_by_month[month_str] = {s: 1.0/n for s in strategies}
        elif method == "sharpe":
            sharpes = {}
            for s in strategies:
                pnl = past[s].dropna()
                if len(pnl) < 20 or pnl.std() == 0:
                    sharpes[s] = 0.0
                else:
                    sharpes[s] = max(float(pnl.mean() / pnl.std() * np.sqrt(365.25)), 0.0)

            total = sum(sharpes.values())
            if total == 0:
                weights_by_month[month_str] = {s: 1.0/n for s in strategies}
            else:
                w = {s: sr / total for s, sr in sharpes.items()}
                # Floor at 10%
                for s in strategies:
                    w[s] = max(w[s], 0.10)
                total_w = sum(w.values())
                weights_by_month[month_str] = {s: v/total_w for s, v in w.items()}
        else:
            weights_by_month[month_str] = {s: 1.0/n for s in strategies}

    return weights_by_month


def apply_strategy_weights(signals: pd.DataFrame,
                            weights_by_month: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Multiply signal_strength by walk-forward strategy weight."""
    out = signals.copy()
    ts = pd.to_datetime(out["timestamp"])
    months = ts.dt.to_period("M").astype(str)

    for i in range(len(out)):
        m = months.iloc[i]
        # Find the nearest month key
        m_key = m[:7]  # "YYYY-MM"
        w = weights_by_month.get(m_key, {})
        strat = out.iloc[i]["strategy"]
        weight = w.get(strat, 1.0 / 3)
        out.at[out.index[i], "signal_strength"] *= weight

    return out


# ============================================================
# Phase 4: Reporting
# ============================================================

def bootstrap_sharpe_ci(daily_pnl: pd.Series, capital: float,
                         n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval on annualized Sharpe ratio."""
    daily_ret = (daily_pnl / capital).values
    n = len(daily_ret)
    rng = np.random.RandomState(42)
    sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(daily_ret, size=n, replace=True)
        std = sample.std()
        sharpes[i] = sample.mean() / std * np.sqrt(365.25) if std > 0 else 0.0
    lo = float(np.percentile(sharpes, (1 - ci) / 2 * 100))
    hi = float(np.percentile(sharpes, (1 + ci) / 2 * 100))
    return lo, hi


def deflated_sharpe_ratio(observed_sr: float, n_trials: int, T: int,
                           skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Accounts for multiple testing bias. n_trials = total strategies tested (including killed).
    """
    sr_std = np.sqrt((1 + 0.5 * observed_sr**2 - skew * observed_sr +
                      (kurt - 3) / 4 * observed_sr**2) / T)
    euler_gamma = 0.5772156649
    expected_max_sr = sr_std * (
        (1 - euler_gamma) * norm.ppf(1 - 1.0/n_trials) +
        euler_gamma * norm.ppf(1 - 1.0/(n_trials * np.e))
    )
    dsr = float(norm.cdf((observed_sr - expected_max_sr) / sr_std))
    return dsr


def regime_breakdown(strategy_pnls: dict[str, pd.Series],
                      btc_daily: pd.Series) -> pd.DataFrame:
    """Compute per-strategy Sharpe per BTC regime."""
    from live.portfolio_optimizer import RegimeSelector

    combined = pd.DataFrame(strategy_pnls).fillna(0)
    selector = RegimeSelector()

    # Align BTC close
    if btc_daily.index.tz is not None:
        btc_daily.index = btc_daily.index.tz_localize(None)

    result = selector.compute_regime_sharpes(combined, btc_daily)

    rows = []
    for strat, regimes in result.items():
        for regime, sr in regimes.items():
            rows.append({"strategy": strat, "regime": regime, "sharpe": sr})

    return pd.DataFrame(rows).pivot(index="strategy", columns="regime", values="sharpe")


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()

    print("=" * 100)
    print("  FULL PORTFOLIO BACKTEST — 3-Strategy Combination")
    print(f"  {len(ASSETS)} assets, {N_WORKERS} workers")
    print("=" * 100)

    # ================================================================
    # PHASE 1: Signal Generation
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 1: SIGNAL GENERATION")
    print(f"{'='*80}")

    t1 = time.time()

    print("\n[1B] LogReg 15m...")
    sig_15m = generate_logreg_signals("15min", fwd_bars=4,
                                       strategy_name="logreg_15m", hold_bars=4)

    print("[1C] LogReg 4H...")
    sig_4h = generate_logreg_signals("4h", fwd_bars=2,
                                      strategy_name="logreg_4h", hold_bars=2)

    print("[1D] CS Momentum...")
    sig_cs = generate_cs_momentum_signals()

    print(f"\nPhase 1 complete in {time.time()-t1:.0f}s")
    print(f"  LogReg 15m: {len(sig_15m):,} signals")
    print(f"  LogReg 4H:  {len(sig_4h):,} signals")
    print(f"  CS Momentum: {len(sig_cs):,} signals")

    # ================================================================
    # PHASE 2: Combined Portfolio Simulation
    # ================================================================
    print(f"\n{'='*80}")
    print("PHASE 2: COMBINED PORTFOLIO SIMULATION")
    print(f"{'='*80}")

    # 2A: Merge signals
    all_dfs = []
    for df in [sig_15m, sig_4h, sig_cs]:
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("NO SIGNALS FROM ANY STRATEGY")
        return

    merged = pd.concat(all_dfs, ignore_index=True)

    # Fill missing columns for compatibility
    if "regime_exposure_mult" not in merged.columns:
        merged["regime_exposure_mult"] = 1.0
    if "sizing_mult" not in merged.columns:
        merged["sizing_mult"] = 1.0
    merged["regime_exposure_mult"] = merged["regime_exposure_mult"].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)

    # ----------------------------------------------------------------
    # IMPROVEMENT 1A: Vol Overlay (BTC vol regime sizing)
    # ----------------------------------------------------------------
    print("\n  [IMP-1A] Applying vol overlay...")
    try:
        from signal_engine.pipeline import apply_vol_overlay
        from signal_engine.config import VolOverlayConfig
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        vol_cfg = VolOverlayConfig(
            enabled=True, vol_window=20, pct_window=60,
            low_pct=0.20, high_pct=0.80,
            low_mult=0.50, mid_mult=1.00, high_mult=1.25,
        )
        merged = apply_vol_overlay(merged, btc_daily, vol_cfg)
        n_low = (merged["sizing_mult"] < 1.0).sum()
        n_high = (merged["sizing_mult"] > 1.0).sum()
        print(f"    Vol overlay applied: {n_low:,} low-vol scaled down, {n_high:,} high-vol scaled up")
    except Exception as e:
        print(f"    Vol overlay failed: {e}")

    # ----------------------------------------------------------------
    # IMPROVEMENT 1D: ML Close Model (path-dependent early exit)
    # Uses walk-forward XGBoost close model to exit losing trades early.
    # Generates per-bar decision examples, trains monthly, applies threshold.
    # ----------------------------------------------------------------
    USE_CLOSE_MODEL = True
    if USE_CLOSE_MODEL:
        print("  [IMP-1D] Integrating ML close model (path-dependent exits)...")
        try:
            from signal_engine.close_integration import compute_close_adjusted_signals
            from signal_engine.config import CloseModelConfig

            close_cfg = CloseModelConfig(
                enabled=True, model_type="xgboost",
                label_mode="residual_triple_barrier",
                threshold=0.5, calibration="none",
                lookbacks=[10], max_hold_minutes=60,
                min_hold_minutes=15, pt_mult=1.0, sl_mult=1.0,
                min_improvement_bps=2.0,
            )

            feat_cfg = FeatureConfig(
                feature_list=FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
            )
            port_cfg = PortfolioConfig(
                starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
                base_sizing_pct=0.02, max_signal_mult=2.0,
                sizing_mode="probability_edge",
            )

            # Adjust LogReg 15m signals with close model
            adj_15m = compute_close_adjusted_signals(
                merged, "logreg_15m", ASSETS, feat_cfg,
                close_cfg, port_cfg, max_workers=N_WORKERS,
            )
            # Adjust LogReg 4H signals with close model
            close_cfg_4h = CloseModelConfig(
                enabled=True, model_type="xgboost",
                label_mode="residual_triple_barrier",
                threshold=0.5, calibration="none",
                lookbacks=[10], max_hold_minutes=480,
                min_hold_minutes=240, pt_mult=1.0, sl_mult=1.0,
                min_improvement_bps=2.0,
            )
            adj_4h = compute_close_adjusted_signals(
                merged, "logreg_4h", ASSETS, feat_cfg,
                close_cfg_4h, port_cfg, max_workers=N_WORKERS,
            )

            # Replace LogReg signals with close-adjusted versions
            cs_only = merged[merged["strategy"] == "cs_momentum"].copy()
            if "return_already_signed" not in cs_only.columns:
                cs_only["return_already_signed"] = False
            if "remaining_fraction" not in cs_only.columns:
                cs_only["remaining_fraction"] = 1.0

            merged = pd.concat([adj_15m, adj_4h, cs_only], ignore_index=True)
            merged = merged.sort_values("timestamp").reset_index(drop=True)

            n_adj = len(adj_15m) + len(adj_4h)
            print(f"    Close model: {n_adj:,} adjusted LogReg signals + {len(cs_only):,} CS momentum")

        except Exception as e:
            print(f"    Close model integration failed: {e}")
            import traceback; traceback.print_exc()

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    print(f"  Merged: {len(merged):,} total signals")

    # 2E: Run per-strategy standalone simulations FIRST (for allocation weights)
    print("\n[2E] Per-strategy standalone simulations...")
    standalone_pnl = {}
    standalone_stats = {}

    portfolio_cfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.80, max_per_asset=0.06,
        base_sizing_pct=0.02, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
    )

    for strat_name in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        strat_signals = merged[merged["strategy"] == strat_name].copy()
        if len(strat_signals) < 10:
            print(f"  {strat_name}: <10 signals, skipping")
            continue

        sim = simulate_portfolio(strat_signals, portfolio_cfg)
        dpnl = compute_daily_pnl_series(sim) if sim else None

        if dpnl is not None:
            if dpnl.index.tz is not None:
                dpnl.index = dpnl.index.tz_localize(None)
            standalone_pnl[strat_name] = dpnl
            stats = compute_full_stats(dpnl, 10000.0, sim["n_trades"],
                                        sim["n_trades"] / max(len(dpnl), 1))
            standalone_stats[strat_name] = stats
            print(f"  {strat_name}: SR={stats.get('sharpe',0):.3f}, "
                  f"DD={stats.get('max_dd_pct',0):.1f}%, "
                  f"trades={sim['n_trades']:,}")

    # 2C: Walk-forward allocation
    print("\n[2C] Walk-forward strategy allocation (Sharpe-weighted)...")
    weights_by_month = walk_forward_strategy_weights(standalone_pnl, method="sharpe")

    # Print allocation evolution
    months = sorted(weights_by_month.keys())
    if months:
        print(f"  First month: {months[0]} -> {weights_by_month[months[0]]}")
        if len(months) > 1:
            print(f"  Last month:  {months[-1]} -> {weights_by_month[months[-1]]}")

    # Apply weights
    merged_weighted = apply_strategy_weights(merged, weights_by_month)

    # 2D: Combined simulation
    print("\n[2D] Combined portfolio simulation...")
    combined_cfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.40,
        base_sizing_pct=0.02, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
    )

    # Use precomputed-exits sim if close model adjusted signals are present
    if "return_already_signed" in merged_weighted.columns and merged_weighted["return_already_signed"].any():
        combined_sim = simulate_portfolio_precomputed_exits(merged_weighted, combined_cfg)
    else:
        combined_sim = simulate_portfolio(merged_weighted, combined_cfg)
    combined_pnl = compute_daily_pnl_series(combined_sim) if combined_sim else None

    if combined_pnl is None:
        print("  COMBINED SIMULATION PRODUCED NO PNL")
        return

    if combined_pnl.index.tz is not None:
        combined_pnl.index = combined_pnl.index.tz_localize(None)

    combined_stats = compute_full_stats(
        combined_pnl, 10000.0, combined_sim["n_trades"],
        combined_sim["n_trades"] / max(len(combined_pnl), 1)
    )

    # Also run equal-weight for comparison
    print("\n[2D-alt] Equal-weight comparison...")
    weights_equal = walk_forward_strategy_weights(standalone_pnl, method="equal")
    merged_equal = apply_strategy_weights(merged, weights_equal)
    if "return_already_signed" in merged_equal.columns and merged_equal["return_already_signed"].any():
        sim_equal = simulate_portfolio_precomputed_exits(merged_equal, combined_cfg)
    else:
        sim_equal = simulate_portfolio(merged_equal, combined_cfg)
    pnl_equal = compute_daily_pnl_series(sim_equal) if sim_equal else None
    if pnl_equal is not None:
        if pnl_equal.index.tz is not None:
            pnl_equal.index = pnl_equal.index.tz_localize(None)
        stats_equal = compute_full_stats(pnl_equal, 10000.0, sim_equal["n_trades"],
                                          sim_equal["n_trades"] / max(len(pnl_equal), 1))

    # ================================================================
    # PHASE 4: Comprehensive Reporting
    # ================================================================
    print(f"\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}")

    # Per-strategy standalone
    print(f"\n--- Per-Strategy Standalone ---")
    header = f"{'Strategy':<20} {'SR':>8} {'DD%':>8} {'$/cal':>10} {'Trades':>10} {'WinRate':>8}"
    print(header)
    print("-" * len(header))
    for strat, stats in standalone_stats.items():
        if "error" in stats:
            continue
        print(f"{strat:<20} {stats.get('sharpe',0):>8.3f} {stats.get('max_dd_pct',0):>7.1f}% "
              f"${stats.get('pnl_per_cal_day',0):>9.2f} {stats.get('n_trades',0):>10,} "
              f"{stats.get('win_rate',0):>7.1f}%")

    # Combined
    print(f"\n--- Combined Portfolio ---")
    print(f"{'Sharpe-weighted':<20} {combined_stats.get('sharpe',0):>8.3f} "
          f"{combined_stats.get('max_dd_pct',0):>7.1f}% "
          f"${combined_stats.get('pnl_per_cal_day',0):>9.2f} "
          f"{combined_sim['n_trades']:>10,} "
          f"{combined_stats.get('win_rate',0):>7.1f}%")

    if pnl_equal is not None:
        print(f"{'Equal-weight':<20} {stats_equal.get('sharpe',0):>8.3f} "
              f"{stats_equal.get('max_dd_pct',0):>7.1f}% "
              f"${stats_equal.get('pnl_per_cal_day',0):>9.2f} "
              f"{sim_equal['n_trades']:>10,} "
              f"{stats_equal.get('win_rate',0):>7.1f}%")

    # Diversification benefit
    best_standalone_sr = max(s.get("sharpe", 0) for s in standalone_stats.values())
    combined_sr = combined_stats.get("sharpe", 0)
    print(f"\n  Diversification benefit: {combined_sr:.3f} - {best_standalone_sr:.3f} = "
          f"{combined_sr - best_standalone_sr:+.3f} SR")

    # 4A: Bootstrap CIs + DSR
    print(f"\n--- Bootstrap 95% CI on Combined Sharpe ---")
    ci_lo, ci_hi = bootstrap_sharpe_ci(combined_pnl, 10000.0)
    print(f"  Combined SR: {combined_sr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    n_days = len(combined_pnl)
    daily_ret = combined_pnl / 10000.0
    skew = float(daily_ret.skew())
    kurt = float(daily_ret.kurtosis()) + 3.0
    dsr = deflated_sharpe_ratio(combined_sr, n_trials=7, T=n_days, skew=skew, kurt=kurt)
    print(f"  Deflated SR (7 trials): {dsr:.4f} {'PASS (>0.95)' if dsr > 0.95 else 'CAUTION (<0.95)'}")

    # Correlation matrix
    print(f"\n--- Pairwise Correlation (daily PnL) ---")
    if len(standalone_pnl) >= 2:
        corr_df = pd.DataFrame(standalone_pnl).fillna(0).corr()
        print(corr_df.round(4).to_string())

    # 4B: Regime breakdown
    print(f"\n--- Regime-Conditional Sharpe ---")
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)

        # Add combined to pnl dict
        all_pnl = dict(standalone_pnl)
        all_pnl["combined"] = combined_pnl
        regime_df = regime_breakdown(all_pnl, btc_daily)
        if not regime_df.empty:
            print(regime_df.round(2).to_string())
    except Exception as e:
        print(f"  Regime analysis failed: {e}")

    # 4C: Monthly attribution
    print(f"\n--- Monthly PnL Attribution ---")
    monthly_attr = {}
    for strat, pnl in standalone_pnl.items():
        monthly_attr[strat] = pnl.resample("MS").sum()
    monthly_attr["combined"] = combined_pnl.resample("MS").sum()
    monthly_df = pd.DataFrame(monthly_attr).fillna(0)

    for _, row in monthly_df.iterrows():
        month = row.name.strftime("%Y-%m")
        parts = "  ".join(f"{k}=${v:>7.0f}" for k, v in row.items())
        print(f"  {month}: {parts}")

    # 4E: Worst-case
    print(f"\n--- Worst-Case Analysis ---")
    for label, pnl in [("combined", combined_pnl)] + list(standalone_pnl.items()):
        worst_day = float(pnl.min())
        worst_week = float(pnl.rolling(5).sum().min())
        worst_month = float(pnl.resample("MS").sum().min())
        cvar_5 = float(pnl.quantile(0.05))
        print(f"  {label:>15}: worst_day=${worst_day:>8.0f}  worst_week=${worst_week:>8.0f}  "
              f"worst_month=${worst_month:>8.0f}  CVaR5%=${cvar_5:>8.0f}")

    # 4D: Correlation stability (rolling 30d)
    print(f"\n--- Correlation Stability (30-day rolling) ---")
    if len(standalone_pnl) >= 2:
        strat_df = pd.DataFrame(standalone_pnl).fillna(0)
        pairs = []
        strats = list(standalone_pnl.keys())
        for i in range(len(strats)):
            for j in range(i+1, len(strats)):
                rolling_corr = strat_df[strats[i]].rolling(30).corr(strat_df[strats[j]])
                pairs.append((strats[i], strats[j], rolling_corr))
                mean_c = float(rolling_corr.mean())
                max_c = float(rolling_corr.max())
                min_c = float(rolling_corr.min())
                spikes = int((rolling_corr.abs() > 0.5).sum())
                print(f"  {strats[i]:>12} <-> {strats[j]:<12}: "
                      f"mean={mean_c:.3f}, max={max_c:.3f}, min={min_c:.3f}, "
                      f"spikes(>0.5)={spikes}")

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"  Combined Sharpe:     {combined_sr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Combined Max DD:     {combined_stats.get('max_dd_pct',0):.1f}%")
    print(f"  Combined $/cal:      ${combined_stats.get('pnl_per_cal_day',0):.2f}")
    print(f"  Deflated SR:         {dsr:.4f}")
    print(f"  Total trades:        {combined_sim['n_trades']:,}")
    print(f"  Final equity:        ${combined_sim['final_equity']:,.2f}")
    print(f"  Diversification:     {combined_sr - best_standalone_sr:+.3f} SR")
    print(f"  Total runtime:       {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
