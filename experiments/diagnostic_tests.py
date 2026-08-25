#!/usr/bin/env python -u
"""Diagnostic tests for ML Close Model — validates whether edge is real or artifactual.

Tests:
1. Shuffle Test: randomize close_probability across trades
2. Random Labels: retrain on shuffled labels
3. Feature Ablation: retrain without current_unrealized_return
4. Threshold Sensitivity: SR vs threshold [0.3-0.8]
5. Year-by-Year Stability: per-year breakdown
6. Oracle Benchmark: perfect foresight exit timing

All tests use the same portfolio sim as production_backtest.py for fair comparison.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
os.environ["XGB_DEVICE"] = "cpu"
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.close_model import close_feature_columns
from pathlib import Path

CAPITAL = 100_000.0
PORT = PortfolioConfig(
    starting_capital=CAPITAL,
    max_dd=0.15, delever_start=0.075,
    slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.40,
    base_sizing_pct=0.02, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
CACHE = Path("results/close_cache")
SIGNAL_CACHE = Path("results/signals_cache")
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]


def load_examples():
    """Load raw close model examples (with labels and all features)."""
    path = CACHE / "examples_logreg_15m.parquet"
    if not path.exists():
        print(f"MISSING: {path} — run full_backtest_optimized.py first")
        sys.exit(1)
    return pd.read_parquet(path)


def load_btc_daily():
    """Load BTC daily close for vol overlay."""
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        return btc_daily
    except Exception as e:
        print(f"Warning: could not load BTC daily: {e}")
        return None


def select_exits(examples, threshold=0.6, return_col="exit_return"):
    """Given examples with close_probability, select one exit per trade."""
    ex = examples.copy()
    ex["_triggered"] = ex["close_probability"] > threshold
    ex["_row_idx"] = np.arange(len(ex))

    triggered = ex[ex["_triggered"]].groupby("trade_id")["_row_idx"].first()
    last_bar = ex.groupby("trade_id")["_row_idx"].last()

    exit_idx = last_bar.copy()
    exit_idx.loc[triggered.index] = triggered
    selected = ex.iloc[exit_idx.values].copy()
    selected["fwd_return"] = selected[return_col]
    selected["return_already_signed"] = True
    return selected


def build_portfolio_signals(selected):
    """Convert selected close model trades into signals for simulate_portfolio."""
    out = pd.DataFrame()
    out["timestamp"] = selected["timestamp"]
    out["asset"] = selected["asset"]
    out["direction"] = selected["direction"]
    # Unsigned return for portfolio sim
    out["fwd_return"] = selected["exit_return"] / selected["direction"]
    out["probability"] = selected["entry_probability"]
    out["confidence"] = selected["entry_confidence"]
    out["signal_strength"] = selected["entry_signal_strength"].abs()
    out["strategy"] = "logreg_15m"
    out["hold_bars"] = 4
    out["regime_exposure_mult"] = selected.get("regime_exposure_mult", 1.0)
    out["sizing_mult"] = 1.0
    return out


def merge_with_other_strategies(logreg_15m_signals):
    """Merge with 4H and CS momentum signals."""
    parts = [logreg_15m_signals]
    for name in ["logreg_4h", "cs_momentum"]:
        path = SIGNAL_CACHE / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            for col in ["regime_exposure_mult", "sizing_mult", "strategy", "hold_bars"]:
                if col not in df.columns:
                    if col == "strategy":
                        df[col] = name
                    elif col == "hold_bars":
                        df[col] = 4
                    else:
                        df[col] = 1.0
            df["probability"] = df["probability"].astype(float)
            parts.append(df)

    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return merged


def run_portfolio(signals, btc_daily=None, label=""):
    """Run full portfolio sim and return (SR, PnL, max_dd, n_trades)."""
    # Vol overlay
    if btc_daily is not None:
        try:
            vol_cfg = VolOverlayConfig(
                enabled=True, vol_window=20, pct_window=60,
                low_pct=0.20, high_pct=0.80,
                low_mult=0.50, mid_mult=1.00, high_mult=1.25,
            )
            signals = apply_vol_overlay(signals, btc_daily, vol_cfg)
        except Exception:
            pass

    sim = simulate_portfolio(signals, PORT)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 30:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    cum_eq = CAPITAL + dpnl.cumsum()
    max_dd = float(((cum_eq.cummax() - cum_eq) / cum_eq.cummax()).max() * 100)
    total_pnl = float(dpnl.sum())
    n_trades = sim["n_trades"]
    return {"sr": sr, "pnl": total_pnl, "max_dd": max_dd, "n_trades": n_trades, "dpnl": dpnl}


def print_result(label, result):
    if result is None:
        print(f"  {label:<40}: FAILED")
    else:
        print(f"  {label:<40}: SR={result['sr']:6.3f}  PnL=${result['pnl']:>10,.0f}  DD={result['max_dd']:.1f}%  Trades={result['n_trades']:,}")


# ============================================================
# TEST 1: SHUFFLE TEST
# ============================================================
def test_shuffle(examples, btc_daily, n_shuffles=5):
    """Shuffle close_probability across trades within each month."""
    print(f"\n{'='*80}")
    print("  TEST 1: SHUFFLE TEST")
    print(f"  Shuffling close_probability across trades within each month")
    print(f"  If model has real edge: SR should drop to ~baseline (2.3)")
    print(f"  If systematic bias: SR stays high")
    print(f"{'='*80}")

    results = []
    rng = np.random.RandomState(42)

    for trial in range(n_shuffles):
        ex = examples.copy()
        # Shuffle close_probability within each decision_month
        for month in ex["decision_month"].unique():
            mask = ex["decision_month"] == month
            probs = ex.loc[mask, "close_probability"].values.copy()
            rng.shuffle(probs)
            ex.loc[mask, "close_probability"] = probs

        selected = select_exits(ex)
        signals = build_portfolio_signals(selected)
        merged = merge_with_other_strategies(signals)
        r = run_portfolio(merged, btc_daily, f"Shuffle {trial+1}")
        if r:
            results.append(r["sr"])
            print_result(f"  Shuffle trial {trial+1}", r)

    if results:
        print(f"\n  Shuffle mean SR: {np.mean(results):.3f} (std: {np.std(results):.3f})")
        print(f"  Original SR: compare against production_backtest.py output")


# ============================================================
# TEST 2: RANDOM LABEL TRAINING
# ============================================================
def test_random_labels(examples, btc_daily):
    """Retrain close model with randomized labels."""
    print(f"\n{'='*80}")
    print("  TEST 2: RANDOM LABEL TRAINING")
    print(f"  Retraining XGBoost with shuffled close_label")
    print(f"  If model has real edge: SR should drop to ~baseline (2.3)")
    print(f"  If data channel leak: SR stays high")
    print(f"{'='*80}")

    from signal_engine.parallel_close_wf import parallel_walk_forward_close

    ex = examples.copy()
    feat_cols = close_feature_columns(FEATURES, regime_timeframes=["1h"])

    # Shuffle labels within each month
    rng = np.random.RandomState(99)
    for month in ex["decision_month"].unique():
        mask = ex["decision_month"] == month
        labels = ex.loc[mask, "close_label"].values.copy()
        rng.shuffle(labels)
        ex.loc[mask, "close_label"] = labels

    print("  Retraining with random labels (this takes ~15s)...")
    ex_pred = parallel_walk_forward_close(
        ex, feat_cols, lookback_months=10,
        test_start="2021-01", max_workers=os.cpu_count(),
        use_gpu=False,
    )

    selected = select_exits(ex_pred)
    signals = build_portfolio_signals(selected)
    merged = merge_with_other_strategies(signals)
    r = run_portfolio(merged, btc_daily)
    print_result("  Random labels", r)


# ============================================================
# TEST 3: FEATURE ABLATION
# ============================================================
def test_ablation(examples, btc_daily):
    """Retrain without current_unrealized_return."""
    print(f"\n{'='*80}")
    print("  TEST 3: FEATURE ABLATION — remove current_unrealized_return")
    print(f"  If model uses diverse features: SR stays well above baseline")
    print(f"  If model is just a loss-cutting rule: SR drops to baseline")
    print(f"{'='*80}")

    from signal_engine.parallel_close_wf import parallel_walk_forward_close

    feat_cols_full = close_feature_columns(FEATURES, regime_timeframes=["1h"])
    feat_cols_ablated = [c for c in feat_cols_full if c != "current_unrealized_return"]

    print(f"  Full features: {len(feat_cols_full)}, Ablated: {len(feat_cols_ablated)}")
    print("  Retraining without current_unrealized_return (this takes ~15s)...")

    ex = examples.copy()
    ex_pred = parallel_walk_forward_close(
        ex, feat_cols_ablated, lookback_months=10,
        test_start="2021-01", max_workers=os.cpu_count(),
        use_gpu=False,
    )

    selected = select_exits(ex_pred)
    signals = build_portfolio_signals(selected)
    merged = merge_with_other_strategies(signals)
    r = run_portfolio(merged, btc_daily)
    print_result("  Without current_unrealized_return", r)


# ============================================================
# TEST 4: THRESHOLD SENSITIVITY
# ============================================================
def test_threshold_sensitivity(examples, btc_daily):
    """Test SR at multiple thresholds."""
    print(f"\n{'='*80}")
    print("  TEST 4: THRESHOLD SENSITIVITY")
    print(f"  Running exit selection at thresholds [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]")
    print(f"  If broad peak: threshold is robust")
    print(f"  If sharp spike at 0.6: threshold is overfit")
    print(f"{'='*80}")

    thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    results = {}

    for t in thresholds:
        selected = select_exits(examples, threshold=t)
        n_early = int((selected["_triggered"] if "_triggered" in selected.columns else pd.Series(False)).sum())
        signals = build_portfolio_signals(selected)
        merged = merge_with_other_strategies(signals)
        r = run_portfolio(merged, btc_daily)
        results[t] = r
        early_pct = n_early / max(len(selected), 1) * 100
        if r:
            print(f"  Threshold={t:.2f}: SR={r['sr']:6.3f}  PnL=${r['pnl']:>10,.0f}  DD={r['max_dd']:.1f}%  Early={early_pct:.0f}%")
        else:
            print(f"  Threshold={t:.2f}: FAILED")

    if results:
        srs = {t: r["sr"] for t, r in results.items() if r}
        if srs:
            best_t = max(srs, key=srs.get)
            print(f"\n  Best threshold: {best_t} (SR={srs[best_t]:.3f})")
            if abs(best_t - 0.6) < 0.01:
                # Check if spike
                neighbors = [srs.get(0.5, 0), srs.get(0.55, 0), srs.get(0.65, 0), srs.get(0.7, 0)]
                neighbors = [x for x in neighbors if x > 0]
                if neighbors:
                    avg_neighbor = np.mean(neighbors)
                    ratio = srs[0.6] / avg_neighbor if avg_neighbor > 0 else 0
                    if ratio > 1.15:
                        print(f"  WARNING: SR at 0.6 is {ratio:.0%} of neighbors — possible overfit")
                    else:
                        print(f"  OK: SR at 0.6 is {ratio:.0%} of neighbors — broad peak")


# ============================================================
# TEST 5: YEAR-BY-YEAR STABILITY
# ============================================================
def test_yearly_stability(examples, btc_daily):
    """Report per-year performance."""
    print(f"\n{'='*80}")
    print("  TEST 5: YEAR-BY-YEAR STABILITY")
    print(f"  If edge is robust: consistent SR across years")
    print(f"  If regime-dependent: edge concentrated in 1-2 years")
    print(f"{'='*80}")

    # Get the full ML close selected signals and run portfolio sim once
    selected = select_exits(examples, threshold=0.6)
    signals = build_portfolio_signals(selected)
    merged = merge_with_other_strategies(signals)
    full_result = run_portfolio(merged, btc_daily)

    if full_result is None:
        print("  FAILED: could not compute portfolio")
        return

    dpnl = full_result["dpnl"]

    # Per-year breakdown
    print(f"\n  {'Year':<8} {'SR':>8} {'PnL':>12} {'DD':>8} {'Days':>8} {'Win%':>8}")
    print(f"  {'-'*52}")

    for year in sorted(dpnl.index.year.unique()):
        yr_pnl = dpnl[dpnl.index.year == year]
        if len(yr_pnl) < 30:
            continue
        dr = yr_pnl / CAPITAL
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        cum = CAPITAL + yr_pnl.cumsum()
        dd = float(((cum.cummax() - cum) / cum.cummax()).max() * 100)
        win_pct = float((yr_pnl > 0).sum() / len(yr_pnl) * 100)
        print(f"  {year:<8} {sr:>8.3f} ${yr_pnl.sum():>11,.0f} {dd:>7.1f}% {len(yr_pnl):>8} {win_pct:>7.1f}%")

    # Coefficient of variation of per-year SR
    year_srs = []
    for year in sorted(dpnl.index.year.unique()):
        yr_pnl = dpnl[dpnl.index.year == year]
        if len(yr_pnl) < 30:
            continue
        dr = yr_pnl / CAPITAL
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        year_srs.append(sr)

    if len(year_srs) > 1:
        cv = np.std(year_srs) / np.mean(year_srs) if np.mean(year_srs) > 0 else 0
        print(f"\n  SR stability: mean={np.mean(year_srs):.2f}, std={np.std(year_srs):.2f}, CV={cv:.2f}")
        if cv > 0.5:
            print(f"  WARNING: High CV={cv:.2f} — edge may be regime-dependent")
        else:
            print(f"  OK: CV={cv:.2f} — edge is reasonably stable")


# ============================================================
# TEST 6: ORACLE BENCHMARK
# ============================================================
def test_oracle(examples, btc_daily):
    """Select best exit per trade (perfect foresight) as upper bound."""
    print(f"\n{'='*80}")
    print("  TEST 6: ORACLE BENCHMARK (perfect foresight)")
    print(f"  If Oracle >> ML >> Baseline: hierarchy is correct")
    print(f"  If ML ~= Oracle: model has near-perfect information (suspicious)")
    print(f"{'='*80}")

    ex = examples.copy()
    ex["_row_idx"] = np.arange(len(ex))

    # Oracle: select bar with highest exit_return per trade
    oracle_idx = ex.groupby("trade_id").apply(
        lambda g: g.loc[g["exit_return"].idxmax()], include_groups=False
    )
    # Actually, use _row_idx for proper selection
    best_idx = ex.loc[ex.groupby("trade_id")["exit_return"].idxmax()]["_row_idx"].values
    oracle_selected = ex.iloc[best_idx].copy()

    oracle_signals = build_portfolio_signals(oracle_selected)
    oracle_merged = merge_with_other_strategies(oracle_signals)
    r_oracle = run_portfolio(oracle_merged, btc_daily)
    print_result("  Oracle (best exit per trade)", r_oracle)

    # Worst case: select bar with lowest exit_return per trade
    worst_idx = ex.loc[ex.groupby("trade_id")["exit_return"].idxmin()]["_row_idx"].values
    worst_selected = ex.iloc[worst_idx].copy()

    worst_signals = build_portfolio_signals(worst_selected)
    worst_merged = merge_with_other_strategies(worst_signals)
    r_worst = run_portfolio(worst_merged, btc_daily)
    print_result("  Worst (worst exit per trade)", r_worst)

    # ML close model (for comparison)
    ml_selected = select_exits(ex, threshold=0.6)
    ml_signals = build_portfolio_signals(ml_selected)
    ml_merged = merge_with_other_strategies(ml_signals)
    r_ml = run_portfolio(ml_merged, btc_daily)
    print_result("  ML Close (threshold=0.6)", r_ml)

    # Baseline (last bar per trade = fixed hold)
    last_idx = ex.groupby("trade_id")["_row_idx"].last().values
    baseline_selected = ex.iloc[last_idx].copy()

    baseline_signals = build_portfolio_signals(baseline_selected)
    baseline_merged = merge_with_other_strategies(baseline_signals)
    r_baseline = run_portfolio(baseline_merged, btc_daily)
    print_result("  Baseline (fixed 4-bar hold)", r_baseline)

    # Compare
    if r_oracle and r_ml and r_baseline:
        print(f"\n  Hierarchy check:")
        print(f"    Worst < Baseline < ML < Oracle ?")
        w = r_worst["sr"] if r_worst else -999
        b = r_baseline["sr"]
        m = r_ml["sr"]
        o = r_oracle["sr"]
        chain = w < b < m < o
        print(f"    {w:.2f} < {b:.2f} < {m:.2f} < {o:.2f} -- {'PASS' if chain else 'FAIL'}")

        if o > 0:
            ml_pct_of_oracle = (m - b) / (o - b) * 100 if (o - b) > 0 else 0
            print(f"    ML captures {ml_pct_of_oracle:.0f}% of oracle improvement over baseline")
            if ml_pct_of_oracle > 90:
                print(f"    WARNING: ML is {ml_pct_of_oracle:.0f}% of oracle — suspiciously close to perfect")
            elif ml_pct_of_oracle > 70:
                print(f"    NOTE: ML is {ml_pct_of_oracle:.0f}% of oracle — high but possible with strong features")
            else:
                print(f"    OK: ML is {ml_pct_of_oracle:.0f}% of oracle — reasonable gap")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()

    print("=" * 80)
    print("  ML CLOSE MODEL — DIAGNOSTIC TEST SUITE")
    print("  Purpose: Validate whether SR 7.64 is real or artifactual")
    print("=" * 80)

    # Load cached data
    print("\nLoading cached examples...")
    examples = load_examples()
    print(f"  {len(examples):,} close decision examples")

    # Need close_probability from the parallel WF predictions
    sel_path = CACHE / "selected_logreg_15m_parallel.parquet"
    if sel_path.exists():
        selected_cache = pd.read_parquet(sel_path)
        # Merge close_probability back into examples by trade_id + decision_step
        if "close_probability" in selected_cache.columns:
            # The selected cache only has ONE row per trade; examples have multiple
            # We need the full examples with close_probability from the WF model
            pass

    # Check if examples already have close_probability
    if "close_probability" not in examples.columns:
        print("  close_probability not in examples — running WF prediction...")
        from signal_engine.parallel_close_wf import parallel_walk_forward_close
        feat_cols = close_feature_columns(FEATURES, regime_timeframes=["1h"])
        examples = parallel_walk_forward_close(
            examples, feat_cols, lookback_months=10,
            test_start="2021-01", max_workers=os.cpu_count(),
            use_gpu=False,
        )
    else:
        print(f"  close_probability already present (range: {examples['close_probability'].min():.3f} - {examples['close_probability'].max():.3f})")

    btc_daily = load_btc_daily()

    # ---- Run tests ----
    # Fast tests first (no retraining needed)
    test_oracle(examples, btc_daily)           # Test 6: ~2 min
    test_threshold_sensitivity(examples, btc_daily)  # Test 4: ~3 min
    test_yearly_stability(examples, btc_daily)  # Test 5: ~1 min
    test_shuffle(examples, btc_daily, n_shuffles=5)  # Test 1: ~5 min

    # Expensive tests (require retraining)
    test_random_labels(examples, btc_daily)     # Test 2: ~20s + 2 min sim
    test_ablation(examples, btc_daily)          # Test 3: ~20s + 2 min sim

    print(f"\n{'='*80}")
    print(f"  ALL DIAGNOSTIC TESTS COMPLETE — Total: {time.time()-t0:.0f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
