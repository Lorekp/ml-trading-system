#!/usr/bin/env python
"""Close Model V2 — Phase 2+3: Cost-Aware Labels + Minimal ML Model.

Phase 2: Compute cost-aware labels (close=1 if remaining return < -4.5bps)
Phase 3: Walk-forward LogReg/XGBoost on 8 state features with cost-aware labels

Uses cached close decision examples (1M rows, 250K trades).
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv

EXIT_COST = 0.00045  # 4.5 bps
CAPITAL = 100_000.0
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
VOL_INVERTED = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                 low_pct=0.20, high_pct=0.80,
                                 low_mult=1.25, mid_mult=1.00, high_mult=0.50)

# 8 state features — backward-looking, no market/regime features
STATE_FEATURES = [
    "current_unrealized_return",
    "max_adverse_return_since_entry",
    "distance_from_entry_mfe",
    "realized_vol_since_entry",
    "bars_since_entry",
    "time_to_expiry_minutes",
    "mfe_minus_mae",
    "entry_to_now_trend_slope",
]


def load_examples():
    """Load cached close decision examples."""
    path = Path("results/close_cache/examples_logreg_15m.parquet")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows, {df['trade_id'].nunique():,} trades")
    return df


def compute_cost_aware_labels(examples):
    """Phase 2: Compute cost-aware labels.

    close_label_v2 = 1 if remaining return < -EXIT_COST
    (i.e., holding to expiry loses more than the cost of exiting now)
    """
    remaining = examples["expiry_return"] - examples["exit_return"]
    labels = (remaining < -EXIT_COST).astype(np.int8)
    print(f"  Cost-aware labels: {labels.mean()*100:.1f}% positive "
          f"(vs original {examples['close_label'].mean()*100:.1f}%)")
    return labels


def oracle_analysis(examples, labels):
    """Phase 2: Oracle — perfect prediction of cost-aware labels."""
    # For each trade: if ANY step has label=1, close at FIRST such step
    ex = examples.copy()
    ex["label_v2"] = labels

    # Find first label=1 step per trade
    positive = ex[ex["label_v2"] == 1]
    first_close_idx = positive.groupby("trade_id").first().index
    first_close_ret = positive.groupby("trade_id")["exit_return"].first()

    # Last step per trade (expiry)
    last_step = ex.groupby("trade_id").last()
    expiry_ret = last_step["exit_return"]

    # Oracle: close at first label=1 step (paying EXIT_COST), else hold to expiry
    oracle_ret = expiry_ret.copy()
    oracle_ret.loc[first_close_ret.index] = first_close_ret - EXIT_COST

    print(f"\n  Oracle analysis:")
    print(f"    Hold all (expiry): mean={expiry_ret.mean()*10000:.1f}bps")
    print(f"    Oracle (cost-aware): mean={oracle_ret.mean()*10000:.1f}bps")
    print(f"    Improvement: {(oracle_ret.mean()-expiry_ret.mean())*10000:.1f}bps/trade")
    print(f"    Trades exited early: {len(first_close_ret):,}/{len(expiry_ret):,} "
          f"({len(first_close_ret)/len(expiry_ret)*100:.0f}%)")

    return oracle_ret


def walk_forward_close_v2(examples, labels, model_type="logistic",
                           lookback_months=10, test_start="2021-01"):
    """Phase 3: Walk-forward ML model on state features with cost-aware labels.

    Returns examples with added 'close_prob_v2' column.
    """
    ex = examples.copy()
    ex["label_v2"] = labels.values

    # Extract features and metadata
    feat_cols = [c for c in STATE_FEATURES if c in ex.columns]
    print(f"\n  Walk-forward {model_type} on {len(feat_cols)} features: {feat_cols}")

    feature_matrix = ex[feat_cols].to_numpy(dtype=np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    label_array = ex["label_v2"].to_numpy(dtype=np.int8)

    # Decision timestamps for walk-forward splits
    decision_ts = pd.to_datetime(ex["decision_timestamp"])
    decision_month = decision_ts.dt.to_period("M")
    month_ords = decision_month.astype("int64").to_numpy()

    # Label end timestamps for embargo
    label_end_ts = pd.to_datetime(ex["label_end_timestamp"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    decision_ts_ns = decision_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    test_start_period = pd.Period(test_start, freq="M")
    test_start_ord = test_start_period.ordinal

    unique_months = sorted(set(month_ords))
    predictions = np.full(len(ex), 0.5, dtype=np.float32)

    n_months_done = 0
    for m_ord in unique_months:
        if m_ord < test_start_ord:
            continue

        month_start_ns = pd.Period(ordinal=m_ord, freq="M").start_time.value
        train_start_ord = max(unique_months[0], m_ord - lookback_months)

        # Train mask: past months with labels resolved before test month
        train_mask = (
            (month_ords >= train_start_ord) &
            (month_ords < m_ord) &
            (decision_ts_ns < month_start_ns) &
            (label_end_ts < month_start_ns)
        )
        test_mask = month_ords == m_ord

        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        if len(train_idx) < 200 or len(test_idx) == 0:
            continue

        X_train = feature_matrix[train_idx]
        y_train = label_array[train_idx]
        X_test = feature_matrix[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train model
        if model_type == "logistic":
            model = LogisticRegression(C=0.1, penalty="l1", solver="saga",
                                        max_iter=1000, random_state=42)
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
        elif model_type == "xgboost":
            pos = y_train.sum()
            neg = len(y_train) - pos
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=max(neg / max(pos, 1), 1e-6),
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
            )
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        predictions[test_idx] = proba
        n_months_done += 1

    ex["close_prob_v2"] = predictions
    print(f"  Walk-forward done: {n_months_done} months predicted")
    return ex


def apply_model_exits(examples_with_proba, threshold=0.5):
    """Apply close model: exit at FIRST bar where close_prob_v2 >= threshold.

    Returns one row per trade with selected exit return.
    """
    ex = examples_with_proba.copy()
    triggered = ex["close_prob_v2"] >= threshold

    # First triggered bar per trade
    ex["_idx"] = np.arange(len(ex))
    first_triggered = ex[triggered].groupby("trade_id")["_idx"].first()
    last_bar = ex.groupby("trade_id")["_idx"].last()

    # Merge: use triggered if available, else last bar
    exit_idx = last_bar.copy()
    exit_idx.loc[first_triggered.index] = first_triggered

    selected = ex.iloc[exit_idx.values].copy()

    # Direction-signed return is already in exit_return
    # Apply exit cost for early exits
    max_step = ex.groupby("trade_id")["bars_since_entry"].max()
    is_early = selected.set_index("trade_id")["bars_since_entry"] < max_step
    n_early = int(is_early.sum())

    selected["fwd_return"] = selected["exit_return"].values
    # Early exits pay extra cost
    early_mask = is_early.reindex(selected["trade_id"]).values
    selected.loc[early_mask, "fwd_return"] -= EXIT_COST

    selected["hold_bars"] = selected["bars_since_entry"].fillna(4).astype(int)
    selected["strategy"] = "logreg_15m"
    selected["probability"] = selected["entry_probability"]
    selected["confidence"] = selected["entry_confidence"]
    selected["signal_strength"] = selected["entry_signal_strength"].abs()

    n_total = len(selected)
    print(f"  Threshold {threshold:.2f}: {n_early}/{n_total} early exits ({n_early/max(n_total,1)*100:.0f}%)")
    return selected


def bybit_sim(signals, base_pct=0.04, max_mult=2.0):
    """Bybit sim."""
    sub = signals.sort_values("timestamp").reset_index(drop=True)
    rt = 0.00045  # base Bybit roundtrip cost
    gross = sub["fwd_return"].values  # already includes exit cost for early exits
    net = gross - rt

    prob = sub["probability"].values if "probability" in sub.columns else np.full(len(sub), 0.55)
    sizing_mult = sub["sizing_mult"].values if "sizing_mult" in sub.columns else np.ones(len(sub))
    edge = np.where(np.isfinite(prob), np.abs(2 * prob - 1), 0.1)
    edge = np.clip(edge, 0, 1)

    eq = CAPITAL
    pnl = np.zeros(len(sub))
    for i in range(len(sub)):
        sz = min(eq * base_pct * edge[i] * sizing_mult[i], eq * base_pct * max_mult)
        pnl[i] = sz * net[i]
        eq += pnl[i]

    ts = pd.to_datetime(sub["timestamp"].values)
    if hasattr(ts, 'tz') and ts.tz is not None:
        ts = ts.tz_localize(None)
    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    sr = float(daily.mean() / daily.std() * np.sqrt(365.25)) if len(daily) > 30 and daily.std() > 0 else 0
    cum_eq = CAPITAL + daily.cumsum()
    dd = float(((cum_eq.cummax() - cum_eq) / cum_eq.cummax()).max() * 100) if len(cum_eq) > 0 else 0
    return {"sr": sr, "pnl": float(pnl.sum()), "dd": dd, "daily": daily,
            "n": len(sub), "net_bps": float(net.mean() * 10000)}


def merge_with_other_strategies(logreg_15m_signals):
    """Merge close-model adjusted 15m signals with 4h and cs_momentum."""
    parts = [logreg_15m_signals]
    for name in ["logreg_4h", "cs_momentum"]:
        path = Path(f"results/signals_cache/{name}.parquet")
        if path.exists():
            other = pd.read_parquet(path)
            for col in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
                if col not in other.columns:
                    other[col] = 4 if col == "hold_bars" else 1.0
            other["probability"] = other["probability"].astype(float)
            parts.append(other)

    merged = pd.concat(parts, ignore_index=True)
    if "regime_exposure_mult" not in merged.columns:
        merged["regime_exposure_mult"] = 1.0
    merged["regime_exposure_mult"] = merged["regime_exposure_mult"].fillna(1.0)
    if "sizing_mult" not in merged.columns:
        merged["sizing_mult"] = 1.0
    return merged.sort_values("timestamp").reset_index(drop=True)


def main():
    T0 = time.time()
    print("=" * 80)
    print("  CLOSE V2 PHASE 2+3: Cost-Aware Labels + Minimal ML Model")
    print("=" * 80)

    # Phase 2: Labels + Oracle
    print("\n[Phase 2] Cost-aware labels...")
    examples = load_examples()
    labels = compute_cost_aware_labels(examples)
    oracle_ret = oracle_analysis(examples, labels)

    # Oracle portfolio sim
    oracle_sigs = examples.groupby("trade_id").first().copy()
    oracle_sigs["fwd_return"] = oracle_ret.values
    oracle_sigs["strategy"] = "logreg_15m"
    oracle_sigs["probability"] = oracle_sigs["entry_probability"]
    oracle_sigs["confidence"] = oracle_sigs["entry_confidence"]
    oracle_sigs["signal_strength"] = oracle_sigs["entry_signal_strength"].abs()
    oracle_sigs["hold_bars"] = 4

    # Apply confidence filter
    oracle_filtered = oracle_sigs[oracle_sigs["confidence"] >= 0.08].reset_index(drop=True)
    merged_oracle = merge_with_other_strategies(oracle_filtered)

    # Vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        merged_oracle = apply_vol_overlay(merged_oracle, btc_daily, VOL_INVERTED)
        has_vol = True
    except Exception:
        has_vol = False

    r_oracle = bybit_sim(merged_oracle)
    print(f"\n  Oracle portfolio (Bybit): SR={r_oracle['sr']:.3f}  PnL=${r_oracle['pnl']:,.0f}  DD={r_oracle['dd']:.1f}%")

    # Baseline (fixed hold, expiry return for all)
    baseline_sigs = examples.groupby("trade_id").last().copy()
    baseline_sigs["fwd_return"] = baseline_sigs["exit_return"]
    baseline_sigs["strategy"] = "logreg_15m"
    baseline_sigs["probability"] = baseline_sigs["entry_probability"]
    baseline_sigs["confidence"] = baseline_sigs["entry_confidence"]
    baseline_sigs["signal_strength"] = baseline_sigs["entry_signal_strength"].abs()
    baseline_sigs["hold_bars"] = 4

    baseline_filtered = baseline_sigs[baseline_sigs["confidence"] >= 0.08].reset_index(drop=True)
    merged_baseline = merge_with_other_strategies(baseline_filtered)
    if has_vol:
        merged_baseline = apply_vol_overlay(merged_baseline, btc_daily, VOL_INVERTED)

    r_baseline = bybit_sim(merged_baseline)
    print(f"  Baseline (fixed hold, Bybit): SR={r_baseline['sr']:.3f}  PnL=${r_baseline['pnl']:,.0f}")
    print(f"  Oracle headroom: {r_oracle['sr'] - r_baseline['sr']:.3f} SR")

    # Phase 2 gate check
    if r_oracle['sr'] - r_baseline['sr'] < 1.0:
        print(f"\n  WARNING: Oracle headroom < 1.0 SR ({r_oracle['sr'] - r_baseline['sr']:.3f})")
        print(f"  Close model upside is limited. Proceeding cautiously.")

    # Phase 3: Walk-Forward ML Models
    print(f"\n{'='*80}")
    print("  [Phase 3] Walk-forward ML models on 8 state features")
    print(f"{'='*80}")

    for model_type in ["logistic", "xgboost"]:
        print(f"\n--- {model_type.upper()} ---")
        t0 = time.time()
        ex_with_pred = walk_forward_close_v2(examples, labels, model_type=model_type)
        gen_time = time.time() - t0
        print(f"  Training time: {gen_time:.0f}s")

        # Sweep thresholds
        print(f"\n  Threshold sweep:")
        best_sr = -999
        best_thresh = 0.5
        for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            selected = apply_model_exits(ex_with_pred, threshold=threshold)
            # Apply confidence filter
            conf_filtered = selected[selected["confidence"] >= 0.08].reset_index(drop=True)
            if len(conf_filtered) < 100:
                continue
            merged = merge_with_other_strategies(conf_filtered)
            if has_vol:
                merged = apply_vol_overlay(merged, btc_daily, VOL_INVERTED)
            r = bybit_sim(merged)
            marker = " <--" if r["sr"] > best_sr else ""
            print(f"    thresh={threshold:.2f}: SR={r['sr']:.3f}  PnL=${r['pnl']:,.0f}  "
                  f"DD={r['dd']:.1f}%  net={r['net_bps']:.1f}bps{marker}")
            if r["sr"] > best_sr:
                best_sr = r["sr"]
                best_thresh = threshold
                best_result = r

        print(f"\n  Best: threshold={best_thresh:.2f} SR={best_sr:.3f}")

        # Year-by-year for best threshold
        print(f"\n  Year-by-year ({model_type}, thresh={best_thresh:.2f}):")
        selected = apply_model_exits(ex_with_pred, threshold=best_thresh)
        conf_filtered = selected[selected["confidence"] >= 0.08].reset_index(drop=True)
        merged = merge_with_other_strategies(conf_filtered)
        if has_vol:
            merged = apply_vol_overlay(merged, btc_daily, VOL_INVERTED)
        r = bybit_sim(merged)
        for year in sorted(r["daily"].index.year.unique()):
            y = r["daily"][r["daily"].index.year == year]
            if len(y) > 20:
                sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
                print(f"    {year}: SR={sr_y:.3f}  PnL=${y.sum():,.0f}")

    # Final comparison
    print(f"\n{'='*80}")
    print("  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"  Baseline (fixed hold): SR={r_baseline['sr']:.3f}  PnL=${r_baseline['pnl']:,.0f}")
    print(f"  Oracle (perfect):      SR={r_oracle['sr']:.3f}  PnL=${r_oracle['pnl']:,.0f}")
    print(f"  ML best:               SR={best_sr:.3f} (headroom captured: "
          f"{(best_sr-r_baseline['sr'])/(r_oracle['sr']-r_baseline['sr'])*100:.0f}%)")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
