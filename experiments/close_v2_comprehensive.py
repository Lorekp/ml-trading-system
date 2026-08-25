#!/usr/bin/env python
"""Close Model V2 — Comprehensive Exploration.

Tests multiple dimensions: labels, features, models, exit strategies.
Full stats comparison (SR, DD, Sortino, CVaR, worst day, monthly).
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

EXIT_COST = 0.00045
CAPITAL = 100_000.0
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
VOL_INV = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                            low_pct=0.20, high_pct=0.80,
                            low_mult=1.25, mid_mult=1.00, high_mult=0.50)

# Feature sets
STATE_8 = [
    "current_unrealized_return", "max_adverse_return_since_entry",
    "distance_from_entry_mfe", "realized_vol_since_entry",
    "bars_since_entry", "time_to_expiry_minutes",
    "mfe_minus_mae", "entry_to_now_trend_slope",
]
MARKET_13 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]
REGIME_13 = [f"regime_1h_{f}" for f in MARKET_13]


def load_data():
    """Load examples + BTC daily + other strategy signals."""
    ex = pd.read_parquet("results/close_cache/examples_logreg_15m.parquet")
    print(f"Examples: {len(ex):,} rows, {ex['trade_id'].nunique():,} trades")

    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
    except Exception:
        btc_daily = None

    other_sigs = []
    for name in ["logreg_4h", "cs_momentum"]:
        p = Path(f"results/signals_cache/{name}.parquet")
        if p.exists():
            other_sigs.append(pd.read_parquet(p))

    return ex, btc_daily, other_sigs


def full_stats(daily_pnl, capital=CAPITAL):
    """Compute comprehensive stats from daily PnL."""
    if daily_pnl is None or len(daily_pnl) < 10:
        return {}
    dr = daily_pnl / capital
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd_series = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(dd_series.max() * 100)
    calmar = float(dr.mean() * 365.25 / dd_series.max()) if dd_series.max() > 0 else 0
    worst_day = float(daily_pnl.min())
    weekly = daily_pnl.resample("W").sum()
    worst_week = float(weekly.min()) if len(weekly) > 0 else 0
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    win_rate = float((daily_pnl > 0).sum() / len(daily_pnl) * 100)
    monthly = daily_pnl.resample("MS").sum()
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)
    worst_month = float(monthly.min())
    streak = mx = 0
    for v in daily_pnl.values:
        if v < 0:
            streak += 1
            mx = max(mx, streak)
        else:
            streak = 0
    pf = float(daily_pnl[daily_pnl > 0].sum() / abs(daily_pnl[daily_pnl < 0].sum())) if (daily_pnl < 0).any() else 99
    return {
        "sr": round(sr, 3), "sortino": round(sortino, 3), "calmar": round(calmar, 2),
        "max_dd": round(max_dd, 1), "worst_day": round(worst_day, 0),
        "worst_week": round(worst_week, 0), "worst_month": round(worst_month, 0),
        "cvar5_bps": round(cvar5, 1), "win_rate": round(win_rate, 1),
        "pos_months": f"{pos_months}/{tot_months}",
        "max_lose_streak": mx, "profit_factor": round(min(pf, 99), 2),
        "pnl": round(float(daily_pnl.sum()), 0),
    }


def bybit_sim(signals, base_pct=0.04, max_mult=2.0):
    """Bybit sim returning daily PnL series."""
    sub = signals.sort_values("timestamp").reset_index(drop=True)
    rt = EXIT_COST
    gross = sub["fwd_return"].values
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
    return daily[daily != 0]


def walk_forward(examples, target, features, model_type="xgboost",
                  lookback_months=10, test_start="2021-01", is_regression=False):
    """Generalized walk-forward. Returns predictions array."""
    feat_cols = [c for c in features if c in examples.columns]
    feat_matrix = examples[feat_cols].to_numpy(dtype=np.float32)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)
    target_arr = target.to_numpy(dtype=np.float32)

    decision_ts = pd.to_datetime(examples["decision_timestamp"])
    month_ords = decision_ts.dt.to_period("M").astype("int64").to_numpy()
    label_end_ns = pd.to_datetime(examples["label_end_timestamp"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    dec_ns = decision_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    test_start_ord = pd.Period(test_start, freq="M").ordinal
    unique_months = sorted(set(month_ords))
    preds = np.full(len(examples), 0.5 if not is_regression else 0.0, dtype=np.float32)

    for m_ord in unique_months:
        if m_ord < test_start_ord:
            continue
        ms_ns = pd.Period(ordinal=m_ord, freq="M").start_time.value
        train_start_ord = max(unique_months[0], m_ord - lookback_months)
        train_mask = ((month_ords >= train_start_ord) & (month_ords < m_ord) &
                      (dec_ns < ms_ns) & (label_end_ns < ms_ns))
        test_mask = month_ords == m_ord
        tr_idx = np.flatnonzero(train_mask)
        te_idx = np.flatnonzero(test_mask)
        if len(tr_idx) < 200 or len(te_idx) == 0:
            continue

        X_tr = feat_matrix[tr_idx]
        y_tr = target_arr[tr_idx]
        X_te = feat_matrix[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if is_regression:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
            )
            model.fit(X_tr_s, y_tr)
            preds[te_idx] = model.predict(X_te_s)
        elif model_type == "logistic":
            if len(np.unique(y_tr)) < 2:
                continue
            model = LogisticRegression(C=0.1, penalty="l1", solver="saga",
                                        max_iter=1000, random_state=42)
            model.fit(X_tr_s, y_tr)
            preds[te_idx] = model.predict_proba(X_te_s)[:, 1]
        elif model_type in ("xgboost", "xgb_deeper"):
            if len(np.unique(y_tr)) < 2:
                continue
            pos = y_tr.sum()
            neg = len(y_tr) - pos
            depth = 5 if model_type == "xgb_deeper" else 3
            n_est = 100 if model_type == "xgb_deeper" else 50
            model = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=max(neg / max(pos, 1), 1e-6),
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
            )
            model.fit(X_tr_s, y_tr)
            preds[te_idx] = model.predict_proba(X_te_s)[:, 1]

    return preds


def select_exits(examples, preds, threshold, exit_cost=EXIT_COST, partial_pct=1.0):
    """Select exit bar per trade. partial_pct=1.0 means full close, 0.5 means half close."""
    ex = examples.copy()
    ex["_pred"] = preds
    ex["_idx"] = np.arange(len(ex))

    triggered = ex[ex["_pred"] >= threshold]
    first_triggered = triggered.groupby("trade_id")["_idx"].first()
    last_bar = ex.groupby("trade_id")["_idx"].last()

    exit_idx = last_bar.copy()
    exit_idx.loc[first_triggered.index] = first_triggered
    selected = ex.iloc[exit_idx.values].copy()

    max_step = ex.groupby("trade_id")["bars_since_entry"].max()
    is_early = selected.set_index("trade_id")["bars_since_entry"] < max_step

    if partial_pct >= 1.0:
        # Full close: use exit_return at selected bar, pay exit cost for early exits
        selected["fwd_return"] = selected["exit_return"].values
        early_mask = is_early.reindex(selected["trade_id"]).values
        selected.loc[early_mask, "fwd_return"] -= exit_cost
    else:
        # Partial close: blend of early exit (partial_pct * exit_return) and hold (rest * expiry_return)
        early_mask = is_early.reindex(selected["trade_id"]).values
        early_ret = selected["exit_return"].values - exit_cost * partial_pct
        expiry_ret = selected["expiry_return"].values if "expiry_return" in selected.columns else selected["exit_return"].values
        blended = np.where(early_mask,
                           partial_pct * early_ret + (1 - partial_pct) * expiry_ret,
                           expiry_ret)
        selected["fwd_return"] = blended

    selected["hold_bars"] = selected["bars_since_entry"].fillna(4).astype(int)
    selected["strategy"] = "logreg_15m"
    selected["probability"] = selected["entry_probability"]
    selected["confidence"] = selected["entry_confidence"]
    selected["signal_strength"] = selected["entry_signal_strength"].abs()

    n_early = int(is_early.sum())
    return selected, n_early


def build_portfolio(logreg_15m_sigs, other_sigs, btc_daily, conf_filter=0.08):
    """Merge, filter, apply vol overlay."""
    # Confidence filter on 15m
    if conf_filter > 0 and "confidence" in logreg_15m_sigs.columns:
        logreg_15m_sigs = logreg_15m_sigs[logreg_15m_sigs["confidence"] >= conf_filter].reset_index(drop=True)

    parts = [logreg_15m_sigs]
    for o in other_sigs:
        d = o.copy()
        for col in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
            if col not in d.columns:
                d[col] = 4 if col == "hold_bars" else 1.0
        d["probability"] = d["probability"].astype(float)
        parts.append(d)

    merged = pd.concat(parts, ignore_index=True)
    for col in ["regime_exposure_mult", "sizing_mult"]:
        if col not in merged.columns:
            merged[col] = 1.0
        merged[col] = merged[col].fillna(1.0)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if btc_daily is not None:
        merged = apply_vol_overlay(merged, btc_daily, VOL_INV)
    return merged


def print_stats_table(results_dict):
    """Print a comparison table of full stats."""
    if not results_dict:
        return
    cols = ["sr", "sortino", "max_dd", "cvar5_bps", "worst_day", "worst_month",
            "win_rate", "pos_months", "max_lose_streak", "profit_factor", "pnl"]
    header = f"{'Variant':<35}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, stats in results_dict.items():
        row = f"{name:<35}"
        for c in cols:
            v = stats.get(c, "N/A")
            if isinstance(v, float):
                if c in ("sr", "sortino"):
                    row += f"{v:>12.3f}"
                elif c in ("max_dd", "win_rate"):
                    row += f"{v:>11.1f}%"
                elif c in ("cvar5_bps",):
                    row += f"{v:>12.1f}"
                elif c in ("profit_factor",):
                    row += f"{v:>12.2f}"
                else:
                    row += f"${v:>10,.0f}"
            elif isinstance(v, int):
                row += f"{v:>12d}"
            else:
                row += f"{str(v):>12}"
        print(row)


def main():
    T0 = time.time()
    print("=" * 100)
    print("  CLOSE V2 COMPREHENSIVE SWEEP")
    print("=" * 100)

    examples, btc_daily, other_sigs = load_data()

    # Precompute labels
    remaining = examples["expiry_return"] - examples["exit_return"]
    labels = {
        "cost_4.5bps": (remaining < -0.00045).astype(np.int8),
        "cost_10bps": (remaining < -0.001).astype(np.int8),
        "cost_20bps": (remaining < -0.002).astype(np.int8),
        "lose_overall": (examples["expiry_return"] < 0).astype(np.int8),
    }
    for name, lb in labels.items():
        print(f"  Label '{name}': {lb.mean()*100:.1f}% positive")

    # Feature sets
    feature_sets = {
        "state_8": STATE_8,
        "state_market_21": STATE_8 + MARKET_13,
        "all_safe_34": STATE_8 + MARKET_13 + [c for c in REGIME_13 if c in examples.columns],
    }
    # Add interaction features
    ex = examples.copy()
    if "bar_range" in ex.columns and "current_unrealized_return" in ex.columns:
        ex["ix_unret_x_barrange"] = ex["current_unrealized_return"] * ex["bar_range"]
        ex["ix_bars_x_vol"] = ex["bars_since_entry"] * ex["realized_vol_since_entry"]
        ex["ix_range_x_conf"] = ex["mfe_minus_mae"] * ex["entry_confidence"]
        ex["ix_mfedist_x_atr"] = ex["distance_from_entry_mfe"] * ex["atr_14"]
        feature_sets["state_interact_12"] = STATE_8 + [
            "ix_unret_x_barrange", "ix_bars_x_vol", "ix_range_x_conf", "ix_mfedist_x_atr"
        ]
    examples = ex

    # ================================================================
    # BASELINE: Fixed hold
    # ================================================================
    print(f"\n{'='*100}")
    print("  BASELINE: Fixed hold (expiry return)")
    print(f"{'='*100}")

    baseline_sigs = examples.groupby("trade_id").last().copy()
    baseline_sigs["fwd_return"] = baseline_sigs["exit_return"]
    baseline_sigs["strategy"] = "logreg_15m"
    baseline_sigs["probability"] = baseline_sigs["entry_probability"]
    baseline_sigs["confidence"] = baseline_sigs["entry_confidence"]
    baseline_sigs["signal_strength"] = baseline_sigs["entry_signal_strength"].abs()
    baseline_sigs["hold_bars"] = 4
    baseline_merged = build_portfolio(baseline_sigs, other_sigs, btc_daily)
    baseline_daily = bybit_sim(baseline_merged)
    baseline_stats = full_stats(baseline_daily)

    all_results = {"BASELINE (fixed hold)": baseline_stats}
    print(f"  SR={baseline_stats['sr']}  DD={baseline_stats['max_dd']}%  PnL=${baseline_stats['pnl']:,.0f}")

    # ================================================================
    # ROUND 1: Priority experiments
    # ================================================================
    print(f"\n{'='*100}")
    print("  ROUND 1: Priority experiments")
    print(f"{'='*100}")

    configs = [
        # (name, model_type, feature_set, label, is_regression, partial_pct, thresholds)
        ("Regr+state_mkt21", "regression", "state_market_21", None, True, 1.0,
         [-0.005, -0.003, -0.002, -0.001, 0.0, 0.001]),
        ("XGB_deep+all34+cost20", "xgb_deeper", "all_safe_34", "cost_20bps", False, 1.0,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]),
        ("XGB+state_mkt21+cost4.5", "xgboost", "state_market_21", "cost_4.5bps", False, 1.0,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]),
        ("XGB+state_mkt21+lose", "xgboost", "state_market_21", "lose_overall", False, 1.0,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]),
        ("LogReg+state_mkt21+cost10", "logistic", "state_market_21", "cost_10bps", False, 1.0,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]),
        ("XGB+interact12+cost10", "xgboost", "state_interact_12", "cost_10bps", False, 1.0,
         [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]),
    ]

    for cfg_name, model_type, feat_key, label_key, is_regr, partial, thresholds in configs:
        print(f"\n--- {cfg_name} ---")
        t0 = time.time()

        features = feature_sets[feat_key]
        if is_regr:
            target = remaining  # predict remaining return
            preds = walk_forward(examples, target, features, model_type, is_regression=True)
        else:
            target = labels[label_key]
            preds = walk_forward(examples, target, features, model_type, is_regression=False)

        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.0f}s")

        best_sr = -999
        best_thresh = thresholds[0]
        for thresh in thresholds:
            selected, n_early = select_exits(examples, preds, thresh, partial_pct=partial)
            merged = build_portfolio(selected, other_sigs, btc_daily)
            daily = bybit_sim(merged)
            stats = full_stats(daily)
            marker = " <--" if stats.get("sr", 0) > best_sr else ""
            print(f"  t={thresh:>6.3f}: SR={stats.get('sr',0):.3f} Sortino={stats.get('sortino',0):.3f} "
                  f"DD={stats.get('max_dd',0):.1f}% CVaR={stats.get('cvar5_bps',0):.1f}bps "
                  f"PnL=${stats.get('pnl',0):,.0f} early={n_early:,}{marker}")
            if stats.get("sr", 0) > best_sr:
                best_sr = stats["sr"]
                best_thresh = thresh
                best_stats = stats
                best_daily = daily

        all_results[cfg_name] = best_stats
        print(f"  BEST: t={best_thresh} SR={best_sr:.3f}")

        # Year-by-year for best
        for year in sorted(best_daily.index.year.unique()):
            y = best_daily[best_daily.index.year == year]
            if len(y) > 20:
                sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
                dd_eq = CAPITAL + y.cumsum()
                dd_y = float(((dd_eq.cummax() - dd_eq) / dd_eq.cummax()).max() * 100) if len(dd_eq) > 0 else 0
                print(f"    {year}: SR={sr_y:.3f} DD={dd_y:.1f}%")

    # ================================================================
    # ROUND 1b: Partial exits with best classification model
    # ================================================================
    print(f"\n--- Partial exits (50%) with XGB+state_mkt21+cost4.5 ---")
    target = labels["cost_4.5bps"]
    features = feature_sets["state_market_21"]
    preds = walk_forward(examples, target, features, "xgboost")
    for thresh in [0.40, 0.50, 0.55, 0.60]:
        selected, n_early = select_exits(examples, preds, thresh, partial_pct=0.5)
        merged = build_portfolio(selected, other_sigs, btc_daily)
        daily = bybit_sim(merged)
        stats = full_stats(daily)
        print(f"  partial50 t={thresh:.2f}: SR={stats.get('sr',0):.3f} DD={stats.get('max_dd',0):.1f}% "
              f"PnL=${stats.get('pnl',0):,.0f}")

    # ================================================================
    # ROUND 1c: Vol-based sizing (no ML)
    # ================================================================
    print(f"\n--- Vol-based sizing (no ML) ---")
    last_bars = examples.groupby("trade_id").last().copy()
    last_bars["fwd_return"] = last_bars["exit_return"]
    last_bars["strategy"] = "logreg_15m"
    last_bars["probability"] = last_bars["entry_probability"]
    last_bars["confidence"] = last_bars["entry_confidence"]
    last_bars["signal_strength"] = last_bars["entry_signal_strength"].abs()
    last_bars["hold_bars"] = 4

    # Scale down sizing when bar_range or atr_14 is high
    if "bar_range" in last_bars.columns and "atr_14" in last_bars.columns:
        br_pct = last_bars["bar_range"].rank(pct=True)
        for mult_factor in [0.5, 0.7, 0.3]:
            sigs = last_bars.copy()
            # Reduce sizing for high-vol trades
            sigs["sizing_mult"] = np.where(br_pct > 0.8, mult_factor, 1.0)
            merged = build_portfolio(sigs, other_sigs, btc_daily)
            daily = bybit_sim(merged)
            stats = full_stats(daily)
            print(f"  vol_sizing (high_vol_mult={mult_factor}): SR={stats.get('sr',0):.3f} "
                  f"DD={stats.get('max_dd',0):.1f}% PnL=${stats.get('pnl',0):,.0f}")
            all_results[f"vol_sizing_{mult_factor}"] = stats

    # ================================================================
    # FULL COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*100}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*100}")
    print_stats_table(all_results)

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
