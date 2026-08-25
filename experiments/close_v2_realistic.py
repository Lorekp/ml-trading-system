#!/usr/bin/env python
"""Close Model V2 — Realistic Comparison using simulate_portfolio.

Uses the authoritative simulation: simulate_portfolio() with exposure tracking,
per-asset limits, deleveraging, then Bybit VIP0 cost adjustment.
Same methodology as production_backtest.py.

Tests the best close model variants from the comprehensive sweep.
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

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv

CAPITAL = 100_000.0
BYBIT_EXTRA_BPS = 3.5  # on top of 1bps in PortfolioConfig
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

PORT = PortfolioConfig(
    starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
    slippage=0.0001,  # 1bps idealized
    max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.50,
    base_sizing_pct=0.04, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
VOL_INV = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                            low_pct=0.20, high_pct=0.80,
                            low_mult=1.25, mid_mult=1.00, high_mult=0.50)

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
EXIT_COST = 0.00045


def walk_forward(examples, target, features, model_type="xgboost",
                  lookback_months=10, test_start="2021-01", is_regression=False):
    """Walk-forward training, returns predictions array."""
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
            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                                      tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr_s, y_tr)
            preds[te_idx] = model.predict(X_te_s)
        else:
            if len(np.unique(y_tr)) < 2:
                continue
            pos = y_tr.sum()
            neg = len(y_tr) - pos
            depth = 5 if model_type == "xgb_deeper" else 3
            n_est = 100 if model_type == "xgb_deeper" else 50
            model = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                                       min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=max(neg/max(pos,1), 1e-6),
                                       tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr_s, y_tr)
            preds[te_idx] = model.predict_proba(X_te_s)[:, 1]
    return preds


def select_exits(examples, preds, threshold, partial_pct=1.0):
    """Select exit bar per trade, return portfolio-ready signals."""
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
    early_mask = is_early.reindex(selected["trade_id"]).values
    n_early = int(early_mask.sum())

    if partial_pct >= 1.0:
        # Full close: fwd_return = unsigned exit_return, portfolio sim does direction * fwd_return
        fwd = selected["exit_return"].values / selected["direction"].values
        # Early exits pay extra cost (as additional slippage)
        fwd[early_mask] -= EXIT_COST
    else:
        # Partial: blend early exit + hold to expiry
        exit_ret = selected["exit_return"].values / selected["direction"].values
        expiry_ret = selected["expiry_return"].values / selected["direction"].values if "expiry_return" in selected.columns else exit_ret
        exit_ret_adj = exit_ret - EXIT_COST * partial_pct
        fwd = np.where(early_mask, partial_pct * exit_ret_adj + (1 - partial_pct) * expiry_ret, expiry_ret)

    selected["fwd_return"] = fwd
    selected["hold_bars"] = selected["bars_since_entry"].fillna(4).astype(int)
    selected["strategy"] = "logreg_15m"
    selected["probability"] = selected["entry_probability"]
    selected["confidence"] = selected["entry_confidence"]
    selected["signal_strength"] = selected["entry_signal_strength"].abs()
    if "regime_exposure_mult" not in selected.columns:
        selected["regime_exposure_mult"] = 1.0
    selected["regime_exposure_mult"] = selected["regime_exposure_mult"].fillna(1.0)

    return selected, n_early


def build_portfolio(logreg_15m, conf_filter=0.08):
    """Merge 15m with 4h + cs, apply conf filter + vol overlay."""
    if conf_filter > 0 and "confidence" in logreg_15m.columns:
        logreg_15m = logreg_15m[logreg_15m["confidence"] >= conf_filter].reset_index(drop=True)

    parts = [logreg_15m]
    for name in ["logreg_4h", "cs_momentum"]:
        p = Path(f"results/signals_cache/{name}.parquet")
        if p.exists():
            o = pd.read_parquet(p)
            for col in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
                if col not in o.columns:
                    o[col] = 4 if col == "hold_bars" else 1.0
            o["probability"] = o["probability"].astype(float)
            parts.append(o)

    merged = pd.concat(parts, ignore_index=True)
    for col in ["regime_exposure_mult", "sizing_mult"]:
        if col not in merged.columns:
            merged[col] = 1.0
        merged[col] = merged[col].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        merged = apply_vol_overlay(merged, btc_daily, VOL_INV)
    except Exception:
        pass
    return merged


def run_portfolio(signals, label):
    """Run simulate_portfolio + Bybit cost adjustment. Return full stats."""
    sim = simulate_portfolio(signals, PORT)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Bybit cost adjustment
    n_trades = sim["n_trades"]
    avg_size = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    extra_cost = avg_size * BYBIT_EXTRA_BPS / 10000
    total_extra = n_trades * extra_cost
    daily_cost = total_extra / max(len(dpnl), 1)
    dpnl_adj = dpnl - daily_cost

    # Full stats
    dr = dpnl_adj / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(dd.max() * 100)
    calmar = float(dr.mean() * 365.25 / dd.max()) if dd.max() > 0 else 0
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    worst_day = float(dpnl_adj.min())
    monthly = dpnl_adj.resample("MS").sum()
    worst_month = float(monthly.min())
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    streak = mx = 0
    for v in dpnl_adj.values:
        if v < 0:
            streak += 1; mx = max(mx, streak)
        else:
            streak = 0

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    drv = dr.values
    for b in range(3000):
        s = rng.choice(drv, size=len(drv), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    return {
        "sr": round(sr, 3), "ci": ci, "sortino": round(sortino, 3),
        "calmar": round(calmar, 2), "max_dd": round(max_dd, 1),
        "cvar5_bps": round(cvar5, 1), "worst_day": round(worst_day, 0),
        "worst_month": round(worst_month, 0),
        "pos_months": f"{pos_months}/{tot_months}",
        "max_lose_streak": mx, "profit_factor": round(min(pf, 99), 2),
        "pnl": round(float(dpnl_adj.sum()), 0),
        "n_trades": n_trades, "n_skipped": int(sim["skipped"].sum()),
        "fees": round(total_extra, 0),
        "daily": dpnl_adj,
    }


def print_comparison(results):
    """Print comparison table."""
    cols = ["sr", "sortino", "max_dd", "cvar5_bps", "worst_day", "worst_month",
            "pos_months", "max_lose_streak", "profit_factor", "pnl", "n_trades", "fees"]
    header = f"{'Variant':<32}" + "".join(f"{c:>13}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        row = f"{name:<32}"
        for c in cols:
            v = r.get(c, "N/A")
            if isinstance(v, float):
                if c in ("sr", "sortino"):
                    row += f"{v:>13.3f}"
                elif c in ("max_dd",):
                    row += f"{v:>12.1f}%"
                elif c in ("profit_factor",):
                    row += f"{v:>13.2f}"
                elif c in ("cvar5_bps",):
                    row += f"{v:>13.1f}"
                else:
                    row += f"${v:>11,.0f}"
            elif isinstance(v, int):
                row += f"{v:>13,}"
            else:
                row += f"{str(v):>13}"
        print(row)


def main():
    T0 = time.time()
    print("=" * 100)
    print("  CLOSE V2 — REALISTIC COMPARISON (simulate_portfolio + Bybit)")
    print("=" * 100)

    examples = pd.read_parquet("results/close_cache/examples_logreg_15m.parquet")
    remaining = examples["expiry_return"] - examples["exit_return"]
    print(f"Examples: {len(examples):,} rows, {examples['trade_id'].nunique():,} trades")

    results = {}

    # ================================================================
    # BASELINE: Fixed hold (expiry return)
    # ================================================================
    print(f"\n--- BASELINE: Fixed hold ---")
    baseline = examples.groupby("trade_id").last().copy()
    baseline["fwd_return"] = baseline["exit_return"] / baseline["direction"]
    baseline["strategy"] = "logreg_15m"
    baseline["probability"] = baseline["entry_probability"]
    baseline["confidence"] = baseline["entry_confidence"]
    baseline["signal_strength"] = baseline["entry_signal_strength"].abs()
    baseline["hold_bars"] = 4
    baseline_merged = build_portfolio(baseline)
    r = run_portfolio(baseline_merged, "Baseline")
    if r:
        results["BASELINE (fixed hold)"] = r
        print(f"  SR={r['sr']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  DD={r['max_dd']}%  "
              f"PnL=${r['pnl']:,.0f}  trades={r['n_trades']:,}")

    # ================================================================
    # CLOSE MODEL VARIANTS
    # ================================================================
    configs = [
        # (name, model_type, features, label_key, is_regr, partial_pct, thresholds)
        ("XGB_deep+all34+cost20",
         "xgb_deeper", STATE_8 + MARKET_13 + REGIME_13, "cost_20bps", False, 1.0,
         [0.60, 0.65, 0.70, 0.75]),
        ("XGB+stmkt21+cost4.5",
         "xgboost", STATE_8 + MARKET_13, "cost_4.5bps", False, 1.0,
         [0.55, 0.60, 0.65, 0.70]),
        ("Partial50+XGB+stmkt21",
         "xgboost", STATE_8 + MARKET_13, "cost_4.5bps", False, 0.5,
         [0.50, 0.55, 0.60, 0.65]),
        ("XGB_deep+all34+cost20 P50",
         "xgb_deeper", STATE_8 + MARKET_13 + REGIME_13, "cost_20bps", False, 0.5,
         [0.55, 0.60, 0.65, 0.70]),
    ]

    labels_map = {
        "cost_4.5bps": (remaining < -0.00045).astype(np.int8),
        "cost_20bps": (remaining < -0.002).astype(np.int8),
    }

    for cfg_name, model_type, features, label_key, is_regr, partial, thresholds in configs:
        print(f"\n--- {cfg_name} ---")
        t0 = time.time()
        target = labels_map[label_key]
        preds = walk_forward(examples, target, features, model_type, is_regression=is_regr)
        print(f"  Trained in {time.time()-t0:.0f}s")

        best_sr = -999
        for thresh in thresholds:
            selected, n_early = select_exits(examples, preds, thresh, partial_pct=partial)
            merged = build_portfolio(selected)
            r = run_portfolio(merged, f"{cfg_name} t={thresh}")
            if r:
                marker = " <--" if r["sr"] > best_sr else ""
                print(f"  t={thresh:.2f}: SR={r['sr']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] "
                      f"Sortino={r['sortino']:.3f} DD={r['max_dd']}% CVaR={r['cvar5_bps']}bps "
                      f"PnL=${r['pnl']:,.0f} early={n_early:,}{marker}")
                if r["sr"] > best_sr:
                    best_sr = r["sr"]
                    best_result = r
                    best_thresh = thresh

        if best_sr > -999:
            results[f"{cfg_name} t={best_thresh}"] = best_result

            # Year-by-year
            print(f"  Best: t={best_thresh} SR={best_sr:.3f}")
            daily = best_result["daily"]
            for year in sorted(daily.index.year.unique()):
                y = daily[daily.index.year == year]
                if len(y) > 20:
                    sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
                    cum = CAPITAL + y.cumsum()
                    dd_y = float(((cum.cummax() - cum) / cum.cummax()).max() * 100)
                    print(f"    {year}: SR={sr_y:.3f}  DD={dd_y:.1f}%  PnL=${y.sum():,.0f}")

    # ================================================================
    # ORACLE
    # ================================================================
    print(f"\n--- ORACLE (perfect foresight) ---")
    oracle_labels = (remaining < -EXIT_COST).astype(np.int8)
    oracle_preds = oracle_labels.astype(np.float32)  # perfect prediction
    oracle_selected, oracle_early = select_exits(examples, oracle_preds, 0.5)
    oracle_merged = build_portfolio(oracle_selected)
    r_oracle = run_portfolio(oracle_merged, "Oracle")
    if r_oracle:
        results["ORACLE (perfect)"] = r_oracle
        print(f"  SR={r_oracle['sr']:.3f}  DD={r_oracle['max_dd']}%  PnL=${r_oracle['pnl']:,.0f}  "
              f"early={oracle_early:,}")

    # ================================================================
    # FULL COMPARISON
    # ================================================================
    print(f"\n{'='*100}")
    print("  FULL COMPARISON (simulate_portfolio + Bybit VIP0)")
    print(f"{'='*100}")
    print_comparison(results)

    # Delta from baseline
    if "BASELINE (fixed hold)" in results:
        base_sr = results["BASELINE (fixed hold)"]["sr"]
        oracle_sr = results.get("ORACLE (perfect)", {}).get("sr", base_sr)
        print(f"\n  Headroom: Oracle SR {oracle_sr:.3f} - Baseline SR {base_sr:.3f} = {oracle_sr - base_sr:.3f}")
        for name, r in results.items():
            if name not in ("BASELINE (fixed hold)", "ORACLE (perfect)"):
                delta = r["sr"] - base_sr
                captured = (r["sr"] - base_sr) / max(oracle_sr - base_sr, 0.001) * 100
                print(f"  {name:<40} delta={delta:+.3f} SR  ({captured:.0f}% of headroom)")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
