#!/usr/bin/env python
"""Close Model V2 — Proper holdout evaluation (2022+ only).

LogReg warm-up: 6 months. Close model warm-up: 10 months of signals.
Conservative holdout: evaluate only from 2022-01 onward.
Uses simulate_portfolio + Bybit VIP0 costs.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
EXIT_COST = 0.00045
HOLDOUT_START = "2022-01-01"
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
PORT = PortfolioConfig(
    starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.50, base_sizing_pct=0.04, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
VOL_INV = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                            low_pct=0.20, high_pct=0.80,
                            low_mult=1.25, mid_mult=1.00, high_mult=0.50)

STATE_8 = ["current_unrealized_return", "max_adverse_return_since_entry",
           "distance_from_entry_mfe", "realized_vol_since_entry",
           "bars_since_entry", "time_to_expiry_minutes",
           "mfe_minus_mae", "entry_to_now_trend_slope"]
MARKET_13 = ["ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
             "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
             "dist_swing_high", "dist_swing_low"]


def walk_forward_xgb(examples, target, features, test_start="2021-01", lookback=10):
    """Walk-forward XGBoost, returns predictions array."""
    feat_cols = [c for c in features if c in examples.columns]
    feat_matrix = examples[feat_cols].to_numpy(dtype=np.float32)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)
    target_arr = target.to_numpy(dtype=np.float32)
    decision_ts = pd.to_datetime(examples["decision_timestamp"])
    month_ords = decision_ts.dt.to_period("M").astype("int64").to_numpy()
    label_end_ns = pd.to_datetime(examples["label_end_timestamp"]).to_numpy(
        dtype="datetime64[ns]").astype(np.int64)
    dec_ns = decision_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    test_start_ord = pd.Period(test_start, freq="M").ordinal
    unique_months = sorted(set(month_ords))
    preds = np.full(len(examples), 0.5, dtype=np.float32)

    n_done = 0
    for m_ord in unique_months:
        if m_ord < test_start_ord:
            continue
        ms_ns = pd.Period(ordinal=m_ord, freq="M").start_time.value
        train_start_ord = max(unique_months[0], m_ord - lookback)
        train_mask = ((month_ords >= train_start_ord) & (month_ords < m_ord) &
                      (dec_ns < ms_ns) & (label_end_ns < ms_ns))
        test_mask = month_ords == m_ord
        tr_idx = np.flatnonzero(train_mask)
        te_idx = np.flatnonzero(test_mask)
        if len(tr_idx) < 200 or len(te_idx) == 0:
            continue
        X_tr, y_tr = feat_matrix[tr_idx], target_arr[tr_idx]
        X_te = feat_matrix[te_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        if len(np.unique(y_tr)) < 2:
            continue
        pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=max(neg / max(pos, 1), 1e-6),
            tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_tr_s, y_tr)
        preds[te_idx] = model.predict_proba(X_te_s)[:, 1]
        n_done += 1
    print(f"  Walk-forward: {n_done} months predicted")
    return preds


def select_exits(examples, preds, threshold):
    """Select exit bar per trade based on model predictions."""
    ex = examples.copy()
    ex["_pred"] = preds
    ex["_idx"] = np.arange(len(ex))
    triggered = ex[ex["_pred"] >= threshold]
    first_trig = triggered.groupby("trade_id")["_idx"].first()
    last_bar = ex.groupby("trade_id")["_idx"].last()
    exit_idx = last_bar.copy()
    exit_idx.loc[first_trig.index] = first_trig
    sel = ex.iloc[exit_idx.values].copy()

    max_step = ex.groupby("trade_id")["bars_since_entry"].max()
    is_early = sel.set_index("trade_id")["bars_since_entry"] < max_step
    early_mask = is_early.reindex(sel["trade_id"]).values
    n_early = int(early_mask.sum())

    # Convert to unsigned fwd_return for simulate_portfolio
    fwd = sel["exit_return"].values / sel["direction"].values
    fwd[early_mask] -= EXIT_COST  # extra cost for early exit
    sel["fwd_return"] = fwd
    sel["hold_bars"] = sel["bars_since_entry"].fillna(4).astype(int)
    sel["strategy"] = "logreg_15m"
    sel["probability"] = sel["entry_probability"]
    sel["confidence"] = sel["entry_confidence"]
    sel["signal_strength"] = sel["entry_signal_strength"].abs()
    if "regime_exposure_mult" not in sel.columns:
        sel["regime_exposure_mult"] = 1.0
    sel["regime_exposure_mult"] = sel["regime_exposure_mult"].fillna(1.0)
    return sel, n_early


def build_portfolio(logreg_15m_sigs):
    """Merge with other strategies, apply conf filter + vol overlay."""
    filt = logreg_15m_sigs[logreg_15m_sigs["confidence"] >= 0.08].reset_index(drop=True)
    parts = [filt]
    for name in ["logreg_4h", "cs_momentum"]:
        p = Path(f"results/signals_cache/{name}.parquet")
        if p.exists():
            o = pd.read_parquet(p)
            for c in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
                if c not in o.columns:
                    o[c] = 4 if c == "hold_bars" else 1.0
            o["probability"] = o["probability"].astype(float)
            parts.append(o)
    merged = pd.concat(parts, ignore_index=True)
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
        merged = apply_vol_overlay(merged, btc_daily, VOL_INV)
    except Exception:
        pass
    return merged


def run_and_evaluate(signals, label):
    """Run simulate_portfolio + Bybit costs, return stats for full and holdout periods."""
    merged = build_portfolio(signals)
    sim = simulate_portfolio(merged, PORT)
    if sim is None:
        print(f"  {label}: simulate_portfolio returned None")
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print(f"  {label}: no daily PnL")
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Bybit cost adjustment
    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    daily_cost = total_extra / max(len(dpnl), 1)
    dpnl_adj = dpnl - daily_cost

    def stats_for_period(d, period_label):
        if len(d) < 20:
            return None
        dr = d / CAPITAL
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        down = dr[dr < 0]
        sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
        eq = (1 + dr).cumprod()
        dd_s = (eq.cummax() - eq) / eq.cummax()
        max_dd = float(dd_s.max() * 100)
        cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
        monthly = d.resample("MS").sum()
        # Bootstrap CI
        rng = np.random.RandomState(42)
        bsr = np.empty(3000)
        drv = dr.values
        for b in range(3000):
            s = rng.choice(drv, size=len(drv), replace=True)
            bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
        ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))
        return {
            "sr": sr, "ci": ci, "sortino": sortino, "max_dd": max_dd,
            "cvar5": cvar5, "worst_day": float(d.min()),
            "worst_month": float(monthly.min()),
            "pos_months": f"{int((monthly > 0).sum())}/{len(monthly)}",
            "pnl": float(d.sum()), "daily": d,
        }

    full = stats_for_period(dpnl_adj, "full")
    holdout = stats_for_period(dpnl_adj[dpnl_adj.index >= HOLDOUT_START], "holdout")
    return {"full": full, "holdout": holdout, "n_trades": n_trades}


def print_stats(label, s):
    if s is None:
        print(f"  {label}: N/A")
        return
    print(f"  {label}: SR={s['sr']:.3f} [{s['ci'][0]:.2f},{s['ci'][1]:.2f}]  "
          f"Sortino={s['sortino']:.3f}  DD={s['max_dd']:.1f}%  CVaR5={s['cvar5']:.1f}bps  "
          f"worst_day=${s['worst_day']:,.0f}  worst_month=${s['worst_month']:,.0f}  "
          f"pos_months={s['pos_months']}  PnL=${s['pnl']:,.0f}")


def main():
    T0 = time.time()
    print("=" * 100)
    print(f"  CLOSE V2 — HOLDOUT EVALUATION (testing only {HOLDOUT_START}+)")
    print("=" * 100)

    ex = pd.read_parquet("results/close_cache/examples_logreg_15m.parquet")
    remaining = ex["expiry_return"] - ex["exit_return"]
    print(f"Examples: {len(ex):,} rows, {ex['trade_id'].nunique():,} trades")

    # ================================================================
    # BASELINE: Fixed hold
    # ================================================================
    print("\n[BASELINE] Fixed hold (T4 expiry)")
    base = ex.groupby("trade_id").last().copy()
    base["fwd_return"] = base["exit_return"] / base["direction"]
    base["strategy"] = "logreg_15m"
    base["probability"] = base["entry_probability"]
    base["confidence"] = base["entry_confidence"]
    base["signal_strength"] = base["entry_signal_strength"].abs()
    base["hold_bars"] = 4
    if "regime_exposure_mult" not in base.columns:
        base["regime_exposure_mult"] = 1.0
    base["regime_exposure_mult"] = base["regime_exposure_mult"].fillna(1.0)

    r_base = run_and_evaluate(base, "Baseline")
    if r_base:
        print_stats("  Full period", r_base["full"])
        print_stats("  HOLDOUT 2022+", r_base["holdout"])

    # ================================================================
    # CLOSE MODEL: XGB + state_market_21 + cost_4.5bps at threshold 0.60
    # ================================================================
    print("\n[CLOSE MODEL] XGB + state+market 21 features + cost_4.5bps labels")
    labels = (remaining < -EXIT_COST).astype(np.int8)
    features = STATE_8 + MARKET_13
    t0 = time.time()
    preds = walk_forward_xgb(ex, labels, features)
    print(f"  Training time: {time.time()-t0:.0f}s")

    for threshold in [0.55, 0.60, 0.65, 0.70]:
        sel, n_early = select_exits(ex, preds, threshold)
        r_model = run_and_evaluate(sel, f"t={threshold}")
        if r_model:
            print(f"\n  Threshold {threshold:.2f} ({n_early:,} early exits):")
            print_stats("    Full period", r_model["full"])
            print_stats("    HOLDOUT 2022+", r_model["holdout"])

    # ================================================================
    # YEAR-BY-YEAR for best threshold (0.60)
    # ================================================================
    sel_best, n_early_best = select_exits(ex, preds, 0.60)
    r_best = run_and_evaluate(sel_best, "best")
    if r_best and r_best["holdout"]:
        print(f"\n{'='*100}")
        print(f"  YEAR-BY-YEAR COMPARISON (holdout 2022+, Bybit costs)")
        print(f"{'='*100}")
        base_daily = r_base["holdout"]["daily"]
        model_daily = r_best["holdout"]["daily"]
        for year in sorted(set(base_daily.index.year) | set(model_daily.index.year)):
            yb = base_daily[base_daily.index.year == year]
            ym = model_daily[model_daily.index.year == year]
            if len(yb) > 20 and len(ym) > 20:
                sr_b = float(yb.mean() / yb.std() * np.sqrt(365.25)) if yb.std() > 0 else 0
                sr_m = float(ym.mean() / ym.std() * np.sqrt(365.25)) if ym.std() > 0 else 0
                cum_b = CAPITAL + yb.cumsum()
                cum_m = CAPITAL + ym.cumsum()
                dd_b = float(((cum_b.cummax() - cum_b) / cum_b.cummax()).max() * 100)
                dd_m = float(((cum_m.cummax() - cum_m) / cum_m.cummax()).max() * 100)
                print(f"  {year}:  Base SR={sr_b:.3f} DD={dd_b:.1f}%  |  "
                      f"Model SR={sr_m:.3f} DD={dd_m:.1f}%  |  "
                      f"delta={sr_m - sr_b:+.3f}")

    # ================================================================
    # DELTA SUMMARY
    # ================================================================
    if r_base and r_best and r_base["holdout"] and r_best["holdout"]:
        b = r_base["holdout"]
        m = r_best["holdout"]
        print(f"\n{'='*100}")
        print(f"  HOLDOUT SUMMARY (2022-01 to present, Bybit VIP0)")
        print(f"{'='*100}")
        print(f"  {'Metric':<25} {'Baseline':>12} {'Close Model':>12} {'Delta':>12}")
        print(f"  {'-'*61}")
        print(f"  {'SR':<25} {b['sr']:>12.3f} {m['sr']:>12.3f} {m['sr']-b['sr']:>+12.3f}")
        print(f"  {'Sortino':<25} {b['sortino']:>12.3f} {m['sortino']:>12.3f} {m['sortino']-b['sortino']:>+12.3f}")
        print(f"  {'Max DD':<25} {b['max_dd']:>11.1f}% {m['max_dd']:>11.1f}% {m['max_dd']-b['max_dd']:>+11.1f}%")
        print(f"  {'CVaR 5% (bps)':<25} {b['cvar5']:>12.1f} {m['cvar5']:>12.1f} {m['cvar5']-b['cvar5']:>+12.1f}")
        print(f"  {'Worst day':<25} ${b['worst_day']:>10,.0f} ${m['worst_day']:>10,.0f} ${m['worst_day']-b['worst_day']:>+10,.0f}")
        print(f"  {'Worst month':<25} ${b['worst_month']:>10,.0f} ${m['worst_month']:>10,.0f} ${m['worst_month']-b['worst_month']:>+10,.0f}")
        print(f"  {'Positive months':<25} {b['pos_months']:>12} {m['pos_months']:>12}")
        print(f"  {'PnL':<25} ${b['pnl']:>10,.0f} ${m['pnl']:>10,.0f} ${m['pnl']-b['pnl']:>+10,.0f}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
