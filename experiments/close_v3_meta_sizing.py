#!/usr/bin/env python
"""Close Model V3 — AFML Meta-Labeling Bet Sizing.

Instead of binary close/hold, uses regression to predict remaining return,
then AFML bet sizing to continuously adjust position size. Proportional
exit costs (30% reduction costs 1.35 bps, not 4.5 bps).

Per-step models: separate regressor for step 1, 2, 3.
Walk-forward monthly with return-weighted samples.
Evaluated on 2022+ holdout only.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
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
FEATURES = STATE_8 + MARKET_13


def afml_bet_size(prob):
    """AFML Ch 10: convert probability to bet size [0, 1]."""
    prob = np.clip(prob, 0.001, 0.999)
    z = (prob - 0.5) / np.sqrt(prob * (1 - prob))
    return np.clip(2 * norm.cdf(z) - 1, 0, 1)


def walk_forward_regression_per_step(examples, features, test_start="2021-01",
                                      lookback=10, return_weighted=False):
    """Walk-forward regression per decision step. Returns dict of predictions."""
    feat_cols = [c for c in features if c in examples.columns]
    feat_matrix = examples[feat_cols].to_numpy(dtype=np.float32)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

    decision_ts = pd.to_datetime(examples["decision_timestamp"])
    month_ords = decision_ts.dt.to_period("M").astype("int64").to_numpy()
    label_end_ns = pd.to_datetime(examples["label_end_timestamp"]).to_numpy(
        dtype="datetime64[ns]").astype(np.int64)
    dec_ns = decision_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    steps = examples["bars_since_entry"].to_numpy()

    remaining = (examples["expiry_return"] - examples["exit_return"]).to_numpy(dtype=np.float32)

    test_start_ord = pd.Period(test_start, freq="M").ordinal
    unique_months = sorted(set(month_ords))
    preds = np.zeros(len(examples), dtype=np.float32)
    train_stds = {}  # store training std per step for calibration

    for step in [1, 2, 3]:
        step_mask = steps == step
        step_idx = np.flatnonzero(step_mask)
        if len(step_idx) == 0:
            continue

        n_done = 0
        for m_ord in unique_months:
            if m_ord < test_start_ord:
                continue
            ms_ns = pd.Period(ordinal=m_ord, freq="M").start_time.value
            train_start_ord = max(unique_months[0], m_ord - lookback)

            train_mask = (step_mask &
                          (month_ords >= train_start_ord) & (month_ords < m_ord) &
                          (dec_ns < ms_ns) & (label_end_ns < ms_ns))
            test_mask = step_mask & (month_ords == m_ord)

            tr_idx = np.flatnonzero(train_mask)
            te_idx = np.flatnonzero(test_mask)
            if len(tr_idx) < 100 or len(te_idx) == 0:
                continue

            X_tr = feat_matrix[tr_idx]
            y_tr = remaining[tr_idx]
            X_te = feat_matrix[te_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Sample weights: AFML return-weighted (big moves matter more)
            if return_weighted:
                weights = np.abs(y_tr) + 1e-6
                weights = weights / weights.mean()  # normalize
            else:
                weights = None

            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X_tr_s, y_tr, sample_weight=weights)
            preds[te_idx] = model.predict(X_te_s)

            # Store training std for calibration
            train_stds[(step, m_ord)] = float(np.std(y_tr))
            n_done += 1

        print(f"    Step {step}: {n_done} months trained")

    return preds, train_stds


def compute_meta_sizing(examples, preds, scale_factor=1.0, discretize_step=0.1,
                         min_keep=0.0):
    """Convert regression predictions to AFML sizing per trade.

    Returns DataFrame with one row per trade and blended effective_return.
    """
    ex = examples.copy()
    ex["pred_remaining"] = preds

    # Calibrate: convert prediction to P(remaining > 0)
    # Use sigmoid: P = sigmoid(pred / scale)
    # scale controls how aggressively the model sizes
    scale = ex.groupby("bars_since_entry")["pred_remaining"].transform("std") * scale_factor
    scale = scale.clip(lower=1e-6)
    prob_positive = 1 / (1 + np.exp(-ex["pred_remaining"] / scale))

    # AFML bet sizing
    sizing_keep = afml_bet_size(prob_positive.values)

    # Discretize (AFML Ch 10: prevents overtrading)
    if discretize_step > 0:
        sizing_keep = np.round(sizing_keep / discretize_step) * discretize_step

    # Floor
    sizing_keep = np.clip(sizing_keep, min_keep, 1.0)

    ex["sizing_keep"] = sizing_keep

    # Compute blended return per trade (path-level)
    trades = []
    for trade_id, group in ex.groupby("trade_id"):
        group = group.sort_values("bars_since_entry")
        rows = list(group.itertuples())

        current_size = 1.0
        effective_return = 0.0
        total_cost = 0.0

        for i, row in enumerate(rows):
            new_size = row.sizing_keep
            # Only reduce, never increase mid-trade
            new_size = min(new_size, current_size)

            delta = current_size - new_size
            if delta > 0.001:  # meaningful size change
                # Close delta fraction at current return, paying proportional exit cost
                effective_return += delta * row.exit_return
                total_cost += delta * EXIT_COST
                current_size = new_size

        # Remaining held to expiry
        expiry_ret = rows[-1].exit_return  # last step = expiry
        effective_return += current_size * expiry_ret
        effective_return -= total_cost

        first = rows[0]
        trades.append({
            "trade_id": trade_id,
            "timestamp": first.timestamp,
            "asset": first.asset,
            "direction": first.direction,
            "fwd_return": effective_return / first.direction,  # unsigned for portfolio sim
            "probability": first.entry_probability,
            "confidence": first.entry_confidence,
            "signal_strength": abs(first.entry_signal_strength),
            "strategy": "logreg_15m",
            "hold_bars": 4,
            "regime_exposure_mult": getattr(first, "regime_exposure_mult", 1.0),
            "avg_sizing_keep": np.mean([r.sizing_keep for r in rows]),
            "total_exit_cost_bps": total_cost * 10000,
        })

    return pd.DataFrame(trades)


def build_portfolio(logreg_15m):
    """Merge with other strategies, apply filters + vol overlay."""
    filt = logreg_15m[logreg_15m["confidence"] >= 0.08].reset_index(drop=True)
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


def evaluate(signals, label):
    """simulate_portfolio + Bybit costs, return holdout stats."""
    merged = build_portfolio(signals)
    sim = simulate_portfolio(merged, PORT)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Bybit cost
    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    daily_cost = total_extra / max(len(dpnl), 1)
    dpnl_adj = dpnl - daily_cost

    # Holdout only
    dpnl_h = dpnl_adj[dpnl_adj.index >= HOLDOUT_START]
    if len(dpnl_h) < 20:
        return None

    dr = dpnl_h / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd_s = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(dd_s.max() * 100)
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    monthly = dpnl_h.resample("MS").sum()
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    drv = dr.values
    for b in range(3000):
        s = rng.choice(drv, size=len(drv), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    return {
        "sr": sr, "ci": ci, "sortino": sortino, "max_dd": max_dd,
        "cvar5": cvar5, "worst_day": float(dpnl_h.min()),
        "worst_month": float(monthly.min()),
        "pos_months": f"{int((monthly > 0).sum())}/{len(monthly)}",
        "pnl": float(dpnl_h.sum()), "n_trades": n_trades,
        "daily": dpnl_h,
    }


def print_result(label, r):
    if r is None:
        print(f"  {label}: N/A")
        return
    print(f"  {label}: SR={r['sr']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}]  "
          f"Sortino={r['sortino']:.3f}  DD={r['max_dd']:.1f}%  CVaR5={r['cvar5']:.1f}bps  "
          f"worst_day=${r['worst_day']:,.0f}  PnL=${r['pnl']:,.0f}")


def main():
    T0 = time.time()
    print("=" * 100)
    print(f"  CLOSE V3 — AFML META-LABELING BET SIZING (holdout {HOLDOUT_START}+)")
    print("=" * 100)

    ex = pd.read_parquet("results/close_cache/examples_logreg_15m.parquet")
    print(f"Examples: {len(ex):,} rows, {ex['trade_id'].nunique():,} trades")

    # ================================================================
    # BASELINE: Fixed hold
    # ================================================================
    print("\n[BASELINE] Fixed hold")
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
    r_base = evaluate(base, "Baseline")
    print_result("Baseline (2022+)", r_base)

    # ================================================================
    # V3: Meta-sizing with per-step regression
    # ================================================================
    for rw_label, return_weighted in [("equal_weight", False), ("return_weighted", True)]:
        print(f"\n[V3] Per-step regression ({rw_label})")
        t0 = time.time()
        preds, train_stds = walk_forward_regression_per_step(
            ex, FEATURES, return_weighted=return_weighted)
        print(f"  Training time: {time.time()-t0:.0f}s")

        # Sweep scale factors and discretization
        for scale_factor in [0.5, 1.0, 2.0]:
            for disc_step in [0.1, 0.2]:
                for min_keep in [0.0, 0.2]:
                    print(f"\n  scale={scale_factor} disc={disc_step} min_keep={min_keep} ({rw_label})")
                    meta_sigs = compute_meta_sizing(
                        ex, preds, scale_factor=scale_factor,
                        discretize_step=disc_step, min_keep=min_keep)

                    avg_keep = meta_sigs["avg_sizing_keep"].mean()
                    avg_cost = meta_sigs["total_exit_cost_bps"].mean()
                    print(f"    avg_sizing_keep={avg_keep:.2f}  avg_exit_cost={avg_cost:.1f}bps")

                    r = evaluate(meta_sigs, "Meta-sizing")
                    if r:
                        delta = r["sr"] - r_base["sr"] if r_base else 0
                        print_result(f"    2022+ holdout", r)
                        print(f"    delta vs baseline: {delta:+.3f} SR")

                        # Year-by-year
                        for year in sorted(r["daily"].index.year.unique()):
                            y = r["daily"][r["daily"].index.year == year]
                            if len(y) > 20:
                                sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
                                print(f"      {year}: SR={sr_y:.3f}  PnL=${y.sum():,.0f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print("  SUMMARY")
    print(f"{'='*100}")
    if r_base:
        print_result("Baseline", r_base)
    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
