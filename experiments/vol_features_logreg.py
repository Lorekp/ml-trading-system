#!/usr/bin/env python
"""Test adding vol-pipeline features to BTC logistic regression.

Compares walk-forward logreg accuracy:
  A) Standard 13 OHLCV features (baseline)
  B) Standard 13 + 15 vol-pipeline features (expanded)
  C) Vol features only (15 features, no OHLCV)
  D) Top 5 vol features only

Uses same walk-forward methodology as signal_engine: monthly retraining,
6-month expanding window, 4-bar embargo.
"""
import sys, os, warnings, time
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Load and merge data ──────────────────────────────────────

def load_data():
    """Load BTC 15m OHLCV, compute standard features, merge vol features."""
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.config import FeatureConfig

    # Load BTC 15m OHLCV
    raw = load_binance_1m("BTCUSDT", [os.path.join("data", "binance-1m")])
    ohlcv = resample_ohlcv(raw, "15min")

    # Standard 13 features
    feats = compute_features(ohlcv, FeatureConfig())
    standard_cols = [c for c in feats.columns if c not in ["open", "high", "low", "close", "volume"]]

    # Forward return (target)
    close = ohlcv["close"]
    feats["fwd_ret"] = close.pct_change(4).shift(-4)  # 1h forward
    feats["direction"] = (feats["fwd_ret"] > 0).astype(int)

    # Vol-pipeline features
    vol = pd.read_parquet("vol_arb/output/BTC_vol_arb_svi.parquet")
    if vol.index.tz is not None:
        vol.index = vol.index.tz_localize(None)
    if feats.index.tz is not None:
        feats.index = feats.index.tz_localize(None)

    vol_feature_cols = [
        'up_jump_fraction', 'down_jump_fraction',
        'svi_atm_1d', 'iv_bull', 'svi_iv_matched',
        'svi_25d_skew_1w', 'trailing_1h_volume', 'iv', 'iv_bear',
        'delta', 'delta_bear',
        'svi_atm_1w', 'svi_atm_2w', 'svi_atm_1m', 'slip_s1_v2',
    ]

    top5_cols = [
        'up_jump_fraction', 'down_jump_fraction',
        'svi_atm_1d', 'iv_bull', 'svi_iv_matched',
    ]

    # Merge vol features onto OHLCV features (backward asof)
    merged = feats.join(vol[vol_feature_cols], how="left")

    # Also add derived features
    merged["jump_imbalance"] = merged.get("up_jump_fraction", 0) - merged.get("down_jump_fraction", 0)
    merged["iv_skew"] = merged.get("iv_bull", 0) - merged.get("iv_bear", 0)

    vol_feature_cols_ext = vol_feature_cols + ["jump_imbalance", "iv_skew"]
    top5_cols_ext = top5_cols + ["jump_imbalance"]

    return merged, standard_cols, vol_feature_cols_ext, top5_cols_ext


# ── Walk-forward evaluation ──────────────────────────────────

def walk_forward_test(df, feature_cols, label="test", fwd_bars=4, min_train_months=6):
    """Monthly walk-forward logreg. Returns per-month accuracy and signal stats."""
    df = df.dropna(subset=feature_cols + ["direction", "fwd_ret"])

    months = pd.period_range(df.index.min().to_period("M") + min_train_months,
                              df.index.max().to_period("M"), freq="M")

    all_preds = []
    monthly_stats = []

    for month in months:
        ms = month.start_time
        me = (month + 1).start_time

        # Train: all data before this month (with embargo)
        embargo = pd.Timedelta(hours=fwd_bars * 0.25 + 1)  # 4 bars + 1h buffer
        train = df[df.index < ms - embargo]
        test = df[(df.index >= ms) & (df.index < me)]

        if len(train) < 500 or len(test) < 10:
            continue

        X_train = train[feature_cols].values
        y_train = train["direction"].values
        X_test = test[feature_cols].values
        y_test = test["direction"].values
        fwd_rets = test["fwd_ret"].values

        # Handle NaN
        mask_train = np.all(np.isfinite(X_train), axis=1)
        mask_test = np.all(np.isfinite(X_test), axis=1)
        if mask_train.sum() < 200 or mask_test.sum() < 5:
            continue

        X_tr, y_tr = X_train[mask_train], y_train[mask_train]
        X_te, y_te = X_test[mask_test], y_test[mask_test]
        fwd_te = fwd_rets[mask_test]

        # Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Train logreg
        model = LogisticRegression(
            penalty="l1", C=1.0, solver="saga", max_iter=2000, random_state=42,
        )
        model.fit(X_tr, y_tr)

        # Predict
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs > 0.5).astype(int)
        accuracy = (preds == y_te).mean()

        # Signal-based PnL (simplified)
        directions = np.where(probs > 0.5, 1, -1)
        confidences = np.abs(probs - 0.5)
        # Only trade when confidence > 0.05
        trade_mask = confidences > 0.05
        if trade_mask.sum() > 0:
            signal_rets = directions[trade_mask] * fwd_te[trade_mask]
            avg_ret = signal_rets.mean()
            win_rate = (signal_rets > 0).mean()
            n_signals = trade_mask.sum()
        else:
            avg_ret = 0
            win_rate = 0.5
            n_signals = 0

        monthly_stats.append({
            "month": str(month),
            "accuracy": accuracy,
            "n_test": len(y_te),
            "n_signals": n_signals,
            "avg_ret": avg_ret,
            "win_rate": win_rate,
            "avg_confidence": confidences.mean(),
        })

        for j in range(len(y_te)):
            all_preds.append({
                "timestamp": test.index[mask_test][j],
                "prob": probs[j],
                "direction": directions[j] if j < len(directions) else 0,
                "fwd_ret": fwd_te[j],
                "actual": y_te[j],
            })

    if not monthly_stats:
        print(f"  {label}: no valid months")
        return None

    stats_df = pd.DataFrame(monthly_stats)
    preds_df = pd.DataFrame(all_preds)

    # Summary
    avg_acc = stats_df["accuracy"].mean()
    avg_wr = stats_df["win_rate"].mean()
    avg_ret = stats_df["avg_ret"].mean()
    avg_conf = stats_df["avg_confidence"].mean()
    total_signals = stats_df["n_signals"].sum()

    # Compute overall Sharpe from signal returns
    traded = preds_df[preds_df["prob"].apply(lambda p: abs(p - 0.5) > 0.05)]
    if len(traded) > 100:
        sig_rets = traded["direction"] * traded["fwd_ret"]
        daily = sig_rets.groupby(traded["timestamp"].dt.date).sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(365.25) if daily.std() > 0 else 0
    else:
        sharpe = 0

    # Per-year accuracy
    stats_df["year"] = pd.to_datetime(stats_df["month"]).dt.year
    yearly = stats_df.groupby("year").agg(
        acc=("accuracy", "mean"), wr=("win_rate", "mean"),
        signals=("n_signals", "sum"),
    )

    return {
        "label": label,
        "avg_accuracy": avg_acc,
        "avg_win_rate": avg_wr,
        "avg_return_per_signal": avg_ret,
        "signal_sharpe": sharpe,
        "total_signals": total_signals,
        "avg_confidence": avg_conf,
        "n_months": len(stats_df),
        "yearly": yearly,
        "monthly": stats_df,
    }


def main():
    T0 = time.time()

    print("Loading data...")
    df, standard_cols, vol_cols, top5_cols = load_data()
    print(f"  Total bars: {len(df):,}")
    print(f"  Standard features ({len(standard_cols)}): {standard_cols[:5]}...")
    print(f"  Vol features ({len(vol_cols)}): {vol_cols[:5]}...")
    print(f"  Top 5 vol ({len(top5_cols)}): {top5_cols}")
    print()

    configs = {
        "A: Standard 13 OHLCV": standard_cols,
        "B: Standard + All Vol (30)": standard_cols + vol_cols,
        "C: Vol features only (17)": vol_cols,
        "D: Top 5 vol only (6)": top5_cols,
        "E: Standard + Top 5 (19)": standard_cols + top5_cols,
    }

    results = {}
    for name, cols in configs.items():
        # Filter to columns that exist
        cols = [c for c in cols if c in df.columns]
        print(f"Running {name} ({len(cols)} features)...")
        r = walk_forward_test(df, cols, label=name)
        if r:
            results[name] = r
            print(f"  Acc={r['avg_accuracy']:.3f}  WinRate={r['avg_win_rate']:.3f}  "
                  f"Sharpe={r['signal_sharpe']:.2f}  Signals={r['total_signals']:,}")
        print()

    # ── Comparison table ──
    print("=" * 110)
    print("COMPARISON: Feature Set Impact on BTC Direction Prediction")
    print("=" * 110)
    print(f"{'Config':35s} {'Feats':>5s} {'Accuracy':>9s} {'WinRate':>8s} {'Sharpe':>8s} {'Signals':>8s} {'Conf':>6s}")
    print("-" * 110)

    baseline_acc = results.get("A: Standard 13 OHLCV", {}).get("avg_accuracy", 0.5)
    for name, r in results.items():
        delta = r["avg_accuracy"] - baseline_acc
        delta_str = f"({delta:+.3f})" if name != "A: Standard 13 OHLCV" else ""
        n_feats = len([c for c in configs[name] if c in df.columns])
        print(f"  {name:33s} {n_feats:>5d} {r['avg_accuracy']:>9.4f} {r['avg_win_rate']:>8.4f} "
              f"{r['signal_sharpe']:>8.2f} {r['total_signals']:>8,} {r['avg_confidence']:>6.3f} {delta_str}")

    # ── Per-year breakdown for best config ──
    print()
    print("Per-Year Accuracy:")
    print(f"{'Year':>6s}", end="")
    for name in results:
        short = name.split(":")[0]
        print(f" {short:>12s}", end="")
    print()

    all_years = set()
    for r in results.values():
        all_years.update(r["yearly"].index)

    for yr in sorted(all_years):
        print(f"{yr:>6d}", end="")
        for name, r in results.items():
            if yr in r["yearly"].index:
                print(f" {r['yearly'].loc[yr, 'acc']:>12.4f}", end="")
            else:
                print(f" {'--':>12s}", end="")
        print()

    print(f"\nRuntime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
