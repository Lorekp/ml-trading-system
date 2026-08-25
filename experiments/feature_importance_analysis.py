"""Feature importance analysis (MDI, MDA, SFI) on fixed (no-leakage) pipeline.

Trains on 2022-2023, tests on 2024. Reports which features have genuine
predictive power now that the regime feature leakage is fixed.
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, log_loss

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

FWD = 4


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("Loading BTC...")
    ohlcv_1m = load_binance_1m("BTCUSDT", [Path("data/binance-1m")])
    ohlcv = resample_ohlcv(ohlcv_1m, "15min")
    n = len(ohlcv)
    ts = ohlcv.index

    # Fixed features (shift regime by 1)
    base = compute_features(ohlcv, cfg.features)
    regime = []
    for rtf in cfg.close_model.regime_timeframes:
        ro = resample_ohlcv(ohlcv_1m, rtf)
        rf = compute_features(ro, cfg.features)
        al = rf.shift(1).reindex(ohlcv.index, method="ffill")
        al = al.add_prefix(f"signal_regime_{rtf}_")
        regime.append(al)
    feat_df = pd.concat([base] + regime, axis=1)
    feat_cols = list(feat_df.columns)
    feat_arr = feat_df.values.astype(np.float32)

    close = ohlcv["close"].values.astype(np.float64)
    fwd_ret = np.full(n, np.nan)
    v = np.arange(n - FWD)
    m = close[v] > 0
    fwd_ret[v[m]] = (close[v[m] + FWD] - close[v[m]]) / close[v[m]]
    labels = np.full(n, -1, dtype=np.int8)
    fi = np.isfinite(fwd_ret)
    labels[fi] = (fwd_ret[fi] > 0).astype(np.int8)

    mk = (ts.year * 12 + ts.month).values

    # Train: 2022-2023, Test: 2024
    valid_feat = np.isfinite(feat_arr).all(axis=1)
    train_mask = (mk >= 2022 * 12 + 1) & (mk <= 2023 * 12 + 12) & (labels >= 0) & valid_feat
    test_mask = (mk >= 2024 * 12 + 1) & (mk <= 2024 * 12 + 12) & (labels >= 0) & valid_feat
    train_idx = np.where(train_mask)[0][:-5]  # embargo
    test_idx = np.where(test_mask)[0]

    X_train = feat_arr[train_idx]
    y_train = labels[train_idx]
    X_test = feat_arr[test_idx]
    y_test = labels[test_idx]
    sw = compute_sample_weights(np.arange(len(train_idx), dtype=np.int64), fwd_bars=FWD)

    print(f"Train: {len(train_idx)} bars (2022-2023), Test: {len(test_idx)} bars (2024)")
    print(f"Train pos rate: {y_train.mean():.4f}, Test pos rate: {y_test.mean():.4f}")
    print(f"Features: {len(feat_cols)}")
    print(f"Data loaded in {time.time() - t0:.0f}s")

    # === Baseline model ===
    print(f"\n{'='*70}")
    print("BASELINE MODEL")
    print(f"{'='*70}")
    mdl = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", random_state=42,
        n_jobs=max(1, os.cpu_count() or 4), verbosity=0,
        eval_metric="logloss", objective="binary:logistic",
    )
    mdl.fit(X_train, y_train, sample_weight=sw, verbose=False)
    probs = mdl.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, (probs > 0.5).astype(int))
    ll = log_loss(y_test, probs)
    dirs = np.where(probs > 0.5, 1.0, -1.0)
    pnls = dirs * fwd_ret[test_idx]
    wr = (pnls > 0).mean()
    bps = pnls.mean() * 10000
    print(f"  Accuracy: {acc:.4f}  LogLoss: {ll:.4f}  Win: {wr * 100:.1f}%  bps: {bps:.2f}")

    # === 1. MDI ===
    print(f"\n{'='*70}")
    print("1. MDI (Mean Decrease Impurity) — in-sample, from XGBoost splits")
    print(f"{'='*70}")
    importances = mdl.feature_importances_
    mdi = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
    print("  Top 15:")
    for feat, imp in mdi.head(15).items():
        print(f"    {feat:<42s} {imp:.4f}")
    print("  Bottom 5:")
    for feat, imp in mdi.tail(5).items():
        print(f"    {feat:<42s} {imp:.4f}")

    # === 2. MDA (Permutation Importance) ===
    print(f"\n{'='*70}")
    print("2. MDA (Permutation Importance) — out-of-sample")
    print(f"{'='*70}")
    baseline_ll = ll
    rng = np.random.RandomState(42)
    mda_rows = []

    for i, col in enumerate(feat_cols):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, i])
        probs_perm = mdl.predict_proba(X_perm)[:, 1]
        ll_perm = log_loss(y_test, probs_perm)
        acc_perm = accuracy_score(y_test, (probs_perm > 0.5).astype(int))
        dirs_perm = np.where(probs_perm > 0.5, 1.0, -1.0)
        bps_perm = (dirs_perm * fwd_ret[test_idx]).mean() * 10000
        mda_rows.append({
            "feature": col,
            "ll_increase": ll_perm - baseline_ll,
            "acc_decrease": acc - acc_perm,
            "bps_decrease": bps - bps_perm,
        })

    mda_df = pd.DataFrame(mda_rows).set_index("feature").sort_values("ll_increase", ascending=False)
    print(f"  {'Feature':<42s} {'LL+':<10s} {'Acc-':<10s} {'bps-':<10s}")
    print("  " + "-" * 72)
    for feat, row in mda_df.head(20).iterrows():
        marker = " ***" if row["bps_decrease"] > 0.5 else (" **" if row["bps_decrease"] > 0.2 else "")
        print(f"  {feat:<42s} {row['ll_increase']:>8.5f}  {row['acc_decrease']:>8.5f}  {row['bps_decrease']:>8.3f}{marker}")
    print("\n  Bottom 5 (least important / harmful when shuffled):")
    for feat, row in mda_df.tail(5).iterrows():
        print(f"  {feat:<42s} {row['ll_increase']:>8.5f}  {row['acc_decrease']:>8.5f}  {row['bps_decrease']:>8.3f}")

    # === 3. SFI (Single Feature Importance) ===
    print(f"\n{'='*70}")
    print("3. SFI (Single Feature Importance) — train with one feature at a time")
    print(f"{'='*70}")
    sfi_rows = []
    for i, col in enumerate(feat_cols):
        X_tr_s = X_train[:, i:i + 1]
        X_te_s = X_test[:, i:i + 1]
        mdl_s = xgb.XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            tree_method="hist", random_state=42, n_jobs=4, verbosity=0,
            eval_metric="logloss", objective="binary:logistic",
        )
        mdl_s.fit(X_tr_s, y_train, sample_weight=sw, verbose=False)
        probs_s = mdl_s.predict_proba(X_te_s)[:, 1]
        dirs_s = np.where(probs_s > 0.5, 1.0, -1.0)
        pnls_s = dirs_s * fwd_ret[test_idx]
        sfi_rows.append({
            "feature": col,
            "bps": pnls_s.mean() * 10000,
            "win_rate": (pnls_s > 0).mean(),
            "accuracy": accuracy_score(y_test, (probs_s > 0.5).astype(int)),
            "logloss": log_loss(y_test, probs_s),
        })

    sfi_df = pd.DataFrame(sfi_rows).set_index("feature").sort_values("bps", ascending=False)
    print(f"  {'Feature':<42s} {'bps':>7s} {'Win%':>6s} {'Acc':>7s} {'LL':>7s}")
    print("  " + "-" * 72)
    for feat, row in sfi_df.iterrows():
        marker = " ***" if row["bps"] > 1.0 else (" **" if row["bps"] > 0.5 else ("  *" if row["bps"] > 0.2 else ""))
        print(f"  {feat:<42s} {row['bps']:>7.2f} {row['win_rate'] * 100:>5.1f}% {row['accuracy']:>6.4f} {row['logloss']:>6.4f}{marker}")

    # === 4. Feature group summary ===
    print(f"\n{'='*70}")
    print("4. FEATURE GROUP SUMMARY (SFI mean bps)")
    print(f"{'='*70}")
    groups = {
        "Base returns": [c for c in feat_cols if c.startswith("ret_") and "regime" not in c],
        "Base indicators": [c for c in feat_cols if not c.startswith("signal_regime") and c in ["rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14"]],
        "Base structure": [c for c in feat_cols if not c.startswith("signal_regime") and c in ["dist_swing_high", "dist_swing_low", "entropy", "parkinson_vol_20", "cs_spread", "ffd_close", "ffd_ret_1", "bar_range", "vol_zscore"]],
        "1h returns": [c for c in feat_cols if c.startswith("signal_regime_1h_ret_")],
        "1h indicators": [c for c in feat_cols if "signal_regime_1h_" in c and any(c.endswith(x) for x in ["rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14"])],
        "1h structure": [c for c in feat_cols if "signal_regime_1h_" in c and any(c.endswith(x) for x in ["dist_swing_high", "dist_swing_low", "entropy", "parkinson_vol_20", "cs_spread", "ffd_close", "ffd_ret_1", "bar_range", "vol_zscore"])],
        "4h returns": [c for c in feat_cols if c.startswith("signal_regime_4h_ret_")],
        "4h indicators": [c for c in feat_cols if "signal_regime_4h_" in c and any(c.endswith(x) for x in ["rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14"])],
        "4h structure": [c for c in feat_cols if "signal_regime_4h_" in c and any(c.endswith(x) for x in ["dist_swing_high", "dist_swing_low", "entropy", "parkinson_vol_20", "cs_spread", "ffd_close", "ffd_ret_1", "bar_range", "vol_zscore"])],
    }
    print(f"  {'Group':<25s} {'Mean bps':>10s} {'Max bps':>10s} {'Min bps':>10s} {'n':>4s}")
    print("  " + "-" * 55)
    for gname, gcols in groups.items():
        if gcols:
            gbps = [sfi_df.loc[c, "bps"] for c in gcols if c in sfi_df.index]
            if gbps:
                print(f"  {gname:<25s} {np.mean(gbps):>10.2f} {np.max(gbps):>10.2f} {np.min(gbps):>10.2f} {len(gbps):>4d}")

    # === 5. PCA validation ===
    print(f"\n{'='*70}")
    print("5. PCA VALIDATION (MDI rank vs PCA rank correlation)")
    print(f"{'='*70}")
    X_all = np.vstack([X_train, X_test])
    X_all = np.nan_to_num(X_all, nan=0.0)
    pca = PCA(n_components=min(54, len(feat_cols)))
    pca.fit(X_all)
    pca_importance = np.abs(pca.components_[0])
    pca_rank = np.argsort(np.argsort(-pca_importance))
    mdi_rank = np.argsort(np.argsort(-importances))
    tau, pval = kendalltau(mdi_rank, pca_rank)
    print(f"  Kendall tau: {tau:.4f}  p-value: {pval:.4f}")
    if tau > 0.3:
        print("  Strong correlation — features with high MDI also load heavily on PC1")
    elif tau > 0.1:
        print("  Weak correlation — some genuine structure")
    else:
        print("  No correlation — MDI importance may be noise-driven")

    # === 6. Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_pos_sfi = (sfi_df["bps"] > 0).sum()
    n_strong_sfi = (sfi_df["bps"] > 0.5).sum()
    n_pos_mda = (mda_df["bps_decrease"] > 0).sum()
    print(f"  SFI: {n_pos_sfi}/{len(feat_cols)} features have positive bps alone")
    print(f"  SFI: {n_strong_sfi}/{len(feat_cols)} features have >0.5 bps alone")
    print(f"  MDA: {n_pos_mda}/{len(feat_cols)} features decrease bps when shuffled")
    print(f"  Baseline model: {bps:.2f} bps, {wr * 100:.1f}% win rate")

    best_sfi = sfi_df["bps"].idxmax()
    print(f"  Best single feature: {best_sfi} ({sfi_df.loc[best_sfi, 'bps']:.2f} bps)")

    if bps < 1.0 and n_strong_sfi < 5:
        print("\n  VERDICT: No feature has strong standalone predictive power.")
        print("  The residual signal (~0.5 Sharpe) comes from weak ensemble effects.")
    elif bps > 1.0:
        print(f"\n  VERDICT: Genuine signal present ({bps:.1f} bps). Key features above.")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
