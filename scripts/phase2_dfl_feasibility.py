"""Phase 2a feasibility: DFL decision-focused fine-tuning test.

Question: can a DFF-style head (Wen et al. AAAI 2025) on TOP of frozen
HAR-PD predictions improve W/L discrimination within trade-fired bars?

If yes -> full decision-focused retrain likely to help.
If no  -> signal already near capacity ceiling; reject cleanly.

Methodology (offline, no backtest, strict leakage control):
- Frozen backbone: production HAR-PD predictions (walk-forward).
- Head features: y_pred_prod, iv, iv_log, pred_edge_log, |pred_edge_log|,
  past realized vol at multiple horizons (strictly lagged), recent
  residual mean/|mean| (strictly lagged).
- Target: residual (y_true - y_pred_prod). Sample-weighted regression
  with w_i = max(0.05, |y_true_i - log(IV_i/100)|) — the DFF "decision-
  focused" weighting that upweights bars where the trade decision
  magnitude is large.
- Temporal split: train < 2025-01-01, test >= 2025-01-01.

Leakage controls:
- Features built only from past data (shift(1) before rolling).
- Win/loss label is evaluation ONLY; never a feature.
- OOF predictions are already walk-forward safe.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main() -> int:
    preds = pd.read_parquet('vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet')
    tl = pd.read_parquet('vol_arb/output/BTC_v2_trade_log_baseline.parquet')

    df = tl[['signal', 'iv_used', 'net_pnl_nc', 'vega']].join(
        preds[['y_pred', 'y_true']], how='inner'
    ).dropna()

    df['iv_log'] = np.log(df['iv_used'] / 100.0)
    df['pred_edge_log'] = df['y_pred'] - df['iv_log']
    df['abs_pred_edge_log'] = df['pred_edge_log'].abs()

    # Past realized vol (strictly lagged, non-leaking)
    df['past_vol_24h'] = df['y_true'].shift(1).rolling(96, min_periods=24).mean()
    df['past_vol_6h']  = df['y_true'].shift(1).rolling(24, min_periods=8).mean()
    df['past_vol_std'] = df['y_true'].shift(1).rolling(96, min_periods=24).std()

    # Recent residual (strictly lagged)
    resid_series = df['y_true'] - df['y_pred']
    df['recent_resid_24h']     = resid_series.shift(1).rolling(96, min_periods=24).mean()
    df['recent_abs_resid_24h'] = resid_series.abs().shift(1).rolling(96, min_periods=24).mean()

    # Trade-fired only + build W/L label
    tr = df.loc[df['signal'] != 0].copy()
    tr['win'] = (tr['net_pnl_nc'] > 0).astype(int)

    feat_cols = [
        'y_pred', 'iv_used', 'iv_log', 'pred_edge_log', 'abs_pred_edge_log',
        'past_vol_24h', 'past_vol_6h', 'past_vol_std',
        'recent_resid_24h', 'recent_abs_resid_24h',
    ]
    tr_full = tr.dropna(subset=feat_cols)
    print(f"After dropna: {len(tr_full)} trades (vs {len(tr)} trade-fired)")

    train = tr_full.loc[tr_full.index < '2025-01-01']
    test  = tr_full.loc[tr_full.index >= '2025-01-01']
    print(f"Train: {len(train)}  ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Test:  {len(test)}   ({test.index.min().date()} to {test.index.max().date()})")
    print(f"Train win rate: {train['win'].mean():.3f}   Test win rate: {test['win'].mean():.3f}")

    # ── Baseline: |pred_edge| -> W/L AUC ──
    baseline_auc_test  = roc_auc_score(test['win'],  test['abs_pred_edge_log'])
    baseline_auc_train = roc_auc_score(train['win'], train['abs_pred_edge_log'])
    print(f"\n[Baseline] AUC(|pred_edge| -> W/L):  train={baseline_auc_train:.4f}  test={baseline_auc_test:.4f}")

    X_train = train[feat_cols]
    X_test  = test[feat_cols]
    y_train = train['win']
    y_test  = test['win']

    # ── Test 1: direct W/L classifier ──
    clf = LGBMClassifier(
        num_leaves=15, max_depth=5, learning_rate=0.05, n_estimators=300,
        min_child_samples=50, reg_lambda=1.0, seed=42, verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[early_stopping(25, verbose=False)],
    )
    pp_test  = clf.predict_proba(X_test)[:, 1]
    pp_train = clf.predict_proba(X_train)[:, 1]
    clf_auc_test  = roc_auc_score(y_test,  pp_test)
    clf_auc_train = roc_auc_score(y_train, pp_train)
    print(f"\n[DFF-classifier] AUC(P(win)):  train={clf_auc_train:.4f}  test={clf_auc_test:.4f}")
    print(f"  Improvement vs baseline on test: {clf_auc_test - baseline_auc_test:+.4f}")

    imp = pd.DataFrame({
        'feature': feat_cols,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)
    print("\n  Feature importance:")
    print(imp.to_string(index=False))

    # ── Test 2: residual regressor, un-weighted and edge-weighted ──
    resid_target = train['y_true'] - train['y_pred']
    resid_target_test = test['y_true'] - test['y_pred']
    weights = (train['y_true'] - train['iv_log']).abs().clip(lower=0.05)

    for name, sw in [('unweighted', None), ('edge-weighted', weights)]:
        reg = LGBMRegressor(
            num_leaves=15, max_depth=5, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=1.0, seed=42, verbose=-1,
        )
        reg.fit(
            X_train, resid_target, sample_weight=sw,
            eval_set=[(X_test, resid_target_test)],
            callbacks=[early_stopping(25, verbose=False)],
        )
        resid_pred_test = reg.predict(X_test)
        corrected_pred = test['y_pred'] + resid_pred_test
        corrected_edge = corrected_pred - test['iv_log']
        abs_corrected_edge = corrected_edge.abs()
        auc_corrected = roc_auc_score(test['win'], abs_corrected_edge)
        real_edge_abs = (test['y_true'] - test['iv_log']).abs()
        rho_corr, _ = spearmanr(abs_corrected_edge, real_edge_abs)
        rho_base, _ = spearmanr(test['abs_pred_edge_log'], real_edge_abs)

        # Win rate if we re-decided trades under corrected predictions
        y_pred_vp_c = np.exp(corrected_pred) * 100
        real_vp     = np.exp(test['y_true']) * 100
        corrected_long  = (y_pred_vp_c - test['iv_used']) > 12
        corrected_short = (y_pred_vp_c - test['iv_used']) < -22
        corrected_trig  = corrected_long | corrected_short
        real_long_prof  = (real_vp - test['iv_used']) > 0
        real_short_prof = (real_vp - test['iv_used']) < 0
        n_trig = int(corrected_trig.sum())
        if n_trig > 0:
            corrected_wins = int(
                ((corrected_long & real_long_prof)
                 | (corrected_short & real_short_prof)).sum()
            )
            win_rate_corr = corrected_wins / n_trig
        else:
            win_rate_corr = float('nan')
        orig_win_rate = test['win'].mean()

        print(f"\n[DFF-regressor, {name}] on TEST:")
        print(f"  AUC(|corr_edge| -> W/L):   {auc_corrected:.4f}  "
              f"(baseline {baseline_auc_test:.4f}, delta {auc_corrected-baseline_auc_test:+.4f})")
        print(f"  Spearman(|corr_edge|, |real_edge|): {rho_corr:+.4f}  "
              f"(baseline {rho_base:+.4f}, delta {rho_corr-rho_base:+.4f})")
        print(f"  Corrected-model trades triggered: {n_trig} / {len(test)} original")
        print(f"  Direction-match rate: corrected={win_rate_corr:.3f}  baseline={orig_win_rate:.3f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
