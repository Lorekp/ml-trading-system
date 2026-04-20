"""Phase 2b: economic-value test of the DFF W/L classifier.

AUC improved +0.08 but test AUC is still 0.55 — weak signal. Test whether
this weak signal translates to meaningful P&L-ranking in the hold-out period.

Specifically: rank test bars by classifier P(win), take top-K% and compare
the captured P&L to random ranking and to baseline |pred_edge| ranking.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score


def build_features(df):
    df = df.copy()
    df['iv_log'] = np.log(df['iv_used'] / 100.0)
    df['pred_edge_log'] = df['y_pred'] - df['iv_log']
    df['abs_pred_edge_log'] = df['pred_edge_log'].abs()
    df['past_vol_24h'] = df['y_true'].shift(1).rolling(96, min_periods=24).mean()
    df['past_vol_6h']  = df['y_true'].shift(1).rolling(24, min_periods=8).mean()
    df['past_vol_std'] = df['y_true'].shift(1).rolling(96, min_periods=24).std()
    r = df['y_true'] - df['y_pred']
    df['recent_resid_24h']     = r.shift(1).rolling(96, min_periods=24).mean()
    df['recent_abs_resid_24h'] = r.abs().shift(1).rolling(96, min_periods=24).mean()
    return df


def main():
    preds = pd.read_parquet('vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet')
    tl = pd.read_parquet('vol_arb/output/BTC_v2_trade_log_baseline.parquet')
    df = tl[['signal', 'iv_used', 'net_pnl_nc', 'vega']].join(
        preds[['y_pred','y_true']], how='inner'
    ).dropna()
    df = build_features(df)

    feat_cols = [
        'y_pred', 'iv_used', 'iv_log', 'pred_edge_log', 'abs_pred_edge_log',
        'past_vol_24h', 'past_vol_6h', 'past_vol_std',
        'recent_resid_24h', 'recent_abs_resid_24h',
    ]
    tr = df.loc[df['signal'] != 0].dropna(subset=feat_cols).copy()
    tr['win'] = (tr['net_pnl_nc'] > 0).astype(int)

    # TRUE walk-forward by YEAR to validate OOS robustness
    print("=" * 70)
    print("YEAR-BY-YEAR walk-forward OOS — DFL classifier value")
    print("=" * 70)
    print(f"{'test_year':<10} {'n_test':<7} {'n_train':<8} {'base_AUC':<10} {'clf_AUC':<9} {'delta':<8} {'top10_pnl_clf':<14} {'top10_pnl_base':<16}")

    years = sorted(set(tr.index.year))
    all_deltas = []
    for test_y in years:
        # Train on all prior years
        train = tr.loc[tr.index.year < test_y]
        test  = tr.loc[tr.index.year == test_y]
        if len(train) < 500 or len(test) < 100:
            continue

        X_train = train[feat_cols]
        X_test  = test[feat_cols]
        y_train = train['win']
        y_test  = test['win']

        base_auc = roc_auc_score(y_test, test['abs_pred_edge_log'])
        clf = LGBMClassifier(
            num_leaves=15, max_depth=5, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=1.0, seed=42, verbose=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                callbacks=[early_stopping(25, verbose=False)])
        pp = clf.predict_proba(X_test)[:, 1]
        clf_auc = roc_auc_score(y_test, pp)

        # Top-10% by classifier score vs by |pred_edge|
        k = max(1, int(len(test) * 0.10))
        # rank by clf proba descending
        idx_clf = np.argsort(-pp)[:k]
        pnl_clf_top = test.iloc[idx_clf]['net_pnl_nc'].sum()
        idx_base = np.argsort(-test['abs_pred_edge_log'].values)[:k]
        pnl_base_top = test.iloc[idx_base]['net_pnl_nc'].sum()

        delta = clf_auc - base_auc
        all_deltas.append(delta)
        print(f"{test_y:<10} {len(test):<7} {len(train):<8} {base_auc:<10.4f} {clf_auc:<9.4f} "
              f"{delta:<+8.4f} ${pnl_clf_top:<13.0f} ${pnl_base_top:<15.0f}")

    print()
    print(f"Mean AUC delta across {len(all_deltas)} test-years: {np.mean(all_deltas):+.4f}")
    print(f"Years with improvement: {sum(1 for d in all_deltas if d > 0)} / {len(all_deltas)}")
    print(f"Years with improvement > 0.01: {sum(1 for d in all_deltas if d > 0.01)} / {len(all_deltas)}")

    # ── Now the economic value test: take top-K% by classifier, report P&L ratios ──
    print()
    print("=" * 70)
    print("ECONOMIC VALUE: P&L capture at top-K% (walk-forward concatenated)")
    print("=" * 70)

    all_test_preds = []
    all_test = []
    for test_y in years:
        train = tr.loc[tr.index.year < test_y]
        test  = tr.loc[tr.index.year == test_y]
        if len(train) < 500 or len(test) < 100:
            continue
        X_train = train[feat_cols]
        X_test  = test[feat_cols]
        y_train = train['win']
        y_test  = test['win']
        clf = LGBMClassifier(
            num_leaves=15, max_depth=5, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=1.0, seed=42, verbose=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                callbacks=[early_stopping(25, verbose=False)])
        pp = clf.predict_proba(X_test)[:, 1]
        all_test_preds.append(pp)
        all_test.append(test.assign(clf_pwin=pp))

    all_test_df = pd.concat(all_test)
    print(f"Total OOS trades: {len(all_test_df)}, total PnL: ${all_test_df['net_pnl_nc'].sum():.0f}")
    print()
    print(f"{'Top K%':<8} {'pnl_by_clf':<14} {'pnl_by_|edge|':<16} {'pnl_by_random':<15}")

    rng = np.random.default_rng(42)
    for pct in [10, 25, 50, 75]:
        k = int(len(all_test_df) * pct / 100)
        # By clf score
        top_clf = all_test_df.sort_values('clf_pwin', ascending=False).head(k)
        pnl_clf = top_clf['net_pnl_nc'].sum()
        # By |pred_edge|
        top_edge = all_test_df.sort_values('abs_pred_edge_log', ascending=False).head(k)
        pnl_edge = top_edge['net_pnl_nc'].sum()
        # Random baseline (mean over 100 samples)
        pnl_rand = np.mean([
            all_test_df['net_pnl_nc'].sample(k, random_state=i).sum()
            for i in range(100)
        ])
        print(f"  top {pct:>2}%  ${pnl_clf:<12.0f}  ${pnl_edge:<14.0f}  ${pnl_rand:<13.0f}")

    # Also: filter-style — drop the bars with LOWEST P(win), keep the rest.
    # This simulates "use clf as a gate: only trade if P(win) > threshold".
    print()
    print("FILTER-STYLE: drop bottom-K% P(win), report retained P&L")
    print(f"{'Drop K%':<8} {'retained_pnl':<14} {'vs_full ratio':<16} {'n_dropped':<11}")
    total_pnl = all_test_df['net_pnl_nc'].sum()
    for pct in [10, 25, 50]:
        k = int(len(all_test_df) * pct / 100)
        bottom = all_test_df.sort_values('clf_pwin', ascending=True).head(k)
        kept = all_test_df.drop(bottom.index)
        kept_pnl = kept['net_pnl_nc'].sum()
        ratio = kept_pnl / total_pnl if total_pnl != 0 else float('nan')
        print(f"  drop {pct:>2}%  ${kept_pnl:<12.0f}  {ratio:<15.3f}  {k}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
