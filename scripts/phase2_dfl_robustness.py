"""Phase 2c: bootstrap CI + robustness checks for the DFL W/L head.

- Bootstrap CI on top-10% PnL-capture ratio
- Sanity check: permutation test — shuffle classifier scores, does
  the top-10% still beat random?
- Seed sweep: are the results seed-stable?
- Per-month breakdown of classifier value
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


FEAT_COLS = [
    'y_pred', 'iv_used', 'iv_log', 'pred_edge_log', 'abs_pred_edge_log',
    'past_vol_24h', 'past_vol_6h', 'past_vol_std',
    'recent_resid_24h', 'recent_abs_resid_24h',
]


def run_walkforward(tr, seed=42):
    years = sorted(set(tr.index.year))
    out = []
    for test_y in years:
        train = tr.loc[tr.index.year < test_y]
        test  = tr.loc[tr.index.year == test_y]
        if len(train) < 500 or len(test) < 100:
            continue
        X_train = train[FEAT_COLS]
        X_test  = test[FEAT_COLS]
        y_train = train['win']
        y_test  = test['win']
        clf = LGBMClassifier(
            num_leaves=15, max_depth=5, learning_rate=0.05, n_estimators=300,
            min_child_samples=50, reg_lambda=1.0, seed=seed, verbose=-1,
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                callbacks=[early_stopping(25, verbose=False)])
        pp = clf.predict_proba(X_test)[:, 1]
        out.append(test.assign(clf_pwin=pp))
    return pd.concat(out)


def top_k_pnl(df, col, pct, ascending=False):
    k = int(len(df) * pct / 100)
    return df.sort_values(col, ascending=ascending).head(k)['net_pnl_nc'].sum()


def main():
    preds = pd.read_parquet('vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet')
    tl = pd.read_parquet('vol_arb/output/BTC_v2_trade_log_baseline.parquet')
    df = tl[['signal', 'iv_used', 'net_pnl_nc', 'vega']].join(
        preds[['y_pred','y_true']], how='inner'
    ).dropna()
    df = build_features(df)
    tr = df.loc[df['signal'] != 0].dropna(subset=FEAT_COLS).copy()
    tr['win'] = (tr['net_pnl_nc'] > 0).astype(int)

    # ── Main walk-forward ──
    print("=" * 70)
    print("MAIN walk-forward (seed=42)")
    print("=" * 70)
    result = run_walkforward(tr, seed=42)
    total = result['net_pnl_nc'].sum()
    print(f"OOS trades={len(result)}, OOS PnL=${total:.0f}")
    for pct in [10, 25, 50]:
        pnl_clf  = top_k_pnl(result, 'clf_pwin', pct)
        pnl_edge = top_k_pnl(result, 'abs_pred_edge_log', pct)
        print(f"  top {pct}%:  clf=${pnl_clf:.0f}  edge=${pnl_edge:.0f}  ratio_clf/edge={pnl_clf/pnl_edge:.2f}")

    # ── Seed sweep (5 seeds) ──
    print()
    print("=" * 70)
    print("SEED SWEEP — is top-10% ratio stable across LGBM seeds?")
    print("=" * 70)
    ratios = []
    for seed in [42, 123, 456, 789, 1000]:
        rs = run_walkforward(tr, seed=seed)
        pnl_clf = top_k_pnl(rs, 'clf_pwin', 10)
        pnl_edge = top_k_pnl(rs, 'abs_pred_edge_log', 10)
        ratio = pnl_clf / pnl_edge
        print(f"  seed={seed}: clf=${pnl_clf:.0f}  edge=${pnl_edge:.0f}  ratio={ratio:.3f}")
        ratios.append(ratio)
    print(f"  ratio across seeds: mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}  "
          f"min={np.min(ratios):.3f}  max={np.max(ratios):.3f}")

    # ── Bootstrap CI on main-seed top-10% PnL capture ──
    print()
    print("=" * 70)
    print("BOOTSTRAP CI on top-10% PnL capture (seed=42 classifier)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    n = len(result)
    n_boot = 1000
    boot_ratios = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = result.iloc[idx]
        if len(sample) < 100:
            continue
        pnl_clf = top_k_pnl(sample, 'clf_pwin', 10)
        pnl_edge = top_k_pnl(sample, 'abs_pred_edge_log', 10)
        if pnl_edge > 0:
            boot_ratios.append(pnl_clf / pnl_edge)
    boot_ratios = np.array(boot_ratios)
    print(f"  mean ratio    = {np.mean(boot_ratios):.3f}")
    print(f"  95% CI        = [{np.quantile(boot_ratios, 0.025):.3f}, {np.quantile(boot_ratios, 0.975):.3f}]")
    print(f"  P(ratio > 1.2) = {np.mean(boot_ratios > 1.2):.3f}")
    print(f"  P(ratio > 1.5) = {np.mean(boot_ratios > 1.5):.3f}")

    # ── Permutation test: shuffle classifier scores ──
    print()
    print("=" * 70)
    print("PERMUTATION TEST — shuffle clf_pwin, does top-10% still beat random?")
    print("=" * 70)
    real_pnl_clf = top_k_pnl(result, 'clf_pwin', 10)
    null_pnls = []
    for b in range(1000):
        shuffled = result['clf_pwin'].sample(frac=1.0, random_state=b).to_numpy()
        tmp = result.copy()
        tmp['shuf'] = shuffled
        null_pnls.append(top_k_pnl(tmp, 'shuf', 10))
    null_pnls = np.array(null_pnls)
    p_value = np.mean(null_pnls >= real_pnl_clf)
    print(f"  Real top-10%:       ${real_pnl_clf:.0f}")
    print(f"  Null distribution:  mean=${np.mean(null_pnls):.0f}  "
          f"p95=${np.quantile(null_pnls, 0.95):.0f}")
    print(f"  Permutation p-value: {p_value:.4f}")

    # ── Per-month breakdown ──
    print()
    print("=" * 70)
    print("PER-MONTH top-10% PnL (clf vs baseline)")
    print("=" * 70)
    result['month'] = result.index.strftime("%Y-%m")
    months = sorted(result['month'].unique())
    print(f"{'month':<10} {'n':<6} {'total_$':<10} {'top10_clf':<11} {'top10_edge':<11} {'ratio':<7}")
    n_better = 0
    n_worse = 0
    for m in months:
        sub = result.loc[result['month'] == m]
        if len(sub) < 20:
            continue
        pnl_clf  = top_k_pnl(sub, 'clf_pwin', 10)
        pnl_edge = top_k_pnl(sub, 'abs_pred_edge_log', 10)
        ratio = pnl_clf / pnl_edge if pnl_edge != 0 else float('nan')
        print(f"{m:<10} {len(sub):<6} ${sub['net_pnl_nc'].sum():<9.0f} ${pnl_clf:<10.0f} ${pnl_edge:<10.0f} {ratio:<7.2f}")
        if ratio > 1:
            n_better += 1
        elif ratio < 1:
            n_worse += 1
    print(f"\nMonths where clf > edge: {n_better}  |  clf < edge: {n_worse}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
