"""Phase 3: single-shot backtest of all pre-committed DFL strategies.

Policy: these strategies were committed BEFORE running. Parameters
are frozen. We run each strategy ONCE, report all metrics, then pick
the most robust — no post-hoc re-tuning.

All strategies are POST-PROCESSORS of the baseline V2 trade log. This
is first-order-faithful (P&L is linear in position size for vega-P&L)
and avoids rerunning the backtester multiple times (which would
itself be a form of overfitting).

Pre-committed strategies:
- A1 Filter-25: drop bars with P(win) < train_q25
- A2 Filter-10: drop bars with P(win) < train_q10
- B1 Linear: size *= clip(2*P(win), 0.25, 1.5)
- B2 Binary-median: size *= 0.5 if P(win) < train_median else 1.0
- B3 Normalized: size *= clip(P(win) / train_mean, 0.5, 2.0)

Baseline: BTC_v2_trade_log_baseline.parquet (invalidated historical V2 reference).

Success criteria (pre-registered):
- Retain as a historical research candidate: NC Sharpe up, NC PnL within 10% of baseline
- Reject otherwise
- If multiple strategies pass, pick the SIMPLEST (Occam)

These gates compare candidates only within the invalidated V2 setup. They do
not establish PnL realism or authorize deployment.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd

from vol_arb.dfl_head import (
    compute_walkforward_scores, pre_committed_strategies,
)


def compute_metrics(tl: pd.DataFrame, capital: float = 100_000.0) -> dict:
    t = tl.loc[tl['signal'] != 0]
    # Total PnL (NC)
    total_pnl = float(t['net_pnl_nc'].sum())
    n_trades = int((t['net_pnl_nc'] != 0).sum())
    # Daily P&L
    daily = t['net_pnl_nc'].resample('1D').sum()
    daily_active = daily.loc[daily != 0]
    sharpe = (
        float(daily_active.mean() / daily_active.std() * np.sqrt(365))
        if daily_active.std() > 0 else float("nan")
    )
    # Drawdown
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    # Worst day
    worst_day = float(daily.min())
    # Win rate (trade-level, non-zero trades)
    wins = (t.loc[t['net_pnl_nc'] != 0, 'net_pnl_nc'] > 0).sum()
    n_nonzero = (t['net_pnl_nc'] != 0).sum()
    win_rate = float(wins / n_nonzero) if n_nonzero > 0 else float("nan")
    # Trade Sharpe (per-trade $)
    nonzero_pnl = t.loc[t['net_pnl_nc'] != 0, 'net_pnl_nc']
    trade_sharpe = (
        float(nonzero_pnl.mean() / nonzero_pnl.std() * np.sqrt(365 * 96))
        if nonzero_pnl.std() > 0 and len(nonzero_pnl) > 0 else float("nan")
    )
    return dict(
        total_pnl=total_pnl,
        n_trades_total=n_trades,
        n_trades_nonzero=int(n_nonzero),
        sharpe_daily=sharpe,
        max_dd=float(dd),
        worst_day=worst_day,
        win_rate=win_rate,
        trade_sharpe=trade_sharpe,
    )


def main():
    t0 = time.time()
    print("Loading baseline trade log + predictions...")
    tl = pd.read_parquet('vol_arb/output/BTC_v2_trade_log_baseline.parquet')
    preds = pd.read_parquet(
        'vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet'
    )
    print(f"  Baseline: {len(tl)} rows, "
          f"{(tl['signal']!=0).sum()} trade-fired bars")

    print("\nComputing walk-forward DFL scores (all trade-fired bars)...")
    scores = compute_walkforward_scores(tl, preds, min_train_trades=500)
    print(f"  Scored: {scores.notna().sum()} bars "
          f"(unscored: {scores.isna().sum() - (tl['signal']==0).sum()} trade-fired "
          f"bars lacked training history)")
    # Save scores for reproducibility
    scores.to_frame('p_win').to_parquet(
        'vol_arb/output/BTC_v2_dfl_scores.parquet'
    )
    print(f"  Saved: vol_arb/output/BTC_v2_dfl_scores.parquet")

    # Baseline metrics
    print("\nBaseline metrics:")
    base = compute_metrics(tl)
    for k, v in base.items():
        print(f"  {k:18}: {v:.3f}" if isinstance(v, float) else f"  {k:18}: {v}")

    # Run each pre-committed strategy
    print("\n" + "=" * 100)
    print("PRE-COMMITTED STRATEGIES — one run each, no re-tuning")
    print("=" * 100)
    strategies = pre_committed_strategies()
    results = {}
    for strat in strategies:
        t1 = time.time()
        print(f"\n[{strat.name}] {strat.description}")
        modified_tl = strat.apply(tl, scores)
        metrics = compute_metrics(modified_tl)
        # Save modified trade log
        out_path = f'vol_arb/output/BTC_v2_trade_log_dfl_{strat.name}.parquet'
        modified_tl.to_parquet(out_path)
        print(f"  Saved: {out_path} ({time.time()-t1:.1f}s)")
        results[strat.name] = metrics

    # Summary table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE (NC, $100K capital)")
    print("=" * 100)
    header = f"{'strategy':<22} {'PnL':>10} {'dPnL':>9} {'Sharpe':>7} {'dShp':>6} {'MaxDD':>8} {'WorstD':>8} {'WR':>5} {'tradeSh':>8} {'nTrades':>8}"
    print(header)
    print("-" * len(header))
    print(f"{'baseline':<22} ${base['total_pnl']:>9.0f} {'—':>9} "
          f"{base['sharpe_daily']:>7.2f} {'—':>6} ${base['max_dd']:>7.0f} "
          f"${base['worst_day']:>7.0f} {base['win_rate']:>5.1%} "
          f"{base['trade_sharpe']:>8.2f} {base['n_trades_nonzero']:>8}")
    for name, m in results.items():
        print(f"{name:<22} ${m['total_pnl']:>9.0f} "
              f"{m['total_pnl']-base['total_pnl']:>+9.0f} "
              f"{m['sharpe_daily']:>7.2f} "
              f"{m['sharpe_daily']-base['sharpe_daily']:>+6.2f} "
              f"${m['max_dd']:>7.0f} "
              f"${m['worst_day']:>7.0f} "
              f"{m['win_rate']:>5.1%} "
              f"{m['trade_sharpe']:>8.2f} "
              f"{m['n_trades_nonzero']:>8}")

    # Save comparison CSV
    df = pd.DataFrame({'baseline': base, **results}).T
    df.to_csv('vol_arb/output/BTC_v2_dfl_comparison.csv')
    print(f"\nSaved comparison CSV: vol_arb/output/BTC_v2_dfl_comparison.csv")

    # Pre-registered within-model candidate decision
    print("\n" + "=" * 100)
    print("HISTORICAL CANDIDATE DECISION (PRE-REGISTERED GATES)")
    print("=" * 100)
    print("Sidecar-candidate gate: NC Sharpe >= baseline AND PnL >= 0.90 * baseline")
    print("Preferred-candidate gate: NC Sharpe >= baseline + 0.5 AND PnL >= baseline")
    print()
    shipped = []
    for name, m in results.items():
        sharpe_ok_sidecar = m['sharpe_daily'] >= base['sharpe_daily']
        pnl_ok_sidecar = m['total_pnl'] >= 0.90 * base['total_pnl']
        sharpe_ok_default = m['sharpe_daily'] >= base['sharpe_daily'] + 0.5
        pnl_ok_default = m['total_pnl'] >= base['total_pnl']
        if sharpe_ok_default and pnl_ok_default:
            verdict = "PREFERRED HISTORICAL CANDIDATE"
            shipped.append((name, verdict))
        elif sharpe_ok_sidecar and pnl_ok_sidecar:
            verdict = "OPTIONAL HISTORICAL CANDIDATE"
            shipped.append((name, verdict))
        else:
            verdict = "REJECT"
        sharpe_diff = m['sharpe_daily'] - base['sharpe_daily']
        pnl_pct = 100 * (m['total_pnl'] - base['total_pnl']) / base['total_pnl']
        print(f"  {name:<22} dSharpe={sharpe_diff:+.2f}  dPnL={pnl_pct:+.1f}%  -> {verdict}")

    if not shipped:
        print("\nAll strategies rejected. Document this within-model comparison as a negative result.")
    else:
        print(f"\n{len(shipped)} strateg{'ies' if len(shipped)>1 else 'y'} passed gate:")
        for name, verdict in shipped:
            print(f"  - {name}: {verdict}")
        print("\nPer Occam, if multiple pass, pick the SIMPLEST "
              "(filter > binary size > continuous size).")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
