#!/usr/bin/env python
"""Apples-to-apples correlation: LogReg PROD vs Vol Arb (BTC options).

Both strategies normalized to non-compound $200K base.
Three comparison levels:
  A) BTC-only LogReg vs Vol Arb (true apples-to-apples)
  B) Full LogReg portfolio vs Vol Arb
  C) LogReg non-BTC vs Vol Arb

Usage:
    PYTHONPATH="src;." python experiments/correlation_analysis.py
"""
import sys, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

LR_PATH = Path("experiments/trade_logs/trade_log_prod.csv")
VA_PATH = Path("vol_arb/output/backtest_tradelog_production.parquet")
CAP = 200_000.0


def load_logreg():
    lr = pd.read_csv(LR_PATH, parse_dates=["timestamp"])
    if lr["timestamp"].dt.tz is not None:
        lr["timestamp"] = lr["timestamp"].dt.tz_localize(None)
    # Non-compound PnL: scale each trade's return to fixed $200K
    lr["pre_equity"] = lr["equity"] - lr["pnl"]
    lr["r_trade"] = lr["pnl"] / lr["pre_equity"]
    lr["pnl_nc"] = lr["r_trade"] * CAP
    lr["date"] = lr["timestamp"].dt.date
    return lr


def load_vol_arb():
    va = pd.read_parquet(VA_PATH)
    va = va.reset_index()
    va["timestamp"] = pd.to_datetime(va["timestamp"])
    if va["timestamp"].dt.tz is not None:
        va["timestamp"] = va["timestamp"].dt.tz_localize(None)
    va["date"] = va["timestamp"].dt.date
    return va


def daily_nc(df, pnl_col="pnl_nc"):
    return df.groupby("date")[pnl_col].sum()


def correlation_report(lr_daily, va_daily, label):
    """Compute and print correlation metrics for two daily PnL series."""
    combined = pd.DataFrame({"lr": lr_daily, "va": va_daily}).dropna()
    if len(combined) < 30:
        print(f"  {label}: insufficient overlap ({len(combined)} days)")
        return None

    pearson = combined["lr"].corr(combined["va"])
    spearman = combined["lr"].corr(combined["va"], method="spearman")

    print(f"\n  {label}")
    print(f"  {'='*60}")
    print(f"    Overlapping days:        {len(combined):,}")
    print(f"    Pearson correlation:      {pearson:+.4f}")
    print(f"    Spearman correlation:     {spearman:+.4f}")

    # Rolling 60-day
    rolling = combined["lr"].rolling(60).corr(combined["va"]).dropna()
    if len(rolling) > 0:
        print(f"    Rolling 60d: mean={rolling.mean():+.4f}  "
              f"median={rolling.median():+.4f}  "
              f"[{rolling.quantile(0.05):+.4f}, {rolling.quantile(0.95):+.4f}]")

    # Yearly
    combined.index = pd.to_datetime(combined.index)
    print(f"\n    Yearly:")
    for yr in sorted(combined.index.year.unique()):
        yc = combined[combined.index.year == yr]
        if len(yc) > 20:
            c = yc["lr"].corr(yc["va"])
            cs = yc["lr"].corr(yc["va"], method="spearman")
            print(f"      {yr}: Pearson={c:+.4f}  Spearman={cs:+.4f}  ({len(yc)} days)")

    # Drawdown overlap
    both_neg = ((combined.lr < 0) & (combined.va < 0)).sum()
    both_pos = ((combined.lr > 0) & (combined.va > 0)).sum()
    mixed = len(combined) - both_neg - both_pos
    print(f"\n    Drawdown overlap:")
    print(f"      Both positive:  {both_pos:>4d} ({both_pos/len(combined)*100:.1f}%)")
    print(f"      Both negative:  {both_neg:>4d} ({both_neg/len(combined)*100:.1f}%)")
    print(f"      Mixed (hedge):  {mixed:>4d} ({mixed/len(combined)*100:.1f}%)")

    # Worst days hedge
    print(f"\n    Worst 5 LogReg days -> Vol Arb:")
    for dt, row in combined.nsmallest(5, "lr").iterrows():
        print(f"      {dt.strftime('%Y-%m-%d')}  LR: ${row.lr:>+10,.0f}  VA: ${row.va:>+10,.0f}")
    print(f"    Worst 5 Vol Arb days -> LogReg:")
    for dt, row in combined.nsmallest(5, "va").iterrows():
        print(f"      {dt.strftime('%Y-%m-%d')}  VA: ${row.va:>+10,.0f}  LR: ${row.lr:>+10,.0f}")

    return combined


def portfolio_optimization(combined):
    """Sweep allocation weights and find optimal combined Sharpe."""
    lr_mean = combined["lr"].mean()
    va_mean = combined["va"].mean()
    lr_std = combined["lr"].std()
    va_std = combined["va"].std()
    cov = combined["lr"].cov(combined["va"])

    ann = np.sqrt(365.25)
    lr_sr = lr_mean / lr_std * ann if lr_std > 0 else 0
    va_sr = va_mean / va_std * ann if va_std > 0 else 0

    print(f"\n  Individual (non-compound, $200K each):")
    print(f"    LogReg:  mean=${lr_mean:>+8,.0f}/day  std=${lr_std:>8,.0f}  "
          f"Sharpe={lr_sr:.2f}  total=${combined['lr'].sum():>+12,.0f}")
    print(f"    Vol Arb: mean=${va_mean:>+8,.0f}/day  std=${va_std:>8,.0f}  "
          f"Sharpe={va_sr:.2f}  total=${combined['va'].sum():>+12,.0f}")

    # Sweep w = fraction allocated to vol arb (0 to 1)
    # Combined PnL = (1-w)*LR + w*VA  (on $200K total)
    best_sr = -999
    best_w = 0
    results = []
    for w_pct in range(0, 101, 5):
        w = w_pct / 100.0
        c_mean = (1 - w) * lr_mean + w * va_mean
        c_var = (1-w)**2 * lr_std**2 + w**2 * va_std**2 + 2*w*(1-w)*cov
        c_std = np.sqrt(c_var)
        c_sr = c_mean / c_std * ann if c_std > 0 else 0
        results.append((w_pct, c_mean, c_std, c_sr))
        if c_sr > best_sr:
            best_sr = c_sr
            best_w = w_pct

    print(f"\n  Allocation sweep ($200K total, rebalanced daily):")
    print(f"    {'VA%':>4s} {'LR%':>4s} {'Mean$/d':>10s} {'Std$/d':>10s} {'Sharpe':>8s}")
    print(f"    {'-'*40}")
    for w_pct, c_mean, c_std, c_sr in results:
        marker = " <-- optimal" if w_pct == best_w else ""
        if w_pct % 10 == 0 or w_pct == best_w:
            print(f"    {w_pct:>3d}% {100-w_pct:>3d}% ${c_mean:>+8,.0f} ${c_std:>8,.0f} "
                  f"{c_sr:>7.2f}{marker}")

    # Also show equal-dollar (additive, $200K each = $400K total)
    add_pnl = combined["lr"] + combined["va"]
    add_sr = add_pnl.mean() / add_pnl.std() * ann if add_pnl.std() > 0 else 0
    div_ratio = (lr_std + va_std) / add_pnl.std()
    vol_reduction = (1 - add_pnl.std() / (lr_std + va_std)) * 100

    print(f"\n  Equal-dollar ($200K each, $400K total):")
    print(f"    Combined: mean=${add_pnl.mean():>+8,.0f}/day  std=${add_pnl.std():>8,.0f}  "
          f"Sharpe={add_sr:.2f}")
    print(f"    Diversification ratio: {div_ratio:.3f}")
    print(f"    Vol reduction: {vol_reduction:.1f}%")


def main():
    print("Loading data...")
    lr = load_logreg()
    va = load_vol_arb()

    # Build daily NC series
    lr_btc_daily = daily_nc(lr[lr.asset == "BTC"])
    lr_nonbtc_daily = daily_nc(lr[lr.asset != "BTC"])
    lr_all_daily = daily_nc(lr)
    va_daily = daily_nc(va, pnl_col="net_pnl_nc")

    # Summary
    overlap_start = max(lr_all_daily.index.min(), va_daily.index.min())
    overlap_end = min(lr_all_daily.index.max(), va_daily.index.max())

    print(f"\n{'='*70}")
    print(f"  CORRELATION ANALYSIS: LogReg PROD vs Vol Arb")
    print(f"  Non-compound, both on $200K base")
    print(f"{'='*70}")
    print(f"\n  Overlap: {overlap_start} to {overlap_end}")
    print(f"  LogReg: {len(lr):,} trades ({len(lr[lr.asset=='BTC']):,} BTC)")
    print(f"  Vol Arb: {(va.signal != 0).sum():,} active bars")

    n_btc = lr[lr.asset == "BTC"]
    n_all = lr
    print(f"\n  NC totals (overlap period):")
    lr_btc_ol = lr_btc_daily[lr_btc_daily.index >= overlap_start]
    lr_all_ol = lr_all_daily[lr_all_daily.index >= overlap_start]
    va_ol = va_daily[va_daily.index >= overlap_start]
    print(f"    LogReg BTC:     ${lr_btc_ol.sum():>+12,.0f}")
    print(f"    LogReg all:     ${lr_all_ol.sum():>+12,.0f}")
    print(f"    Vol Arb:        ${va_ol.sum():>+12,.0f}")

    # A) BTC-only vs BTC-only
    c_btc = correlation_report(lr_btc_daily, va_daily,
                                "A) BTC LogReg vs BTC Vol Arb (apples-to-apples)")

    # B) Full portfolio vs Vol Arb
    c_all = correlation_report(lr_all_daily, va_daily,
                                "B) Full LogReg portfolio vs Vol Arb")

    # C) Non-BTC LogReg vs Vol Arb
    c_nonbtc = correlation_report(lr_nonbtc_daily, va_daily,
                                   "C) Non-BTC LogReg vs Vol Arb")

    # Portfolio optimization (use full portfolio vs vol arb)
    if c_all is not None:
        print(f"\n{'='*70}")
        print(f"  PORTFOLIO OPTIMIZATION: Full LogReg + Vol Arb")
        print(f"{'='*70}")
        portfolio_optimization(c_all)

    # Also do BTC-only optimization
    if c_btc is not None:
        print(f"\n{'='*70}")
        print(f"  PORTFOLIO OPTIMIZATION: BTC LogReg + Vol Arb")
        print(f"{'='*70}")
        portfolio_optimization(c_btc)

    # Bottom line
    print(f"\n{'='*70}")
    print(f"  BOTTOM LINE")
    print(f"{'='*70}")
    if c_btc is not None:
        p_btc = c_btc["lr"].corr(c_btc["va"])
        p_all = c_all["lr"].corr(c_all["va"]) if c_all is not None else 0
        p_nonbtc = c_nonbtc["lr"].corr(c_nonbtc["va"]) if c_nonbtc is not None else 0
        print(f"  BTC-vs-BTC correlation:       {p_btc:+.4f}")
        print(f"  Full portfolio-vs-VA:          {p_all:+.4f}")
        print(f"  Non-BTC-vs-VA:                {p_nonbtc:+.4f}")
        print(f"\n  Vol arb trades BTC options. LogReg's BTC perp trades show")
        print(f"  {abs(p_btc):.1%} correlation with vol arb. The alt/meme basket")
        print(f"  ({abs(p_nonbtc):.1%} corr) provides additional diversification.")


if __name__ == "__main__":
    main()
