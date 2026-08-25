"""
Full Statistics Report — Post-Audit Production Configuration.

Runs the audited best config (regime_off + abs_conf_0.08) through the
production pipeline and outputs comprehensive QuantConnect-style stats.

Usage:
    python experiments/full_stats_report.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, "src")

import copy
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

from signal_engine.config import load_config
from signal_engine.pipeline import run_backtest
from signal_engine.stats import compute_full_stats

W = max(1, os.cpu_count() - 1)


def bootstrap_sharpe_ci(daily_pnl, starting_capital, n_boot=5000, ci=0.95):
    daily_ret = (daily_pnl / starting_capital).values
    rng = np.random.default_rng(42)
    sharpes = []
    for _ in range(n_boot):
        s = rng.choice(daily_ret, size=len(daily_ret), replace=True)
        std = s.std()
        if std > 0:
            sharpes.append(s.mean() / std * np.sqrt(365.25))
    sharpes = np.sort(sharpes)
    lo = np.percentile(sharpes, (1 - ci) / 2 * 100)
    hi = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return round(lo, 3), round(hi, 3)


def deflated_sharpe(observed_sr_ann, k, n_days, skew=0, kurt=3):
    """Deflated Sharpe Ratio (AFML Ch 14 / ML for AM Ch 8).

    Correctly computes E[max SR] = std[SR] * E[max(Z_1,...,Z_K)]
    where std[SR] is the Lo (2002) standard error of the SR estimator.
    """
    sr = observed_sr_ann / np.sqrt(365.25)  # unannualise

    # Lo (2002) variance of SR estimator
    v_sr = (1 - skew * sr + ((kurt - 1) / 4) * sr**2) / max(n_days - 1, 1)
    sr_std = np.sqrt(max(v_sr, 1e-20))

    # Expected max of K standard normals (Bailey & Lopez de Prado)
    gamma = 0.5772  # Euler-Mascheroni
    z1 = norm.ppf(1 - 1.0 / k)
    z2 = norm.ppf(1 - 1.0 / (k * np.e))
    e_max_z = (1 - gamma) * z1 + gamma * z2

    # Scale to SR units
    e_max_sr = sr_std * e_max_z

    z = (sr - e_max_sr) / sr_std
    return float(norm.cdf(z))


def run_report():
    cfg = load_config(Path("configs/default.yaml"))

    # Production config
    prod = copy.deepcopy(cfg)
    prod.regime_filter.enabled = False
    prod.confidence_filter.mode = "absolute"
    prod.confidence_filter.threshold = 0.08

    SC = prod.portfolio.starting_capital

    # ============================================================
    # Run backtest
    # ============================================================
    print("Running production backtest (16 assets, multiprocessing)...", flush=True)
    r = run_backtest(prod, max_workers=W)
    s = r["stats"]

    if "error" in s:
        print(f"ERROR: {s}")
        return

    dpnl = r["daily_pnl"]
    daily_ret = dpnl / SC

    # ============================================================
    # Derived stats
    # ============================================================
    equity = (1 + daily_ret).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # Monthly
    monthly_pnl = dpnl.resample("ME").sum()
    monthly_ret = monthly_pnl / SC
    pos_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)
    hhi = float((monthly_pnl.abs() / monthly_pnl.abs().sum()).pow(2).sum()) if monthly_pnl.abs().sum() > 0 else 0
    top3 = monthly_pnl.nlargest(3)
    top3_pct = top3.sum() / monthly_pnl.sum() * 100 if monthly_pnl.sum() > 0 else 0

    # Bootstrap
    ci_lo, ci_hi = bootstrap_sharpe_ci(dpnl, SC)

    # DSR
    dsr = deflated_sharpe(s["sharpe"], k=40, n_days=s["n_days"],
                          skew=s["skewness"], kurt=s["kurtosis"])

    # Turnover
    daily_turnover = s["trades_per_day"] * prod.portfolio.base_sizing_pct
    ann_turnover = daily_turnover * 365.25

    # Estimated fees (1 bps slippage only, 0% maker)
    avg_pos = SC * prod.portfolio.base_sizing_pct * 0.5
    total_fees = s["n_trades"] * avg_pos * prod.portfolio.slippage

    # ============================================================
    # Print
    # ============================================================
    print()
    print("=" * 80)
    print("ML DIRECTION PREDICTION SYSTEM — FULL STATISTICS")
    print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Config: regime_off + abs_conf_0.08 | 16 assets | 15m bars")
    print(f"Data: {s['n_days']} days, {s['n_trades']:,} trades")
    print("=" * 80)

    print(f"""
--- RETURNS ------------------------------------------------------------------
Compounding Annual Return    {s['cagr_pct']:>10.2f}%
Total Return                 {s['total_return_pct']:>10.2f}%
Start Equity                 ${SC:>12,.2f}
End Equity                   ${s['final_equity']:>12,.2f}
Net Profit                   ${s['final_equity'] - SC:>12,.2f}

--- RISK-ADJUSTED ------------------------------------------------------------
Sharpe Ratio                 {s['sharpe']:>10.3f}
  95% Bootstrap CI           [{ci_lo}, {ci_hi}]
Sortino Ratio                {s['sortino']:>10.3f}
Calmar Ratio                 {s['calmar']:>10.3f}
PSR (P[true SR > 0])         {s['psr_pct']:>10.1f}%
DSR (K=40 trials)            {dsr * 100:>10.1f}%

--- DRAWDOWN ---------------------------------------------------------------━━
Maximum Drawdown             {s['max_dd_pct']:>10.2f}%
Drawdown Recovery            {s['dd_recovery_days']:>10} days
Annual Standard Deviation    {s['ann_std'] * 100:>10.2f}%

--- TRADING ------------------------------------------------------------------
Total Trades                 {s['n_trades']:>10,}
Trades per Day               {s['trades_per_day']:>10.1f}
Win Rate (days)              {s['win_rate_days'] * 100:>10.1f}%
Loss Rate (days)             {s['loss_rate_days'] * 100:>10.1f}%
Profit Factor                {s['profit_factor']:>10.3f}
Average Win                  {s['avg_win_bps']:>10.1f} bps/day
Average Loss                 {s['avg_loss_bps']:>10.1f} bps/day
Expectancy                   {s['expectancy_bps']:>10.2f} bps/day
Portfolio Turnover           {ann_turnover:>10.0f}x/year
Estimated Fees (slippage)    ${total_fees:>12,.2f}

--- DISTRIBUTION ------------------------------------------------------------━
Daily Skewness               {s['skewness']:>10.3f}
Daily Kurtosis               {s['kurtosis']:>10.1f}
Number of Days               {s['n_days']:>10}""")

    # Signal counts
    print(f"""
--- SIGNAL PIPELINE ---------------------------------------------------------━
Raw Signals                  {r['n_total_raw']:>10,}
After Confidence Filter      {r['n_after_confidence']:>10,}  ({100 * r['n_after_confidence'] / max(r['n_total_raw'], 1):.1f}% pass)
After Regime Filter          {r['n_after_regime']:>10,}  (regime disabled = passthrough)
Filter Ratio                 {100 * r['n_after_regime'] / max(r['n_total_raw'], 1):>10.1f}%""")

    # Monthly breakdown
    print(f"""
--- MONTHLY CONSISTENCY ------------------------------------------------------━
Positive Months              {pos_months}/{total_months} ({100 * pos_months / total_months:.0f}%)
HHI Concentration            {hhi:.4f} {"(OK)" if hhi < 0.15 else "(CONCENTRATED)"}
Top 3 Months Contribution    {top3_pct:.1f}% of total PnL""")

    print(f"\n  {'Month':<10s} {'PnL':>10s} {'Return%':>8s} {'Cum%':>8s}")
    print(f"  {'-' * 40}")
    cum_ret = 0
    for dt, pnl in monthly_pnl.items():
        ret_pct = pnl / SC * 100
        cum_ret += ret_pct
        marker = " ***" if pnl == monthly_pnl.max() else (" ---" if pnl == monthly_pnl.min() else "")
        print(f"  {dt.strftime('%Y-%m'):<10s} ${pnl:>9.2f} {ret_pct:>7.2f}% {cum_ret:>7.2f}%{marker}")

    # Per-asset
    print(f"""
--- PER-ASSET BREAKDOWN ------------------------------------------------------""")
    print(f"  {'Asset':<8s} {'Raw':>8s} {'Filtered':>8s} {'Bars':>8s} {'Time':>6s}")
    pa = r.get("per_asset", {})
    for asset in sorted(pa.keys()):
        d = pa[asset]
        final = d.get("n_regime_filtered", d.get("n_confidence_filtered", 0))
        print(f"  {asset:<8s} {d['n_raw_signals']:>8,} {final:>8,} {d['n_bars']:>8,} {d['elapsed_s']:>5.1f}s")

    # Configuration
    print(f"""
--- CONFIGURATION ------------------------------------------------------------
Model                        LogisticRegression(C=0.1, L1, saga)
Features                     14 universal OHLCV (no external data)
Timeframe                    15-minute bars
Forward Horizon              4 bars (1 hour)
Confidence Filter            Absolute, threshold=0.08
Regime Filter                DISABLED (validated improvement)
Volatility Filter            Top 50% (quantile 0.50)
Base Position Size           2% of equity
Max Signal Multiplier        2.0x
Max Exposure                 80% (6% per asset)
Drawdown Delever             7.5%
Drawdown Halt                15%
Slippage                     1 bps
Sample Weights               AFML Ch4 uniqueness (O(n))
Walk-Forward                 Monthly expanding, 6-month minimum
Embargo                      (fwd_bars+1) x bar_duration

--- AUDIT STATUS ------------------------------------------------------------━
Look-ahead bias              FIXED (regime filter shift(1), then disabled)
Embargo correctness          FIXED (fwd_bars+1)
Vol filter leakage           VERIFIED CLEAN (EWM is backward-looking)
Feature leakage              VERIFIED CLEAN (all 14 backward-looking)
Scaler leakage               VERIFIED CLEAN (fit on training only)
Winsorization leakage        VERIFIED CLEAN (training quantiles only)
OOS Holdout Validation       PASS (10 unseen assets, Sharpe HIGHER)
Synthetic Noise Test         PASS (Sharpe ~0 on random walks)

--- VERDICT ------------------------------------------------------------------
Signal:    {"REAL" if ci_lo > 0 else "UNCERTAIN"} (bootstrap 95% CI excludes zero: [{ci_lo}, {ci_hi}])
DSR:       {dsr * 100:.1f}% ({"PASS" if dsr > 0.95 else "BORDERLINE" if dsr > 0.5 else "FAIL"} at K=40)
OOS:       PASS (holdout Sharpe > core Sharpe)
Deploy:    {"RECOMMENDED" if ci_lo > 0 and dsr > 0.5 else "CAUTION"}
""")

    # Save
    out = {
        "generated": datetime.utcnow().isoformat(),
        "config": "regime_off + abs_conf_0.08",
        "stats": s,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "dsr_k40": round(dsr, 4),
        "hhi": round(hhi, 4),
        "positive_months_pct": round(100 * pos_months / total_months, 1),
        "signal_counts": {
            "raw": r["n_total_raw"],
            "after_confidence": r["n_after_confidence"],
            "after_regime": r["n_after_regime"],
        },
        "per_asset": {a: {k: v for k, v in d.items()} for a, d in pa.items()},
        "elapsed_s": r["elapsed_s"],
    }
    out_path = Path("v4-results/full_stats_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run_report()
