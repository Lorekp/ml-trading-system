"""Full deep stats report for the EXACT homeserver production config."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "src")

import warnings, os, json, time
import numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path
warnings.filterwarnings("ignore")

from signal_engine.config import load_config
from signal_engine.pipeline import run_backtest


def main():
    W = max(1, os.cpu_count() - 1)
    cfg = load_config(Path("configs/default.yaml"))

    # EXACT homeserver production config
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08
    cfg.portfolio.base_sizing_pct = 0.04
    cfg.portfolio.slippage = 0.0005  # 5 bps
    cfg.portfolio.max_exposure = 0.90
    cfg.portfolio.max_per_asset = 0.03
    cfg.portfolio.kelly_sizing = True  # Per-asset Kelly weighting
    cfg.assets = {
        "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
        "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
        "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
        "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
        "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
        "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
        "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
        "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
    }

    SC = cfg.portfolio.starting_capital
    print("Running 30-asset production backtest...", flush=True)
    r = run_backtest(cfg, max_workers=W)
    s = r["stats"]
    dpnl = r["daily_pnl"]
    pa = r.get("per_asset", {})

    daily_ret = dpnl / SC
    equity = (1 + daily_ret).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(5000):
        sample = rng.choice(daily_ret.values, size=len(daily_ret), replace=True)
        std = sample.std()
        if std > 0:
            boot.append(sample.mean() / std * np.sqrt(365.25))
    ci_lo = round(np.percentile(boot, 2.5), 3)
    ci_hi = round(np.percentile(boot, 97.5), 3)

    # DSR
    sr = s["sharpe"] / np.sqrt(365.25)
    v_sr = (1 - s["skewness"] * sr + ((s["kurtosis"] - 1) / 4) * sr ** 2) / max(s["n_days"] - 1, 1)
    sr_std = np.sqrt(max(v_sr, 1e-20))
    gamma = 0.5772
    e_max_sr = sr_std * ((1 - gamma) * norm.ppf(1 - 1 / 40) + gamma * norm.ppf(1 - 1 / (40 * np.e)))
    dsr = float(norm.cdf((sr - e_max_sr) / sr_std))

    # Monthly
    monthly_pnl = dpnl.resample("ME").sum()
    pos_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)
    hhi = float((monthly_pnl.abs() / monthly_pnl.abs().sum()).pow(2).sum()) if monthly_pnl.abs().sum() > 0 else 0
    top3 = monthly_pnl.nlargest(3)
    top3_pct = top3.sum() / monthly_pnl.sum() * 100 if monthly_pnl.sum() > 0 else 0

    # DD detail
    max_dd_idx = drawdown.idxmin()
    dd_start = running_max[:max_dd_idx].idxmax() if len(running_max[:max_dd_idx]) > 0 else None
    post_dd = equity.loc[max_dd_idx:]
    recovery = post_dd[post_dd >= running_max.loc[max_dd_idx]]
    recovery_date = recovery.index[0] if len(recovery) > 0 else None
    dd_duration = (max_dd_idx - dd_start).days if dd_start else 0
    dd_recovery = (recovery_date - max_dd_idx).days if recovery_date else -1

    # Underwater
    uw_periods = []
    in_dd = False
    dd_s = None
    for t, v in (drawdown < -0.001).items():
        if v and not in_dd:
            dd_s = t
            in_dd = True
        elif not v and in_dd:
            uw_periods.append((t - dd_s).days)
            in_dd = False
    avg_uw = np.mean(uw_periods) if uw_periods else 0
    max_uw = max(uw_periods) if uw_periods else 0

    # Tail risk
    var_95 = np.percentile(daily_ret, 5) * 100
    cvar_95 = daily_ret[daily_ret <= np.percentile(daily_ret, 5)].mean() * 100 if len(daily_ret) > 20 else 0

    # Streaks
    max_win = max_loss = cur = 0
    for v in (daily_ret > 0):
        cur = cur + 1 if v else 0
        max_win = max(max_win, cur)
    cur = 0
    for v in (daily_ret <= 0):
        cur = cur + 1 if v else 0
        max_loss = max(max_loss, cur)

    # Turnover
    ann_turnover = s["trades_per_day"] * cfg.portfolio.base_sizing_pct * 365.25
    pnl_day = (s["final_equity"] - SC) / max(s["n_days"], 1)

    # Weekly / quarterly Sharpe
    weekly_ret = dpnl.resample("W").sum() / SC
    weekly_sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if weekly_ret.std() > 0 else 0
    quarterly_ret = dpnl.resample("QE").sum() / SC
    quarterly_sharpe = quarterly_ret.mean() / quarterly_ret.std() * np.sqrt(4) if quarterly_ret.std() > 0 else 0

    print()
    print("=" * 80)
    print("FULL PRODUCTION STATS REPORT")
    print(f"Config: 30 assets | regime off | abs conf 0.08 | 4% sizing | 5 bps slip")
    print(f"Data: {s['n_days']} days, {s['n_trades']:,} trades")
    print("=" * 80)

    print(f"""
--- RETURNS -------------------------------------------------------------------
Compounding Annual Return    {s['cagr_pct']:>10.2f}%
Total Return                 {s['total_return_pct']:>10.2f}%
Start Equity                 ${SC:>12,.2f}
End Equity                   ${s['final_equity']:>12,.2f}
Net Profit                   ${s['final_equity']-SC:>12,.2f}
PnL per Day                  ${pnl_day:>12.2f}

--- RISK-ADJUSTED (multi-horizon) ---------------------------------------------
Sharpe (daily)               {s['sharpe']:>10.3f}
  95% Bootstrap CI           [{ci_lo}, {ci_hi}]
Sharpe (weekly)              {weekly_sharpe:>10.3f}
Sharpe (quarterly)           {quarterly_sharpe:>10.3f}
Sortino Ratio                {s['sortino']:>10.3f}
Calmar Ratio                 {s['calmar']:>10.3f}
PSR (P[true SR > 0])         {s['psr_pct']:>10.1f}%
DSR (K=40 trials)            {dsr*100:>10.1f}%

--- DRAWDOWN ------------------------------------------------------------------
Maximum Drawdown             {s['max_dd_pct']:>10.2f}%
DD Peak-to-Trough            {dd_duration:>10} days
DD Recovery                  {dd_recovery:>10} days
Avg Underwater Period        {avg_uw:>10.1f} days
Max Underwater Period        {max_uw:>10} days
Annual Std Dev               {s['ann_std']*100:>10.2f}%

--- TAIL RISK -----------------------------------------------------------------
VaR (95%, daily)             {var_95:>10.4f}%
CVaR (95%, daily)            {cvar_95:>10.4f}%
Daily Skewness               {s['skewness']:>10.3f}
Daily Kurtosis               {s['kurtosis']:>10.1f}

--- TRADING -------------------------------------------------------------------
Total Trades                 {s['n_trades']:>10,}
Trades per Day               {s['trades_per_day']:>10.1f}
Win Rate (days)              {s['win_rate_days']*100:>10.1f}%
Profit Factor                {s['profit_factor']:>10.3f}
Average Win                  {s['avg_win_bps']:>10.1f} bps/day
Average Loss                 {s['avg_loss_bps']:>10.1f} bps/day
Expectancy                   {s['expectancy_bps']:>10.2f} bps/day
Max Consecutive Wins         {max_win:>10} days
Max Consecutive Losses       {max_loss:>10} days
Portfolio Turnover           {ann_turnover:>10.0f}x/year

--- SIGNAL PIPELINE -----------------------------------------------------------
Raw Signals                  {r['n_total_raw']:>10,}
After Confidence Filter      {r['n_after_confidence']:>10,}  ({100*r['n_after_confidence']/max(r['n_total_raw'],1):.1f}% pass)
Filter Ratio                 {100*r['n_after_regime']/max(r['n_total_raw'],1):>10.1f}%

--- MONTHLY CONSISTENCY -------------------------------------------------------
Positive Months              {pos_months}/{total_months} ({100*pos_months/total_months:.0f}%)
HHI Concentration            {hhi:.4f} {"(OK)" if hhi < 0.15 else "(CONCENTRATED)"}
Top 3 Months Contribution    {top3_pct:.1f}% of total PnL
Best Month                   ${monthly_pnl.max():.2f} ({monthly_pnl.idxmax().strftime("%Y-%m")})
Worst Month                  ${monthly_pnl.min():.2f} ({monthly_pnl.idxmin().strftime("%Y-%m")})""")

    print(f"\n  {'Month':<10s} {'PnL':>10s} {'Ret%':>8s} {'Cum%':>8s}")
    print(f"  {'-'*40}")
    cum = 0
    for dt, pnl in monthly_pnl.items():
        ret = pnl / SC * 100
        cum += ret
        mark = " ***" if pnl == monthly_pnl.max() else (" ---" if pnl == monthly_pnl.min() else "")
        print(f"  {dt.strftime('%Y-%m'):<10s} ${pnl:>9.2f} {ret:>7.2f}% {cum:>7.2f}%{mark}")

    print(f"\n--- PER-ASSET BREAKDOWN -------------------------------------------------------")
    print(f"  {'Asset':<8s} {'Raw':>8s} {'Filtered':>8s} {'Bars':>8s} {'Time':>6s}")
    for asset in sorted(pa.keys()):
        d = pa[asset]
        final = d.get("n_regime_filtered", d.get("n_confidence_filtered", 0))
        print(f"  {asset:<8s} {d['n_raw_signals']:>8,} {final:>8,} {d['n_bars']:>8,} {d['elapsed_s']:>5.1f}s")

    print(f"""
--- CONFIGURATION -------------------------------------------------------------
Model                        LogReg L1 C=0.1 saga (sklearn {__import__('sklearn').__version__})
Features                     14 universal OHLCV
Timeframe                    15m bars, 4-bar (1h) horizon
Confidence Filter            Absolute >= 0.08
Regime Filter                DISABLED
Base Sizing                  4% of equity (prop firm conservative)
Max Exposure                 90% (3% per asset, 30 assets)
Slippage                     5 bps (conservative)
Delever Start / Halt         7.5% / 15%
Embargo                      (fwd_bars+1) x bar_duration
Assets                       {len(cfg.assets)} (volume-ranked, no meme coins)

--- VERDICT -------------------------------------------------------------------
Signal:    {"REAL" if ci_lo > 0 else "UNCERTAIN"} (CI [{ci_lo}, {ci_hi}])
DSR:       {dsr*100:.1f}%
Deploy:    {"RECOMMENDED" if ci_lo > 0 and dsr > 0.5 else "CAUTION"}
""")

    # Save
    out = {
        "config": "homeserver_production_30assets_5bps_4pct",
        "stats": s,
        "bootstrap_ci": [ci_lo, ci_hi],
        "dsr": round(dsr, 4),
        "weekly_sharpe": round(weekly_sharpe, 3),
        "quarterly_sharpe": round(quarterly_sharpe, 3),
        "var_95_pct": round(var_95, 4),
        "cvar_95_pct": round(cvar_95, 4),
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
        "avg_underwater_days": round(avg_uw, 1),
        "max_underwater_days": max_uw,
        "hhi": round(hhi, 4),
        "monthly_positive_pct": round(100 * pos_months / total_months, 1),
        "pnl_per_day": round(pnl_day, 2),
    }
    with open("v4-results/baseline_before_tier1.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved to v4-results/baseline_before_tier1.json")


if __name__ == "__main__":
    main()
