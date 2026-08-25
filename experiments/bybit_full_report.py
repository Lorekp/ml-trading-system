"""Full stat report: 3-strategy portfolio, $100K, Bybit VIP0 maker fees."""

import sys, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

CAPITAL = 100_000.0
MAKER_FEE_BPS = 2.0
SPREAD_BPS = 0.5
RT_COST = (SPREAD_BPS + 2 * MAKER_FEE_BPS) / 10000.0  # 4.5bps


def run_sim(sigs, capital=CAPITAL, rt_cost=RT_COST):
    sub = sigs.copy()
    gross_ret = sub["direction"].values * sub["fwd_return"].values
    net_ret = gross_ret - rt_cost

    edge = np.where(
        np.isfinite(sub["probability"].values),
        np.abs(2.0 * sub["probability"].values - 1.0),
        np.clip(np.abs(sub["signal_strength"].values), 0, 1),
    )
    edge = np.clip(edge, 0, 1)

    equity = capital
    peak = capital
    pnl_arr = np.zeros(len(sub))
    size_arr = np.zeros(len(sub))

    for i in range(len(sub)):
        size = equity * 0.02 * edge[i]
        size = min(size, equity * 0.04)
        pnl_arr[i] = size * net_ret[i]
        size_arr[i] = size
        equity += pnl_arr[i]
        if equity > peak:
            peak = equity

    ts = pd.to_datetime(sub["timestamp"].values)
    if hasattr(ts, "tz") and ts.tz is not None:
        ts = ts.tz_localize(None)

    daily = pd.Series(pnl_arr, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    daily_ret = daily / capital
    monthly = daily.resample("MS").sum()

    n_days = len(daily)
    n_cal = (daily.index.max() - daily.index.min()).days if n_days > 1 else 1
    n_years = n_cal / 365.25
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    sr = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0
    sortino_d = daily_ret[daily_ret < 0].std()
    sortino = mean_r / sortino_d * np.sqrt(365.25) if sortino_d > 0 else 0
    total_pnl = float(pnl_arr.sum())
    final_eq = capital + total_pnl
    total_ret = total_pnl / capital * 100
    ann_ret = ((final_eq / capital) ** (1 / max(n_years, 0.1)) - 1) * 100

    cum_eq = capital + np.cumsum(pnl_arr)
    rm = np.maximum.accumulate(cum_eq)
    dd = (rm - cum_eq) / rm
    max_dd = float(dd.max()) * 100
    calmar = ann_ret / max(max_dd, 0.01)

    n_trades = len(sub)
    win_mask = pnl_arr > 0
    n_win = int(win_mask.sum())
    win_rate = n_win / n_trades * 100
    avg_win = float(pnl_arr[win_mask].mean()) if n_win > 0 else 0
    lose_mask = ~win_mask & (pnl_arr != 0)
    avg_loss = float(pnl_arr[lose_mask].mean()) if lose_mask.sum() > 0 else 0
    pf = abs(pnl_arr[win_mask].sum() / pnl_arr[lose_mask].sum()) if pnl_arr[lose_mask].sum() != 0 else 0
    avg_bps = float(net_ret.mean() * 10000)
    tpd = n_trades / max(n_cal, 1)

    worst_day = float(daily.min())
    worst_week = float(daily.rolling(5).sum().min()) if n_days >= 5 else worst_day
    worst_month = float(monthly.min()) if len(monthly) > 0 else 0
    best_day = float(daily.max())
    best_month = float(monthly.max()) if len(monthly) > 0 else 0

    cvar5 = float(daily.quantile(0.05))
    streak = 0
    max_streak = 0
    for v in daily.values:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(10000)
    dr = daily_ret.values
    for b in range(10000):
        s = rng.choice(dr, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci_lo, ci_hi = float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))

    skew = float(daily_ret.skew())
    kurt = float(daily_ret.kurtosis()) + 3.0
    psr = float(norm.cdf(sr * np.sqrt(n_days) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))) * 100
    sr_std = np.sqrt((1 + 0.5 * sr ** 2 - skew * sr + (kurt - 3) / 4 * sr ** 2) / n_days)
    euler = 0.5772156649
    exp_max = sr_std * ((1 - euler) * norm.ppf(1 - 1 / 7) + euler * norm.ppf(1 - 1 / (7 * np.e)))
    dsr = float(norm.cdf((sr - exp_max) / max(sr_std, 1e-6))) * 100

    notional = float(size_arr.sum())
    fees = notional * rt_cost
    gross_pnl = float((pnl_arr + size_arr * rt_cost).sum())
    fee_pct = fees / max(abs(gross_pnl), 1) * 100

    return {
        "sr": sr, "sortino": sortino, "calmar": calmar,
        "total_pnl": total_pnl, "final_eq": final_eq,
        "total_ret": total_ret, "ann_ret": ann_ret, "max_dd": max_dd,
        "n_trades": n_trades, "n_days": n_days, "n_cal": n_cal,
        "win_rate": win_rate, "pf": pf, "avg_bps": avg_bps, "tpd": tpd,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "worst_day": worst_day, "worst_week": worst_week, "worst_month": worst_month,
        "best_day": best_day, "best_month": best_month,
        "cvar5": cvar5, "max_streak": max_streak,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "psr": psr, "dsr": dsr,
        "skew": skew, "kurt": kurt,
        "notional": notional, "fees": fees, "fee_pct": fee_pct,
        "pnl_day": total_pnl / max(n_cal, 1),
        "pnl_month": total_pnl / max(n_cal / 30.4, 1),
        "daily": daily, "monthly": monthly,
    }


def main():
    # Load signals
    sigs = []
    for name in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        df = pd.read_parquet(f"results/signals_cache/{name}.parquet")
        sigs.append(df)
    merged = pd.concat(sigs, ignore_index=True)

    r = run_sim(merged)

    print("=" * 110)
    print("  FULL PORTFOLIO STAT REPORT")
    print(f"  Capital: ${CAPITAL:,.0f}  |  Exchange: Bybit VIP0  |  Execution: PostOnly (maker)")
    print(f"  Maker fee: {MAKER_FEE_BPS}bps  |  Spread: {SPREAD_BPS}bps  |  Roundtrip: {RT_COST*10000:.1f}bps")
    print("=" * 110)

    print(f"""
  Period:              {r['n_cal']:,} cal days ({r['n_cal']/365.25:.1f} years)
  Active trading days: {r['n_days']:,}
  Starting capital:    ${CAPITAL:,.0f}
  Final equity:        ${r['final_eq']:,.0f}

  --- RETURNS ---
  Total return:        {r['total_ret']:.1f}%
  Annualized return:   {r['ann_ret']:.1f}%
  Total PnL:           ${r['total_pnl']:,.0f}
  PnL / calendar day:  ${r['pnl_day']:.2f}
  PnL / month:         ${r['pnl_month']:,.0f}

  --- RISK-ADJUSTED ---
  Sharpe Ratio:        {r['sr']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] 95% CI
  Sortino Ratio:       {r['sortino']:.3f}
  Calmar Ratio:        {r['calmar']:.2f}
  Max Drawdown:        {r['max_dd']:.2f}%  (${CAPITAL * r['max_dd']/100:,.0f})
  CVaR (5%):           ${r['cvar5']:,.0f} / day

  --- STATISTICAL VALIDITY ---
  PSR:                 {r['psr']:.1f}%
  Deflated SR (7 trials): {r['dsr']:.1f}%
  Skewness:            {r['skew']:.3f}
  Kurtosis:            {r['kurt']:.2f}

  --- TRADE STATISTICS ---
  Total trades:        {r['n_trades']:,}
  Trades / day:        {r['tpd']:.1f}
  Win rate:            {r['win_rate']:.1f}%
  Profit factor:       {r['pf']:.2f}
  Avg winning trade:   ${r['avg_win']:.2f}
  Avg losing trade:    ${r['avg_loss']:.2f}
  Avg trade return:    {r['avg_bps']:.2f} bps (net of costs)

  --- EXECUTION COSTS ---
  Roundtrip cost:      {RT_COST*10000:.1f} bps
  Total notional:      ${r['notional']:,.0f}
  Total fees paid:     ${r['fees']:,.0f}
  Fees as % of gross:  {r['fee_pct']:.1f}%

  --- WORST CASE ---
  Worst day:           ${r['worst_day']:,.0f}
  Worst week:          ${r['worst_week']:,.0f}
  Worst month:         ${r['worst_month']:,.0f}
  Best day:            ${r['best_day']:,.0f}
  Best month:          ${r['best_month']:,.0f}
  Max losing streak:   {r['max_streak']} consecutive days
""")

    # Per-strategy
    print("=" * 110)
    print("  PER-STRATEGY BREAKDOWN")
    print("=" * 110)

    hdr = f"{'Strategy':<16} {'SR':>7} {'Ann%':>7} {'DD%':>6} {'Trades':>9} {'Win%':>6} {'PF':>5} {'bps':>7} {'PnL':>11} {'$/day':>8}"
    print(hdr)
    print("-" * len(hdr))

    for sn in ["logreg_15m", "logreg_4h", "cs_momentum", "COMBINED"]:
        if sn == "COMBINED":
            s = r
        else:
            s = run_sim(merged[merged["strategy"] == sn])
        print(f"{sn:<16} {s['sr']:>7.3f} {s['ann_ret']:>6.1f}% {s['max_dd']:>5.1f}% "
              f"{s['n_trades']:>9,} {s['win_rate']:>5.1f}% {s['pf']:>5.2f} "
              f"{s['avg_bps']:>7.2f} ${s['total_pnl']:>10,.0f} ${s['pnl_day']:>7.2f}")

    # Correlation
    print(f"\n{'='*110}")
    print("  CORRELATION MATRIX")
    print("=" * 110)

    strat_pnls = {}
    for sn in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        s = run_sim(merged[merged["strategy"] == sn])
        strat_pnls[sn] = s["daily"]
    print(pd.DataFrame(strat_pnls).fillna(0).corr().round(4).to_string())

    # Monthly last 14
    print(f"\n{'='*110}")
    print("  MONTHLY PnL (last 14 months)")
    print("=" * 110)

    monthly_all = {}
    for sn in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        s = run_sim(merged[merged["strategy"] == sn])
        monthly_all[sn] = s["monthly"]
    monthly_all["COMBINED"] = r["monthly"]
    mdf = pd.DataFrame(monthly_all).fillna(0)

    recent = mdf.iloc[-14:]
    print(f"{'Month':<10} {'15m':>10} {'4H':>10} {'CS_Mom':>10} {'COMBINED':>10}")
    print("-" * 54)
    for idx, row in recent.iterrows():
        m = idx.strftime("%Y-%m")
        print(f"{m:<10} ${row.get('logreg_15m',0):>9,.0f} ${row.get('logreg_4h',0):>9,.0f} "
              f"${row.get('cs_momentum',0):>9,.0f} ${row.get('COMBINED',0):>9,.0f}")

    # Annual
    print(f"\n{'='*110}")
    print("  ANNUAL PnL")
    print("=" * 110)
    yearly = mdf.resample("YE").sum()
    for idx, row in yearly.iterrows():
        y = idx.strftime("%Y")
        print(f"  {y}: 15m=${row.get('logreg_15m',0):>9,.0f}  "
              f"4H=${row.get('logreg_4h',0):>9,.0f}  "
              f"CS=${row.get('cs_momentum',0):>9,.0f}  "
              f"COMBINED=${row.get('COMBINED',0):>9,.0f}")


if __name__ == "__main__":
    main()
