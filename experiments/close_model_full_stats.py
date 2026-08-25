"""Full stats report: ML Close Model backtest on Bybit $100K."""
import sys, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.config import PortfolioConfig

CAPITAL = 100_000.0
RT_BPS = 4.5
PORT_CFG = PortfolioConfig(
    starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.40, base_sizing_pct=0.02, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)


def run_sim(df, capital=CAPITAL, rt_bps=RT_BPS):
    sub = df.copy().sort_values("timestamp").reset_index(drop=True)
    gross = sub["direction"].values * sub["fwd_return"].values
    rt = rt_bps / 10000.0
    net = gross - rt

    edge = np.where(np.isfinite(sub["probability"].values),
                    np.abs(2.0 * sub["probability"].values - 1.0),
                    np.clip(np.abs(sub["signal_strength"].values), 0, 1))
    edge = np.clip(edge, 0, 1)

    eq = capital
    peak = capital
    pnl = np.zeros(len(sub))
    sizes = np.zeros(len(sub))
    for i in range(len(sub)):
        sz = min(eq * 0.02 * edge[i], eq * 0.04)
        pnl[i] = sz * net[i]
        sizes[i] = sz
        eq += pnl[i]
        if eq > peak:
            peak = eq

    ts = pd.to_datetime(sub["timestamp"].values)
    if ts.tz is not None:
        ts = ts.tz_localize(None)

    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    dr = daily / capital
    monthly = daily.resample("MS").sum()

    n_days = len(daily)
    n_cal = (daily.index.max() - daily.index.min()).days if n_days > 1 else 1
    n_years = n_cal / 365.25
    mean_r = dr.mean()
    std_r = dr.std()
    sr = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0
    sortino_d = dr[dr < 0].std()
    sortino = mean_r / sortino_d * np.sqrt(365.25) if sortino_d > 0 else 0
    total_pnl = float(pnl.sum())
    final_eq = capital + total_pnl
    total_ret = total_pnl / capital * 100
    ann_ret = ((final_eq / capital) ** (1 / max(n_years, 0.1)) - 1) * 100

    cum = capital + np.cumsum(pnl)
    rm = np.maximum.accumulate(cum)
    dd = (rm - cum) / rm
    maxdd = float(dd.max()) * 100
    calmar = ann_ret / max(maxdd, 0.01)

    n_trades = len(sub)
    win_mask = pnl > 0
    lose_mask = (pnl < 0) & (pnl != 0)
    wr = win_mask.sum() / (win_mask.sum() + lose_mask.sum()) * 100 if (win_mask.sum() + lose_mask.sum()) > 0 else 0
    pf = abs(pnl[win_mask].sum() / pnl[lose_mask].sum()) if pnl[lose_mask].sum() != 0 else 0
    avg_win = float(pnl[win_mask].mean()) if win_mask.sum() > 0 else 0
    avg_loss = float(pnl[lose_mask].mean()) if lose_mask.sum() > 0 else 0
    avg_bps = float(net.mean() * 10000)
    gross_bps = float(gross.mean() * 10000)

    streak = 0
    mx_streak = 0
    for v in daily.values:
        if v < 0:
            streak += 1
            mx_streak = max(mx_streak, streak)
        else:
            streak = 0

    pos_months = int((monthly > 0).sum())
    neg_months = int((monthly < 0).sum())

    rng = np.random.RandomState(42)
    bsr = np.empty(5000)
    drv = dr.values
    for b in range(5000):
        s = rng.choice(drv, size=len(drv), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci_lo, ci_hi = float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))

    skew = float(dr.skew())
    kurt = float(dr.kurtosis()) + 3.0
    psr = float(norm.cdf(sr * np.sqrt(n_days) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))) * 100

    notional = float(sizes.sum())
    fees = notional * rt / 1  # approximate

    return {
        "sr": sr, "ci": (ci_lo, ci_hi), "sortino": sortino, "calmar": calmar,
        "ann_ret": ann_ret, "maxdd": maxdd, "total_pnl": total_pnl, "final_eq": final_eq,
        "total_ret": total_ret,
        "n_trades": n_trades, "n_days": n_days, "n_cal": n_cal,
        "wr": wr, "pf": pf, "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_bps": avg_bps, "gross_bps": gross_bps,
        "worst_day": float(daily.min()), "worst_month": float(monthly.min()),
        "best_day": float(daily.max()), "best_month": float(monthly.max()),
        "max_streak": mx_streak, "pos_months": pos_months, "neg_months": neg_months,
        "pnl_day": total_pnl / max(n_cal, 1), "pnl_month": total_pnl / max(n_cal / 30.4, 1),
        "psr": psr, "skew": skew, "kurt": kurt,
        "notional": notional, "fees": fees,
        "daily": daily, "monthly": monthly,
    }


def main():
    base_15m = pd.read_parquet("results/signals_cache/logreg_15m.parquet")
    base_4h = pd.read_parquet("results/signals_cache/logreg_4h.parquet")
    base_cs = pd.read_parquet("results/signals_cache/cs_momentum.parquet")
    sel = pd.read_parquet("results/close_cache/selected_logreg_15m_parallel.parquet")

    # Build ML Close variant
    ml = sel.copy()
    ml["fwd_return"] = ml["exit_return"] / ml["direction"]
    ml["probability"] = ml["entry_probability"]
    ml["confidence"] = ml["entry_confidence"]
    ml["signal_strength"] = ml["entry_signal_strength"].abs()
    ml["strategy"] = "logreg_15m"
    ml["hold_bars"] = 4
    if "regime_exposure_mult" not in ml.columns:
        ml["regime_exposure_mult"] = 1.0
    ml["sizing_mult"] = 1.0

    # Baseline
    bl = base_15m.copy()
    bl["strategy"] = "logreg_15m"

    configs = [
        ("BASELINE (fixed hold)", bl),
        ("+ ML CLOSE MODEL", ml),
    ]

    for label, logreg_df in configs:
        merged = pd.concat([logreg_df, base_4h, base_cs], ignore_index=True)
        for col in ["regime_exposure_mult", "sizing_mult"]:
            if col not in merged.columns:
                merged[col] = 1.0
            merged[col] = merged[col].fillna(1.0)
        merged["probability"] = merged["probability"].astype(float)

        r = run_sim(merged)

        print("=" * 110)
        print(f"  {label}")
        print(f"  Capital: ${CAPITAL:,.0f}  |  Exchange: Bybit VIP0  |  Roundtrip: {RT_BPS}bps")
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
  Sharpe Ratio:        {r['sr']:.3f}  [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] 95% CI
  Sortino Ratio:       {r['sortino']:.3f}
  Calmar Ratio:        {r['calmar']:.2f}
  Max Drawdown:        {r['maxdd']:.2f}%  (${CAPITAL * r['maxdd']/100:,.0f})
  PSR:                 {r['psr']:.1f}%
  Skewness:            {r['skew']:.3f}
  Kurtosis:            {r['kurt']:.2f}

  --- TRADE STATISTICS ---
  Total trades:        {r['n_trades']:,}
  Trades / day:        {r['n_trades']/max(r['n_cal'],1):.1f}
  Win rate:            {r['wr']:.1f}%
  Profit factor:       {r['pf']:.2f}
  Avg winning trade:   ${r['avg_win']:.2f}
  Avg losing trade:    ${r['avg_loss']:.2f}
  Avg gross return:    {r['gross_bps']:.2f} bps
  Avg net return:      {r['avg_bps']:.2f} bps

  --- WORST CASE ---
  Worst day:           ${r['worst_day']:,.0f}
  Worst month:         ${r['worst_month']:,.0f}
  Best day:            ${r['best_day']:,.0f}
  Best month:          ${r['best_month']:,.0f}
  Max losing streak:   {r['max_streak']} consecutive days
  Positive months:     {r['pos_months']} / {r['pos_months']+r['neg_months']} ({r['pos_months']/(r['pos_months']+r['neg_months'])*100:.0f}%)
""")

    # Per-strategy breakdown for ML Close
    print("=" * 110)
    print("  PER-STRATEGY (ML CLOSE, Bybit VIP0)")
    print("=" * 110)

    for strat_name, strat_df in [("logreg_15m (ML close)", ml),
                                   ("logreg_4h", base_4h),
                                   ("cs_momentum", base_cs)]:
        r = run_sim(strat_df)
        print(f"  {strat_name:<25}: SR={r['sr']:.3f}  Ann={r['ann_ret']:.1f}%  DD={r['maxdd']:.1f}%  "
              f"Trades={r['n_trades']:,}  Win={r['wr']:.1f}%  PF={r['pf']:.2f}  "
              f"Net={r['avg_bps']:.2f}bps  PnL=${r['total_pnl']:,.0f}")

    # Monthly PnL for ML Close
    print(f"\n{'='*110}")
    print("  MONTHLY PnL (ML CLOSE, last 14 months)")
    print("=" * 110)

    merged_ml = pd.concat([ml, base_4h, base_cs], ignore_index=True)
    for col in ["regime_exposure_mult", "sizing_mult"]:
        if col not in merged_ml.columns:
            merged_ml[col] = 1.0
        merged_ml[col] = merged_ml[col].fillna(1.0)
    merged_ml["probability"] = merged_ml["probability"].astype(float)
    r_ml = run_sim(merged_ml)

    merged_bl = pd.concat([bl, base_4h, base_cs], ignore_index=True)
    for col in ["regime_exposure_mult", "sizing_mult"]:
        if col not in merged_bl.columns:
            merged_bl[col] = 1.0
        merged_bl[col] = merged_bl[col].fillna(1.0)
    merged_bl["probability"] = merged_bl["probability"].astype(float)
    r_bl = run_sim(merged_bl)

    recent_ml = r_ml["monthly"].iloc[-14:]
    recent_bl = r_bl["monthly"].iloc[-14:]
    print(f"{'Month':<10} {'Baseline':>12} {'ML Close':>12} {'Delta':>12}")
    print("-" * 48)
    for idx in recent_ml.index:
        m = idx.strftime("%Y-%m")
        bl_val = recent_bl.get(idx, 0)
        ml_val = recent_ml.get(idx, 0)
        print(f"{m:<10} ${bl_val:>11,.0f} ${ml_val:>11,.0f} ${ml_val-bl_val:>+11,.0f}")


if __name__ == "__main__":
    main()
