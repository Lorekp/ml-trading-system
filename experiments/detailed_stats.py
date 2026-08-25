#!/usr/bin/env python
"""Detailed stats for the optimized portfolio config."""

import sys, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
PORT = PortfolioConfig(
    starting_capital=CAPITAL, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.50, base_sizing_pct=0.04, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)
VOL_INV = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                            low_pct=0.20, high_pct=0.80,
                            low_mult=1.25, mid_mult=1.00, high_mult=0.50)


def main():
    # Load and merge
    sigs = []
    for name in ["logreg_15m", "logreg_4h", "cs_momentum"]:
        sigs.append(pd.read_parquet(f"results/signals_cache/{name}.parquet"))
    merged = pd.concat(sigs, ignore_index=True)
    for c in ["regime_exposure_mult", "sizing_mult", "hold_bars"]:
        if c not in merged.columns:
            merged[c] = 4 if c == "hold_bars" else 1.0
        merged[c] = merged[c].fillna(4 if c == "hold_bars" else 1.0)
    merged["probability"] = merged["probability"].astype(float)

    # Confidence filter
    merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Vol overlay
    btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
    btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
    if btc_daily.index.tz is not None:
        btc_daily.index = btc_daily.index.tz_localize(None)
    merged = apply_vol_overlay(merged, btc_daily, VOL_INV)

    # Simulate
    sim = simulate_portfolio(merged, PORT)
    dpnl = compute_daily_pnl_series(sim)
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Bybit cost
    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean())
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    daily_cost = total_extra / max(len(dpnl), 1)
    dpnl = dpnl - daily_cost

    dr = dpnl / CAPITAL
    eq = (1 + dr).cumprod()
    dd_series = (eq.cummax() - eq) / eq.cummax()

    # Core metrics
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25))
    calmar = float(dr.mean() * 365.25 / dd_series.max()) if dd_series.max() > 0 else 0
    max_dd = float(dd_series.max() * 100)
    cagr = float((eq.iloc[-1]) ** (365.25 / len(dr)) - 1) * 100

    # Tail risk
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    cvar1 = float(-dr[dr <= dr.quantile(0.01)].mean() * 10000) if len(dr) > 100 else 0
    var5 = float(-dr.quantile(0.05) * 10000)
    var1 = float(-dr.quantile(0.01) * 10000)
    skew = float(dr.skew())
    kurt = float(dr.kurtosis() + 3)

    # Daily
    win_days = int((dpnl > 0).sum())
    lose_days = int((dpnl < 0).sum())
    win_rate = win_days / max(win_days + lose_days, 1) * 100
    avg_win = float(dpnl[dpnl > 0].mean())
    avg_loss = float(dpnl[dpnl < 0].mean())
    pf = float(dpnl[dpnl > 0].sum() / abs(dpnl[dpnl < 0].sum())) if (dpnl < 0).any() else 99

    # Streaks
    mx_lose = mx_win = cur_l = cur_w = 0
    for v in dpnl.values:
        if v < 0:
            cur_l += 1; mx_lose = max(mx_lose, cur_l); cur_w = 0
        else:
            cur_w += 1; mx_win = max(mx_win, cur_w); cur_l = 0

    # Monthly
    monthly = dpnl.resample("MS").sum()
    pos_m = int((monthly > 0).sum())
    neg_m = int((monthly < 0).sum())
    monthly_sr = float(monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() > 0 else 0

    # Weekly
    weekly = dpnl.resample("W").sum()

    # DD recovery
    dd_idx = dd_series.idxmax()
    post = eq.loc[dd_idx:]
    recovery = post[post >= eq.cummax().loc[dd_idx]]
    dd_recovery = (recovery.index[0] - dd_idx).days if len(recovery) > 0 else -1

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(5000)
    drv = dr.values
    for b in range(5000):
        s = rng.choice(drv, size=len(drv), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    # PSR
    sr_unann = sr / np.sqrt(365.25)
    denom_inner = (1 - skew * sr_unann + ((kurt - 1) / 4) * sr_unann ** 2) / max(len(dr) - 1, 1)
    psr = norm.cdf(sr_unann / np.sqrt(max(denom_inner, 1e-10))) * 100

    n_cal = (dpnl.index.max() - dpnl.index.min()).days
    years = n_cal / 365.25

    print("=" * 70)
    print("  OPTIMIZED PORTFOLIO -- DETAILED STATS")
    print("  conf >= 0.08 | inverted vol | 4% sizing | Bybit VIP0")
    print("=" * 70)

    print(f"""
  PERIOD
    Calendar days:         {n_cal:,} ({years:.1f} years)
    Active trading days:   {len(dpnl):,}
    Trades executed:       {n_trades:,} ({n_trades/n_cal:.1f}/day)

  RETURNS
    Total PnL:             ${dpnl.sum():,.0f}
    CAGR:                  {cagr:.1f}%
    PnL / day:             ${dpnl.mean():,.1f}
    PnL / month:           ${monthly.mean():,.0f}
    Final equity:          ${CAPITAL + dpnl.sum():,.0f}

  RISK-ADJUSTED
    Sharpe Ratio:          {sr:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}] 95% CI
    Sortino Ratio:         {sortino:.3f}
    Calmar Ratio:          {calmar:.2f}
    Monthly Sharpe:        {monthly_sr:.3f}
    PSR (SR > 0):          {psr:.1f}%

  DRAWDOWN
    Max Drawdown:          {max_dd:.2f}%  (${CAPITAL * max_dd / 100:,.0f})
    DD Recovery:           {dd_recovery} days
    Current DD:            {dd_series.iloc[-1]*100:.2f}%

  TAIL RISK
    VaR 5% (daily):        {var5:.1f} bps
    CVaR 5% (daily):       {cvar5:.1f} bps
    VaR 1% (daily):        {var1:.1f} bps
    CVaR 1% (daily):       {cvar1:.1f} bps
    Skewness:              {skew:.3f}
    Kurtosis:              {kurt:.1f}

  DAILY
    Win rate:              {win_rate:.1f}%
    Avg winning day:       ${avg_win:,.0f}
    Avg losing day:        ${avg_loss:,.0f}
    Profit factor:         {pf:.2f}
    Best day:              ${dpnl.max():,.0f}
    Worst day:             ${dpnl.min():,.0f}
    Max win streak:        {mx_win} days
    Max lose streak:       {mx_lose} days

  WEEKLY
    Best week:             ${weekly.max():,.0f}
    Worst week:            ${weekly.min():,.0f}

  MONTHLY
    Positive months:       {pos_m}/{pos_m+neg_m} ({pos_m/(pos_m+neg_m)*100:.0f}%)
    Best month:            ${monthly.max():,.0f}
    Worst month:           ${monthly.min():,.0f}
    Avg month:             ${monthly.mean():,.0f}

  COSTS
    Total Bybit fees:      ${total_extra:,.0f} (${daily_cost:.0f}/day)
    Avg trade size:        ${avg_sz:,.0f}
    Fee drag:              {total_extra/(dpnl.sum()+total_extra)*100:.1f}% of gross PnL
""")

    # Year-by-year
    print("  YEAR-BY-YEAR")
    print(f"  {'Year':<6} {'SR':>7} {'Sortino':>8} {'DD':>6} {'PnL':>11} {'Days':>5}")
    print("  " + "-" * 44)
    for year in sorted(dpnl.index.year.unique()):
        y = dpnl[dpnl.index.year == year]
        if len(y) > 10:
            dr_y = y / CAPITAL
            sr_y = float(dr_y.mean() / dr_y.std() * np.sqrt(365.25)) if dr_y.std() > 0 else 0
            eq_y = (1 + dr_y).cumprod()
            dd_y = float(((eq_y.cummax() - eq_y) / eq_y.cummax()).max() * 100)
            down_y = dr_y[dr_y < 0]
            sort_y = float(dr_y.mean() / down_y.std() * np.sqrt(365.25)) if len(down_y) > 0 and down_y.std() > 0 else 0
            print(f"  {year:<6} {sr_y:>7.3f} {sort_y:>8.3f} {dd_y:>5.1f}% ${y.sum():>9,.0f} {len(y):>5}")

    # Per-asset
    print(f"\n  PER-ASSET (by PnL contribution)")
    print(f"  {'Asset':<8} {'PnL':>11} {'Trades':>7}")
    print("  " + "-" * 28)
    active = ~sim["skipped"]
    assets_arr = merged["asset"].values if "asset" in merged.columns else np.full(len(merged), "SINGLE")
    asset_pnl = {}
    asset_n = {}
    for i in range(len(merged)):
        if active[i]:
            a = assets_arr[i]
            asset_pnl[a] = asset_pnl.get(a, 0) + sim["pnl"][i]
            asset_n[a] = asset_n.get(a, 0) + 1
    for a, p in sorted(asset_pnl.items(), key=lambda x: -x[1]):
        print(f"  {a:<8} ${p:>9,.0f} {asset_n.get(a,0):>7,}")

    # Per-strategy
    print(f"\n  PER-STRATEGY")
    strats = merged["strategy"].values if "strategy" in merged.columns else np.full(len(merged), "default")
    strat_pnl = {}
    strat_n = {}
    for i in range(len(merged)):
        if active[i]:
            s = strats[i]
            strat_pnl[s] = strat_pnl.get(s, 0) + sim["pnl"][i]
            strat_n[s] = strat_n.get(s, 0) + 1
    for s, p in sorted(strat_pnl.items(), key=lambda x: -x[1]):
        print(f"  {s:<15} PnL=${p:>10,.0f}  trades={strat_n.get(s,0):>7,}")


if __name__ == "__main__":
    main()
