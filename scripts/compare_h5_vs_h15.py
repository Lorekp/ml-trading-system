"""In-depth comparison of h=5 vs h=15 V2-stack backtests.

Both runs share the full V2 stack: aligner-corrected predictions +
Kelly K1 long-side boost. The horizon is the only intentional axis
of difference. Caveats and known asymmetries:

- h=15 is the production trade log (BTC_v2_trade_log.parquet); used the
  legacy h=15 vol_arb_svi.parquet IV source.
- h=5 (BTC_v2_trade_log_h5m.parquet) used the h=5 native-bucketed
  BTC_vol_arb_svi_h5m.parquet (SVI calibrated at 5-min cadence).
- Both used realistic Deribit 2026 cost widening.

Output sections:
1. Headline NC + compound metrics
2. Per-year breakdown
3. Daily PnL distribution + tails
4. Drawdown analysis
5. Trade-level economics (long vs short, win rate, avg win/loss)
6. Concentration (% trades driving X% PnL)
7. Cost decomposition (slippage, fees, hedge costs)
8. Daily PnL correlation + portfolio analysis
9. Block bootstrap 95% CI on ΔNC PnL (h=5 - h=15)
10. Permutation test under "horizon doesn't matter" null
11. Monthly jackknife stability
"""
from __future__ import annotations

import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CAPITAL = 100_000.0
SEED = 42
H15 = "vol_arb/output/BTC_v2_trade_log.parquet"
H5  = "vol_arb/output/BTC_v2_trade_log_h5m.parquet"


def daily_pnl(tl: pd.DataFrame, col: str = "net_pnl_nc") -> pd.Series:
    t = tl.loc[tl["signal"] != 0]
    return t[col].resample("1D").sum()


def annual_sharpe(daily: pd.Series) -> float:
    a = daily.loc[daily != 0]
    if a.std() == 0 or len(a) < 2:
        return float("nan")
    return float(a.mean() / a.std() * np.sqrt(365))


def max_drawdown(daily: pd.Series, capital: float = CAPITAL) -> tuple[float, float]:
    """Returns (NC dollar MDD, compound % MDD)."""
    eq = daily.cumsum()
    nc_dd = float((eq - eq.cummax()).min())
    cp_eq = daily.cumsum() + capital  # approximate
    cp_dd = float(((cp_eq - cp_eq.cummax()) / cp_eq.cummax()).min())
    return nc_dd, cp_dd * 100


def calmar(daily: pd.Series) -> float:
    eq = daily.cumsum()
    mdd = abs((eq - eq.cummax()).min())
    if mdd == 0:
        return float("nan")
    annual_return = daily.sum() / (len(daily) / 365.0)
    return float(annual_return / mdd)


def headline(tl: pd.DataFrame) -> dict:
    t = tl.loc[tl["signal"] != 0]
    nc_d = daily_pnl(tl, "net_pnl_nc")
    cp_d = daily_pnl(tl, "net_pnl")
    nc_mdd, cp_mdd_pct = max_drawdown(nc_d)
    n_tr = int((t["net_pnl_nc"] != 0).sum())
    nc_pnl = float(t["net_pnl_nc"].sum())
    return dict(
        nc_pnl=nc_pnl,
        cp_pnl=float(t["net_pnl"].sum()),
        nc_sharpe=annual_sharpe(nc_d),
        cp_sharpe=annual_sharpe(cp_d),
        nc_mdd=nc_mdd,
        cp_mdd_pct=cp_mdd_pct,
        nc_calmar=calmar(nc_d),
        nc_worst_day=float(nc_d.min()),
        nc_best_day=float(nc_d.max()),
        n_trades=n_tr,
        per_trade_pnl=nc_pnl / max(n_tr, 1),
        n_days_active=int((nc_d != 0).sum()),
    )


def yearly_breakdown(tl: pd.DataFrame) -> pd.DataFrame:
    t = tl.loc[tl["signal"] != 0]
    nc_d = daily_pnl(tl, "net_pnl_nc")
    rows = []
    for year, group in t.groupby(t.index.year):
        yd = nc_d.loc[nc_d.index.year == year]
        yda = yd.loc[yd != 0]
        rows.append(dict(
            year=int(year),
            nc_pnl=float(group["net_pnl_nc"].sum()),
            n_trades=int((group["net_pnl_nc"] != 0).sum()),
            sharpe=float(yda.mean() / yda.std() * np.sqrt(365)) if yda.std() > 0 else float("nan"),
            mdd=float((yd.cumsum() - yd.cumsum().cummax()).min()),
            worst_day=float(yd.min()),
        ))
    return pd.DataFrame(rows).set_index("year")


def trade_economics(tl: pd.DataFrame) -> dict:
    t = tl.loc[(tl["signal"] != 0) & (tl["net_pnl_nc"] != 0)]
    longs = t.loc[t["signal"] == 1]
    shorts = t.loc[t["signal"] == -1]
    wins = t.loc[t["net_pnl_nc"] > 0]
    losses = t.loc[t["net_pnl_nc"] < 0]
    return dict(
        n_long=len(longs), long_pnl=float(longs["net_pnl_nc"].sum()),
        n_short=len(shorts), short_pnl=float(shorts["net_pnl_nc"].sum()),
        n_wins=len(wins), n_losses=len(losses),
        win_rate=len(wins) / max(len(t), 1),
        avg_win=float(wins["net_pnl_nc"].mean()) if len(wins) else float("nan"),
        avg_loss=float(losses["net_pnl_nc"].mean()) if len(losses) else float("nan"),
        profit_factor=(
            float(wins["net_pnl_nc"].sum() / abs(losses["net_pnl_nc"].sum()))
            if len(losses) and losses["net_pnl_nc"].sum() < 0 else float("nan")
        ),
        # Trade-level Sharpe (per-trade $, annualised by trade rate)
        trade_sharpe=(
            float(t["net_pnl_nc"].mean() / t["net_pnl_nc"].std() * np.sqrt(len(t) / 4.17))
            if t["net_pnl_nc"].std() > 0 else float("nan")
        ),
    )


def concentration_curve(tl: pd.DataFrame) -> dict:
    t = tl.loc[(tl["signal"] != 0) & (tl["net_pnl_nc"] != 0)]
    pnls = t["net_pnl_nc"].abs().sort_values(ascending=False)
    cum = pnls.cumsum() / pnls.sum()
    # Find threshold trades for 50% and 80% of total |PnL|
    n50 = int((cum < 0.5).sum() + 1)
    n80 = int((cum < 0.8).sum() + 1)
    return dict(
        n_trades=len(pnls),
        pct_trades_50pnl=100 * n50 / len(pnls),
        pct_trades_80pnl=100 * n80 / len(pnls),
    )


def cost_decomposition(tl: pd.DataFrame) -> dict:
    t = tl.loc[tl["signal"] != 0]
    return dict(
        gross_pnl=float(t.get("gross_pnl_nc", t["net_pnl_nc"]).sum()),
        net_pnl=float(t["net_pnl_nc"].sum()),
        option_fees=float(t.get("option_fees_nc", t.get("option_fees", pd.Series([0]))).sum()),
        slippage=float(t.get("slippage_nc", t.get("slippage", pd.Series([0]))).sum()),
        hedge_costs=float(t.get("hedge_costs_nc", t.get("hedge_costs", pd.Series([0]))).sum()),
    )


def block_bootstrap_delta(d5: pd.Series, d15: pd.Series, n: int = 1000) -> tuple[float, float, float]:
    """Daily-block bootstrap 95% CI on cumulative ΔNC PnL (h=5 - h=15)."""
    rng = np.random.default_rng(SEED)
    diff = (d5 - d15).reindex(d5.index.union(d15.index), fill_value=0.0)
    n_days = len(diff)
    cum_deltas = []
    for _ in range(n):
        idx = rng.integers(0, n_days, n_days)
        cum_deltas.append(diff.iloc[idx].sum())
    arr = np.sort(np.array(cum_deltas))
    return float(arr[int(0.025 * n)]), float(diff.sum()), float(arr[int(0.975 * n)])


def permutation_test(d5: pd.Series, d15: pd.Series, n: int = 1000) -> float:
    """Under H0 'horizon doesn't matter', shuffle horizon labels per day and
    compute ΔNC. Returns one-sided p-value on observed ΔNC PnL."""
    rng = np.random.default_rng(SEED)
    aligned = pd.concat([d5.rename("h5"), d15.rename("h15")], axis=1).fillna(0.0)
    obs_delta = float(aligned["h5"].sum() - aligned["h15"].sum())
    null_deltas = []
    for _ in range(n):
        flips = rng.integers(0, 2, len(aligned)).astype(bool)
        h5_perm = np.where(flips, aligned["h15"], aligned["h5"])
        h15_perm = np.where(flips, aligned["h5"], aligned["h15"])
        null_deltas.append(h5_perm.sum() - h15_perm.sum())
    null_arr = np.array(null_deltas)
    return float((null_arr >= obs_delta).mean())


def monthly_jackknife(tl5: pd.DataFrame, tl15: pd.DataFrame) -> pd.DataFrame:
    """Drop each month in turn; recompute ΔNC PnL. Returns per-month
    leave-one-out impact."""
    rows = []
    months = sorted({(ts.year, ts.month) for ts in tl5.index} | {(ts.year, ts.month) for ts in tl15.index})
    for y, m in months:
        mask5 = ~((tl5.index.year == y) & (tl5.index.month == m))
        mask15 = ~((tl15.index.year == y) & (tl15.index.month == m))
        nc5 = tl5.loc[mask5 & (tl5["signal"] != 0), "net_pnl_nc"].sum()
        nc15 = tl15.loc[mask15 & (tl15["signal"] != 0), "net_pnl_nc"].sum()
        rows.append(dict(year=y, month=m, delta_nc=float(nc5 - nc15)))
    return pd.DataFrame(rows)


def main() -> int:
    tl5 = pd.read_parquet(H5)
    tl15 = pd.read_parquet(H15)

    print("=" * 100)
    print("h=5m vs h=15m — IN-DEPTH BACKTEST COMPARISON (V2 + aligner + Kelly K1)")
    print("=" * 100)

    # 1. Headline
    h5 = headline(tl5)
    h15 = headline(tl15)
    print("\n[1] HEADLINE METRICS")
    print(f"{'metric':<22}{'h=5m':>15}{'h=15m':>15}{'Δ (h=5 − h=15)':>20}")
    print("-" * 72)
    for k in ["nc_pnl","cp_pnl","nc_sharpe","cp_sharpe","nc_mdd","cp_mdd_pct",
              "nc_calmar","nc_worst_day","nc_best_day","n_trades","per_trade_pnl","n_days_active"]:
        v5, v15 = h5[k], h15[k]
        delta = v5 - v15
        if "pnl" in k or "mdd" in k.replace("pct","") or "day" in k or "_per_" in k:
            print(f"{k:<22}{'$'+format(v5,',.0f'):>15}{'$'+format(v15,',.0f'):>15}{'$'+format(delta,'+,.0f'):>20}")
        elif "pct" in k:
            print(f"{k:<22}{v5:>14.2f}%{v15:>14.2f}%{delta:>+19.2f}pp")
        elif isinstance(v5, int):
            print(f"{k:<22}{v5:>15,}{v15:>15,}{delta:>+20,}")
        else:
            print(f"{k:<22}{v5:>15.3f}{v15:>15.3f}{delta:>+20.3f}")

    # 2. Yearly breakdown
    print("\n[2] PER-YEAR BREAKDOWN — NC dollars")
    yb5 = yearly_breakdown(tl5)
    yb15 = yearly_breakdown(tl15)
    years = sorted(set(yb5.index) | set(yb15.index))
    print(f"{'year':<8}{'h=5 PnL':>14}{'h=15 PnL':>14}{'Δ':>14}{'h=5 Shp':>10}{'h=15 Shp':>10}{'h=5 trades':>12}{'h=15 trades':>12}")
    print("-" * 96)
    for y in years:
        r5  = yb5.loc[y]  if y in yb5.index  else None
        r15 = yb15.loc[y] if y in yb15.index else None
        p5  = r5["nc_pnl"]   if r5  is not None else 0
        p15 = r15["nc_pnl"]  if r15 is not None else 0
        s5  = r5["sharpe"]   if r5  is not None else float("nan")
        s15 = r15["sharpe"]  if r15 is not None else float("nan")
        n5  = r5["n_trades"] if r5  is not None else 0
        n15 = r15["n_trades"] if r15 is not None else 0
        print(f"{y:<8}${p5:>11,.0f} ${p15:>11,.0f} ${p5-p15:>+11,.0f}{s5:>10.2f}{s15:>10.2f}{n5:>12,}{n15:>12,}")

    # 3. Daily PnL distribution
    print("\n[3] DAILY NC PnL DISTRIBUTION")
    d5 = daily_pnl(tl5, "net_pnl_nc")
    d15 = daily_pnl(tl15, "net_pnl_nc")
    print(f"{'pct':<8}{'h=5':>14}{'h=15':>14}")
    print("-" * 36)
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"p{p:<7}${np.percentile(d5, p):>11,.0f}  ${np.percentile(d15, p):>11,.0f}")
    print(f"{'std':<8}${d5.std():>11,.0f}  ${d15.std():>11,.0f}")
    print(f"{'mean':<8}${d5.mean():>11,.0f}  ${d15.mean():>11,.0f}")
    print(f"{'skew':<8}{d5.skew():>14.3f}{d15.skew():>14.3f}")
    print(f"{'kurt':<8}{d5.kurt():>14.3f}{d15.kurt():>14.3f}")

    # 4. Trade economics
    print("\n[4] TRADE-LEVEL ECONOMICS")
    e5 = trade_economics(tl5)
    e15 = trade_economics(tl15)
    print(f"{'metric':<20}{'h=5m':>15}{'h=15m':>15}")
    print("-" * 50)
    for k in ["n_long", "long_pnl", "n_short", "short_pnl", "n_wins", "n_losses",
              "win_rate", "avg_win", "avg_loss", "profit_factor", "trade_sharpe"]:
        v5, v15 = e5[k], e15[k]
        if isinstance(v5, int):
            print(f"{k:<20}{v5:>15,}{v15:>15,}")
        elif "pnl" in k or "win" in k and "rate" not in k or "loss" in k:
            print(f"{k:<20}{'$'+format(v5,',.0f'):>15}{'$'+format(v15,',.0f'):>15}")
        elif "rate" in k:
            print(f"{k:<20}{v5:>14.2%}{v15:>14.2%}")
        else:
            print(f"{k:<20}{v5:>15.3f}{v15:>15.3f}")

    # 5. PnL concentration
    print("\n[5] PnL CONCENTRATION (% of trades driving X% of |PnL|)")
    c5 = concentration_curve(tl5)
    c15 = concentration_curve(tl15)
    print(f"{'metric':<22}{'h=5m':>12}{'h=15m':>12}")
    print(f"{'% trades for 50% PnL':<22}{c5['pct_trades_50pnl']:>11.2f}%{c15['pct_trades_50pnl']:>11.2f}%")
    print(f"{'% trades for 80% PnL':<22}{c5['pct_trades_80pnl']:>11.2f}%{c15['pct_trades_80pnl']:>11.2f}%")

    # 6. Daily correlation
    print("\n[6] DAILY NC PnL CORRELATION")
    df = pd.concat([d5.rename("h5"), d15.rename("h15")], axis=1).fillna(0.0)
    df = df.loc[(df != 0).any(axis=1)]
    rho_p = df["h5"].corr(df["h15"])
    rho_s = spearmanr(df["h5"], df["h15"])
    print(f"  Pearson  ρ = {rho_p:.3f}")
    print(f"  Spearman ρ = {rho_s.statistic:.3f}  (p={rho_s.pvalue:.2e})")

    # 7. Portfolio analysis
    print("\n[7] PORTFOLIO SHARPE ANALYSIS")
    s5_solo = annual_sharpe(d5)
    s15_solo = annual_sharpe(d15)
    print(f"  Solo h=5  Sharpe: {s5_solo:.2f}")
    print(f"  Solo h=15 Sharpe: {s15_solo:.2f}")
    sigma5, sigma15 = d5.std(), d15.std()
    mu5, mu15 = d5.mean(), d15.mean()
    print(f"  {'w15':<6}{'w5':<6}{'Sharpe':<10}{'NC PnL ($)':<15}")
    for w15 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w5 = 1 - w15
        mu_p = w5 * mu5 + w15 * mu15
        var_p = (w5 * sigma5)**2 + (w15 * sigma15)**2 + 2 * w5 * w15 * rho_p * sigma5 * sigma15
        sharpe_p = mu_p / np.sqrt(var_p) * np.sqrt(365) if var_p > 0 else float("nan")
        pnl_p = w5 * d5.sum() + w15 * d15.sum()
        print(f"  {w15:<6.2f}{w5:<6.2f}{sharpe_p:<10.2f}${pnl_p:>13,.0f}")

    # 8. Block bootstrap CI on ΔNC PnL
    print("\n[8] BLOCK BOOTSTRAP CI on ΔNC PnL (h=5 − h=15)")
    print("    Methodology: daily-block bootstrap, 1000 resamples, seed=42")
    lo, obs, hi = block_bootstrap_delta(d5, d15)
    print(f"    Observed ΔNC PnL: ${obs:>+12,.0f}")
    print(f"    95% CI: [${lo:>+,.0f}, ${hi:>+,.0f}]")
    print(f"    LCL > $0? {'YES (significant gain)' if lo > 0 else 'NO (CI straddles zero)'}")

    # 9. Permutation test
    print("\n[9] PERMUTATION TEST (under H0: horizon doesn't matter)")
    print("    1000 random per-day horizon-label flips")
    p = permutation_test(d5, d15)
    print(f"    Observed ΔNC PnL ranks at p = {p:.4f}")
    print(f"    Reject H0 at α=0.05? {'YES' if p < 0.05 else 'NO'}")

    # 10. Monthly jackknife
    print("\n[10] MONTHLY JACKKNIFE STABILITY (drop each month, recompute Δ)")
    jk = monthly_jackknife(tl5, tl15)
    obs_full = float(tl5.loc[tl5["signal"]!=0, "net_pnl_nc"].sum() - tl15.loc[tl15["signal"]!=0, "net_pnl_nc"].sum())
    pos = (jk["delta_nc"] > 0).sum()
    print(f"    Months with delta > 0 after drop: {pos}/{len(jk)}  ({100*pos/len(jk):.1f}%)")
    print(f"    Median jackknife ΔNC PnL: ${jk['delta_nc'].median():>+,.0f}")
    print(f"    Min jackknife ΔNC PnL:    ${jk['delta_nc'].min():>+,.0f}")
    print(f"    Max jackknife ΔNC PnL:    ${jk['delta_nc'].max():>+,.0f}")
    print(f"    Full-sample ΔNC PnL:      ${obs_full:>+,.0f}")
    # Find single-month-drop most damaging
    worst = jk.loc[jk["delta_nc"].idxmin()]
    print(f"    Most fragile drop: {int(worst['year'])}-{int(worst['month']):02d} (drops Δ to ${worst['delta_nc']:+,.0f})")

    # 11. Decision summary
    print("\n[11] DECISION SUMMARY (pre-committed gates)")
    nc_delta = h5["nc_pnl"] - h15["nc_pnl"]
    sh_delta = h5["nc_sharpe"] - h15["nc_sharpe"]
    mdd_delta = h5["nc_mdd"] - h15["nc_mdd"]
    print(f"  ΔNC PnL    = ${nc_delta:>+,.0f}")
    print(f"  ΔSharpe    = {sh_delta:>+.3f}")
    print(f"  ΔMDD       = ${mdd_delta:>+,.0f}")
    print()
    # Plan's gate notation "Δ ≤ -0.3" means "Sharpe drop must be at MOST
    # 0.3" — i.e. delta must stay above -0.3 (delta ≥ -0.3). Same for MDD:
    # "ΔMDD ≤ +$1K" means MDD may worsen by at most $1K (delta ≥ -$1K).
    df_pass = (nc_delta >= 150_000) and (sh_delta >= -0.3) and (mdd_delta >= -1_000)
    sp_pass = (nc_delta >= 50_000)  and (sh_delta >= -0.7) and (mdd_delta >= -3_000)
    print(f"  Default-flip gate (ΔNC≥+$150K, ΔSharpe≥-0.3, ΔMDD≥-$1K): {'PASS' if df_pass else 'FAIL'}")
    print(f"  Ship-as-preset gate (ΔNC≥+$50K, ΔSharpe≥-0.7, ΔMDD≥-$3K): {'PASS' if sp_pass else 'FAIL'}")
    if not df_pass:
        fails = []
        if nc_delta < 150_000: fails.append(f"NC PnL ${nc_delta:+,.0f} < +$150K")
        if sh_delta < -0.3: fails.append(f"Sharpe drop {-sh_delta:.2f} > 0.3")
        if mdd_delta < -1_000: fails.append(f"MDD worsens by ${-mdd_delta:,.0f} > $1K")
        print(f"    Default-flip fails on: {', '.join(fails)}")
    if not sp_pass:
        fails = []
        if nc_delta < 50_000: fails.append(f"NC PnL ${nc_delta:+,.0f} < +$50K")
        if sh_delta < -0.7: fails.append(f"Sharpe drop {-sh_delta:.2f} > 0.7")
        if mdd_delta < -3_000: fails.append(f"MDD worsens by ${-mdd_delta:,.0f} > $3K")
        print(f"    Ship-as-preset fails on: {', '.join(fails)}")
    print()
    print("  Note: Kelly K1 precedent (2026-04-21) accepted a similar Sharpe-for-PnL")
    print("  tradeoff at ΔSharpe -0.94 / +$114K. h=5 is ΔSharpe %.2f / +$%dK." % (sh_delta, nc_delta/1000))
    print("  $/Sharpe-pt ratio: $%dK (vs Kelly K1: $121K)" % (abs(nc_delta) / abs(sh_delta) / 1000))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
