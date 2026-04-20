"""Compare NC and Compound PnL across all production + overlay configs.

Configs:
- baseline V2             (backtester: BTC_v2_trade_log_baseline.parquet)
- V2 + aligner (bias)     (backtester: BTC_v2_trade_log_aligned_bias.parquet)
- V2 + throttle (sc=0.5)  (post-processor, compound approximate)
- V2 + DFL A1 filter-25   (backtester: BTC_v2_trade_log_dfl_A1_backtester.parquet)

Notes:
- Baseline, aligner, DFL A1 are proper backtester runs (compound exact).
- Throttle is post-processor only; compound is approximate.
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')


CONFIGS = [
    ("V2 baseline (pre-aligner)", "vol_arb/output/BTC_v2_trade_log_baseline.parquet", "backtester"),
    ("V2 + aligner(bias)", "vol_arb/output/BTC_v2_trade_log_aligned_bias.parquet", "backtester"),
    ("V2 + throttle(sc=0.5)", "vol_arb/output/BTC_v2_trade_log_throttled_static.parquet", "post-processor"),
    ("V2 + DFL A1 filter-25", "vol_arb/output/BTC_v2_trade_log_dfl_A1_backtester.parquet", "backtester"),
    # NEW DEFAULT (2026-04-21): V2 + aligner + Kelly K1 long-side boost.
    ("** DEFAULT: V2+aligner+KellyK1 **", "vol_arb/output/BTC_v2_trade_log.parquet", "backtester"),
]


def metrics(path: str, capital: float = 100_000.0) -> dict:
    """NC and compound metrics from a trade log."""
    tl = pd.read_parquet(path)
    t = tl.loc[tl['signal'] != 0]
    # NC
    nc_daily = t['net_pnl_nc'].resample('1D').sum()
    nc_daily_active = nc_daily.loc[nc_daily != 0]
    nc_eq = nc_daily.cumsum()
    nc_dd = (nc_eq - nc_eq.cummax()).min()
    nc_sharpe = (
        nc_daily_active.mean() / nc_daily_active.std() * np.sqrt(365)
        if nc_daily_active.std() > 0 else float('nan')
    )
    # Compound
    cp_daily = t['net_pnl'].resample('1D').sum()
    cp_eq = cp_daily.cumsum() + capital
    cp_dd_series = (cp_eq - cp_eq.cummax()) / cp_eq.cummax()
    cp_dd = cp_dd_series.min()
    cp_daily_active = cp_daily.loc[cp_daily != 0]
    cp_sharpe = (
        cp_daily_active.mean() / cp_daily_active.std() * np.sqrt(365)
        if cp_daily_active.std() > 0 else float('nan')
    )
    # Profitable months (compound)
    cp_monthly = cp_daily.resample('ME').sum()
    pct_prof = (cp_monthly > 0).sum() / len(cp_monthly) if len(cp_monthly) > 0 else float('nan')
    return dict(
        nc_pnl=float(t['net_pnl_nc'].sum()),
        nc_sharpe=float(nc_sharpe),
        nc_max_dd=float(nc_dd),
        nc_worst_day=float(nc_daily.min()),
        cp_pnl=float(t['net_pnl'].sum()),
        cp_sharpe=float(cp_sharpe),
        cp_max_dd_pct=float(cp_dd * 100),
        cp_worst_day=float(cp_daily.min()),
        pct_profitable_months=float(pct_prof * 100),
        n_trades=int((t['net_pnl_nc'] != 0).sum()),
    )


def main():
    rows = []
    for name, path, source in CONFIGS:
        m = metrics(path)
        m['config'] = name
        m['source'] = source
        rows.append(m)
    df = pd.DataFrame(rows).set_index('config')

    # Print side-by-side
    print("=" * 118)
    print("ALL CONFIGS — NC and COMPOUND PnL comparison")
    print("=" * 118)
    print(f"{'config':<28} {'src':<14} {'NC PnL':>12} {'NC Shp':>8} {'NC MDD$':>10} "
          f"{'CP PnL':>13} {'CP Shp':>8} {'CP MDD%':>8} {'%ProfM':>8} {'nT':>6}")
    print("-" * 118)
    for name, path, source in CONFIGS:
        m = df.loc[name]
        print(f"{name:<28} {source:<14} "
              f"${m['nc_pnl']:>11,.0f} {m['nc_sharpe']:>8.2f} ${m['nc_max_dd']:>9,.0f} "
              f"${m['cp_pnl']:>12,.0f} {m['cp_sharpe']:>8.2f} {m['cp_max_dd_pct']:>7.2f}% "
              f"{m['pct_profitable_months']:>7.1f}% {m['n_trades']:>6}")

    # Delta vs baseline
    print()
    print("=" * 118)
    print("DELTA vs baseline V2")
    print("=" * 118)
    base = df.loc["V2 baseline (pre-aligner)"]
    print(f"{'config':<28} {'dNC PnL$':>12} {'dNC %':>8} {'dNC Shp':>9} "
          f"{'dCP PnL$':>13} {'dCP %':>8} {'dCP Shp':>9} {'dCP MDD pp':>12}")
    print("-" * 118)
    for name, path, source in CONFIGS:
        m = df.loc[name]
        d_nc_pnl = m['nc_pnl'] - base['nc_pnl']
        d_nc_pct = 100 * d_nc_pnl / base['nc_pnl']
        d_nc_shp = m['nc_sharpe'] - base['nc_sharpe']
        d_cp_pnl = m['cp_pnl'] - base['cp_pnl']
        d_cp_pct = 100 * d_cp_pnl / base['cp_pnl']
        d_cp_shp = m['cp_sharpe'] - base['cp_sharpe']
        d_cp_mdd = m['cp_max_dd_pct'] - base['cp_max_dd_pct']
        print(f"{name:<28} ${d_nc_pnl:>+11,.0f} {d_nc_pct:>+7.1f}% {d_nc_shp:>+9.2f} "
              f"${d_cp_pnl:>+12,.0f} {d_cp_pct:>+7.1f}% {d_cp_shp:>+9.2f} {d_cp_mdd:>+11.2f}pp")

    # Save CSV
    df.to_csv("vol_arb/output/ALL_CONFIGS_COMPARISON.csv")
    print(f"\nSaved: vol_arb/output/ALL_CONFIGS_COMPARISON.csv")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
