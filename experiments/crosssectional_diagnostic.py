"""Diagnostic: Cross-Sectional Mean Reversion vs Momentum in Crypto

Quick focused test:
1. Signal direction (MR vs momentum) per period
2. BTC-hedged vs raw L/S spreads
3. Vol-normalized residuals
4. Period-specific analysis (pre-2024 vs post-2024)
5. Momentum direction test (flip the signal)
"""

import sys, os, time, warnings
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")


def load_daily_panel(min_bars=400):
    """Load all assets, return daily close panel + returns panel."""
    closes = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        try:
            ohlcv_1m = load_binance_1m(d.name, [DATA_DIR])
            daily = resample_ohlcv(ohlcv_1m, "1D")
            if len(daily) < min_bars:
                continue
            closes[d.name] = daily["close"]
        except Exception:
            continue
    close_panel = pd.DataFrame(closes)
    ret_panel = close_panel.pct_change()
    return close_panel, ret_panel


def rolling_beta_panel(ret_panel, btc_col="BTCUSDT", window=60):
    """Compute rolling beta for all assets vs BTC. Returns DataFrame."""
    btc_ret = ret_panel[btc_col]
    btc_var = btc_ret.rolling(window, min_periods=30).var()
    betas = {}
    for col in ret_panel.columns:
        if col == btc_col:
            continue
        cov = ret_panel[col].rolling(window, min_periods=30).cov(btc_ret)
        betas[col] = (cov / btc_var.replace(0, np.nan)).clip(-3, 5)
    return pd.DataFrame(betas)


def residual_panel(ret_panel, beta_panel, btc_col="BTCUSDT"):
    """Compute residual = r_i - beta_i * r_btc."""
    btc_ret = ret_panel[btc_col]
    resids = {}
    for col in beta_panel.columns:
        resids[col] = ret_panel[col] - beta_panel[col] * btc_ret
    return pd.DataFrame(resids)


def analyze_signal_per_period(
    resid_panel, ret_panel, btc_ret,
    periods, lookbacks=[3, 5, 7, 10],
    fwd_days=[3, 5, 7],
    top_n=30,
    direction="reversion",  # "reversion" or "momentum"
):
    """Analyze cross-sectional signal per time period."""
    results = []

    for period_label, start, end in periods:
        mask = (resid_panel.index >= pd.Timestamp(start, tz="UTC")) & \
               (resid_panel.index <= pd.Timestamp(end, tz="UTC"))
        dates = resid_panel.index[mask]

        for lb in lookbacks:
            cum_resid = resid_panel.rolling(lb, min_periods=max(2, lb-1)).sum()

            for fd in fwd_days:
                spreads = []
                spreads_hedged = []
                n_obs = 0

                for date in dates:
                    row = cum_resid.loc[date].dropna()
                    if len(row) < 12:
                        continue

                    # Rank
                    ranked = row.sort_values()
                    n_leg = max(3, len(row) // 5)

                    if direction == "reversion":
                        # Long oversold (bottom), short overbought (top)
                        longs = ranked.index[:n_leg]
                        shorts = ranked.index[-n_leg:]
                    else:
                        # Momentum: long overbought (top), short oversold (bottom)
                        longs = ranked.index[-n_leg:]
                        shorts = ranked.index[:n_leg]

                    # Forward returns
                    future_slice = ret_panel.loc[date:].iloc[1:fd+1]
                    if len(future_slice) < fd - 1:
                        continue

                    long_fwd = future_slice[longs].mean(axis=1).sum()
                    short_fwd = future_slice[shorts].mean(axis=1).sum()
                    btc_fwd = btc_ret.loc[date:].iloc[1:fd+1].sum()

                    raw_spread = long_fwd - short_fwd
                    # BTC-hedged: subtract market from each leg, then spread
                    hedged_spread = (long_fwd - btc_fwd) - (short_fwd - btc_fwd)
                    # Hedged spread = raw spread (BTC cancels in L-S)

                    spreads.append(raw_spread)
                    n_obs += 1

                if n_obs < 20:
                    continue

                arr = np.array(spreads)
                mean_s = arr.mean()
                std_s = arr.std()
                t = mean_s / (std_s / np.sqrt(n_obs)) if std_s > 0 else 0
                sharpe = mean_s / std_s * np.sqrt(365.25 / fd) if std_s > 0 else 0
                wr = np.mean(arr > 0) * 100

                results.append({
                    "period": period_label,
                    "lookback": lb,
                    "fwd_days": fd,
                    "n_obs": n_obs,
                    "spread_bps": mean_s * 10000,
                    "t_stat": t,
                    "sharpe": sharpe,
                    "win_rate": wr,
                })

    return pd.DataFrame(results)


def quick_ls_pnl(
    resid_panel, ret_panel, start, end,
    lookback=5, rebal_days=7, n_leg=6, direction="reversion",
    leverage=1.0, starting_capital=10000.0,
):
    """Quick L/S portfolio PnL computation."""
    dates = ret_panel.index[
        (ret_panel.index >= pd.Timestamp(start, tz="UTC")) &
        (ret_panel.index <= pd.Timestamp(end, tz="UTC"))
    ]
    cum_resid = resid_panel.rolling(lookback, min_periods=max(2, lookback-1)).sum()

    equity = starting_capital
    equity_curve = [equity]
    positions = {}
    last_rebal = -rebal_days

    for i, date in enumerate(dates):
        # PnL from held positions
        day_pnl = 0.0
        for sym, w in positions.items():
            if sym in ret_panel.columns and date in ret_panel.index:
                r = ret_panel.loc[date, sym]
                if not np.isnan(r):
                    day_pnl += equity * leverage * w * r
        equity += day_pnl

        # Rebalance
        if i - last_rebal >= rebal_days:
            row = cum_resid.loc[date].dropna()
            if len(row) >= 2 * n_leg:
                ranked = row.sort_values()
                if direction == "reversion":
                    longs = ranked.index[:n_leg]
                    shorts = ranked.index[-n_leg:]
                else:
                    longs = ranked.index[-n_leg:]
                    shorts = ranked.index[:n_leg]

                new_pos = {}
                for s in longs:
                    new_pos[s] = 1.0 / n_leg
                for s in shorts:
                    new_pos[s] = -1.0 / n_leg

                # Transaction costs
                turnover = sum(
                    abs(new_pos.get(s, 0) - positions.get(s, 0))
                    for s in set(list(new_pos.keys()) + list(positions.keys()))
                )
                equity -= 0.001 * turnover * leverage * equity  # 10 bps

                positions = new_pos
                last_rebal = i

        equity_curve.append(equity)

        # DD halt
        peak = max(equity_curve)
        if (equity - peak) / peak < -0.15:
            break

    final = equity_curve[-1]
    pnl = final - starting_capital
    peak = max(equity_curve)
    dd = min((e - peak) / peak for e, peak in zip(equity_curve, [max(equity_curve[:i+1]) for i in range(len(equity_curve))]))
    daily_rets = np.diff(equity_curve) / np.array(equity_curve[:-1])
    worst_day = abs(min(daily_rets)) if len(daily_rets) > 0 else 0
    sharpe = 0
    if len(daily_rets) > 5 and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(365.25)

    return {
        "pnl": pnl,
        "sharpe": sharpe,
        "max_dd": abs(dd),
        "worst_day": worst_day,
        "n_days": len(equity_curve) - 1,
    }


def main():
    t0 = time.time()
    print("=" * 100)
    print("CROSS-SECTIONAL RESIDUAL DIAGNOSTIC")
    print("=" * 100)

    print("\nLoading daily panel...", end=" ", flush=True)
    close_panel, ret_panel = load_daily_panel()
    btc_ret = ret_panel["BTCUSDT"]
    print(f"{len(ret_panel.columns)} assets, {len(ret_panel)} days loaded in {time.time()-t0:.0f}s")

    # Compute betas and residuals
    print("Computing betas and residuals...", end=" ", flush=True)
    beta_panel = rolling_beta_panel(ret_panel, window=60)
    resid_panel = residual_panel(ret_panel, beta_panel)
    print("done")

    # Periods for analysis
    periods = [
        ("2021-22", "2021-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024-H1", "2024-01-01", "2024-06-30"),
        ("2024-H2", "2024-07-01", "2024-12-31"),
        ("2025-H1", "2025-01-01", "2025-06-30"),
        ("2025-H2", "2025-07-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-02-28"),
        ("Full", "2021-01-01", "2026-02-28"),
    ]

    # ---- Test 1: Mean Reversion (original hypothesis) ----
    print("\n" + "=" * 80)
    print("TEST 1: RESIDUAL MEAN REVERSION (long oversold, short overbought)")
    print("=" * 80)

    mr_results = analyze_signal_per_period(
        resid_panel, ret_panel, btc_ret, periods,
        lookbacks=[5, 10], fwd_days=[5, 7], direction="reversion"
    )
    if len(mr_results) > 0:
        # Best config per period
        best = mr_results.loc[mr_results.groupby("period")["t_stat"].idxmax()]
        print(f"\n{'Period':<10s} {'LB':>3s} {'Fwd':>3s} {'N':>5s} {'Spread(bps)':>12s} "
              f"{'t-stat':>8s} {'Sharpe':>8s} {'Win%':>6s}")
        print("-" * 65)
        for _, row in best.iterrows():
            t = row['t_stat']
            sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.65 else ''))
            print(f"{row['period']:<10s} {int(row['lookback']):>3d} {int(row['fwd_days']):>3d} "
                  f"{int(row['n_obs']):>5d} {row['spread_bps']:>12.1f} "
                  f"{t:>7.2f}{sig} {row['sharpe']:>7.2f} {row['win_rate']:>5.1f}%")

    # ---- Test 2: Momentum (flip direction) ----
    print("\n" + "=" * 80)
    print("TEST 2: RESIDUAL MOMENTUM (long outperformers, short underperformers)")
    print("=" * 80)

    mom_results = analyze_signal_per_period(
        resid_panel, ret_panel, btc_ret, periods,
        lookbacks=[5, 10], fwd_days=[5, 7], direction="momentum"
    )
    if len(mom_results) > 0:
        best = mom_results.loc[mom_results.groupby("period")["t_stat"].idxmax()]
        print(f"\n{'Period':<10s} {'LB':>3s} {'Fwd':>3s} {'N':>5s} {'Spread(bps)':>12s} "
              f"{'t-stat':>8s} {'Sharpe':>8s} {'Win%':>6s}")
        print("-" * 65)
        for _, row in best.iterrows():
            t = row['t_stat']
            sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.65 else ''))
            print(f"{row['period']:<10s} {int(row['lookback']):>3d} {int(row['fwd_days']):>3d} "
                  f"{int(row['n_obs']):>5d} {row['spread_bps']:>12.1f} "
                  f"{t:>7.2f}{sig} {row['sharpe']:>7.2f} {row['win_rate']:>5.1f}%")

    # ---- Test 3: Vol-Normalized Residuals ----
    print("\n" + "=" * 80)
    print("TEST 3: VOL-NORMALIZED RESIDUALS (z-score of residual)")
    print("=" * 80)

    # Normalize residuals by rolling std
    resid_vol = resid_panel.rolling(30, min_periods=15).std()
    resid_z = resid_panel / resid_vol.replace(0, np.nan)

    vnorm_results = analyze_signal_per_period(
        resid_z, ret_panel, btc_ret, periods,
        lookbacks=[5, 10], fwd_days=[5, 7], direction="reversion"
    )
    if len(vnorm_results) > 0:
        best = vnorm_results.loc[vnorm_results.groupby("period")["t_stat"].idxmax()]
        print(f"\n{'Period':<10s} {'LB':>3s} {'Fwd':>3s} {'N':>5s} {'Spread(bps)':>12s} "
              f"{'t-stat':>8s} {'Sharpe':>8s} {'Win%':>6s}")
        print("-" * 65)
        for _, row in best.iterrows():
            t = row['t_stat']
            sig = '***' if abs(t) > 2.58 else ('**' if abs(t) > 1.96 else ('*' if abs(t) > 1.65 else ''))
            print(f"{row['period']:<10s} {int(row['lookback']):>3d} {int(row['fwd_days']):>3d} "
                  f"{int(row['n_obs']):>5d} {row['spread_bps']:>12.1f} "
                  f"{t:>7.2f}{sig} {row['sharpe']:>7.2f} {row['win_rate']:>5.1f}%")

    # ---- Test 4: Quick P&L for both directions ----
    print("\n" + "=" * 80)
    print("TEST 4: WALK-FORWARD P&L (both directions)")
    print("=" * 80)

    windows = [
        ("W1", "2024-07-01", "2025-02-28", 245),
        ("W2", "2025-03-01", "2025-10-31", 244),
        ("W3", "2025-11-01", "2026-02-28", 122),
    ]

    for direction_label, direction in [("REVERSION", "reversion"), ("MOMENTUM", "momentum")]:
        print(f"\n  Direction: {direction_label}")
        print(f"  {'Window':<6s} {'PnL':>8s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} {'Days':>5s}")
        print(f"  " + "-" * 55)

        total_pnl = 0
        total_cal = 0
        max_dd = 0
        max_wd = 0

        for wl, start, end, cal in windows:
            r = quick_ls_pnl(resid_panel, ret_panel, start, end,
                            lookback=5, rebal_days=7, n_leg=6,
                            direction=direction, leverage=1.0)
            pnl_cal = r["pnl"] / cal
            total_pnl += r["pnl"]
            total_cal += cal
            max_dd = max(max_dd, r["max_dd"])
            max_wd = max(max_wd, r["worst_day"])
            print(f"  {wl:<6s} ${r['pnl']:>7.0f} ${pnl_cal:>6.2f} {r['sharpe']:>7.2f} "
                  f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% {r['n_days']:>5d}")

        if total_cal > 0:
            ok = max_dd < 0.10 and max_wd < 0.05
            print(f"  {'TOTAL':<6s} ${total_pnl:>7.0f} ${total_pnl/total_cal:>6.2f} "
                  f"{'':>7s} {max_dd*100:>5.1f}% {max_wd*100:>5.2f}%  "
                  f"{'OK' if ok else 'FAIL'}")

    # ---- Test 5: Higher leverage (if base works) ----
    print("\n" + "=" * 80)
    print("TEST 5: LEVERAGE SCALING (best direction)")
    print("=" * 80)

    # Pick whichever direction had higher total PnL
    for lev in [1.0, 2.0, 3.0, 5.0]:
        for direction in ["reversion", "momentum"]:
            total_pnl = 0
            total_cal = 0
            max_dd = 0
            max_wd = 0
            for wl, start, end, cal in windows:
                r = quick_ls_pnl(resid_panel, ret_panel, start, end,
                                lookback=5, rebal_days=7, n_leg=6,
                                direction=direction, leverage=lev)
                total_pnl += r["pnl"]
                total_cal += cal
                max_dd = max(max_dd, r["max_dd"])
                max_wd = max(max_wd, r["worst_day"])

            if total_cal > 0:
                ok = max_dd < 0.10 and max_wd < 0.05
                label = f"{direction[:3].upper()} {lev:.0f}x"
                print(f"  {label:<12s} ${total_pnl/total_cal:>7.2f}/cal  "
                      f"DD={max_dd*100:.1f}%  WD={max_wd*100:.2f}%  "
                      f"{'OK' if ok else 'FAIL'}")

    # ---- Correlation analysis ----
    print("\n" + "=" * 80)
    print("CORRELATION: AVERAGE PAIRWISE CORRELATION PER PERIOD")
    print("=" * 80)

    for label, start, end in periods:
        mask = (ret_panel.index >= pd.Timestamp(start, tz="UTC")) & \
               (ret_panel.index <= pd.Timestamp(end, tz="UTC"))
        period_rets = ret_panel[mask].dropna(axis=1, how="all")
        if len(period_rets) < 30 or period_rets.shape[1] < 10:
            continue
        corr_matrix = period_rets.corr()
        # Average pairwise correlation (excluding diagonal)
        n = len(corr_matrix)
        avg_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))
        print(f"  {label:<10s} avg_corr={avg_corr:.3f}  n_assets={n}")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")

    # ---- Verdict ----
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("""
Key questions answered:
1. Does residual mean reversion exist? → Check t-stats above
2. Does residual momentum exist? → Check t-stats above
3. Is it profitable after costs? → Check P&L above
4. Does it meet risk constraints? → Check DD/WD above
5. Is it regime-dependent? → Compare across periods
""")


if __name__ == "__main__":
    main()
