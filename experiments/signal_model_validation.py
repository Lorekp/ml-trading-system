"""HISTORICAL EXPERIMENT — its deployment verdict is not current evidence.

Signal-model diagnostic suite retained for provenance. Any `validated` or
`deployment` text emitted below predates the current validation ledger.

Loads cached combined signals from the walk-forward report and runs:
1. Per-asset Sharpe decomposition
2. Monthly rolling Sharpe stability
3. Cost sensitivity curve
4. Holdout stress tests (leave-one-out)
5. PBO (Probability of Backtest Overfitting)
6. Year-over-year persistence
7. Rank IC per month
8. HHI concentration
9. Bootstrap CI on holdout Sharpe
10. DSR / PSR summary
"""

import sys

sys.path.insert(0, "src")

import glob
import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from afml.evaluation import deflated_sharpe_ratio, hhi_concentration, rank_ic
from signal_engine.config import PortfolioConfig, load_config
from signal_engine.portfolio import compute_daily_pnl_series, simulate_portfolio
from signal_engine.stats import compute_full_stats

warnings.filterwarnings("ignore")

SEARCH_START = "2022-01"
SEARCH_END = "2024-12"
HOLDOUT_START = "2025-01"
SCOPE = "a30_08d04ff1"
CACHE_DIR = Path("experiments/cache/ml_close_walkforward/combined") / SCOPE
BEST_LOOKBACK = 2  # from report: lb=2 had best search Sharpe


def load_signals(lookback: int) -> pd.DataFrame:
    pattern = str(CACHE_DIR / f"lb{lookback}_v10_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No cached signals for lb={lookback}: {pattern}")
    return pd.read_parquet(files[0])


def filter_period(df, start, end=None):
    ts = pd.to_datetime(df["timestamp"], utc=True)
    months = ts.dt.strftime("%Y-%m")
    mask = months >= start
    if end:
        mask &= months <= end
    return df.loc[mask].reset_index(drop=True)


def quick_sharpe(dpnl, capital):
    if dpnl is None or len(dpnl) < 5:
        return 0.0
    dr = dpnl / capital
    std = dr.std()
    return float(dr.mean() / std * np.sqrt(365.25)) if std > 0 else 0.0


def simulate(signals, pcfg):
    if len(signals) < 10:
        return None, None
    sim = simulate_portfolio(signals, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None, None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None, None
    return sim, dpnl


def main():
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = PortfolioConfig(**cfg.portfolio.model_dump())
    capital = pcfg.starting_capital

    print("=" * 78)
    print("SIGNAL MODEL VALIDATION REPORT")
    print("=" * 78)
    print(f"  Lookback: {BEST_LOOKBACK}mo | Scope: {SCOPE}")
    print(f"  Search: {SEARCH_START} to {SEARCH_END} | Holdout: {HOLDOUT_START}+")
    print(f"  Capital: ${capital:,.0f} | Slippage: {pcfg.slippage}")
    print()

    signals = load_signals(BEST_LOOKBACK)
    search = filter_period(signals, SEARCH_START, SEARCH_END)
    holdout = filter_period(signals, HOLDOUT_START)
    print(f"  Loaded {len(signals)} signals ({len(search)} search, {len(holdout)} holdout)")
    print(f"  Assets: {signals['asset'].nunique()} | Date range: {signals['timestamp'].min()} to {signals['timestamp'].max()}")
    print()

    # Full simulation
    sim_s, dpnl_s = simulate(search, pcfg)
    sim_h, dpnl_h = simulate(holdout, pcfg)
    sharpe_s = quick_sharpe(dpnl_s, capital) if dpnl_s is not None else 0.0
    sharpe_h = quick_sharpe(dpnl_h, capital) if dpnl_h is not None else 0.0
    print(f"  Portfolio Sharpe: search={sharpe_s:.3f}, holdout={sharpe_h:.3f}")
    print()

    passed = 0
    failed = 0
    total = 0

    def check(name, condition, detail):
        nonlocal passed, failed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        print(f"  {total:>2}. {name:<35s} {status}  {detail}")

    # ===== TEST 1: Per-asset Sharpe decomposition =====
    print("-" * 78)
    print("TEST 1: Per-Asset Sharpe Decomposition")
    print("-" * 78)
    asset_sharpes = {}
    for asset in sorted(search["asset"].unique()):
        a_sigs = search[search["asset"] == asset]
        rets = a_sigs["direction"].values * a_sigs["fwd_return"].values
        if len(rets) > 20:
            std = rets.std()
            sr = rets.mean() / std * np.sqrt(365.25 * 24 / 1) if std > 0 else 0.0
            asset_sharpes[asset] = sr
    positive = sum(1 for v in asset_sharpes.values() if v > 0)
    median_sr = float(np.median(list(asset_sharpes.values()))) if asset_sharpes else 0
    check("Per-asset Sharpe (search)", positive >= len(asset_sharpes) * 0.7,
          f"{positive}/{len(asset_sharpes)} positive, median={median_sr:.2f}")
    top5 = sorted(asset_sharpes.items(), key=lambda x: x[1], reverse=True)[:5]
    bot5 = sorted(asset_sharpes.items(), key=lambda x: x[1])[:5]
    print(f"    Top 5: {', '.join(f'{a}={s:.2f}' for a, s in top5)}")
    print(f"    Bot 5: {', '.join(f'{a}={s:.2f}' for a, s in bot5)}")

    # Holdout per-asset
    asset_sharpes_h = {}
    for asset in sorted(holdout["asset"].unique()):
        a_sigs = holdout[holdout["asset"] == asset]
        rets = a_sigs["direction"].values * a_sigs["fwd_return"].values
        if len(rets) > 20:
            std = rets.std()
            sr = rets.mean() / std * np.sqrt(365.25 * 24 / 1) if std > 0 else 0.0
            asset_sharpes_h[asset] = sr
    positive_h = sum(1 for v in asset_sharpes_h.values() if v > 0)
    median_sr_h = float(np.median(list(asset_sharpes_h.values()))) if asset_sharpes_h else 0
    check("Per-asset Sharpe (holdout)", positive_h >= len(asset_sharpes_h) * 0.7,
          f"{positive_h}/{len(asset_sharpes_h)} positive, median={median_sr_h:.2f}")
    print()

    # ===== TEST 2: Monthly rolling Sharpe stability =====
    print("-" * 78)
    print("TEST 2: Monthly Rolling Sharpe Stability")
    print("-" * 78)
    if dpnl_s is not None and len(dpnl_s) > 90:
        daily_ret = dpnl_s / capital
        rolling_sr = daily_ret.rolling(90).apply(
            lambda x: x.mean() / x.std() * np.sqrt(365.25) if x.std() > 0 else 0, raw=True
        ).dropna()
        min_rolling = float(rolling_sr.min())
        max_rolling = float(rolling_sr.max())
        pct_positive = float((rolling_sr > 0).mean() * 100)
        check("Rolling 90d Sharpe > 0", pct_positive > 80,
              f"{pct_positive:.0f}% positive, range [{min_rolling:.2f}, {max_rolling:.2f}]")
    else:
        check("Rolling 90d Sharpe", False, "insufficient data")

    if dpnl_s is not None:
        monthly = dpnl_s.resample("ME").sum()
        neg_months = int((monthly < 0).sum())
        total_months = len(monthly)
        check("Monthly consistency", neg_months < total_months * 0.4,
              f"{neg_months}/{total_months} negative months ({neg_months/max(total_months,1)*100:.0f}%)")
    print()

    # ===== TEST 3: Cost sensitivity curve =====
    print("-" * 78)
    print("TEST 3: Cost Sensitivity Curve")
    print("-" * 78)
    slippages = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    breakeven_slip = None
    for slip in slippages:
        test_pcfg = PortfolioConfig(**{**pcfg.model_dump(), "slippage": slip})
        _, dpnl_test = simulate(search, test_pcfg)
        sr = quick_sharpe(dpnl_test, capital) if dpnl_test is not None else 0.0
        bps = slip * 10000
        marker = " <-- baseline" if slip == pcfg.slippage else ""
        print(f"    {bps:>6.1f} bps: Sharpe {sr:.3f}{marker}")
        if sr < 2.0 and breakeven_slip is None:
            breakeven_slip = slip
    if breakeven_slip is not None:
        check("Cost robustness (Sharpe>2)", breakeven_slip > 0.001,
              f"Sharpe drops <2.0 at {breakeven_slip*10000:.1f} bps")
    else:
        check("Cost robustness (Sharpe>2)", True, "Sharpe>2.0 at all tested slippages")
    print()

    # ===== TEST 4: Holdout stress tests =====
    print("-" * 78)
    print("TEST 4: Holdout Stress Tests")
    print("-" * 78)
    # Remove best asset
    if asset_sharpes_h:
        best_asset = max(asset_sharpes_h, key=asset_sharpes_h.get)
        stressed = holdout[holdout["asset"] != best_asset]
        _, dpnl_stress = simulate(stressed, pcfg)
        sr_stress = quick_sharpe(dpnl_stress, capital) if dpnl_stress is not None else 0.0
        check("Remove best asset", sr_stress > 1.0,
              f"Without {best_asset}: Sharpe {sr_stress:.3f} (was {sharpe_h:.3f})")

    # Remove best month
    if dpnl_h is not None:
        monthly_h = dpnl_h.resample("ME").sum()
        best_month = monthly_h.idxmax()
        dpnl_no_best = dpnl_h.copy()
        dpnl_no_best.loc[dpnl_no_best.index.to_period("M") == best_month.to_period("M")] = 0.0
        sr_no_best = quick_sharpe(dpnl_no_best, capital)
        check("Remove best month", sr_no_best > 1.0,
              f"Without {best_month.strftime('%Y-%m')}: Sharpe {sr_no_best:.3f}")

    # Invert all directions (anti-signal test)
    anti = holdout.copy()
    anti["direction"] = -anti["direction"]
    _, dpnl_anti = simulate(anti, pcfg)
    sr_anti = quick_sharpe(dpnl_anti, capital) if dpnl_anti is not None else 0.0
    check("Anti-signal (inverted)", sr_anti < -0.5,
          f"Inverted Sharpe {sr_anti:.3f} (should be strongly negative)")
    print()

    # ===== TEST 5: PBO (CSCV) =====
    print("-" * 78)
    print("TEST 5: PBO (Probability of Backtest Overfitting)")
    print("-" * 78)
    # Load all lookbacks and compute CSCV
    lookbacks = [2, 4, 6, 8, 10, 12]
    lb_monthly: dict[int, pd.Series] = {}
    for lb in lookbacks:
        try:
            s = load_signals(lb)
            s_search = filter_period(s, SEARCH_START, SEARCH_END)
            _, d = simulate(s_search, pcfg)
            if d is not None and len(d) > 0:
                lb_monthly[lb] = d.resample("ME").sum()
        except Exception:
            pass

    if len(lb_monthly) >= 2:
        all_months = sorted(set().union(*[set(s.index) for s in lb_monthly.values()]))
        n_partitions = 6
        if len(all_months) >= n_partitions * 2:
            month_groups = np.array_split(np.array(all_months, dtype="datetime64[ns]"), n_partitions)
            labels = list(lb_monthly.keys())
            perf = np.full((len(labels), n_partitions), np.nan)
            for i, lb in enumerate(labels):
                for j, group in enumerate(month_groups):
                    idx = pd.to_datetime(group, utc=True)
                    perf[i, j] = float(lb_monthly[lb].reindex(idx, fill_value=0.0).sum())
            half = n_partitions // 2
            pbo_scores = []
            for train_idx in combinations(range(n_partitions), half):
                test_idx = [i for i in range(n_partitions) if i not in train_idx]
                train_perf = np.array([perf[i, list(train_idx)].sum() for i in range(len(labels))])
                test_perf = np.array([perf[i, list(test_idx)].sum() for i in range(len(labels))])
                winner = int(np.argmax(train_perf))
                oos_rank = int(np.sum(test_perf <= test_perf[winner]))
                pbo_scores.append(oos_rank / len(labels))
            pbo = float(np.mean(np.array(pbo_scores) <= 0.5))
            check("PBO < 0.50", pbo < 0.50, f"PBO={pbo:.3f} across {len(pbo_scores)} splits")
        else:
            check("PBO", False, "insufficient months for CSCV")
    else:
        check("PBO", False, "insufficient lookback variants")
    print()

    # ===== TEST 6: Year-over-year persistence =====
    print("-" * 78)
    print("TEST 6: Year-over-Year Persistence")
    print("-" * 78)
    years = ["2022", "2023", "2024", "2025"]
    year_sharpes = {}
    for year in years:
        start = f"{year}-01"
        end = f"{year}-12"
        y_sigs = filter_period(signals, start, end)
        if len(y_sigs) < 100:
            continue
        _, d = simulate(y_sigs, pcfg)
        sr = quick_sharpe(d, capital) if d is not None else 0.0
        year_sharpes[year] = sr
        print(f"    {year}: Sharpe {sr:.3f} ({len(y_sigs)} signals)")
    all_positive = all(v > 0 for v in year_sharpes.values())
    check("All years positive Sharpe", all_positive,
          f"{sum(1 for v in year_sharpes.values() if v > 0)}/{len(year_sharpes)} years positive")
    print()

    # ===== TEST 7: Rank IC per month =====
    print("-" * 78)
    print("TEST 7: Rank IC Per Month (Cross-Sectional Signal Quality)")
    print("-" * 78)
    search_ts = pd.to_datetime(search["timestamp"], utc=True)
    search_months = search_ts.dt.to_period("M").unique()
    rics = []
    for month in sorted(search_months):
        mask = search_ts.dt.to_period("M") == month
        m = search.loc[mask]
        if len(m) < 20:
            continue
        pred = pd.Series(m["probability"].values * m["direction"].values)
        actual = pd.Series((m["direction"].values * m["fwd_return"].values))
        common = pred.dropna().index.intersection(actual.dropna().index)
        if len(common) < 10:
            continue
        rho, _ = spearmanr(pred.loc[common], actual.loc[common])
        if np.isfinite(rho):
            rics.append(rho)
    if rics:
        mean_ric = float(np.mean(rics))
        pct_positive = float(np.mean(np.array(rics) > 0) * 100)
        check("Rank IC > 0.02", mean_ric > 0.02,
              f"mean={mean_ric:.4f}, {pct_positive:.0f}% positive months")
    else:
        check("Rank IC", False, "no valid months")
    print()

    # ===== TEST 8: HHI Concentration =====
    print("-" * 78)
    print("TEST 8: Return Concentration (HHI)")
    print("-" * 78)
    if dpnl_s is not None:
        monthly_s = dpnl_s.resample("ME").sum()
        pos_months = monthly_s[monthly_s > 0]
        if len(pos_months) > 2:
            shares = (pos_months / pos_months.sum()).values
            hhi = float(np.sum(shares ** 2))
            check("HHI < 0.10 (diversified)", hhi < 0.10,
                  f"HHI={hhi:.4f} across {len(pos_months)} profitable months")
        else:
            check("HHI", False, "insufficient profitable months")
    print()

    # ===== TEST 9: Bootstrap CI on holdout Sharpe =====
    print("-" * 78)
    print("TEST 9: Bootstrap Confidence Interval (Holdout)")
    print("-" * 78)
    if dpnl_h is not None and len(dpnl_h) > 30:
        daily_h = (dpnl_h / capital).values
        rng = np.random.RandomState(42)
        boot_sharpes = []
        for _ in range(10000):
            sample = rng.choice(daily_h, size=len(daily_h), replace=True)
            s = sample.std()
            if s > 0:
                boot_sharpes.append(sample.mean() / s * np.sqrt(365.25))
        ci_lo = float(np.percentile(boot_sharpes, 2.5))
        ci_hi = float(np.percentile(boot_sharpes, 97.5))
        check("Bootstrap CI lower > 1.0", ci_lo > 1.0,
              f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    else:
        check("Bootstrap CI", False, "insufficient holdout data")
    print()

    # ===== TEST 10: DSR / PSR summary =====
    print("-" * 78)
    print("TEST 10: DSR & PSR")
    print("-" * 78)
    if dpnl_s is not None and len(dpnl_s) > 30:
        daily_s = dpnl_s / capital
        sr = float(daily_s.mean() / daily_s.std() * np.sqrt(365.25))
        sr_daily = sr / np.sqrt(365.25)
        skew = float(daily_s.skew())
        kurt = float(daily_s.kurtosis() + 3)
        n = len(daily_s)

        # PSR
        denom_inner = (1 - skew * sr_daily + ((kurt - 1) / 4) * sr_daily ** 2) / max(n - 1, 1)
        denom = np.sqrt(max(denom_inner, 0))
        psr = float(norm.cdf(sr_daily / denom) * 100) if denom > 0 else 50.0
        check("PSR > 95%", psr > 95, f"PSR={psr:.1f}%")

        # DSR (across lookback trials)
        n_trials = len(lookbacks)
        lb_sharpes = []
        for lb in lookbacks:
            try:
                s = load_signals(lb)
                s_search = filter_period(s, SEARCH_START, SEARCH_END)
                _, d = simulate(s_search, pcfg)
                if d is not None:
                    dr = d / capital
                    lb_sharpes.append(float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0)
            except Exception:
                pass
        if len(lb_sharpes) >= 2:
            sr_var = float(np.var(np.array(lb_sharpes) / np.sqrt(365.25), ddof=1))
            dsr = deflated_sharpe_ratio(
                observed_sr=sr / np.sqrt(365.25),
                num_trials=max(len(lb_sharpes), 1),
                num_observations=n,
                skewness=skew,
                kurtosis=kurt,
                sr_variance=sr_var if np.isfinite(sr_var) and sr_var > 0 else 0.1,
            )
            check("DSR > 0.95", dsr > 0.95,
                  f"DSR={dsr:.3f} across {len(lb_sharpes)} lookback trials")
        else:
            check("DSR", False, "insufficient trials")
    print()

    # ===== SUMMARY =====
    print("=" * 78)
    print(f"VALIDATION SUMMARY: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 78)
    if failed == 0:
        print("  ALL TESTS PASSED. Signal model validated for deployment.")
    else:
        print(f"  {failed} test(s) failed. Investigate before deployment.")


if __name__ == "__main__":
    main()
