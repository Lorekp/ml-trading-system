"""Final Production Report — Full stats for all lookback periods with NCO allocation.

Generates:
1. Lookback comparison table (Version A + B) with NCO k=8
2. Full stats report for the production config (8mo, NCO)
3. Monthly PnL breakdown
4. Bootstrap CI, DSR, VaR/CVaR, underwater analysis

Fully parallelized signal generation.
"""
import sys
sys.path.insert(0, "src")

import copy, json, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.allocation import compute_nco_weights, compute_kelly_weights
from afml.bet_sizing import get_signal, discretize_signal, kelly_size

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LOOKBACKS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 18]

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate signals for ALL lookbacks for one asset."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, {}

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    results = {lb: [] for lb in LOOKBACKS}
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"), freq="M",
    )

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50:
            continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        month_str = str(month)

        for lb in LOOKBACKS:
            train_start = ms - pd.DateOffset(months=lb)
            if train_start < start_date:
                continue
            train_cutoff = ms - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200:
                continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg()
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception:
                continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            fv = fwd_ret.reindex(te).values
            finite = np.isfinite(fv)
            if not finite.any():
                continue

            signals = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            signals["asset"] = asset
            filtered = apply_confidence_filter(
                signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
            if len(filtered) > 0:
                results[lb].append((month_str, filtered.to_dict("list")))

    return asset, results


def apply_nco_wf(combined):
    """Walk-forward NCO k=8 allocation."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]
        if len(past_sigs) < 50:
            parts.append(km_sigs)
            continue

        # Build correlation matrix from past trade returns
        assets_in_month = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for a in assets_in_month:
            a_past = past_sigs[past_sigs["asset"] == a]
            if len(a_past) < 10:
                continue
            ts_a = pd.to_datetime(a_past["timestamp"])
            rets = pd.Series(
                a_past["direction"].values * a_past["fwd_return"].values, index=ts_a
            )
            daily_rets[a] = rets.resample("D").sum()

        if len(daily_rets) < 3:
            # Fallback to Kelly
            kw = compute_kelly_weights(past_sigs)
            kmc = km_sigs.copy()
            kmc["signal_strength"] = kmc.apply(
                lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
            parts.append(kmc)
            continue

        ret_df = pd.DataFrame(daily_rets).fillna(0)
        corr = ret_df.corr().values
        names = list(ret_df.columns)
        exp_r = np.array([
            float((past_sigs[past_sigs["asset"] == a]["direction"]
                   * past_sigs[past_sigs["asset"] == a]["fwd_return"]).mean())
            if a in past_sigs["asset"].values else 0.0
            for a in names
        ])
        weights = compute_nco_weights(corr, exp_r, names, n_clusters=8)

        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)

    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def get_pcfg():
    return PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )


def run_lookback(lb, all_results, start_after=None):
    """Aggregate signals for one lookback, apply NCO, run sim."""
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results.get(lb, []):
            if start_after and month_str < start_after:
                continue
            sigs.append(pd.DataFrame(sig_dict))
    if not sigs:
        return None
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    combined = apply_nco_wf(combined)
    pcfg = get_pcfg()
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"]/len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {**stats, "n_trades": sim["n_trades"], "pnl_day": pnl_day,
            "dpnl": dpnl, "sim": sim, "combined": combined}


def full_stats_report(label, result, n_trials=40):
    """Print comprehensive stats report."""
    if result is None:
        print(f"  {label}: NO RESULT")
        return

    dpnl = result["dpnl"]
    sim = result["sim"]
    pcfg = get_pcfg()
    stats = result

    daily_returns = dpnl / pcfg.starting_capital
    equity = sim["equity"]
    timestamps = sim["timestamps"]

    # Basic stats
    sharpe = stats["sharpe"]
    sortino = stats.get("sortino", 0)
    calmar = stats.get("calmar", 0)
    total_ret = stats["total_return_pct"]
    max_dd = stats["max_dd_pct"]
    n_trades = stats["n_trades"]
    n_days = stats["n_days"]
    final_eq = stats["final_equity"]

    # Win/loss analysis
    wins = dpnl[dpnl > 0]
    losses = dpnl[dpnl < 0]
    win_rate = len(wins) / len(dpnl) * 100 if len(dpnl) > 0 else 0
    loss_rate = 100 - win_rate
    avg_win_bps = (wins.mean() / pcfg.starting_capital * 10000) if len(wins) > 0 else 0
    avg_loss_bps = (losses.mean() / pcfg.starting_capital * 10000) if len(losses) > 0 else 0
    profit_loss_ratio = abs(avg_win_bps / avg_loss_bps) if avg_loss_bps != 0 else 0
    expectancy_bps = daily_returns.mean() * 10000

    # Annual stats
    annual_ret = (1 + total_ret / 100) ** (365 / max(n_days, 1)) - 1
    annual_std = daily_returns.std() * np.sqrt(365)
    annual_var = annual_std ** 2

    # Turnover
    turnover = n_trades / max(n_days, 1) * 365

    # Drawdown analysis
    eq_series = pd.Series(equity[~sim["skipped"]], index=pd.to_datetime(timestamps[~sim["skipped"]]))
    peak = eq_series.expanding().max()
    dd_series = (peak - eq_series) / peak
    underwater = dd_series > 0.001

    # Underwater periods
    uw_periods = []
    in_dd = False
    dd_start = None
    for i, (ts, is_uw) in enumerate(underwater.items()):
        if is_uw and not in_dd:
            in_dd = True
            dd_start = ts
        elif not is_uw and in_dd:
            in_dd = False
            uw_periods.append((ts - dd_start).days)
    if in_dd and dd_start is not None:
        uw_periods.append((underwater.index[-1] - dd_start).days)

    avg_uw = np.mean(uw_periods) if uw_periods else 0
    max_uw = max(uw_periods) if uw_periods else 0

    # Consecutive wins/losses
    daily_sign = (dpnl > 0).astype(int)
    max_consec_wins = max_consec_losses = 0
    current_streak = 0
    current_is_win = None
    for v in daily_sign:
        if v == current_is_win:
            current_streak += 1
        else:
            if current_is_win == 1:
                max_consec_wins = max(max_consec_wins, current_streak)
            elif current_is_win == 0:
                max_consec_losses = max(max_consec_losses, current_streak)
            current_streak = 1
            current_is_win = v
    if current_is_win == 1:
        max_consec_wins = max(max_consec_wins, current_streak)
    elif current_is_win == 0:
        max_consec_losses = max(max_consec_losses, current_streak)

    # VaR / CVaR
    var_95 = np.percentile(daily_returns, 5) * 100
    cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100 if len(daily_returns) > 0 else 0

    # PSR
    T = len(dpnl)
    skew = stats.get("skew", 0)
    kurt = stats.get("kurtosis", 3)
    sr_std = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / max(T, 1))
    psr = norm.cdf((sharpe - 0) / sr_std) * 100 if sr_std > 0 else 100

    # DSR
    e_z1 = norm.ppf(1 - 1/max(n_trials, 2))
    gamma = 0.5772
    e_max = (1 - gamma) * e_z1 + gamma * norm.ppf(1 - 1/(max(n_trials, 2) * np.e))
    dsr = norm.cdf((sharpe - sr_std * e_max) / sr_std) * 100 if sr_std > 0 else 100

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_sharpes = []
    daily_arr = dpnl.values
    for _ in range(5000):
        sample = rng.choice(daily_arr, size=len(daily_arr), replace=True)
        s_mean = sample.mean()
        s_std = sample.std()
        if s_std > 0:
            boot_sharpes.append(s_mean / s_std * np.sqrt(365))
    ci_lo = np.percentile(boot_sharpes, 2.5)
    ci_hi = np.percentile(boot_sharpes, 97.5)

    # Weekly/quarterly Sharpe
    weekly_pnl = dpnl.resample("W").sum()
    weekly_sharpe = weekly_pnl.mean() / weekly_pnl.std() * np.sqrt(52) if weekly_pnl.std() > 0 else 0
    quarterly_pnl = dpnl.resample("QE").sum()
    quarterly_sharpe = quarterly_pnl.mean() / quarterly_pnl.std() * np.sqrt(4) if len(quarterly_pnl) > 2 and quarterly_pnl.std() > 0 else 0

    # Monthly PnL
    monthly_pnl = dpnl.resample("ME").sum()

    # Positive months
    pos_months = (monthly_pnl > 0).sum() / len(monthly_pnl) * 100 if len(monthly_pnl) > 0 else 0

    # HHI concentration
    if "asset" in result["combined"].columns:
        asset_counts = result["combined"]["asset"].value_counts(normalize=True)
        hhi = (asset_counts ** 2).sum()
    else:
        hhi = 1.0

    # DD recovery
    dd_max_idx = dd_series.idxmax() if len(dd_series) > 0 else None
    dd_recovery = 0
    if dd_max_idx is not None:
        recovery_slice = dd_series[dd_series.index >= dd_max_idx]
        recovered = recovery_slice[recovery_slice < 0.001]
        if len(recovered) > 0:
            dd_recovery = (recovered.index[0] - dd_max_idx).days

    pnl_day = stats["pnl_day"]

    # Print report
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  PSR                          {psr:.1f}%")
    print(f"  Sharpe Ratio                 {sharpe:.3f}")
    print(f"  Sortino Ratio                {sortino:.3f}")
    print(f"  Calmar Ratio                 {calmar:.3f}")
    print(f"  Total Orders                 {n_trades:,d}")
    print(f"  Average Win                  {avg_win_bps:.1f} bps/day")
    print(f"  Average Loss                 {avg_loss_bps:.1f} bps/day")
    print(f"  Compounding Annual Return    {annual_ret*100:.2f}%")
    print(f"  Drawdown                     {max_dd:.2f}%")
    print(f"  Expectancy                   {expectancy_bps:.2f} bps/day")
    print(f"  Start Equity                 ${pcfg.starting_capital:,.2f}")
    print(f"  End Equity                   ${final_eq:,.2f}")
    print(f"  Net Profit                   {total_ret:.2f}%")
    print(f"  Loss Rate                    {loss_rate:.1f}%")
    print(f"  Win Rate                     {win_rate:.1f}%")
    print(f"  Profit-Loss Ratio            {profit_loss_ratio:.3f}")
    print(f"  Annual Standard Deviation    {annual_std*100:.2f}%")
    print(f"  Annual Variance              {annual_var*100:.2f}%")
    print(f"  Portfolio Turnover           {turnover:.0f}x/year")
    print(f"  Drawdown Recovery            {dd_recovery} days")
    print(f"  --- ADDITIONAL STATS ---")
    print(f"  Sharpe (weekly)              {weekly_sharpe:.3f}")
    print(f"  Sharpe (quarterly)           {quarterly_sharpe:.3f}")
    print(f"  Bootstrap 95% CI             [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  DSR (K={n_trials})                   {dsr:.1f}%")
    print(f"  VaR (95%, daily)             {var_95:.4f}%")
    print(f"  CVaR (95%, daily)            {cvar_95:.4f}%")
    print(f"  Skewness                     {skew:.3f}")
    print(f"  Kurtosis                     {kurt:.1f}")
    print(f"  Max Consecutive Wins         {max_consec_wins} days")
    print(f"  Max Consecutive Losses       {max_consec_losses} days")
    print(f"  Avg Underwater Period         {avg_uw:.1f} days")
    print(f"  Max Underwater Period         {max_uw} days")
    print(f"  PnL per Day                  ${pnl_day:.2f}")
    print(f"  Positive Months              {pos_months:.0f}%")
    print(f"  HHI Concentration            {hhi:.4f}{' (OK)' if hhi < 0.15 else ' (HIGH)'}")
    print(f"  Days of Data                 {n_days}")
    print(f"  Allocation                   NCO k=8 (walk-forward)")
    print(f"  Training Window              8-month rolling")
    print(f"  --- MONTHLY PnL ---")
    for date, pnl in monthly_pnl.items():
        month_str = date.strftime("%Y-%m")
        print(f"  {month_str}: ${pnl:>10.2f}")

    return {
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "total_return_pct": total_ret, "max_dd_pct": max_dd,
        "n_trades": n_trades, "pnl_day": pnl_day,
        "bootstrap_ci": [ci_lo, ci_hi], "psr": psr, "dsr": dsr,
        "win_rate": win_rate, "annual_ret": annual_ret * 100,
    }


def main():
    print("=" * 70)
    print("FINAL PRODUCTION REPORT")
    print(f"  13 features, NCO k=8 allocation, {W} workers")
    print(f"  Lookbacks: {LOOKBACKS}")
    print("=" * 70, flush=True)

    t0 = time.time()

    # Phase 1: Generate signals for all lookbacks (parallel)
    print(f"\n[1] Generating signals for all lookbacks ({W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            total = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {total} month-lookback combos", flush=True)

    # Determine Version B start
    cfg = load_config(Path("configs/default.yaml"))
    sample_1m = load_binance_1m("BTCUSDT", [DATA_DIR])
    sample_15m = resample_ohlcv(sample_1m, "15min")
    data_start = sample_15m.index.min()
    fixed_start = str((data_start + pd.DateOffset(months=max(LOOKBACKS) + 1)).to_period("M"))
    print(f"\n  Signal generation: {time.time()-t0:.0f}s")
    print(f"  Version B start: {fixed_start}")

    # Phase 2: Lookback comparison tables
    print(f"\n{'='*70}")
    print("LOOKBACK COMPARISON WITH NCO k=8 ALLOCATION")
    print(f"{'='*70}")

    for version, start_after, vlabel in [
        ("A", None, "Start ASAP"),
        ("B", fixed_start, f"Same test period (start {fixed_start})"),
    ]:
        print(f"\nVERSION {version}: {vlabel}")
        print(f"{'Lookback':<10s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'Trades':>7s} {'Days':>5s} {'PnL/day':>8s}")
        print("-" * 55)

        for lb in LOOKBACKS:
            r = run_lookback(lb, all_results, start_after)
            if r:
                print(
                    f"  {lb:>2d}mo      {r['sharpe']:>7.3f} {r['total_return_pct']:>8.2f} "
                    f"{r['max_dd_pct']:>7.2f} {r['n_trades']:>7d} {r['n_days']:>5d} "
                    f"${r['pnl_day']:>7.2f}",
                    flush=True,
                )
            else:
                print(f"  {lb:>2d}mo      NO SIGNALS", flush=True)

    # Phase 3: Full stats for production config (8mo, Version B)
    print(f"\n{'='*70}")
    print("FULL STATS: 8-MONTH ROLLING + NCO k=8 (Version B)")
    print(f"{'='*70}")
    prod_result = run_lookback(8, all_results, fixed_start)
    report_data = full_stats_report("8-Month Rolling, NCO k=8, Version B", prod_result)

    # Phase 4: Full stats for Version A (longer period)
    print(f"\n{'='*70}")
    print("FULL STATS: 8-MONTH ROLLING + NCO k=8 (Version A - Start ASAP)")
    print(f"{'='*70}")
    prod_a = run_lookback(8, all_results, None)
    full_stats_report("8-Month Rolling, NCO k=8, Version A", prod_a)

    # Save to desktop
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
