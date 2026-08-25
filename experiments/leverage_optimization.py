"""Leverage optimization for prop firm challenge.

Constraints: 5% daily DD, 10% total DD, target 10% profit.
Uses the ACTUAL daily PnL distribution from the production backtest
to simulate expected time-to-target and probability of DD breach.

Runs Monte Carlo with 10,000 paths per leverage level.
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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
LOOKBACK = 8
FIXED_START = "2023-08"

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

# Prop firm constraints
TARGET_PROFIT_PCT = 10.0
MAX_DAILY_DD_PCT = 5.0
MAX_TOTAL_DD_PCT = 10.0

# Leverage levels to test (applied to base_sizing_pct)
LEVERAGE_LEVELS = [1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]

N_SIMULATIONS = 10000


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate 8mo rolling signals for one asset."""
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
        return asset, []
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
    if len(valid) < 3000: return asset, []
    start_date = ohlcv_15m.index.min()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5
    results = []
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=ohlcv_15m.index.max().to_period("M"), freq="M")
    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None: ms = ms.tz_localize(None); me = me.tz_localize(None)
        te = valid[(valid >= ms) & (valid <= me)]
        vts = ms - pd.DateOffset(months=3)
        vt_data = vol.loc[(vol.index >= vts) & (vol.index < ms)].dropna()
        if len(vt_data) < 50: continue
        vt = vt_data.quantile(0.50)
        hv = (vol.reindex(te) > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue
        ts_start = ms - pd.DateOffset(months=LOOKBACK)
        if ts_start < start_date: continue
        tr = valid[(valid >= ts_start) & (valid < ms - embargo_td)]
        if len(tr) < 200: continue
        try:
            sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(fc.loc[tr])
            model = make_logreg()
            model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
            Xte = scaler.transform(fc.loc[te])
            yp = model.predict_proba(Xte)[:, 1]
        except Exception: continue
        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any(): continue
        signals = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
            "probability": yp[finite], "signal_strength": ss[finite]})
        signals["asset"] = asset
        filtered = apply_confidence_filter(
            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered) > 0:
            results.append((str(month), filtered.to_dict("list")))
    return asset, results


def apply_nco_wf(combined):
    """Walk-forward NCO k=8."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]
        if len(past_sigs) < 50:
            parts.append(km_sigs); continue
        assets_in = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for a in assets_in:
            a_past = past_sigs[past_sigs["asset"] == a]
            if len(a_past) < 10: continue
            ts_a = pd.to_datetime(a_past["timestamp"])
            rets = pd.Series(a_past["direction"].values * a_past["fwd_return"].values, index=ts_a)
            daily_rets[a] = rets.resample("D").sum()
        if len(daily_rets) < 3:
            weights = compute_kelly_weights(past_sigs)
        else:
            ret_df = pd.DataFrame(daily_rets).fillna(0)
            corr = ret_df.corr().values
            names = list(ret_df.columns)
            exp_r = np.array([
                float((past_sigs[past_sigs["asset"] == a]["direction"]
                       * past_sigs[past_sigs["asset"] == a]["fwd_return"]).mean())
                if a in past_sigs["asset"].values else 0.0 for a in names])
            weights = compute_nco_weights(corr, exp_r, names, n_clusters=8)
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_at_leverage(combined, leverage):
    """Run portfolio sim at a given leverage (scales base_sizing_pct)."""
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.50,  # Don't halt in backtest — we track DD ourselves
        delever_start=0.40,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04 * leverage, max_signal_mult=2.0)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    return dpnl


def monte_carlo_challenge(daily_returns_pct, leverage, n_sims=N_SIMULATIONS):
    """Monte Carlo simulation of prop firm challenge.

    Samples daily returns (with replacement) to simulate challenge outcomes.
    Returns stats on time-to-target and probability of failure.
    """
    scaled = daily_returns_pct * leverage
    rng = np.random.RandomState(42)

    successes = 0
    failures_total_dd = 0
    failures_daily_dd = 0
    days_to_target = []
    max_dds = []

    for _ in range(n_sims):
        # Random starting point in the daily returns
        indices = rng.randint(0, len(scaled), size=365)  # max 1 year challenge
        equity = 10000.0
        peak = equity
        day_start_equity = equity
        hit_target = False
        hit_dd = False

        for day_i, idx in enumerate(indices):
            day_pnl = equity * scaled[idx] / 100.0  # scaled[idx] is % return
            equity += day_pnl

            # Daily DD check (from day start)
            daily_dd = (day_start_equity - equity) / day_start_equity if day_start_equity > 0 else 0
            if daily_dd >= MAX_DAILY_DD_PCT / 100:
                failures_daily_dd += 1
                hit_dd = True
                break

            # Total DD check
            if equity > peak:
                peak = equity
            total_dd = (peak - equity) / peak if peak > 0 else 0
            if total_dd >= MAX_TOTAL_DD_PCT / 100:
                failures_total_dd += 1
                hit_dd = True
                break

            # Target check
            profit_pct = (equity - 10000) / 10000 * 100
            if profit_pct >= TARGET_PROFIT_PCT:
                successes += 1
                days_to_target.append(day_i + 1)
                hit_target = True
                break

            day_start_equity = equity

        if not hit_target and not hit_dd:
            # Ran out of time (365 days)
            profit = (equity - 10000) / 10000 * 100
            if profit >= TARGET_PROFIT_PCT:
                successes += 1
                days_to_target.append(365)

        max_dds.append((peak - equity) / peak * 100 if peak > 0 else 0)

    return {
        "success_rate": successes / n_sims * 100,
        "failure_total_dd": failures_total_dd / n_sims * 100,
        "failure_daily_dd": failures_daily_dd / n_sims * 100,
        "median_days": np.median(days_to_target) if days_to_target else float("inf"),
        "mean_days": np.mean(days_to_target) if days_to_target else float("inf"),
        "p10_days": np.percentile(days_to_target, 10) if len(days_to_target) > 10 else float("inf"),
        "p90_days": np.percentile(days_to_target, 90) if len(days_to_target) > 10 else float("inf"),
        "median_max_dd": np.median(max_dds),
    }


def main():
    print("=" * 85)
    print("LEVERAGE OPTIMIZATION FOR PROP FIRM CHALLENGE")
    print(f"  Target: {TARGET_PROFIT_PCT}% profit")
    print(f"  Constraints: {MAX_DAILY_DD_PCT}% daily DD, {MAX_TOTAL_DD_PCT}% total DD")
    print(f"  Monte Carlo: {N_SIMULATIONS:,d} simulations per leverage level")
    print(f"  System: 8mo rolling, 13 features, NCO k=8, 30 assets")
    print("=" * 85, flush=True)

    t0 = time.time()

    # Phase 1: Generate signals
    print(f"\n[1] Generating signals ({W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            if done % 10 == 0: print(f"  [{done}/30]", flush=True)

    sigs = []
    for ar in all_results.values():
        for ms, sd in ar:
            if ms < FIXED_START: continue
            sigs.append(pd.DataFrame(sd))
    raw = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    combined = apply_nco_wf(raw)
    print(f"  {len(combined):,d} signals", flush=True)

    # Phase 2: Get baseline daily returns at 1x
    print(f"\n[2] Computing daily returns at 1x...", flush=True)
    dpnl_1x = run_at_leverage(combined, 1.0)
    if dpnl_1x is None:
        print("ERROR: No PnL data")
        return

    # Convert to % returns
    equity_start = 10000.0
    daily_ret_pct = (dpnl_1x / equity_start * 100).values

    # Stats at 1x
    print(f"  Days: {len(daily_ret_pct)}")
    print(f"  Mean daily return: {daily_ret_pct.mean():.4f}%")
    print(f"  Std daily return: {daily_ret_pct.std():.4f}%")
    print(f"  Worst day: {daily_ret_pct.min():.4f}%")
    print(f"  Best day: {daily_ret_pct.max():.4f}%")
    print(f"  Skew: {pd.Series(daily_ret_pct).skew():.2f}")
    print(f"  Kurtosis: {pd.Series(daily_ret_pct).kurtosis():.1f}")

    # Phase 3: Also run actual backtest at each leverage level
    print(f"\n[3] Running actual backtests at each leverage...", flush=True)
    backtest_results = {}
    for lev in LEVERAGE_LEVELS:
        dpnl = run_at_leverage(combined, lev)
        if dpnl is not None:
            total_ret = dpnl.sum() / equity_start * 100
            cumsum = dpnl.cumsum()
            peak = cumsum.expanding().max()
            dd = (peak - cumsum) / (equity_start + peak) * 100
            max_dd = dd.max()
            # Worst single day
            worst_day = (dpnl / equity_start * 100).min()
            # Days to 10%
            target_val = equity_start * TARGET_PROFIT_PCT / 100
            cum = dpnl.cumsum()
            hit = cum[cum >= target_val]
            days_to_target = (hit.index[0] - cum.index[0]).days if len(hit) > 0 else None
            backtest_results[lev] = {
                "total_ret": total_ret, "max_dd": max_dd,
                "worst_day": worst_day, "days_to_target": days_to_target,
                "n_trades": len(dpnl),
            }

    # Phase 4: Monte Carlo
    print(f"\n[4] Monte Carlo simulations ({N_SIMULATIONS:,d} paths per level)...", flush=True)
    mc_results = {}
    for lev in LEVERAGE_LEVELS:
        mc = monte_carlo_challenge(daily_ret_pct, lev)
        mc_results[lev] = mc

    # Phase 5: Results
    print(f"\n{'='*85}")
    print("RESULTS: ACTUAL BACKTEST")
    print(f"{'='*85}")
    print(f"  {'Lev':>5s} {'Sizing':>7s} {'Return%':>8s} {'MaxDD%':>7s} {'WorstDay%':>10s} {'DaysTo10%':>10s} {'DD<10%':>7s} {'Daily<5%':>8s}")
    print(f"  {'-'*72}")
    for lev in LEVERAGE_LEVELS:
        r = backtest_results.get(lev)
        if r is None: continue
        sizing = 4.0 * lev
        dd_ok = "YES" if r["max_dd"] < MAX_TOTAL_DD_PCT else "NO"
        daily_ok = "YES" if abs(r["worst_day"]) < MAX_DAILY_DD_PCT else "NO"
        dtt = f"{r['days_to_target']}" if r["days_to_target"] else "never"
        print(f"  {lev:>5.2f} {sizing:>6.1f}% {r['total_ret']:>8.1f} {r['max_dd']:>7.2f} {r['worst_day']:>10.3f} {dtt:>10s} {dd_ok:>7s} {daily_ok:>8s}")

    print(f"\n{'='*85}")
    print("RESULTS: MONTE CARLO (10K simulations per level)")
    print(f"{'='*85}")
    print(f"  {'Lev':>5s} {'Success%':>9s} {'Median':>7s} {'P10':>5s} {'P90':>5s} {'Fail(DD)':>9s} {'Fail(day)':>10s} {'Med MaxDD':>10s}")
    print(f"  {'-'*70}")
    for lev in LEVERAGE_LEVELS:
        mc = mc_results[lev]
        med = f"{mc['median_days']:.0f}d" if mc["median_days"] < 999 else "n/a"
        p10 = f"{mc['p10_days']:.0f}d" if mc["p10_days"] < 999 else "n/a"
        p90 = f"{mc['p90_days']:.0f}d" if mc["p90_days"] < 999 else "n/a"
        print(f"  {lev:>5.2f} {mc['success_rate']:>8.1f}% {med:>7s} {p10:>5s} {p90:>5s} "
              f"{mc['failure_total_dd']:>8.1f}% {mc['failure_daily_dd']:>9.1f}% {mc['median_max_dd']:>9.2f}%")

    # Find optimal
    print(f"\n{'='*85}")
    print("RECOMMENDATION")
    print(f"{'='*85}")

    # Best = highest success rate with median days < 90
    best_lev = None
    best_score = -1
    for lev in LEVERAGE_LEVELS:
        mc = mc_results[lev]
        bt = backtest_results.get(lev, {})
        if bt.get("max_dd", 99) > MAX_TOTAL_DD_PCT:
            continue  # Would have failed in actual backtest
        score = mc["success_rate"] / max(mc["median_days"], 1) * 100  # success per day
        if score > best_score:
            best_score = score
            best_lev = lev

    if best_lev:
        mc = mc_results[best_lev]
        bt = backtest_results[best_lev]
        sizing = 4.0 * best_lev
        print(f"\n  OPTIMAL: {best_lev:.2f}x leverage (base_sizing = {sizing:.1f}%)")
        print(f"  Success rate: {mc['success_rate']:.1f}%")
        print(f"  Median time to target: {mc['median_days']:.0f} days")
        print(f"  Range (P10-P90): {mc['p10_days']:.0f} - {mc['p90_days']:.0f} days")
        print(f"  Backtest MaxDD: {bt['max_dd']:.2f}%")
        print(f"  Backtest worst day: {bt['worst_day']:.3f}%")
        print(f"  Failure rate (total DD): {mc['failure_total_dd']:.1f}%")
        print(f"  Failure rate (daily DD): {mc['failure_daily_dd']:.1f}%")

    # Also show conservative and aggressive options
    print(f"\n  --- Options ---")
    for label, lev_target in [("Conservative", 1.1), ("Moderate", 1.25), ("Aggressive", 1.4)]:
        # Find closest tested level
        lev = min(LEVERAGE_LEVELS, key=lambda x: abs(x - lev_target))
        mc = mc_results[lev]
        bt = backtest_results.get(lev, {})
        sizing = 4.0 * lev
        print(f"  {label} ({lev:.2f}x, sizing={sizing:.1f}%): "
              f"success={mc['success_rate']:.0f}%, median={mc['median_days']:.0f}d, "
              f"DD={bt.get('max_dd',0):.1f}%, fail={mc['failure_total_dd']:.0f}%")

    print(f"\n  Config to apply:")
    if best_lev:
        sizing = 4.0 * best_lev
        print(f"    base_sizing_pct: {sizing/100:.4f}  # was 0.04")
        print(f"    OR leverage: {best_lev}  # exchange leverage with base_sizing 0.04")
        print(f"    max_dd: 0.10")
        print(f"    delever_start: 0.06")
        print(f"    daily_dd_halt_pct: 0.05")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
