"""Leverage optimization across ALL lookback periods.

For each lookback (1-18mo), finds the optimal leverage for a prop firm
challenge: 5% daily DD, 10% total DD, 10% profit target.

Reuses signals from lookback_fair generation, then runs Monte Carlo
per lookback per leverage level. Fully parallelized.
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
from signal_engine.allocation import compute_nco_weights, compute_kelly_weights
from afml.bet_sizing import get_signal, discretize_signal, kelly_size

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LOOKBACKS = [1, 2, 3, 4, 6, 8, 10, 12, 14, 18]
FIXED_START = "2023-08"
LEVERAGE_LEVELS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0]
N_SIMS = 10000
TARGET = 10.0
MAX_DAILY_DD = 5.0
MAX_TOTAL_DD = 10.0

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
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5
    results = {lb: [] for lb in LOOKBACKS}
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=ohlcv_15m.index.max().to_period("M"), freq="M")
    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)
        te = valid[(valid >= ms) & (valid <= me)]
        vts = ms - pd.DateOffset(months=3)
        vt_data = vol.loc[(vol.index >= vts) & (vol.index < ms)].dropna()
        if len(vt_data) < 50: continue
        vt = vt_data.quantile(0.50)
        hv = (vol.reindex(te) > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue
        month_str = str(month)
        for lb in LOOKBACKS:
            ts_start = ms - pd.DateOffset(months=lb)
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
                results[lb].append((month_str, filtered.to_dict("list")))
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


def get_daily_returns(combined, leverage=1.0):
    """Run portfolio sim and return daily PnL as % of starting capital."""
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.50, delever_start=0.40,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04 * leverage, max_signal_mult=2.0)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5: return None
    return dpnl


def monte_carlo(daily_ret_pct, leverage):
    """Monte Carlo prop challenge simulation."""
    scaled = daily_ret_pct * leverage
    rng = np.random.RandomState(42)
    successes = 0
    fail_total = fail_daily = 0
    days_list = []
    for _ in range(N_SIMS):
        indices = rng.randint(0, len(scaled), size=365)
        equity = 10000.0
        peak = equity
        day_start = equity
        done = False
        for di, idx in enumerate(indices):
            pnl = equity * scaled[idx] / 100.0
            equity += pnl
            if (day_start - equity) / day_start >= MAX_DAILY_DD / 100:
                fail_daily += 1; done = True; break
            if equity > peak: peak = equity
            if (peak - equity) / peak >= MAX_TOTAL_DD / 100:
                fail_total += 1; done = True; break
            if (equity - 10000) / 10000 * 100 >= TARGET:
                successes += 1; days_list.append(di + 1); done = True; break
            day_start = equity
        if not done and (equity - 10000) / 10000 * 100 >= TARGET:
            successes += 1; days_list.append(365)
    return {
        "success": successes / N_SIMS * 100,
        "fail_dd": fail_total / N_SIMS * 100,
        "fail_daily": fail_daily / N_SIMS * 100,
        "median_days": np.median(days_list) if days_list else 999,
        "p10": np.percentile(days_list, 10) if len(days_list) > 10 else 999,
        "p90": np.percentile(days_list, 90) if len(days_list) > 10 else 999,
    }


def main():
    print("=" * 100)
    print("LEVERAGE OPTIMIZATION ACROSS ALL LOOKBACK PERIODS")
    print(f"  Target: {TARGET}% profit | Max DD: {MAX_TOTAL_DD}% | Max daily DD: {MAX_DAILY_DD}%")
    print(f"  NCO k=8 allocation | {N_SIMS:,d} MC sims | {W} workers")
    print(f"  Lookbacks: {LOOKBACKS}")
    print(f"  Leverage: {LEVERAGE_LEVELS}")
    print("=" * 100, flush=True)

    t0 = time.time()

    # Phase 1: Generate all signals
    print(f"\n[1] Generating signals for all lookbacks ({W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            if done % 10 == 0: print(f"  [{done}/30]", flush=True)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # Phase 2: For each lookback, get daily returns at 1x, then Monte Carlo
    print(f"\n[2] Running backtest + Monte Carlo per lookback...", flush=True)

    # Store results
    all_lb_results = {}

    for lb in LOOKBACKS:
        # Aggregate signals (Version B)
        sigs = []
        for ar in all_results.values():
            for ms, sd in ar.get(lb, []):
                if ms < FIXED_START: continue
                sigs.append(pd.DataFrame(sd))
        if not sigs:
            print(f"  {lb:>2d}mo: NO SIGNALS", flush=True)
            continue
        combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
        combined = apply_nco_wf(combined)

        # Get 1x daily returns
        dpnl = get_daily_returns(combined, 1.0)
        if dpnl is None:
            print(f"  {lb:>2d}mo: NO PnL DATA", flush=True)
            continue

        daily_pct = (dpnl / 10000.0 * 100).values
        worst_day = daily_pct.min()
        best_day = daily_pct.max()
        mean_day = daily_pct.mean()
        std_day = daily_pct.std()

        # Backtest stats at 1x
        total_ret = dpnl.sum() / 10000 * 100
        cumsum = dpnl.cumsum()
        peak_eq = cumsum.expanding().max()
        max_dd = ((peak_eq - cumsum) / (10000 + peak_eq) * 100).max()

        # Monte Carlo at each leverage
        lb_mc = {}
        for lev in LEVERAGE_LEVELS:
            lb_mc[lev] = monte_carlo(daily_pct, lev)

        # Find optimal leverage (highest success rate)
        best_lev = max(LEVERAGE_LEVELS, key=lambda l: lb_mc[l]["success"])
        # Among equal success rates, pick fastest
        best_success = lb_mc[best_lev]["success"]
        candidates = [l for l in LEVERAGE_LEVELS if lb_mc[l]["success"] >= best_success - 1]
        best_lev = min(candidates, key=lambda l: lb_mc[l]["median_days"])

        all_lb_results[lb] = {
            "mean_day": mean_day, "std_day": std_day,
            "worst_day": worst_day, "total_ret": total_ret,
            "max_dd_1x": max_dd, "n_days": len(daily_pct),
            "mc": lb_mc, "best_lev": best_lev,
        }

        mc_best = lb_mc[best_lev]
        print(
            f"  {lb:>2d}mo: Sharpe~{mean_day/std_day*np.sqrt(365):.1f}  "
            f"MaxDD(1x)={max_dd:.1f}%  WorstDay={worst_day:.2f}%  "
            f"Best={best_lev:.1f}x  Success={mc_best['success']:.0f}%  "
            f"Median={mc_best['median_days']:.0f}d",
            flush=True,
        )

    # Phase 3: Summary tables
    print(f"\n{'='*100}")
    print("SUMMARY: OPTIMAL LEVERAGE PER LOOKBACK")
    print(f"{'='*100}")
    print(f"  {'LB':>4s} {'Sharpe':>7s} {'DD(1x)':>7s} {'Worst':>7s} {'OptLev':>7s} {'Sizing':>7s} "
          f"{'Success':>8s} {'Med.Days':>9s} {'P10-P90':>12s} {'FailDD':>7s} {'FailDay':>8s}")
    print(f"  {'-'*92}")

    for lb in LOOKBACKS:
        r = all_lb_results.get(lb)
        if r is None: continue
        lev = r["best_lev"]
        mc = r["mc"][lev]
        sharpe = r["mean_day"] / r["std_day"] * np.sqrt(365) if r["std_day"] > 0 else 0
        sizing = 4.0 * lev
        p10p90 = f"{mc['p10']:.0f}-{mc['p90']:.0f}d"
        print(
            f"  {lb:>2d}mo {sharpe:>7.2f} {r['max_dd_1x']:>6.1f}% {r['worst_day']:>6.2f}% "
            f"{lev:>6.1f}x {sizing:>6.1f}% "
            f"{mc['success']:>7.0f}% {mc['median_days']:>8.0f}d {p10p90:>12s} "
            f"{mc['fail_dd']:>6.1f}% {mc['fail_daily']:>7.1f}%"
        )

    # Phase 4: Detailed grid for the best lookback (8mo)
    print(f"\n{'='*100}")
    print("DETAILED GRID: 8-MONTH LOOKBACK (production config)")
    print(f"{'='*100}")
    r8 = all_lb_results.get(8)
    if r8:
        print(f"  {'Lev':>5s} {'Sizing':>7s} {'Success':>8s} {'Median':>7s} {'P10':>5s} {'P90':>5s} "
              f"{'FailDD':>7s} {'FailDay':>8s} {'ExpDD':>6s}")
        print(f"  {'-'*60}")
        for lev in LEVERAGE_LEVELS:
            mc = r8["mc"][lev]
            exp_dd = r8["max_dd_1x"] * lev
            sizing = 4.0 * lev
            med = f"{mc['median_days']:.0f}d" if mc["median_days"] < 999 else "n/a"
            p10 = f"{mc['p10']:.0f}d" if mc["p10"] < 999 else "n/a"
            p90 = f"{mc['p90']:.0f}d" if mc["p90"] < 999 else "n/a"
            marker = " <<<" if lev == r8["best_lev"] else ""
            print(
                f"  {lev:>5.2f} {sizing:>6.1f}% {mc['success']:>7.0f}% {med:>7s} {p10:>5s} {p90:>5s} "
                f"{mc['fail_dd']:>6.1f}% {mc['fail_daily']:>7.1f}% {exp_dd:>5.1f}%{marker}"
            )

    # Phase 5: Recommendation
    print(f"\n{'='*100}")
    print("RECOMMENDATION")
    print(f"{'='*100}")

    # Find overall best (highest success rate, then fastest)
    best_lb = None
    best_score = -1
    for lb, r in all_lb_results.items():
        mc = r["mc"][r["best_lev"]]
        score = mc["success"]
        if score > best_score or (score == best_score and mc["median_days"] < all_lb_results[best_lb]["mc"][all_lb_results[best_lb]["best_lev"]]["median_days"]):
            best_score = score
            best_lb = lb

    if best_lb:
        r = all_lb_results[best_lb]
        lev = r["best_lev"]
        mc = r["mc"][lev]
        sizing = 4.0 * lev
        print(f"\n  BEST CONFIG: {best_lb}-month lookback, {lev:.1f}x leverage (sizing {sizing:.1f}%)")
        print(f"  Success rate: {mc['success']:.1f}%")
        print(f"  Median days to 10%: {mc['median_days']:.0f}")
        print(f"  Range (P10-P90): {mc['p10']:.0f} - {mc['p90']:.0f} days")
        print(f"  Backtest MaxDD at {lev}x: {r['max_dd_1x']*lev:.1f}%")
        print(f"  DD failure rate: {mc['fail_dd']:.1f}%")
        print(f"  Daily DD failure rate: {mc['fail_daily']:.1f}%")
        print(f"\n  Config:")
        print(f"    lookback_months: {best_lb}")
        print(f"    base_sizing_pct: {sizing/100:.4f}")
        print(f"    max_dd: 0.10")
        print(f"    delever_start: 0.06")
        print(f"    daily_dd_halt_pct: 0.05")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
