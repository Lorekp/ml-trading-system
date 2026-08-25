"""In-depth NCO vs Kelly comparison.

Both walk-forward. Same signals, same pipeline, only allocation differs.
Compares: Sharpe, MaxDD, monthly stability, per-asset attribution,
drawdown profiles, rolling Sharpe, and regime performance.
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
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
    if len(valid) < 3000:
        return asset, []
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


def apply_kelly_wf(combined):
    """Walk-forward Kelly."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(combined["timestamp"])
    months = sorted(ts.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_sigs = combined[ts.dt.to_period("M") == km]
        past_sigs = combined[ts.dt.to_period("M") < km]
        if len(past_sigs) < 50:
            parts.append(km_sigs); continue
        weights = compute_kelly_weights(past_sigs)
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


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
        assets_in_month = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for a in assets_in_month:
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
                if a in past_sigs["asset"].values else 0.0
                for a in names])
            weights = compute_nco_weights(corr, exp_r, names, n_clusters=8)
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_sim(combined, label):
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"]/len(dpnl))
    return {"stats": stats, "dpnl": dpnl, "sim": sim, "combined": combined,
            "n_trades": sim["n_trades"],
            "pnl_day": (stats["final_equity"] - 10000) / max(stats["n_days"], 1)}


def main():
    print("=" * 80)
    print("NCO vs KELLY — IN-DEPTH COMPARISON")
    print(f"  Both walk-forward, 8mo rolling, 13 features, 30 assets, {W} workers")
    print(f"  Version B (start {FIXED_START})")
    print("=" * 80, flush=True)

    t0 = time.time()

    # Phase 1: Generate signals
    print(f"\n[1] Generating signals...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            if done % 10 == 0: print(f"  [{done}/30]", flush=True)

    # Combine (Version B)
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar:
            if ms < FIXED_START: continue
            sigs.append(pd.DataFrame(sd))
    raw = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(raw):,d} signals in {time.time()-t0:.0f}s", flush=True)

    # Phase 2: Apply both allocation methods
    print(f"\n[2] Applying allocations...", flush=True)
    t1 = time.time()
    kelly_combined = apply_kelly_wf(raw.copy())
    nco_combined = apply_nco_wf(raw.copy())
    equal_combined = raw.copy()
    print(f"  Done in {time.time()-t1:.0f}s", flush=True)

    # Phase 3: Run sims
    print(f"\n[3] Running portfolio sims...", flush=True)
    r_equal = run_sim(equal_combined, "Equal")
    r_kelly = run_sim(kelly_combined, "Kelly")
    r_nco = run_sim(nco_combined, "NCO")

    # ================================================================
    # COMPARISON
    # ================================================================
    print(f"\n{'='*80}")
    print("1. HEADLINE COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Metric':<30s} {'Equal':>10s} {'Kelly':>10s} {'NCO k=8':>10s}")
    print(f"  {'-'*62}")
    for name, key in [
        ("Sharpe", "sharpe"), ("Sortino", "sortino"), ("Calmar", "calmar"),
        ("Return %", "total_return_pct"), ("Max DD %", "max_dd_pct"),
        ("Win Rate %", "win_rate"), ("Final Equity", "final_equity"),
    ]:
        ve = r_equal["stats"].get(key, 0) if r_equal else 0
        vk = r_kelly["stats"].get(key, 0) if r_kelly else 0
        vn = r_nco["stats"].get(key, 0) if r_nco else 0
        if key == "final_equity":
            print(f"  {name:<30s} ${ve:>9,.2f} ${vk:>9,.2f} ${vn:>9,.2f}")
        else:
            print(f"  {name:<30s} {ve:>10.3f} {vk:>10.3f} {vn:>10.3f}")
    # Trades and PnL/day
    for lbl, r in [("Equal", r_equal), ("Kelly", r_kelly), ("NCO", r_nco)]:
        if r:
            print(f"  {lbl}: {r['n_trades']:,d} trades, ${r['pnl_day']:.2f}/day")

    # ================================================================
    print(f"\n{'='*80}")
    print("2. MONTHLY PnL COMPARISON")
    print(f"{'='*80}")
    dpnl_e = r_equal["dpnl"] if r_equal else pd.Series(dtype=float)
    dpnl_k = r_kelly["dpnl"] if r_kelly else pd.Series(dtype=float)
    dpnl_n = r_nco["dpnl"] if r_nco else pd.Series(dtype=float)

    monthly_e = dpnl_e.resample("ME").sum()
    monthly_k = dpnl_k.resample("ME").sum()
    monthly_n = dpnl_n.resample("ME").sum()

    # Align
    all_months = sorted(set(monthly_e.index) | set(monthly_k.index) | set(monthly_n.index))
    print(f"  {'Month':<10s} {'Equal':>10s} {'Kelly':>10s} {'NCO':>10s} {'Winner':>10s}")
    print(f"  {'-'*52}")
    kelly_wins = nco_wins = equal_wins = 0
    for m in all_months:
        ve = monthly_e.get(m, 0)
        vk = monthly_k.get(m, 0)
        vn = monthly_n.get(m, 0)
        best = max(ve, vk, vn)
        winner = "Equal" if ve == best else ("Kelly" if vk == best else "NCO")
        if winner == "Kelly": kelly_wins += 1
        elif winner == "NCO": nco_wins += 1
        else: equal_wins += 1
        print(f"  {m.strftime('%Y-%m'):<10s} ${ve:>9.2f} ${vk:>9.2f} ${vn:>9.2f} {winner:>10s}")
    print(f"\n  Monthly wins: Equal={equal_wins}, Kelly={kelly_wins}, NCO={nco_wins}")

    # ================================================================
    print(f"\n{'='*80}")
    print("3. DRAWDOWN COMPARISON")
    print(f"{'='*80}")
    for lbl, r in [("Equal", r_equal), ("Kelly", r_kelly), ("NCO", r_nco)]:
        if not r: continue
        sim = r["sim"]
        eq = sim["equity"][~sim["skipped"]]
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.maximum(peak, 1)
        p95_dd = np.percentile(dd, 95) * 100
        p99_dd = np.percentile(dd, 99) * 100
        max_dd = dd.max() * 100
        avg_dd = dd[dd > 0.001].mean() * 100 if (dd > 0.001).any() else 0
        print(f"  {lbl:<10s} MaxDD={max_dd:.2f}%  95th={p95_dd:.2f}%  99th={p99_dd:.2f}%  AvgDD={avg_dd:.2f}%")

    # ================================================================
    print(f"\n{'='*80}")
    print("4. ROLLING SHARPE (60-day windows)")
    print(f"{'='*80}")
    for lbl, dpnl in [("Kelly", dpnl_k), ("NCO", dpnl_n)]:
        if len(dpnl) < 60: continue
        rolling = dpnl.rolling(60).apply(lambda x: x.mean()/x.std()*np.sqrt(365) if x.std()>0 else 0)
        rolling = rolling.dropna()
        print(f"  {lbl:<10s} Mean={rolling.mean():.3f}  Min={rolling.min():.3f}  "
              f"Max={rolling.max():.3f}  Std={rolling.std():.3f}  "
              f"<1.0: {(rolling < 1.0).sum()} days  <0: {(rolling < 0).sum()} days")

    # ================================================================
    print(f"\n{'='*80}")
    print("5. PER-ASSET ATTRIBUTION (Kelly vs NCO weight differences)")
    print(f"{'='*80}")
    # Get the LAST month's weights for comparison
    ts_k = pd.to_datetime(kelly_combined["timestamp"])
    ts_n = pd.to_datetime(nco_combined["timestamp"])
    last_month = ts_k.dt.to_period("M").max()

    # Compute per-asset PnL
    print(f"  {'Asset':<8s} {'Kelly PnL':>10s} {'NCO PnL':>10s} {'Delta':>10s}")
    print(f"  {'-'*40}")
    asset_pnl_k = {}
    asset_pnl_n = {}
    for r, combined, pnl_dict in [(r_kelly, kelly_combined, asset_pnl_k),
                                    (r_nco, nco_combined, asset_pnl_n)]:
        if r is None: continue
        sim = r["sim"]
        pnl = sim["pnl"]
        skipped = sim["skipped"]
        assets = combined["asset"].values
        for i in range(len(pnl)):
            if not skipped[i]:
                a = assets[i]
                pnl_dict[a] = pnl_dict.get(a, 0) + pnl[i]

    all_assets = sorted(set(asset_pnl_k.keys()) | set(asset_pnl_n.keys()))
    deltas = []
    for a in all_assets:
        pk = asset_pnl_k.get(a, 0)
        pn = asset_pnl_n.get(a, 0)
        d = pn - pk
        deltas.append((a, pk, pn, d))
    deltas.sort(key=lambda x: -x[3])
    for a, pk, pn, d in deltas:
        marker = " <<<" if abs(d) > 500 else ""
        print(f"  {a:<8s} ${pk:>9.2f} ${pn:>9.2f} ${d:>+9.2f}{marker}")

    total_k = sum(v for v in asset_pnl_k.values())
    total_n = sum(v for v in asset_pnl_n.values())
    print(f"  {'TOTAL':<8s} ${total_k:>9.2f} ${total_n:>9.2f} ${total_n-total_k:>+9.2f}")

    # ================================================================
    print(f"\n{'='*80}")
    print("6. REGIME ANALYSIS (high-vol vs low-vol months)")
    print(f"{'='*80}")
    # Classify months by BTC volatility
    monthly_k_dict = {m.strftime("%Y-%m"): v for m, v in monthly_k.items()}
    monthly_n_dict = {m.strftime("%Y-%m"): v for m, v in monthly_n.items()}

    # High-vol months: Nov-Dec 2024, Jan-Feb 2025 (crypto bull)
    # Low-vol months: Apr-Aug 2025 (quieter period)
    high_vol = ["2024-11", "2024-12", "2025-01", "2025-02", "2023-11", "2023-12"]
    low_vol = ["2024-04", "2024-09", "2025-04", "2025-05", "2025-06", "2025-08"]

    for regime, months_list in [("HIGH-VOL", high_vol), ("LOW-VOL", low_vol)]:
        k_total = sum(monthly_k_dict.get(m, 0) for m in months_list)
        n_total = sum(monthly_n_dict.get(m, 0) for m in months_list)
        print(f"  {regime} months ({', '.join(months_list[:3])}...):")
        print(f"    Kelly: ${k_total:>9.2f}   NCO: ${n_total:>9.2f}   Delta: ${n_total-k_total:>+9.2f}")

    # ================================================================
    print(f"\n{'='*80}")
    print("7. BOOTSTRAP SHARPE COMPARISON")
    print(f"{'='*80}")
    rng = np.random.RandomState(42)
    for lbl, dpnl in [("Kelly", dpnl_k), ("NCO", dpnl_n)]:
        arr = dpnl.values
        boots = []
        for _ in range(5000):
            s = rng.choice(arr, size=len(arr), replace=True)
            std = s.std()
            if std > 0:
                boots.append(s.mean() / std * np.sqrt(365))
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {lbl:<10s} 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  Median: {np.median(boots):.3f}")

    # Paired difference test
    k_arr = dpnl_k.values
    n_arr = dpnl_n.reindex(dpnl_k.index).fillna(0).values
    min_len = min(len(k_arr), len(n_arr))
    diff = n_arr[:min_len] - k_arr[:min_len]
    diff_mean = diff.mean()
    diff_se = diff.std() / np.sqrt(len(diff))
    t_stat = diff_mean / diff_se if diff_se > 0 else 0
    p_val = 2 * (1 - norm.cdf(abs(t_stat)))
    print(f"\n  Paired difference (NCO - Kelly):")
    print(f"    Mean daily diff: ${diff_mean:.4f}")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_val:.4f}")
    print(f"    Significant at 5%: {'YES' if p_val < 0.05 else 'NO'}")

    # ================================================================
    print(f"\n{'='*80}")
    print("8. VERDICT")
    print(f"{'='*80}")
    sk = r_kelly["stats"]["sharpe"] if r_kelly else 0
    sn = r_nco["stats"]["sharpe"] if r_nco else 0
    dk = r_kelly["stats"]["max_dd_pct"] if r_kelly else 0
    dn = r_nco["stats"]["max_dd_pct"] if r_nco else 0
    pk = r_kelly["pnl_day"] if r_kelly else 0
    pn = r_nco["pnl_day"] if r_nco else 0
    print(f"  Sharpe:  Kelly {sk:.3f} vs NCO {sn:.3f} (delta {sn-sk:+.3f})")
    print(f"  MaxDD:   Kelly {dk:.2f}% vs NCO {dn:.2f}% (delta {dn-dk:+.2f}%)")
    print(f"  PnL/day: Kelly ${pk:.2f} vs NCO ${pn:.2f} (delta ${pn-pk:+.2f})")
    print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'} (p={p_val:.4f})")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
