"""Tier 2B: Portfolio Allocation Comparison (HRP, NCO, Kelly, Denoised).

Uses PROPER 8-month rolling signals (same as production pipeline).
Only the allocation method differs — signals are identical across tests.

Methods:
  A: Equal weight (baseline)
  B: Walk-forward Kelly (current production)
  C: HRP (AFML Ch16)
  D: Denoised inverse-variance (ML for AM Ch2)
  E: NCO k=5 (ML for AM Ch7)

Fully parallelized signal generation across 30 assets.
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
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
    """Generate 8mo rolling signals + daily returns for one asset."""
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
        return asset, [], None

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
        return asset, [], None

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    # Daily returns for correlation matrix
    daily_ret = close.resample("D").last().pct_change().dropna()

    results = []
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

        train_start = ms - pd.DateOffset(months=LOOKBACK)
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
            results.append((str(month), filtered.to_dict("list")))

    return asset, results, daily_ret.to_dict() if daily_ret is not None else None


# ---- Allocation methods ----

def compute_hrp_weights(corr, names):
    n = len(names)
    if n <= 1:
        return {names[0]: 1.0} if n == 1 else {}
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = leaves_list(link).astype(int)
    cov = np.diag(np.diag(corr))
    weights = np.ones(n)

    def _bisect(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left, right = items[:mid], items[mid:]
        vl = sum(cov[i, i] for i in left)
        vr = sum(cov[i, i] for i in right)
        t = vl + vr
        if t <= 0:
            return
        alpha = 1 - vl / t
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1 - alpha)
        _bisect(left)
        _bisect(right)

    _bisect(list(order))
    weights /= weights.mean()
    return {names[i]: weights[i] for i in range(n)}


def denoise_corr(corr, q):
    evals, evecs = np.linalg.eigh(corr)
    lp = (1 + np.sqrt(1 / q)) ** 2
    noise = evals < lp
    if noise.any():
        evals[noise] = evals[noise].mean()
    d = evecs @ np.diag(evals) @ evecs.T
    diag = np.sqrt(np.diag(d))
    diag[diag == 0] = 1
    d = d / np.outer(diag, diag)
    np.fill_diagonal(d, 1.0)
    return d


def compute_nco_weights(corr, exp_rets, names, n_clusters=5):
    n = len(names)
    if n <= 1:
        return {names[0]: 1.0} if n == 1 else {}
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="ward")
    clusters = fcluster(link, t=min(n_clusters, n), criterion="maxclust")
    weights = np.zeros(n)
    cr, cv = {}, {}
    for c in np.unique(clusters):
        m = np.where(clusters == c)[0]
        if len(m) == 1:
            weights[m[0]] = 1.0
            cr[c] = exp_rets[m[0]]
            cv[c] = np.sqrt(corr[m[0], m[0]])
        else:
            v = np.maximum(np.array([corr[i, i] for i in m]), 1e-10)
            iv = 1.0 / v
            w = iv / iv.sum()
            for j, mi in enumerate(m):
                weights[mi] = w[j]
            cr[c] = sum(exp_rets[mi] * w[j] for j, mi in enumerate(m))
            cv[c] = np.sqrt(sum(v * w**2))
    cids = sorted(cr.keys())
    cs = [max(cr[c] / max(cv[c], 1e-10), 0) for c in cids]
    ts = sum(cs)
    if ts <= 0:
        for c in cids:
            for mi in np.where(clusters == c)[0]:
                weights[mi] *= 1.0 / len(cids)
    else:
        for i, c in enumerate(cids):
            cw = cs[i] / ts
            for mi in np.where(clusters == c)[0]:
                weights[mi] *= cw
    if weights.sum() > 0:
        weights /= weights.mean()
    return {names[i]: weights[i] for i in range(n)}


def apply_weights(combined, weights):
    """Apply per-asset allocation weights to signal_strength."""
    out = combined.copy()
    out["signal_strength"] = out.apply(
        lambda r: r["signal_strength"] * weights.get(r["asset"], 1.0), axis=1)
    return out


def apply_kelly_wf(combined):
    """Walk-forward Kelly (production method)."""
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
        kw = {}
        for an in km_sigs["asset"].unique():
            pa = past_sigs[past_sigs["asset"] == an]
            if len(pa) < 10:
                kw[an] = 1.0
                continue
            tr = pa["direction"].values * pa["fwd_return"].values
            w = tr[tr > 0]; lo = tr[tr <= 0]
            if len(w) < 3 or len(lo) < 3:
                kw[an] = 1.0
                continue
            wp, wr, lr = len(w)/len(tr), float(w.mean()), float(abs(lo.mean()))
            if lr <= 0:
                kw[an] = 1.0
                continue
            kw[an] = max(0.2, min(3.0, kelly_size(wp, wr, lr) * 7.5))
        if kw:
            mw = np.mean(list(kw.values()))
            if mw > 0:
                kw = {k: v/mw for k, v in kw.items()}
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_sim(combined, label):
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label:<45s} NO RESULT", flush=True)
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"]/len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {"sharpe": stats["sharpe"], "return_pct": stats["total_return_pct"],
            "max_dd_pct": stats["max_dd_pct"], "n_trades": sim["n_trades"], "pnl_day": pnl_day}


def main():
    print("=" * 80)
    print("TIER 2B: PORTFOLIO ALLOCATION COMPARISON")
    print(f"  8mo rolling, 13 features, 30 assets, Version B ({FIXED_START}+), {W} workers")
    print(f"  Methods: Equal, Kelly WF, HRP, Denoised Inv-Var, NCO")
    print("=" * 80, flush=True)

    t0 = time.time()

    # Phase 1: Generate signals (parallel)
    print(f"\n[1] Generating 8mo rolling signals...", flush=True)
    all_results = {}
    all_daily_rets = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results, daily = f.result()
            done += 1
            all_results[asset] = results
            if daily is not None:
                all_daily_rets[asset] = pd.Series(daily)
            print(f"  [{done:>2d}/30] {asset}: {len(results)} months", flush=True)

    # Combine signals (Version B)
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results:
            if month_str < FIXED_START:
                continue
            sigs.append(pd.DataFrame(sig_dict))
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(combined):,d} signals in {time.time()-t0:.0f}s", flush=True)

    # Build correlation matrix
    common = sorted(set(all_results.keys()) & set(all_daily_rets.keys()))
    ret_df = pd.DataFrame({a: all_daily_rets[a] for a in common}).dropna(how="all").fillna(0)
    corr = ret_df.corr().values
    n = len(common)
    q = len(ret_df) / n
    print(f"  Correlation: {n}x{n}, T/N={q:.1f}, avg corr={corr[np.triu_indices(n,k=1)].mean():.3f}")

    # Expected returns per asset (from signals)
    exp_rets = np.zeros(n)
    for i, asset in enumerate(common):
        asset_sigs = [pd.DataFrame(d) for ms, d in all_results[asset] if ms >= FIXED_START]
        if asset_sigs:
            df = pd.concat(asset_sigs)
            exp_rets[i] = (df["direction"] * df["fwd_return"]).mean()

    # Phase 2: Test allocation methods
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}", flush=True)

    results = {}

    # A: Equal weight (no Kelly)
    print("\n--- A: Equal weight ---")
    r = run_sim(combined, "Equal weight")
    results["Equal weight"] = r
    if r:
        print(f"  {'Equal weight':<45s} Sharpe {r['sharpe']:.3f}  Ret {r['return_pct']:.2f}%  MaxDD {r['max_dd_pct']:.2f}%  PnL/day ${r['pnl_day']:.2f}")

    # B: Walk-forward Kelly (production)
    print("\n--- B: Walk-forward Kelly (production) ---")
    combined_kelly = apply_kelly_wf(combined)
    r = run_sim(combined_kelly, "WF Kelly")
    results["WF Kelly"] = r
    baseline_sharpe = r["sharpe"] if r else 0
    if r:
        print(f"  {'WF Kelly':<45s} Sharpe {r['sharpe']:.3f}  Ret {r['return_pct']:.2f}%  MaxDD {r['max_dd_pct']:.2f}%  PnL/day ${r['pnl_day']:.2f}")

    # C: HRP
    print("\n--- C: HRP (AFML Ch16) ---")
    hrp_w = compute_hrp_weights(corr, common)
    combined_hrp = apply_weights(combined, hrp_w)
    r = run_sim(combined_hrp, "HRP")
    results["HRP"] = r
    if r:
        delta = r["sharpe"] - baseline_sharpe
        print(f"  {'HRP':<45s} Sharpe {r['sharpe']:.3f} ({delta:+.3f})  Ret {r['return_pct']:.2f}%  MaxDD {r['max_dd_pct']:.2f}%  PnL/day ${r['pnl_day']:.2f}")

    # D: Denoised inverse-variance
    print("\n--- D: Denoised Inv-Var (ML for AM Ch2) ---")
    denoised = denoise_corr(corr, q)
    var = np.maximum(np.diag(denoised), 1e-10)
    iv = 1.0 / var
    iv_w = {common[i]: iv[i] / iv.mean() for i in range(n)}
    combined_div = apply_weights(combined, iv_w)
    r = run_sim(combined_div, "Denoised Inv-Var")
    results["Denoised Inv-Var"] = r
    if r:
        delta = r["sharpe"] - baseline_sharpe
        print(f"  {'Denoised Inv-Var':<45s} Sharpe {r['sharpe']:.3f} ({delta:+.3f})  Ret {r['return_pct']:.2f}%  MaxDD {r['max_dd_pct']:.2f}%  PnL/day ${r['pnl_day']:.2f}")

    # E: NCO
    print("\n--- E: NCO (ML for AM Ch7) ---")
    for k in [3, 5, 8]:
        nco_w = compute_nco_weights(denoised, exp_rets, common, n_clusters=k)
        combined_nco = apply_weights(combined, nco_w)
        r = run_sim(combined_nco, f"NCO k={k}")
        results[f"NCO k={k}"] = r
        if r:
            delta = r["sharpe"] - baseline_sharpe
            print(f"  {f'NCO k={k}':<45s} Sharpe {r['sharpe']:.3f} ({delta:+.3f})  Ret {r['return_pct']:.2f}%  MaxDD {r['max_dd_pct']:.2f}%  PnL/day ${r['pnl_day']:.2f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (sorted by Sharpe)")
    print(f"{'='*80}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["sharpe"] if x[1] else -99, reverse=True):
        if r:
            delta = r["sharpe"] - baseline_sharpe
            print(f"  {name:<25s} Sharpe {r['sharpe']:.3f} ({delta:+.3f})  MaxDD {r['max_dd_pct']:.2f}%  PnL/day ${r['pnl_day']:.2f}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
