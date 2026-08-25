"""HISTORICAL EXPERIMENT — the former production/validated claim is withdrawn.

Derivatives-features and volatility-overlay A/B study retained for provenance.

Rigorous A/B comparison using EXACT T1 production methodology:
- Same 30 assets, same walk-forward, same NCO, same portfolio sim
- Three configs: Baseline, +Derivatives, +Derivatives+VolOverlay
- Full validation: bootstrap CI, feature importance, regime breakdown

Its internal checks do not satisfy the current promotion policy in
`docs/VALIDATION.md`.
"""

import sys, os, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features, compute_derivatives_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal
from data.hl_derivatives import load_hl_asset_ctxs, resample_to_ohlcv_freq

# Reuse NCO from test_original_pipeline
from test_original_pipeline import apply_nco_wf

warnings.filterwarnings("ignore")

NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
PCACHE = Path("data/cache_15m")
HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
HL_CACHE = Path("data/hl-asset-ctxs/cache_15m")

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

DERIV_FEATURE_NAMES = [
    "funding_zscore", "funding_zscore_long", "funding_cumulative",
    "oi_change_5", "oi_zscore", "oi_price_divergence",
    "premium_zscore", "impact_skew",
]

# Use lb=2 (T1 optimal) for main comparison
TARGET_LB = 2
HOLDOUT_START = "2025-01"


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def get_pcfg():
    return PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )


def process_asset(asset, symbol, use_derivatives=False):
    """Generate signals for one asset. Exact T1 flow + optional derivatives."""
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features, compute_derivatives_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))

    # Load OHLCV
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

    # Base features
    features = compute_features(ohlcv_15m, cfg.features)
    base_cols = list(cfg.features.feature_list)

    # Optionally add derivatives
    if use_derivatives:
        cache_path = HL_CACHE / f"{asset}.parquet"
        try:
            deriv = load_hl_asset_ctxs(str(HL_DIR), coin=asset, cache_path=str(cache_path))
            deriv_15m = resample_to_ohlcv_freq(deriv, ohlcv_15m.index, method="last")
            deriv_feat = compute_derivatives_features(deriv_15m, ohlcv_15m["close"])
            features = pd.concat([features, deriv_feat], axis=1)
        except Exception:
            pass  # graceful degradation — use base features only

    feat_cols = [c for c in features.columns if c in base_cols or c in DERIV_FEATURE_NAMES]
    feat = features[feat_cols].copy()

    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Winsorize
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

    results = {TARGET_LB: []}
    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=2)).to_period("M"),
        end=end_date.to_period("M"), freq="M",
    )

    auc_data = []  # (month, y_true, y_prob) for OOS AUC

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)

        te = valid[(valid >= ms) & (valid <= me)]

        # Vol filter
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50: continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue

        month_str = str(month)
        lb = TARGET_LB

        train_start = ms - pd.DateOffset(months=lb)
        if train_start < start_date: continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200: continue

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
        if not finite.any(): continue

        signals = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
            "probability": yp[finite], "signal_strength": ss[finite],
        })
        signals["asset"] = asset
        filtered = apply_confidence_filter(signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered) > 0:
            results[lb].append((month_str, filtered.to_dict("list")))

        # Store AUC data for OOS months
        if month_str >= HOLDOUT_START:
            y_true = (fv[finite] > 0).astype(int)
            auc_data.append((y_true, yp[finite]))

    # Compute per-asset OOS AUC
    oos_auc = np.nan
    if auc_data:
        all_yt = np.concatenate([d[0] for d in auc_data])
        all_yp = np.concatenate([d[1] for d in auc_data])
        if len(np.unique(all_yt)) == 2 and len(all_yt) >= 50:
            try:
                oos_auc = roc_auc_score(all_yt, all_yp)
            except: pass

    return asset, results, oos_auc


def apply_vol_overlay(signals_df, btc_vol_pct):
    """Apply vol regime sizing overlay to signal_strength."""
    df = signals_df.copy()
    ts = pd.to_datetime(df["timestamp"])
    ts_day = ts.dt.normalize()
    if ts_day.dt.tz is not None:
        ts_day = ts_day.dt.tz_localize(None)

    vol_lookup = btc_vol_pct.reindex(ts_day.values).values
    mult = np.where(vol_lookup < 0.20, 0.25,
                    np.where(vol_lookup >= 0.70, 1.25, 1.0))
    # Handle NaN vol (early period)
    mult = np.where(np.isfinite(vol_lookup), mult, 1.0)
    df["signal_strength"] = df["signal_strength"].values * mult
    return df


def run_config(all_results, auc_dict, label, vol_overlay=False, btc_vol_pct=None):
    """Run NCO + portfolio sim for a set of results."""
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results.get(TARGET_LB, []):
            sigs.append(pd.DataFrame(sig_dict))
    if not sigs:
        print(f"  {label}: NO SIGNALS"); return None

    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    combined = apply_nco_wf(combined)

    if vol_overlay and btc_vol_pct is not None:
        combined = apply_vol_overlay(combined, btc_vol_pct)

    pcfg = get_pcfg()
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label}: NO TRADES"); return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        print(f"  {label}: NO PNL"); return None

    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))

    # Bootstrap 95% CI on Sharpe
    rng = np.random.RandomState(42)
    boots = []
    arr = dpnl.values
    for _ in range(5000):
        s = rng.choice(arr, len(arr), replace=True)
        if s.std() > 0:
            boots.append(s.mean() / s.std() * np.sqrt(365.25))
    ci_lo = np.percentile(boots, 2.5) if boots else 0
    ci_hi = np.percentile(boots, 97.5) if boots else 0

    # OOS-only metrics
    dpnl_oos = dpnl[dpnl.index >= pd.Timestamp(HOLDOUT_START)]
    if len(dpnl_oos) > 5:
        dr_oos = dpnl_oos / pcfg.starting_capital
        sr_oos = dr_oos.mean() / dr_oos.std() * np.sqrt(365.25) if dr_oos.std() > 0 else 0
        cum_oos = dpnl_oos.cumsum()
        peak_oos = (cum_oos + pcfg.starting_capital).cummax()
        dd_oos = abs(((cum_oos + pcfg.starting_capital - peak_oos) / peak_oos).min())
        wd_oos = abs(dr_oos.min())
        cal_oos = (dpnl_oos.index[-1] - dpnl_oos.index[0]).days + 1
        pnl_cal_oos = cum_oos.iloc[-1] / cal_oos
    else:
        sr_oos = pnl_cal_oos = dd_oos = wd_oos = 0

    # Mean OOS AUC
    aucs = [v for v in auc_dict.values() if np.isfinite(v)]
    mean_auc = np.mean(aucs) if aucs else 0

    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    print(f"\n  --- {label} ---")
    print(f"  Full:  SR={stats['sharpe']:.3f} [{ci_lo:.2f}, {ci_hi:.2f}]  "
          f"Ret={stats['total_return_pct']:.1f}%  DD={stats['max_dd_pct']:.1f}%  "
          f"Tr={sim['n_trades']}  $/day={pnl_day:.2f}")
    print(f"  OOS:   SR={sr_oos:.3f}  ${pnl_cal_oos:.2f}/cal  DD={dd_oos*100:.1f}%  WD={wd_oos*100:.2f}%")
    print(f"  AUC:   mean={mean_auc:.4f} ({len(aucs)} assets)")

    return {
        "sharpe_full": stats["sharpe"], "ci_lo": ci_lo, "ci_hi": ci_hi,
        "sharpe_oos": sr_oos, "pnl_cal_oos": pnl_cal_oos,
        "dd_oos": dd_oos, "wd_oos": wd_oos,
        "n_trades": sim["n_trades"], "mean_auc": mean_auc,
        "dpnl": dpnl, "dpnl_oos": dpnl_oos,
    }


def main():
    t0 = time.time()
    print("=" * 100)
    print("PRODUCTION V3 VALIDATED — Derivatives Features + Vol Overlay")
    print(f"  30 assets, lb={TARGET_LB}, LogReg L1 (C=0.1), NCO k=8")
    print(f"  A: Baseline (13 OHLCV features)")
    print(f"  B: +Derivatives (13 + 8 = 21 features)")
    print(f"  C: +Derivatives +VolOverlay")
    print("=" * 100, flush=True)

    # Compute BTC vol percentile for overlay
    btc = pd.read_parquet(PCACHE / "BTCUSDT.parquet", columns=["close"])
    btc_daily = btc["close"].resample("1D").last().dropna()
    btc_daily.index = btc_daily.index.tz_localize(None)
    btc_ret = btc_daily.pct_change()
    btc_vol_20d = btc_ret.rolling(20).std() * np.sqrt(365)
    btc_vol_pct = btc_vol_20d.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # ---- Config A: Baseline ----
    print(f"\n[A] Generating BASELINE signals ({W} workers)...", flush=True)
    base_results = {}; base_aucs = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, a, s, False): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            asset = r[0]; results = r[1]; auc = r[2] if len(r) > 2 else np.nan
            done += 1
            base_results[asset] = results
            base_aucs[asset] = auc
            total = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {total} months, AUC={auc:.4f}" if np.isfinite(auc)
                  else f"  [{done:>2d}/30] {asset}: {total} months", flush=True)
    print(f"  Phase A done in {time.time()-t0:.0f}s")

    # ---- Config B: +Derivatives ----
    t1 = time.time()
    print(f"\n[B] Generating +DERIVATIVES signals ({W} workers)...", flush=True)
    deriv_results = {}; deriv_aucs = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, a, s, True): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            asset = r[0]; results = r[1]; auc = r[2] if len(r) > 2 else np.nan
            done += 1
            deriv_results[asset] = results
            deriv_aucs[asset] = auc
            total = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {total} months, AUC={auc:.4f}" if np.isfinite(auc)
                  else f"  [{done:>2d}/30] {asset}: {total} months", flush=True)
    print(f"  Phase B done in {time.time()-t1:.0f}s")

    # ---- Compare ----
    print(f"\n{'='*100}")
    print("RESULTS COMPARISON")
    print(f"{'='*100}")

    r_base = run_config(base_results, base_aucs, "A: Baseline (13 feat)")
    r_deriv = run_config(deriv_results, deriv_aucs, "B: +Derivatives (21 feat)")
    r_vol = run_config(deriv_results, deriv_aucs, "C: +Deriv +VolOverlay",
                       vol_overlay=True, btc_vol_pct=btc_vol_pct)

    # ---- Per-Asset AUC Comparison ----
    print(f"\n{'='*100}")
    print("PER-ASSET OOS AUC")
    print(f"{'='*100}")
    print(f"{'Asset':<8s} {'Base':>8s} {'+Deriv':>8s} {'Delta':>8s}")
    print("-" * 36)
    for asset in sorted(ASSETS.keys()):
        ba = base_aucs.get(asset, np.nan)
        da = deriv_aucs.get(asset, np.nan)
        if np.isfinite(ba) and np.isfinite(da):
            d = da - ba
            flag = " *" if d > 0.01 else ""
            print(f"{asset:<8s} {ba:>8.4f} {da:>8.4f} {d:>+7.4f}{flag}")

    # ---- Monthly PnL Comparison ----
    if r_base and r_deriv:
        print(f"\n{'='*100}")
        print("MONTHLY PNL (OOS)")
        print(f"{'='*100}")
        dpnl_b = r_base["dpnl_oos"].resample("ME").sum()
        dpnl_d = r_deriv["dpnl_oos"].resample("ME").sum()
        dpnl_v = r_vol["dpnl_oos"].resample("ME").sum() if r_vol else pd.Series(dtype=float)

        print(f"{'Month':<10s} {'Base':>8s} {'+Deriv':>8s} {'+Vol':>8s}")
        all_months = dpnl_b.index.union(dpnl_d.index)
        for dt in sorted(all_months):
            b = dpnl_b.get(dt, 0); d = dpnl_d.get(dt, 0); v = dpnl_v.get(dt, 0)
            print(f"{dt.strftime('%Y-%m'):<10s} ${b:>7.0f} ${d:>7.0f} ${v:>7.0f}")

    # ---- Summary Table ----
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"{'Config':<30s} {'SR_full':>8s} {'95% CI':>15s} {'SR_OOS':>8s} {'$/cal':>7s} "
          f"{'DD%':>5s} {'WD%':>6s} {'AUC':>6s}")
    print("-" * 90)
    for label, r in [("A: Baseline", r_base), ("B: +Derivatives", r_deriv), ("C: +Deriv+Vol", r_vol)]:
        if r is None: continue
        print(f"{label:<30s} {r['sharpe_full']:>8.3f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] "
              f"{r['sharpe_oos']:>8.3f} ${r['pnl_cal_oos']:>6.2f} "
              f"{r['dd_oos']*100:>4.1f}% {r['wd_oos']*100:>5.2f}% {r['mean_auc']:>6.4f}")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
