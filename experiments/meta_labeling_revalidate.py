"""Tier 0B: Meta-labeling re-validation with production pipeline.

Re-tests GBM meta-labeler on the CURRENT system (8mo, 13 features, 30 assets,
Kelly, fixed exposure). Previous +0.40 Sharpe claim was before exposure fix.

Architecture:
1. LogReg primary predicts direction (same as production)
2. GBM meta-labeler predicts {correct, incorrect} using features + primary prob
3. Only trade when meta says "correct" with confidence >= threshold
4. Compare: baseline (no meta) vs meta-filtered

Fully parallelized across 30 assets.
"""
import sys
sys.path.insert(0, "src")

import copy, os, time, warnings
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

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
    """Generate signals WITH and WITHOUT meta-labeling for one asset."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from signal_engine.config import load_config, ConfidenceFilterConfig
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from signal_engine.filters import apply_confidence_filter
    from afml.bet_sizing import get_signal, discretize_signal
    from sklearn.ensemble import GradientBoostingClassifier

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, [], []

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
        return asset, [], []

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    baseline_results = []
    meta_results = {}  # {threshold: [(month_str, sig_dict)]}
    for thr in [0.50, 0.55, 0.60]:
        meta_results[thr] = []

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
        month_str = str(month)

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

            # Primary model
            primary = make_logreg()
            primary.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)

            # Primary predictions on TRAINING data (for meta-labeler training)
            train_probs = primary.predict_proba(Xtr)[:, 1]
            train_dirs = np.where(train_probs > 0.5, 1, -1)
            actual_dirs = np.where(y_all.loc[tr].values > 0, 1, -1)
            train_correct = (train_dirs == actual_dirs).astype(int)

            # Meta-labeler: features + primary prob -> correct/incorrect
            meta_X = np.column_stack([Xtr, train_probs])
            meta_scaler = StandardScaler()
            meta_X_s = meta_scaler.fit_transform(meta_X)

            meta_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, min_samples_leaf=20,
                learning_rate=0.05, random_state=42, subsample=0.8,
            )
            meta_model.fit(meta_X_s, train_correct, sample_weight=sw)

            # Test predictions
            Xte = scaler.transform(fc.loc[te])
            p_probs = primary.predict_proba(Xte)[:, 1]

            # Meta predictions on test
            meta_X_te = np.column_stack([Xte, p_probs])
            meta_X_te_s = meta_scaler.transform(meta_X_te)
            meta_probs = meta_model.predict_proba(meta_X_te_s)[:, 1]
        except Exception:
            continue

        conf = np.abs(p_probs - 0.5)
        dirs = np.where(p_probs > 0.5, 1.0, -1.0)
        sigs = get_signal(p_probs, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(p_probs), abs(float(sd)))
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any():
            continue

        # Baseline signals (confidence-filtered, same as production)
        signals = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
            "probability": p_probs[finite], "signal_strength": ss[finite],
        })
        signals["asset"] = asset
        filtered = apply_confidence_filter(
            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered) > 0:
            baseline_results.append((month_str, filtered.to_dict("list")))

        # Meta-filtered signals (confidence filter + meta threshold)
        for thr in meta_results:
            meta_pass = meta_probs[finite] >= thr
            conf_pass = conf[finite] >= 0.08
            combined_mask = meta_pass & conf_pass
            if combined_mask.any():
                meta_sigs = pd.DataFrame({
                    "timestamp": te[finite][combined_mask],
                    "direction": dirs[finite][combined_mask],
                    "fwd_return": fv[finite][combined_mask],
                    "confidence": conf[finite][combined_mask],
                    "probability": p_probs[finite][combined_mask],
                    "signal_strength": ss[finite][combined_mask],
                })
                meta_sigs["asset"] = asset
                if len(meta_sigs) > 0:
                    meta_results[thr].append((month_str, meta_sigs.to_dict("list")))

    return asset, baseline_results, meta_results


def apply_kelly(combined):
    """Walk-forward Kelly."""
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


def run_sim(label, all_results, start_after=None):
    """Aggregate, apply Kelly, run portfolio sim."""
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results:
            if start_after and month_str < start_after:
                continue
            sigs.append(pd.DataFrame(sig_dict))
    if not sigs:
        return None

    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    combined = apply_kelly(combined)

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"]/len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {
        "sharpe": stats["sharpe"], "return_pct": stats["total_return_pct"],
        "max_dd_pct": stats["max_dd_pct"], "n_trades": sim["n_trades"],
        "pnl_day": pnl_day, "n_signals": len(combined),
    }


def main():
    print("=" * 85)
    print("TIER 0B: META-LABELING RE-VALIDATION")
    print(f"  8mo lookback, 13 features, 30 assets, Kelly, {W} workers")
    print(f"  Primary: LogReg L1 | Meta: GBM(100 trees, depth=3)")
    print(f"  Test: meta thresholds [0.50, 0.55, 0.60]")
    print("=" * 85, flush=True)

    t0 = time.time()

    # Generate signals (parallel across assets)
    print(f"\n[1] Generating signals with meta-labeling ({W} workers)...", flush=True)
    all_baseline = {}
    all_meta = {0.50: {}, 0.55: {}, 0.60: {}}

    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(_process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, baseline, meta = f.result()
            done += 1
            all_baseline[asset] = baseline
            for thr in meta:
                all_meta[thr][asset] = meta[thr]
            n_base = sum(len(pd.DataFrame(d)) for _, d in baseline)
            print(f"  [{done:>2d}/30] {asset}: {len(baseline)} months, {n_base} baseline sigs", flush=True)

    print(f"  Signal generation: {time.time()-t0:.0f}s", flush=True)

    # Run portfolio sims
    print(f"\n{'='*85}")
    print("RESULTS (Version B: {})".format(FIXED_START))
    print(f"{'='*85}", flush=True)

    print("\n--- Baseline (no meta, same as production) ---", flush=True)
    base_stats = run_sim("Baseline", all_baseline, FIXED_START)
    baseline_sharpe = base_stats["sharpe"] if base_stats else 0
    if base_stats:
        print(f"  Sharpe {base_stats['sharpe']:.3f}  Ret {base_stats['return_pct']:.2f}%  "
              f"MaxDD {base_stats['max_dd_pct']:.2f}%  Trades {base_stats['n_trades']}  "
              f"PnL/day ${base_stats['pnl_day']:.2f}")

    print("\n--- With meta-labeling ---", flush=True)
    for thr in [0.50, 0.55, 0.60]:
        stats = run_sim(f"Meta >= {thr}", all_meta[thr], FIXED_START)
        if stats:
            delta = stats["sharpe"] - baseline_sharpe
            print(f"  Meta >= {thr:.2f}: Sharpe {stats['sharpe']:.3f} ({delta:+.3f})  "
                  f"Ret {stats['return_pct']:.2f}%  MaxDD {stats['max_dd_pct']:.2f}%  "
                  f"Trades {stats['n_trades']}  PnL/day ${stats['pnl_day']:.2f}")
        else:
            print(f"  Meta >= {thr:.2f}: NO RESULT")

    print(f"\n{'='*85}")
    print(f"Done in {time.time()-t0:.0f}s")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
