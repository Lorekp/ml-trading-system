"""Test feature pruning based on MDA results.

Compares current 14 features vs pruned sets on the production pipeline
(8-month lookback, walk-forward Kelly, Version B 2023-08 to 2026-02).

Feature sets to test:
  A: All 14 (baseline)
  B: Remove HARMFUL (drop dist_nearest_level) = 13
  C: Remove HARMFUL + NOISE (keep only IMPORTANT + useful) = 7
  D: Core only (stoch_rsi, bb_pct_b, rsi_14, ret_14, ret_7, ret_1) = 6
  E: Top 3 only (stoch_rsi, bb_pct_b, rsi_14) = 3
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

from signal_engine.config import load_config, ConfidenceFilterConfig
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
FIXED_START = "2023-08"
LOOKBACK = 8

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

# Feature sets based on MDA ranking
FEATURE_SETS = {
    "A: All 14 (baseline)": [
        "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
        "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
        "dist_swing_high", "dist_swing_low", "dist_nearest_level",
    ],
    "B: Drop harmful (13)": [
        "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
        "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
        "dist_swing_high", "dist_swing_low",
    ],
    "C: Important+useful (7)": [
        "stoch_rsi", "bb_pct_b", "rsi_14", "ret_14", "ret_7", "ret_1",
        "dist_swing_high",
    ],
    "D: Core 6": [
        "stoch_rsi", "bb_pct_b", "rsi_14", "ret_14", "ret_7", "ret_1",
    ],
    "E: Top 3 only": [
        "stoch_rsi", "bb_pct_b", "rsi_14",
    ],
}


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate signals for one asset with a given feature set."""
    asset, symbol, feature_list = args
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
    feat = features[[c for c in feature_list if c in features.columns]].copy()
    if len(feat.columns) < len(feature_list):
        return asset, []

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
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

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
            results.append((month_str, filtered.to_dict("list")))

    return asset, results


def apply_kelly(combined):
    """Walk-forward Kelly (same as production pipeline)."""
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    timestamps_pd = pd.to_datetime(combined["timestamp"])
    months = sorted(timestamps_pd.dt.to_period("M").unique())
    parts = []
    for km in months:
        km_mask = timestamps_pd.dt.to_period("M") == km
        km_sigs = combined[km_mask]
        past_mask = timestamps_pd.dt.to_period("M") < km
        past_sigs = combined[past_mask]
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
            wp = len(w) / len(tr)
            wr = float(w.mean())
            lr = float(abs(lo.mean()))
            if lr <= 0:
                kw[an] = 1.0
                continue
            fs = kelly_size(wp, wr, lr)
            kw[an] = max(0.2, min(3.0, fs * 7.5))
        if kw:
            mw = np.mean(list(kw.values()))
            if mw > 0:
                kw = {k: v / mw for k, v in kw.items()}
        kmc = km_sigs.copy()
        kmc["signal_strength"] = kmc.apply(
            lambda r: r["signal_strength"] * kw.get(r["asset"], 1.0), axis=1)
        parts.append(kmc)
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def run_test(feature_set_name, feature_list):
    """Run full pipeline for one feature set."""
    t0 = time.time()

    # Generate signals (parallel)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {
            pool.submit(_process_asset, (a, s, feature_list)): a
            for a, s in ASSETS.items()
        }
        for f in as_completed(futures):
            asset, results = f.result()
            all_results[asset] = results

    # Aggregate (Version B: start at 2023-08)
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results:
            if month_str < FIXED_START:
                continue
            sigs.append(pd.DataFrame(sig_dict))

    if not sigs:
        return None

    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)

    # Walk-forward Kelly
    combined = apply_kelly(combined)

    # Portfolio sim
    cfg = load_config(Path("configs/default.yaml"))
    pcfg = copy.deepcopy(cfg.portfolio)
    pcfg.base_sizing_pct = 0.04
    pcfg.slippage = 0.0001
    pcfg.max_exposure = 0.90
    pcfg.max_per_asset = 0.03

    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None

    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl))
    elapsed = time.time() - t0
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)

    return {
        "sharpe": stats["sharpe"],
        "return_pct": stats["total_return_pct"],
        "max_dd_pct": stats["max_dd_pct"],
        "n_trades": sim["n_trades"],
        "n_days": stats["n_days"],
        "pnl_day": pnl_day,
        "n_signals": len(combined),
        "elapsed": elapsed,
    }


def main():
    print("=" * 85)
    print("FEATURE PRUNING TEST -- MDA-guided feature selection")
    print(f"  8-month lookback, 30 assets, Version B (2023-08+), Kelly, {W} workers")
    print("=" * 85, flush=True)

    results = {}
    for name, features in FEATURE_SETS.items():
        print(f"\n--- {name} [{len(features)} features] ---", flush=True)
        print(f"  Features: {', '.join(features)}", flush=True)
        r = run_test(name, features)
        results[name] = r
        if r:
            print(
                f"  Sharpe {r['sharpe']:>7.3f}  "
                f"Ret {r['return_pct']:>7.2f}%  "
                f"MaxDD {r['max_dd_pct']:>6.2f}%  "
                f"Trades {r['n_trades']:>6d}  "
                f"PnL/day ${r['pnl_day']:>7.2f}  "
                f"({r['elapsed']:.0f}s)",
                flush=True,
            )
        else:
            print("  NO RESULT", flush=True)

    # Summary table
    print(f"\n{'='*85}")
    print("SUMMARY")
    print(f"{'='*85}")
    baseline_sharpe = results.get("A: All 14 (baseline)", {}).get("sharpe", 0) if results.get("A: All 14 (baseline)") else 0
    print(f"\n  {'Config':<30s} {'Feats':>5s} {'Sharpe':>7s} {'Delta':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'PnL/day':>8s}")
    print(f"  {'-'*75}")
    for name, r in results.items():
        if r is None:
            continue
        n_feats = len(FEATURE_SETS[name])
        delta = r["sharpe"] - baseline_sharpe
        print(
            f"  {name:<30s} {n_feats:>5d} {r['sharpe']:>7.3f} {delta:>+7.3f} "
            f"{r['return_pct']:>8.2f} {r['max_dd_pct']:>7.2f} ${r['pnl_day']:>7.2f}"
        )

    print(f"\n{'='*85}")
    print("Accept pruning only if Sharpe improves AND MaxDD doesn't worsen")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
