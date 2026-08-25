"""Test the original leak-free pipeline (LogReg L1, base features, NCO).

Exact copy of final_production_report.py logic, ported to vectorized
walk-forward with explicit leakage checks.

Key differences from the regime-XGBoost version:
- LogisticRegression L1 (C=0.1) instead of XGBoost
- StandardScaler on features
- Base 18 features only (NO regime features)
- Confidence filter: absolute threshold 0.08
- NCO k=8 portfolio allocation
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.allocation import compute_nco_weights, compute_kelly_weights
from afml.bet_sizing import get_signal, discretize_signal

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LOOKBACKS = [2, 4, 6, 8, 10, 12]

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

SEARCH_START = "2022-01"
SEARCH_END = "2024-12"
HOLDOUT_START = "2025-01"


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def _process_asset(args):
    """Generate signals for ALL lookbacks for one asset. Exact copy of original."""
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
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, {}

    # BASE FEATURES ONLY — no regime features, no leakage risk
    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Winsorize (global — minor, matches original)
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

        # Vol filter from prior 3 months
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
    """Walk-forward NCO k=8 allocation — exact copy from original."""
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
        assets_in_month = sorted(km_sigs["asset"].unique())
        daily_rets = {}
        for a in assets_in_month:
            a_past = past_sigs[past_sigs["asset"] == a]
            if len(a_past) < 10:
                continue
            ts_a = pd.to_datetime(a_past["timestamp"])
            rets = pd.Series(a_past["direction"].values * a_past["fwd_return"].values, index=ts_a)
            daily_rets[a] = rets.resample("D").sum()
        if len(daily_rets) < 3:
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
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    return {**stats, "n_trades": sim["n_trades"], "pnl_day": pnl_day,
            "dpnl": dpnl, "sim": sim, "combined": combined}


def flat_bet_stats(all_results, lb, start_after=None, end_before=None):
    """Simple flat-bet stats for comparison (no portfolio sim)."""
    sigs = []
    for asset_results in all_results.values():
        for month_str, sig_dict in asset_results.get(lb, []):
            if start_after and month_str < start_after:
                continue
            if end_before and month_str > end_before:
                continue
            sigs.append(pd.DataFrame(sig_dict))
    if not sigs:
        return None
    df = pd.concat(sigs)
    pnls = df["direction"].values * df["fwd_return"].values - 0.0001
    ts_arr = pd.to_datetime(df["timestamp"], utc=True)
    daily = pd.Series(5.0 * pnls, index=ts_arr.values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return None
    std = daily.std()
    sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
    return {
        "sharpe": sr, "win_rate": (pnls > 0).mean() * 100,
        "mean_bps": pnls.mean() * 10000, "n_trades": len(df),
        "n_days": len(daily), "n_assets": df["asset"].nunique(),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("ORIGINAL PIPELINE TEST (LogReg L1, base features, NCO k=8)")
    print(f"  30 assets, {W} workers")
    print(f"  Lookbacks: {LOOKBACKS}")
    print(f"  Model: LogisticRegression(C=0.1, L1)")
    print(f"  Features: base 18 only (NO regime)")
    print(f"  Confidence: absolute > 0.08")
    print("=" * 70, flush=True)

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
            total = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {total} month-lookback combos", flush=True)

    print(f"\n  Phase 1 done in {time.time()-t0:.0f}s")

    # Phase 2: Flat-bet comparison (simple, no portfolio sim)
    print(f"\n{'='*80}")
    print("FLAT-BET RESULTS (no portfolio sim, $5/trade)")
    print(f"{'='*80}")

    print(f"\n  SEARCH ({SEARCH_START} to {SEARCH_END}):")
    print(f"  {'LB':<5s} {'Sharpe':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>8s} {'Assets':>6s}")
    print("  " + "-" * 45)
    for lb in LOOKBACKS:
        r = flat_bet_stats(all_results, lb, SEARCH_START, SEARCH_END)
        if r:
            print(f"  {lb:>2d}mo {r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% {r['mean_bps']:>6.2f} {r['n_trades']:>8d} {r['n_assets']:>5d}")
        else:
            print(f"  {lb:>2d}mo  NO DATA")

    print(f"\n  HOLDOUT ({HOLDOUT_START}+):")
    print(f"  {'LB':<5s} {'Sharpe':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>8s} {'Assets':>6s}")
    print("  " + "-" * 45)
    for lb in LOOKBACKS:
        r = flat_bet_stats(all_results, lb, HOLDOUT_START)
        if r:
            print(f"  {lb:>2d}mo {r['sharpe']:>7.3f} {r['win_rate']:>5.1f}% {r['mean_bps']:>6.2f} {r['n_trades']:>8d} {r['n_assets']:>5d}")
        else:
            print(f"  {lb:>2d}mo  NO DATA")

    # Phase 3: NCO portfolio simulation
    print(f"\n{'='*80}")
    print("NCO k=8 PORTFOLIO RESULTS")
    print(f"{'='*80}")

    # Determine common start
    sample_1m = load_binance_1m("BTCUSDT", [DATA_DIR])
    sample_15m = resample_ohlcv(sample_1m, "15min")
    data_start = sample_15m.index.min()
    fixed_start = str((data_start + pd.DateOffset(months=max(LOOKBACKS) + 1)).to_period("M"))
    print(f"  Common start: {fixed_start}")

    print(f"\n  {'LB':<5s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>8s} {'Trades':>8s} {'Days':>6s} {'PnL/day':>9s}")
    print("  " + "-" * 55)
    for lb in LOOKBACKS:
        r = run_lookback(lb, all_results, fixed_start)
        if r:
            print(f"  {lb:>2d}mo {r['sharpe']:>7.3f} {r['total_return_pct']:>8.2f} "
                  f"{r['max_dd_pct']:>8.2f} {r['n_trades']:>8d} {r['n_days']:>6d} "
                  f"${r['pnl_day']:>8.2f}", flush=True)
        else:
            print(f"  {lb:>2d}mo  NO SIGNALS", flush=True)

    # Phase 4: Full stats for best config
    print(f"\n{'='*80}")
    print("FULL STATS: 8-MONTH ROLLING + NCO k=8")
    print(f"{'='*80}")
    prod = run_lookback(8, all_results, fixed_start)
    if prod:
        dpnl = prod["dpnl"]
        pcfg = get_pcfg()
        daily_ret = dpnl / pcfg.starting_capital
        print(f"  Sharpe:     {prod['sharpe']:.3f}")
        print(f"  Total Ret:  {prod['total_return_pct']:.2f}%")
        print(f"  Max DD:     {prod['max_dd_pct']:.2f}%")
        print(f"  Trades:     {prod['n_trades']}")
        print(f"  Days:       {prod['n_days']}")
        print(f"  PnL/day:    ${prod['pnl_day']:.2f}")
        print(f"  Final eq:   ${prod.get('final_equity', 0):.2f}")

        # Monthly PnL
        monthly = dpnl.resample("ME").sum()
        pos_months = (monthly > 0).sum()
        print(f"  Pos months: {pos_months}/{len(monthly)} ({pos_months/len(monthly)*100:.0f}%)")

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boots = []
        arr = dpnl.values
        for _ in range(5000):
            s = rng.choice(arr, len(arr), replace=True)
            if s.std() > 0:
                boots.append(s.mean() / s.std() * np.sqrt(365))
        print(f"  Bootstrap 95% CI: [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")

    print(f"\n  Total runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
