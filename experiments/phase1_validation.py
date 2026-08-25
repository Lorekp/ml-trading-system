"""Phase 1: Validation & Measurement.

Chunks 1.1-1.4: Cost audit, DSR, regime analysis, per-asset contribution.
No model changes — just measuring the existing baseline honestly.
"""

import sys
sys.path.insert(0, "src")

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

NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")

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


def process_asset(args):
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

    results = {8: []}
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
        train_start = ms - pd.DateOffset(months=8)
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
            results[8].append((month_str, filtered.to_dict("list")))

    return asset, results


def main():
    t0 = time.time()
    print("=" * 70)
    print("PHASE 1: VALIDATION & MEASUREMENT")
    print("  LogReg L1, 18 features, lb=8mo, 30 assets")
    print("=" * 70)

    print(f"\nGenerating signals ({W} workers)...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            all_results[asset] = results
            n = sum(len(v) for v in results.values())
            print(f"  [{done:>2d}/30] {asset}: {n} months", flush=True)

    print(f"\n  Signals done in {time.time()-t0:.0f}s")

    # Combine
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get(8, []):
            sigs.append(pd.DataFrame(sd))
    if not sigs:
        print("NO SIGNALS")
        return
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    print(f"  Total: {len(combined)} trades, {combined['asset'].nunique()} assets")

    # ========== CHUNK 1.1: COST AUDIT ==========
    print(f"\n{'='*70}")
    print("CHUNK 1.1: COST SENSITIVITY")
    print(f"{'='*70}")
    print(f"  {'Slip':>5s} {'Sharpe':>7s} {'Ret%':>9s} {'MaxDD%':>8s} {'PnL/day':>9s} {'Trades':>7s}")
    print("  " + "-" * 50)

    for slip_bps in [1, 3, 5, 7, 10, 15, 20]:
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
            slippage=slip_bps / 10000.0, max_exposure=0.90, max_per_asset=0.03,
            base_sizing_pct=0.04, max_signal_mult=2.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim and sim["n_trades"] >= 10:
            dpnl = compute_daily_pnl_series(sim)
            if dpnl is not None and len(dpnl) >= 5:
                stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                                           sim["n_trades"] / len(dpnl))
                pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
                print(f"  {slip_bps:>3d}bp {stats['sharpe']:>7.3f} {stats['total_return_pct']:>8.1f}% "
                      f"{stats['max_dd_pct']:>7.2f}% ${pnl_day:>8.2f} {sim['n_trades']:>7d}")

    # ========== CHUNK 1.2: DSR ==========
    print(f"\n{'='*70}")
    print("CHUNK 1.2: DSR & STATISTICAL VALIDATION")
    print(f"{'='*70}")

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.03,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )
    sim = simulate_portfolio(combined, pcfg)
    dpnl = compute_daily_pnl_series(sim)
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    sr = stats["sharpe"]
    T = stats["n_days"]
    skew = stats.get("skew", 0)
    kurt = stats.get("kurtosis", 3)

    sr_std = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / max(T, 1))
    psr_0 = norm.cdf(sr / sr_std) * 100 if sr_std > 0 else 100
    psr_1 = norm.cdf((sr - 1) / sr_std) * 100 if sr_std > 0 else 100

    K = 40
    emc = 0.5772
    e_z = norm.ppf(1 - 1 / max(K, 2))
    e_max = (1 - emc) * e_z + emc * norm.ppf(1 - 1 / (max(K, 2) * np.e))
    dsr = norm.cdf((sr - sr_std * e_max) / sr_std) * 100 if sr_std > 0 else 100

    print(f"  Sharpe:       {sr:.3f}")
    print(f"  Days:         {T}")
    print(f"  Skewness:     {skew:.3f}")
    print(f"  Kurtosis:     {kurt:.1f}")
    print(f"  SR std err:   {sr_std:.4f}")
    print(f"  PSR(SR*=0):   {psr_0:.1f}%")
    print(f"  PSR(SR*=1):   {psr_1:.1f}%")
    print(f"  DSR(K={K}):    {dsr:.1f}%")

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boots = []
    arr = dpnl.values
    for _ in range(5000):
        s = rng.choice(arr, len(arr), replace=True)
        if s.std() > 0:
            boots.append(s.mean() / s.std() * np.sqrt(365))
    print(f"  Bootstrap CI: [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")

    # ========== CHUNK 1.3: REGIME ==========
    print(f"\n{'='*70}")
    print("CHUNK 1.3: REGIME ANALYSIS (BTC vol terciles)")
    print(f"{'='*70}")

    ohlcv_1m_btc = load_binance_1m("BTCUSDT", [DATA_DIR])
    ohlcv_15m_btc = resample_ohlcv(ohlcv_1m_btc, "15min")
    btc_vol = ohlcv_15m_btc["close"].pct_change().rolling(20 * 24 * 4).std() * np.sqrt(365 * 24 * 4)
    btc_vol_daily = btc_vol.resample("1D").last().dropna()

    common = dpnl.index.intersection(btc_vol_daily.index)
    if len(common) > 100:
        vol_aligned = btc_vol_daily.reindex(common)
        pnl_aligned = dpnl.reindex(common)
        terciles = pd.qcut(vol_aligned, 3, labels=["Low Vol", "Med Vol", "High Vol"])
        for regime in ["Low Vol", "Med Vol", "High Vol"]:
            mask = terciles == regime
            rp = pnl_aligned[mask]
            if len(rp) > 10:
                s = rp.std()
                sr_r = rp.mean() / s * np.sqrt(365.25) if s > 0 else 0
                wr = (rp > 0).mean() * 100
                print(f"  {regime:<10s}: SR={sr_r:>6.3f}  Win={wr:.1f}%  Days={len(rp)}")

    # ========== CHUNK 1.4: PER-ASSET ==========
    print(f"\n{'='*70}")
    print("CHUNK 1.4: PER-ASSET CONTRIBUTION (flat-bet, lb=8)")
    print(f"{'='*70}")
    print(f"  {'Asset':<8s} {'SR(srch)':>8s} {'SR(hold)':>8s} {'Win%':>6s} {'bps':>6s} {'n':>6s}")
    print("  " + "-" * 46)

    for asset in sorted(ASSETS.keys()):
        ar = all_results.get(asset, {})
        parts = []
        for ms, sd in ar.get(8, []):
            parts.append(pd.DataFrame(sd))
        if not parts:
            continue
        df = pd.concat(parts)
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
        df["ym"] = df["ts"].dt.strftime("%Y-%m")

        sr_s = sr_h = 0.0
        wr_s = bps_s = 0.0
        nt = 0

        for period, start, end in [("search", "2022-01", "2024-12"), ("holdout", "2025-01", None)]:
            mask = df["ym"] >= start
            if end:
                mask &= df["ym"] <= end
            pf = df[mask]
            if len(pf) < 10:
                continue
            pnls = pf["direction"].values * pf["fwd_return"].values - 0.0001
            daily = pd.Series(5.0 * pnls, index=pf["ts"].values).resample("1D").sum()
            daily = daily[daily != 0]
            s = daily.std()
            sr_a = daily.mean() / s * np.sqrt(365.25) if len(daily) > 5 and s > 0 else 0
            if period == "search":
                sr_s = sr_a
                wr_s = (pnls > 0).mean() * 100
                bps_s = pnls.mean() * 10000
                nt = len(pf)
            else:
                sr_h = sr_a

        print(f"  {asset:<8s} {sr_s:>8.3f} {sr_h:>8.3f} {wr_s:>5.1f}% {bps_s:>5.2f} {nt:>6d}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
