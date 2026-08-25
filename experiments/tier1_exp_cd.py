"""Tier 1 Experiments C & D — Funding Basis + Asymmetric Barriers.

EXP-C: Ackerer funding basis (observed - theoretical) as feature
EXP-D: Asymmetric triple-barrier [0.5, 2.0] labeling

Both use parallelized per-asset signal generation.
"""
import sys
sys.path.insert(0, "src")

import copy, warnings, os, json, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
warnings.filterwarnings("ignore")

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")

PROD_ASSETS = {
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
    from sklearn.linear_model import LogisticRegression
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def get_prod_portfolio():
    cfg = load_config(Path("configs/default.yaml"))
    p = copy.deepcopy(cfg.portfolio)
    p.base_sizing_pct = 0.04
    p.slippage = 0.0005
    p.max_exposure = 0.90
    p.max_per_asset = 0.03
    return p


# =========================================================================
# EXP-C: Funding Rate Basis Feature
# =========================================================================

def fetch_binance_funding(symbol, months=24):
    """Fetch historical funding rates from Binance futures API."""
    import requests
    all_rates = []
    end_time = int(time.time() * 1000)
    start_time = end_time - months * 30 * 24 * 3600 * 1000

    current = start_time
    while current < end_time:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": symbol, "startTime": current, "limit": 1000}
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if not data:
                break
            all_rates.extend(data)
            current = data[-1]["fundingTime"] + 1
            time.sleep(0.1)
        except Exception:
            break

    if not all_rates:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    return df.set_index("timestamp")["rate"].sort_index()


def _process_asset_funding(args):
    """Process one asset with funding basis feature. Subprocess."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from sklearn.preprocessing import StandardScaler

    cfg = load_config(Path("configs/default.yaml"))
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)

        # Fetch funding rates
        funding = fetch_binance_funding(symbol, months=24)
        if len(funding) < 50:
            return asset, None, "no_funding_data"

        # Compute Ackerer basis feature:
        # theoretical_rate ~ 0 for crypto (no interest rate differential)
        # So basis_deviation = observed_funding - 0 = observed_funding
        # But the KEY insight from Ackerer: use ROLLING MEAN DEVIATION
        # funding_basis = funding_rate - rolling_mean(funding_rate, 30 periods)
        # This captures deviation from the "normal" funding level
        funding_mean = funding.rolling(30, min_periods=5).mean()
        funding_basis = funding - funding_mean

        # Resample to 15m (forward-fill 8h funding to 15m bars)
        funding_15m = funding_basis.reindex(ohlcv_15m.index, method="ffill")

        # Add as feature
        features["funding_basis"] = funding_15m.astype(np.float32)
        feature_cols = cfg.features.feature_list + ["funding_basis"]

        # Walk-forward with the extra feature
        from signal_engine.model import WalkForwardModel
        cfg_ext = copy.deepcopy(cfg)
        cfg_ext.features.feature_list = feature_cols
        wf = WalkForwardModel(cfg_ext)
        raw = wf.generate_signals_batch(ohlcv_15m, features, asset)

        if len(raw) == 0:
            return asset, None, "no_signals"

        filtered = apply_confidence_filter(raw, cfg.confidence_filter)
        if len(filtered) < 5:
            return asset, None, "too_few"

        return asset, filtered.to_dict("list"), "ok"
    except Exception as e:
        return asset, None, str(e)[:60]


# =========================================================================
# EXP-D: Asymmetric Triple-Barrier
# =========================================================================

def _process_asset_barrier(args):
    """Process one asset with asymmetric triple-barrier labels. Subprocess."""
    asset, symbol, pt_sl = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")

    from sklearn.preprocessing import StandardScaler

    cfg = load_config(Path("configs/default.yaml"))
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08

    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)

        close = ohlcv_15m["close"]
        vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
        # Actual 1h forward return (for portfolio sim — same for all)
        fwd_ret_1h = close.pct_change(4).shift(-4)

        # Triple-barrier labels
        pt_mult, sl_mult = pt_sl
        vertical_bars = 16  # 4 hours max hold

        labels = pd.Series(index=close.index, dtype=float)
        for i in range(len(close) - vertical_bars):
            entry = close.iloc[i]
            v = vol.iloc[i]
            if pd.isna(v) or v <= 0:
                labels.iloc[i] = np.nan
                continue

            upper = entry * (1 + pt_mult * v)
            lower = entry * (1 - sl_mult * v)

            future = close.iloc[i + 1: i + 1 + vertical_bars]
            hit_upper = (future >= upper)
            hit_lower = (future <= lower)

            if hit_upper.any() and hit_lower.any():
                labels.iloc[i] = 1 if hit_upper.idxmax() <= hit_lower.idxmax() else -1
            elif hit_upper.any():
                labels.iloc[i] = 1
            elif hit_lower.any():
                labels.iloc[i] = -1
            else:
                ret = (close.iloc[i + vertical_bars] - entry) / entry
                labels.iloc[i] = 1 if ret > 0 else -1

        # Walk-forward with barrier labels (but portfolio uses actual fwd returns)
        feature_cols = cfg.features.feature_list
        feat = features[feature_cols].copy()
        vol_filter = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

        start_date = ohlcv_15m.index.min()
        end_date = ohlcv_15m.index.max()
        if (end_date - start_date).days < 180:
            return asset, None, "insufficient"

        first_test = start_date + pd.DateOffset(months=6)
        all_months = pd.period_range(start=first_test.to_period("M"),
                                      end=end_date.to_period("M"), freq="M")
        median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
        embargo_td = median_gap * 5

        rows = []
        cached_model = cached_scaler = None
        last_retrain = -999

        for mi, month in enumerate(all_months):
            ms = pd.Timestamp(month.start_time, tz="UTC")
            me = pd.Timestamp(month.end_time, tz="UTC")
            if feat.index.tz is None:
                ms = ms.tz_localize(None)
                me = me.tz_localize(None)
            tc = ms - embargo_td

            fc = feat.dropna()
            y_all = labels.reindex(fc.index).dropna()
            fwd_all = fwd_ret_1h.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index).intersection(fwd_all.index)
            tr = valid[valid < tc]
            te = valid[(valid >= ms) & (valid <= me)]
            if len(tr) < 200 or len(te) < 5:
                continue

            tv = vol_filter.loc[vol_filter.index.intersection(tr)].dropna()
            if len(tv) < 20:
                continue
            vt = tv.quantile(0.50)
            va = vol_filter.reindex(te)
            hv = (va > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 3:
                continue

            X_train = fc.loc[tr]
            y_train = (y_all.loc[tr] > 0).astype(int)
            X_test = fc.loc[te]

            need = (mi - last_retrain) >= 2 or cached_model is None
            try:
                if need:
                    sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)
                    scaler = StandardScaler()
                    Xts = scaler.fit_transform(X_train)
                    model = make_logreg()
                    model.fit(Xts, y_train, sample_weight=sw)
                    cached_model, cached_scaler = model, scaler
                    last_retrain = mi

                Xtes = cached_scaler.transform(X_test)
                yp = cached_model.predict_proba(Xtes)[:, 1]
            except Exception:
                continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))

            # Use ACTUAL forward returns for portfolio
            fv = fwd_ret_1h.reindex(te).values
            finite = np.isfinite(fv)
            if finite.any():
                batch = pd.DataFrame({
                    "timestamp": te[finite], "direction": dirs[finite],
                    "fwd_return": fv[finite], "confidence": conf[finite],
                    "probability": yp[finite], "signal_strength": ss[finite],
                })
                batch["asset"] = asset
                rows.append(batch)

        if not rows:
            return asset, None, "no_signals"

        df = pd.concat(rows, ignore_index=True)
        # Apply confidence filter
        filtered = apply_confidence_filter(df, cfg.confidence_filter)
        if len(filtered) < 5:
            return asset, None, "too_few"

        return asset, filtered.to_dict("list"), "ok"
    except Exception as e:
        return asset, None, str(e)[:60]


def combine_and_report(label, signal_dicts, portfolio_cfg):
    frames = [pd.DataFrame(d) for d in signal_dicts if d is not None]
    if not frames:
        print(f"  {label:<45s} FAILED (no signals)", flush=True)
        return

    combined = pd.concat(frames).sort_values("timestamp")
    sim = simulate_portfolio(combined, portfolio_cfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label:<45s} FAILED (too few trades)", flush=True)
        return

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        print(f"  {label:<45s} FAILED (insufficient days)", flush=True)
        return

    stats = compute_full_stats(dpnl, portfolio_cfg.starting_capital,
                                sim["n_trades"], sim["n_trades"] / len(dpnl))
    pnl_day = (stats["final_equity"] - 10000) / max(stats["n_days"], 1)
    print(
        f"  {label:<45s} Sharpe={stats['sharpe']:>6.3f}  Ret={stats['total_return_pct']:>7.2f}%  "
        f"DD={stats['max_dd_pct']:>5.2f}%  Trades={sim['n_trades']:>6d}  "
        f"PnL/day=${pnl_day:>6.2f}",
        flush=True,
    )


def main():
    portfolio_cfg = get_prod_portfolio()

    print("=" * 80)
    print(f"TIER 1 EXPERIMENTS C & D ({W} CPU workers)")
    print("Baseline: Sharpe 2.819, 30 assets, 4% sizing, 5 bps slip")
    print("=" * 80, flush=True)

    # =================================================================
    # EXP-C: Funding Rate Basis Feature
    # =================================================================
    print("\n" + "=" * 80)
    print("EXP-C: ACKERER FUNDING BASIS FEATURE")
    print("Feature = funding_rate - rolling_mean(funding_rate, 30 periods)")
    print("=" * 80, flush=True)

    t0 = time.time()
    funding_signals = []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {
            pool.submit(_process_asset_funding, (a, s)): a
            for a, s in PROD_ASSETS.items()
        }
        done = 0
        for future in as_completed(futures):
            asset, sig, status = future.result()
            done += 1
            if sig is not None:
                funding_signals.append(sig)
            print(f"  [{done:>2d}/30] {asset}: {status}", flush=True)

    print(f"  Funding data fetched + signals generated in {time.time()-t0:.0f}s", flush=True)
    combine_and_report("Funding basis (14+1 features)", funding_signals, portfolio_cfg)

    # =================================================================
    # EXP-D: Asymmetric Triple-Barrier
    # =================================================================
    print("\n" + "=" * 80)
    print("EXP-D: ASYMMETRIC TRIPLE-BARRIER LABELING")
    print("Portfolio sim uses ACTUAL 1h forward returns (fair comparison)")
    print("=" * 80, flush=True)

    barrier_configs = [
        ("Fixed-horizon (baseline)", None),
        ("Barrier [0.5, 2.0] (tight TP, wide SL)", (0.5, 2.0)),
        ("Barrier [0.5, 1.0] (tight TP, normal SL)", (0.5, 1.0)),
        ("Barrier [1.0, 3.0] (normal TP, wide SL)", (1.0, 3.0)),
    ]

    for label, pt_sl in barrier_configs:
        if pt_sl is None:
            # Baseline: use production pipeline
            from signal_engine.pipeline import run_backtest
            cfg = load_config(Path("configs/default.yaml"))
            cfg.regime_filter.enabled = False
            cfg.confidence_filter.mode = "absolute"
            cfg.confidence_filter.threshold = 0.08
            cfg.portfolio = portfolio_cfg
            cfg.assets = PROD_ASSETS
            r = run_backtest(cfg, max_workers=W)
            s = r["stats"]
            if "error" not in s:
                pnl_day = (s["final_equity"] - 10000) / max(s["n_days"], 1)
                print(
                    f"  {label:<45s} Sharpe={s['sharpe']:>6.3f}  Ret={s['total_return_pct']:>7.2f}%  "
                    f"DD={s['max_dd_pct']:>5.2f}%  Trades={s['n_trades']:>6d}  "
                    f"PnL/day=${pnl_day:>6.2f}",
                    flush=True,
                )
            continue

        t0 = time.time()
        barrier_signals = []
        with ProcessPoolExecutor(max_workers=W) as pool:
            futures = {
                pool.submit(_process_asset_barrier, (a, s, pt_sl)): a
                for a, s in PROD_ASSETS.items()
            }
            done = 0
            for future in as_completed(futures):
                asset, sig, status = future.result()
                done += 1
                if sig is not None:
                    barrier_signals.append(sig)
            # Don't print per-asset for barriers (too noisy)

        combine_and_report(label, barrier_signals, portfolio_cfg)

    print("\n" + "=" * 80)
    print("TIER 1 C+D COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
