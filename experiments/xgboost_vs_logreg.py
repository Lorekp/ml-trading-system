"""XGBoost vs LogReg — Head-to-Head Walk-Forward Comparison

Tests whether XGBoost captures non-linear feature interactions that
LogReg misses. Same assets, features, walk-forward windows, filtering.

Kill criteria:
- XGBoost SR < 1.5 OOS
- Daily return correlation with LogReg > 0.7
- AUC not > LogReg + 0.005
"""

import sys, os, time, warnings, copy
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from signal_engine.config import SystemConfig, ModelConfig, FeatureConfig, PortfolioConfig
from signal_engine.config import RegimeFilterConfig, ConfidenceFilterConfig, DataConfig, SignalConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.filters import apply_confidence_filter, apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.weights import compute_sample_weights
from afml.bet_sizing import discretize_signal, get_signal

warnings.filterwarnings("ignore")

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "SOL": "SOLUSDT", "ADA": "ADAUSDT",
    "AVAX": "AVAXUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "BCH": "BCHUSDT", "FIL": "FILUSDT",
    "FTM": "FTMUSDT", "HBAR": "HBARUSDT", "TRX": "TRXUSDT",
    "SUI": "SUIUSDT", "APT": "APTUSDT", "ENJ": "ENJUSDT",
    "SXP": "SXPUSDT", "ZEC": "ZECUSDT", "XLM": "XLMUSDT",
    "FET": "FETUSDT", "WLD": "WLDUSDT", "BNX": "BNXUSDT",
    "ANKR": "ANKRUSDT", "TAO": "TAOUSDT", "ENA": "ENAUSDT",
}
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]


def _generate_signals_xgb(ohlcv, features, asset_name, fcfg, fwd_bars=4):
    """Walk-forward XGBoost signal generation — mirrors WalkForwardModel exactly."""
    feature_cols = [c for c in FEATURES if c in features.columns]
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()
    if (end_date - start_date).days < 180:
        return pd.DataFrame(), []

    first_test = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(start=first_test.to_period("M"),
                                  end=end_date.to_period("M"), freq="M")
    if len(all_months) == 0:
        return pd.DataFrame(), []

    median_gap = ohlcv.index.to_series().diff().dropna().median()
    embargo_td = median_gap * (fwd_bars + 1)

    rows = []
    aucs = []
    cached_model = None
    cached_scaler = None
    last_retrain = -999

    retrain_every = 1
    if len(ohlcv) > 500_000:
        retrain_every = 3
    elif len(ohlcv) > 200_000:
        retrain_every = 2

    for midx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        embargo_cutoff = ms - embargo_td
        feat = features[feature_cols].copy()

        # Winsorize
        train_mask = feat.index < embargo_cutoff
        for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
            if col in feat.columns:
                tv = feat.loc[train_mask, col]
                if len(tv.dropna()) > 10:
                    lo, hi = tv.quantile(fcfg.winsor_lo), tv.quantile(fcfg.winsor_hi)
                    feat[col] = feat[col].clip(lo, hi)

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)
        tr = valid[valid < embargo_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 200 or len(te) < 10:
            continue

        # Vol filter
        tv = vol.loc[vol.index.intersection(tr)].dropna()
        if len(tv) < 20:
            continue
        vt = tv.quantile(fcfg.vol_filter_quantile)
        vol_aligned = vol.reindex(te)
        hv = (vol_aligned > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        X_train = fc.loc[tr]
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]
        y_test = (y_all.loc[te] > 0).astype(int)

        need_retrain = (midx - last_retrain) >= retrain_every or cached_model is None

        try:
            if need_retrain:
                sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=fwd_bars)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)

                model = xgb.XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    tree_method="hist", random_state=42, n_jobs=1, verbosity=0,
                )
                model.fit(X_tr_s, y_train, sample_weight=sw)
                cached_model = model
                cached_scaler = scaler
                last_retrain = midx
            else:
                model = cached_model
                scaler = cached_scaler

            X_te_s = scaler.transform(X_test)
            yp = model.predict_proba(X_te_s)[:, 1]
        except Exception:
            continue

        # AUC
        if len(np.unique(y_test)) > 1:
            aucs.append(roc_auc_score(y_test, yp))

        # Build signals
        fwd_rets = fwd_ret.reindex(te).values
        for i, ts in enumerate(te):
            prob = float(yp[i])
            direction = 1.0 if prob > 0.5 else -1.0
            conf = abs(prob - 0.5)
            sig = get_signal(prob, num_classes=2)
            sig_disc = discretize_signal(sig, step_size=0.1)
            rows.append({
                "timestamp": ts, "direction": direction,
                "fwd_return": float(fwd_rets[i]) if np.isfinite(fwd_rets[i]) else 0.0,
                "confidence": conf, "probability": prob,
                "signal_strength": abs(float(sig_disc)),
                "asset": asset_name,
            })

    if not rows:
        return pd.DataFrame(), aucs
    return pd.DataFrame(rows), aucs


def _process_asset(asset_name, symbol, config, model_type):
    """Process one asset with either LogReg or XGBoost."""
    warnings.filterwarnings("ignore")
    t0 = time.time()
    try:
        ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
    except FileNotFoundError:
        return {"asset": asset_name, "error": True, "signals": None, "auc": []}

    ohlcv_tf = resample_ohlcv(ohlcv_1m, config.data.timeframe)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    features = compute_features(ohlcv_tf, config.features)

    if model_type == "xgboost":
        raw_signals, aucs = _generate_signals_xgb(
            ohlcv_tf, features, asset_name, config.features, config.model.fwd_bars
        )
    else:
        from signal_engine.model import WalkForwardModel
        wf = WalkForwardModel(config)
        raw_signals = wf.generate_signals_batch(ohlcv_tf, features, asset_name)
        aucs = []  # LogReg AUC computed separately if needed

    n_raw = len(raw_signals)
    if n_raw > 0:
        conf = apply_confidence_filter(raw_signals, config.confidence_filter)
    else:
        conf = pd.DataFrame()

    if len(conf) > 0 and config.regime_filter.enabled:
        regime = apply_regime_filter(conf, ohlcv_4h, config.regime_filter)
    elif len(conf) > 0:
        regime = conf
    else:
        regime = pd.DataFrame()

    elapsed = round(time.time() - t0, 1)
    return {
        "asset": asset_name, "error": False,
        "signals": regime.to_dict("list") if len(regime) > 0 else None,
        "n_raw": n_raw, "n_filtered": len(regime),
        "auc": aucs, "elapsed": elapsed,
    }


def run_model(config, model_type, label, max_workers=8):
    """Run full pipeline for a given model type."""
    print(f"\n{'='*80}")
    print(f"  {label} ({model_type})")
    print(f"{'='*80}")

    assets = list(config.assets.items())
    all_signals = []
    all_aucs = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_asset, name, sym, config, model_type): name
            for name, sym in assets
        }
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            if r["error"]:
                continue
            all_aucs.extend(r["auc"])
            if r["signals"]:
                all_signals.append(pd.DataFrame(r["signals"]))
            print(f"  [{i:2d}/{len(assets)}] {r['asset']}: raw={r['n_raw']}, "
                  f"filt={r['n_filtered']}, AUC={np.mean(r['auc']):.4f} ({r['elapsed']}s)"
                  if r["auc"] else
                  f"  [{i:2d}/{len(assets)}] {r['asset']}: raw={r['n_raw']}, "
                  f"filt={r['n_filtered']} ({r['elapsed']}s)")

    if not all_signals:
        return {"stats": {"error": "no_signals"}, "daily_pnl": None, "auc": 0}

    filtered = pd.concat(all_signals, ignore_index=True)
    sim = simulate_portfolio(filtered, config.portfolio)
    daily_pnl = compute_daily_pnl_series(sim) if sim else None

    if daily_pnl is not None and sim:
        stats = compute_full_stats(
            daily_pnl, config.portfolio.starting_capital,
            sim["n_trades"], sim["n_trades"] / max(len(daily_pnl), 1)
        )
    else:
        stats = {"error": "no_trades"}

    mean_auc = float(np.mean(all_aucs)) if all_aucs else 0
    print(f"\n  Mean OOS AUC: {mean_auc:.4f} ({len(all_aucs)} windows)")

    return {"stats": stats, "daily_pnl": daily_pnl, "auc": mean_auc,
            "n_signals": len(filtered)}


def main():
    t0 = time.time()
    n_workers = min(os.cpu_count() or 4, 11)

    config = SystemConfig(
        model=ModelConfig(C=0.1, penalty="l1", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        portfolio=PortfolioConfig(starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
                                  slippage=0.0001, max_exposure=0.80, max_per_asset=0.06,
                                  base_sizing_pct=0.02, max_signal_mult=2.0,
                                  sizing_mode="probability_edge",
                                  use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe="15min", regime_timeframe="4h"),
        assets=ASSETS,
    )

    print("=" * 80)
    print("  XGBoost vs LogReg — Head-to-Head Comparison")
    print(f"  {len(ASSETS)} assets, {n_workers} workers")
    print("=" * 80)

    # Run both
    res_lr = run_model(config, "logreg", "A: LogReg L1 (baseline)", n_workers)
    res_xgb = run_model(config, "xgboost", "B: XGBoost (depth=4)", n_workers)

    # Results
    print(f"\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}")

    header = f"{'Model':<25} {'SR':>8} {'AUC':>8} {'DD%':>8} {'Signals':>10}"
    print(header)
    print("-" * len(header))

    for label, res in [("A: LogReg L1", res_lr), ("B: XGBoost", res_xgb)]:
        s = res["stats"]
        if "error" in s:
            print(f"{label:<25} ERROR: {s['error']}")
            continue
        print(f"{label:<25} {s.get('sharpe',0):>8.3f} {res['auc']:>8.4f} "
              f"{s.get('max_dd_pct',0):>7.1f}% {res.get('n_signals',0):>10,}")

    # Correlation
    pnl_lr = res_lr.get("daily_pnl")
    pnl_xgb = res_xgb.get("daily_pnl")

    if pnl_lr is not None and pnl_xgb is not None:
        combined = pd.DataFrame({"LR": pnl_lr, "XGB": pnl_xgb}).fillna(0)
        mask = (combined["LR"] != 0) | (combined["XGB"] != 0)
        combined = combined[mask]
        corr = float(combined["LR"].corr(combined["XGB"])) if len(combined) > 30 else float("nan")

        print(f"\n  Daily return correlation: {corr:.4f}")
        if corr > 0.7:
            print(f"  FAIL: correlation > 0.7 — XGBoost is just a noisier LogReg")
        else:
            print(f"  PASS: correlation < 0.7")

        # Combined portfolio
        combined_pnl = combined["LR"] * 0.5 + combined["XGB"] * 0.5
        combined_nonzero = combined_pnl[combined_pnl != 0]
        if len(combined_nonzero) >= 10:
            cs = compute_full_stats(combined_nonzero, 10000.0, 0, 0)
            print(f"\n  Combined 50/50 SR: {cs.get('sharpe',0):.3f} "
                  f"(vs {res_lr['stats'].get('sharpe',0):.3f} LogReg alone)")

    # Decision
    print(f"\n{'='*80}")
    print("DECISION")
    print(f"{'='*80}")

    sr_xgb = res_xgb["stats"].get("sharpe", 0) if "error" not in res_xgb["stats"] else 0
    auc_lr = res_lr.get("auc", 0)
    auc_xgb = res_xgb.get("auc", 0)
    corr_val = corr if pnl_lr is not None and pnl_xgb is not None else float("nan")

    reasons = []
    if sr_xgb < 1.5:
        reasons.append(f"SR={sr_xgb:.2f} < 1.5")
    if corr_val > 0.7:
        reasons.append(f"corr={corr_val:.3f} > 0.7")
    if auc_xgb < auc_lr + 0.005:
        reasons.append(f"AUC improvement={auc_xgb-auc_lr:.4f} < 0.005")

    if reasons:
        print(f"  XGBoost: KILL ({', '.join(reasons)})")
    else:
        print(f"  XGBoost: PASS (SR={sr_xgb:.2f}, corr={corr_val:.3f}, "
              f"AUC +{auc_xgb-auc_lr:.4f})")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
