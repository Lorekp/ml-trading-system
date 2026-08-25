"""Train and export the live model + Kelly weights.

Trains a LogReg L1 on the most recent N months of multi-asset data,
saves model.joblib + scaler.joblib + metadata.json to data/live/model/.
Computes walk-forward Kelly weights and saves to data/live/kelly_weights.json.

Usage:
    python experiments/train_live_model.py [--lookback 8]

To swap models: just re-run this script and restart the bot.
"""
import sys
sys.path.insert(0, "src")

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from signal_engine.config import load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from afml.bet_sizing import get_signal, discretize_signal, kelly_size

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

DATA_DIR = Path("data/binance-1m")
MODEL_DIR = Path("data/live/model")
KELLY_PATH = Path("data/live/kelly_weights.json")

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


def train_model(lookback_months: int):
    """Train model on the most recent N months of BTC data.

    Uses BTC only (not pooled multi-asset) because:
    - Backtest trains per-asset models, each asset gets tailored coefficients
    - Pooling 30 assets creates a weak averaged model (probs ~0.50)
    - BTC-only matches the original working deployment
    - Per-asset differentiation comes from Kelly/NCO weights, not the model
    """
    print(f"\n{'='*70}")
    print(f"TRAINING LIVE MODEL ({lookback_months}-month lookback, BTC-based)")
    print(f"{'='*70}\n")

    cfg = load_config(Path("configs/default.yaml"))
    feature_list = cfg.features.feature_list

    # Train on BTC only (matches backtest per-asset approach)
    symbol = "BTCUSDT"
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception as e:
        print(f"ERROR: Cannot load BTC data: {e}")
        return None

    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[feature_list].copy()
    close = ohlcv_15m["close"]
    fwd_ret = close.pct_change(4).shift(-4)

    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)

    end = valid[-1]
    start = end - pd.DateOffset(months=lookback_months)

    train_mask = (valid >= start) & (valid <= end)
    tr = valid[train_mask]
    if len(tr) < 200:
        print("ERROR: Not enough training data!")
        return None

    data_start_actual = tr[0]
    data_end = tr[-1]

    sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
    X_combined = fc.loc[tr].values
    y_combined = (y_all.loc[tr] > 0).astype(int).values
    sw_combined = sw

    print(f"  BTC: {len(tr):,d} bars ({tr[0].date()} to {tr[-1].date()})")
    print(f"  Training window: {data_start_actual.date()} to {data_end.date()}")

    # Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    model = make_logreg()
    model.fit(X_scaled, y_combined, sample_weight=sw_combined)

    train_acc = model.score(X_scaled, y_combined, sample_weight=sw_combined)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Features: {feature_list}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    metadata = {
        "train_start": str(data_start_actual.date()),
        "train_end": str(data_end.date()),
        "train_bars": int(len(X_combined)),
        "train_accuracy": round(float(train_acc), 4),
        "features": feature_list,
        "lookback_months": lookback_months,
        "n_assets_trained": len(all_X),
        "created": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to {MODEL_DIR}/")
    print(f"    model.joblib ({(MODEL_DIR / 'model.joblib').stat().st_size:,d} bytes)")
    print(f"    scaler.joblib ({(MODEL_DIR / 'scaler.joblib').stat().st_size:,d} bytes)")
    print(f"    metadata.json")

    return model, scaler, metadata


def _generate_asset_signals(args):
    """Generate walk-forward OOS signals for one asset (for Kelly computation)."""
    asset, symbol, lookback = args
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

        train_start = ms - pd.DateOffset(months=lookback)
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

        dirs = np.where(yp > 0.5, 1.0, -1.0)
        conf = np.abs(yp - 0.5)
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any():
            continue

        signals = pd.DataFrame({
            "timestamp": te[finite], "direction": dirs[finite],
            "fwd_return": fv[finite], "confidence": conf[finite],
        })
        signals["asset"] = asset
        filtered = apply_confidence_filter(
            signals, ConfidenceFilterConfig(mode="absolute", threshold=0.08))
        if len(filtered) > 0:
            results.append(filtered)

    return asset, results


def compute_kelly_weights(lookback_months: int):
    """Compute per-asset Kelly weights from walk-forward OOS signals."""
    print(f"\n{'='*70}")
    print(f"COMPUTING KELLY WEIGHTS (walk-forward, {lookback_months}-month)")
    print(f"{'='*70}\n")

    W = max(1, os.cpu_count() - 1)

    all_asset_signals = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {
            pool.submit(_generate_asset_signals, (a, s, lookback_months)): a
            for a, s in ASSETS.items()
        }
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            n_sigs = sum(len(r) for r in results) if results else 0
            all_asset_signals[asset] = results
            print(f"  [{done:>2d}/30] {asset}: {n_sigs:,d} OOS signals", flush=True)

    # Compute Kelly fraction per asset
    weights = {}
    for asset, sig_list in all_asset_signals.items():
        if not sig_list:
            weights[asset] = 1.0
            continue
        combined = pd.concat(sig_list)
        trade_rets = combined["direction"].values * combined["fwd_return"].values
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets <= 0]

        if len(wins) < 5 or len(losses) < 5:
            weights[asset] = 1.0
            continue

        w_prob = len(wins) / len(trade_rets)
        w_ret = float(wins.mean())
        l_ret = float(abs(losses.mean()))
        if l_ret <= 0:
            weights[asset] = 1.0
            continue

        f_star = kelly_size(w_prob, w_ret, l_ret)
        weights[asset] = max(0.2, min(3.0, f_star * 7.5))

    # Normalize to mean=1
    if weights:
        mean_w = np.mean(list(weights.values()))
        if mean_w > 0:
            weights = {k: round(v / mean_w, 4) for k, v in weights.items()}

    # Save
    kelly_data = {
        "weights": weights,
        "computed": datetime.now(timezone.utc).isoformat(),
        "n_assets": len(weights),
        "lookback_months": lookback_months,
        "source": "walk_forward_historical",
    }
    KELLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KELLY_PATH, "w") as f:
        json.dump(kelly_data, f, indent=2)

    print(f"\n  Kelly weights saved to {KELLY_PATH}")
    print(f"  Range: [{min(weights.values()):.3f}, {max(weights.values()):.3f}]")
    print(f"  Assets: {len(weights)}")

    # Print sorted by weight
    print(f"\n  {'Asset':<8s} {'Weight':>7s}")
    print(f"  {'-'*16}")
    for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
        marker = " ***" if w > 1.5 or w < 0.5 else ""
        print(f"  {asset:<8s} {w:>7.3f}{marker}")

    return weights


def compute_nco_weights_from_signals(lookback_months: int):
    """Compute NCO weights from walk-forward OOS signals."""
    from signal_engine.allocation import compute_nco_weights as _nco

    print(f"\n{'='*70}")
    print(f"COMPUTING NCO WEIGHTS (k=8, walk-forward, {lookback_months}-month)")
    print(f"{'='*70}\n")

    W_local = max(1, os.cpu_count() - 1)
    all_asset_signals = {}
    with ProcessPoolExecutor(max_workers=W_local) as pool:
        futures = {
            pool.submit(_generate_asset_signals, (a, s, lookback_months)): a
            for a, s in ASSETS.items()
        }
        done = 0
        for f in as_completed(futures):
            asset, results = f.result()
            done += 1
            n_sigs = sum(len(r) for r in results) if results else 0
            all_asset_signals[asset] = results
            print(f"  [{done:>2d}/30] {asset}: {n_sigs:,d} OOS signals", flush=True)

    # Build per-asset daily returns and expected returns
    daily_rets = {}
    expected_returns = {}
    for asset, sig_list in all_asset_signals.items():
        if not sig_list:
            continue
        combined = pd.concat(sig_list)
        trade_rets = combined["direction"].values * combined["fwd_return"].values
        ts = pd.to_datetime(combined["timestamp"])
        daily = pd.Series(trade_rets, index=ts).resample("D").sum()
        daily_rets[asset] = daily
        expected_returns[asset] = float(trade_rets.mean())

    common = sorted(daily_rets.keys())
    if len(common) < 3:
        print("  ERROR: Too few assets for NCO")
        return {}

    ret_df = pd.DataFrame({a: daily_rets[a] for a in common}).fillna(0)
    corr = ret_df.corr().values
    exp_r = np.array([expected_returns.get(a, 0) for a in common])

    weights = _nco(corr, exp_r, common, n_clusters=8)

    # Save
    nco_path = Path("data/live/nco_weights.json")
    nco_data = {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "computed": datetime.now(timezone.utc).isoformat(),
        "n_assets": len(weights),
        "n_clusters": 8,
        "lookback_months": lookback_months,
        "source": "walk_forward_historical",
    }
    with open(nco_path, "w") as f:
        json.dump(nco_data, f, indent=2)

    print(f"\n  NCO weights saved to {nco_path}")
    print(f"  Range: [{min(weights.values()):.3f}, {max(weights.values()):.3f}]")
    print(f"\n  {'Asset':<8s} {'Weight':>7s}")
    print(f"  {'-'*16}")
    for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
        marker = " ***" if w > 1.5 or w < 0.5 else ""
        print(f"  {asset:<8s} {w:>7.3f}{marker}")

    return weights


def main():
    parser = argparse.ArgumentParser(description="Train live model + weights")
    parser.add_argument("--lookback", type=int, default=8, help="Lookback months (default: 8)")
    parser.add_argument("--allocation", type=str, default="nco", choices=["kelly", "nco", "both"],
                        help="Allocation method (default: nco)")
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: Train and save model
    result = train_model(args.lookback)
    if result is None:
        return

    # Step 2: Compute weights
    if args.allocation in ("kelly", "both"):
        compute_kelly_weights(args.lookback)
    if args.allocation in ("nco", "both"):
        compute_nco_weights_from_signals(args.lookback)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"  Model: {MODEL_DIR}/model.joblib ({args.lookback}-month lookback)")
    print(f"  Allocation: {args.allocation}")
    print(f"\nTo deploy: copy data/live/ to homeserver and restart bot")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
