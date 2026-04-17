"""Train and save the production model for live trading.

Runs the walk-forward pipeline with production config (C=0.01, 18 features,
14-month rolling lookback) and saves the final model, scaler, and NCO weights
for use by the live engine.

Output:
    data/live/model/model.joblib    — trained LogisticRegression
    data/live/model/scaler.joblib   — fitted StandardScaler
    data/live/model/metadata.json   — training metadata
    data/live/nco_weights.json      — NCO k=8 portfolio weights

Usage:
    python scripts/train_production_model.py [--data-dir data/binance-1m]
"""

import sys
import os

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.allocation import compute_nco_weights

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)

# Production config
C = 0.01
LOOKBACK_MONTHS = 14
CONF_THR = 0.14
N_CLUSTERS = 8


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(
            C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42
        )
    return LogisticRegression(
        C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42
    )


def main():
    parser = argparse.ArgumentParser(description="Train production model")
    parser.add_argument(
        "--data-dir",
        default="data/binance-1m",
        help="Path to Binance 1m data directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/live/model",
        help="Output directory for model artifacts",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top-volume assets to train on",
    )
    args = parser.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 70)
    print("PRODUCTION MODEL TRAINING")
    print(f"  C={C}, lookback={LOOKBACK_MONTHS}mo, threshold={CONF_THR}")
    print(f"  Features: {len(cfg.features.feature_list)}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Phase 1: Load data and compute volumes for asset selection
    print("\n[1] Loading data...", flush=True)
    asset_data = {}

    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [data_dir])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000:
                continue

            close = ohlcv_15m["close"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(
                        feat[col].quantile(0.01), feat[col].quantile(0.99)
                    )
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 3000:
                continue

            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

            # Dollar volume for asset ranking
            dollar_vol = (close * ohlcv_15m["volume"]).resample("D").sum()
            median_dvol = dollar_vol.rolling(30).median().iloc[-1]

            asset_data[symbol] = {
                "fc": fc,
                "y_all": y_all,
                "valid": valid,
                "vol": vol,
                "start_date": ohlcv_15m.index.min(),
                "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
                "dollar_vol": median_dvol,
            }
            print(f"  {symbol:<16s} {len(ohlcv_15m):>7d} bars", flush=True)
        except Exception:
            pass

    print(f"\n  Loaded {len(asset_data)} assets in {time.time() - t0:.0f}s")

    # Phase 2: Select top-N by volume
    ranked = sorted(asset_data.items(), key=lambda x: x[1]["dollar_vol"], reverse=True)
    selected = [sym for sym, _ in ranked[: args.top_n]]
    print(f"\n[2] Selected top-{args.top_n}: {', '.join(s.replace('USDT','') for s in selected[:10])}...")

    # Phase 3: Train on ALL available data (most recent month as validation)
    print("\n[3] Training final model...", flush=True)

    # Collect all training data from selected assets
    all_X_train = []
    all_y_train = []
    all_sw_train = []
    all_signals = []  # for NCO weight computation

    # Use all data except last month as train, last month as validation
    end_date = max(ad["end_date"] for ad in asset_data.values())
    val_start = end_date - pd.DateOffset(months=1)
    train_end = val_start

    for symbol in selected:
        ad = asset_data[symbol]
        fc = ad["fc"]
        y_all = ad["y_all"]
        valid = ad["valid"]
        vol = ad["vol"]

        # Training window: last LOOKBACK_MONTHS before validation
        train_start = train_end - pd.DateOffset(months=LOOKBACK_MONTHS)
        if train_start < ad["start_date"]:
            train_start = ad["start_date"]

        embargo_td = ad["median_gap"] * 5
        train_cutoff = train_end - embargo_td

        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200:
            continue

        try:
            sw = compute_sample_weights(
                fc.loc[tr].index.values.astype(np.int64), fwd_bars=4
            )
            all_X_train.append(fc.loc[tr].values)
            all_y_train.append((y_all.loc[tr] > 0).astype(int).values)
            all_sw_train.append(sw)
        except Exception:
            continue

        # Also collect validation signals for NCO weights
        te = valid[(valid >= val_start) & (valid <= end_date)]
        if len(te) < 5:
            continue

        vol_train_start = val_start - pd.DateOffset(months=3)
        vol_train = vol.loc[
            (vol.index >= vol_train_start) & (vol.index < val_start)
        ].dropna()
        if len(vol_train) < 50:
            continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        # Predict on validation
        try:
            scaler_tmp = StandardScaler()
            X_tr_tmp = scaler_tmp.fit_transform(fc.loc[tr])
            model_tmp = make_logreg()
            sw_tmp = compute_sample_weights(
                fc.loc[tr].index.values.astype(np.int64), fwd_bars=4
            )
            model_tmp.fit(X_tr_tmp, (y_all.loc[tr] > 0).astype(int), sample_weight=sw_tmp)
            X_te = scaler_tmp.transform(fc.loc[te])
            yp = model_tmp.predict_proba(X_te)[:, 1]

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            mask = conf > CONF_THR
            fv = y_all.reindex(te).values
            finite = np.isfinite(fv)

            for i in range(len(te)):
                if not mask[i] or not finite[i]:
                    continue
                all_signals.append({
                    "asset": symbol.replace("USDT", ""),
                    "direction": dirs[i],
                    "fwd_return": fv[i],
                    "timestamp": te[i],
                })
        except Exception:
            pass

    if not all_X_train:
        print("ERROR: No training data collected")
        return

    # Concatenate and train final model
    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    sw_train = np.concatenate(all_sw_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = make_logreg()
    model.fit(X_scaled, y_train, sample_weight=sw_train)

    print(f"  Trained on {len(X_train)} samples from {len(selected)} assets")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Non-zero coefficients: {(np.abs(model.coef_[0]) > 1e-6).sum()}")

    # Phase 4: Compute NCO weights
    print("\n[4] Computing NCO weights...", flush=True)
    nco_weights = {}
    if all_signals:
        sdf = pd.DataFrame(all_signals)
        daily_rets = {}
        for asset in sdf["asset"].unique():
            a_sigs = sdf[sdf["asset"] == asset]
            if len(a_sigs) < 10:
                continue
            ts = pd.to_datetime(a_sigs["timestamp"])
            rets = pd.Series(
                a_sigs["direction"].values * a_sigs["fwd_return"].values,
                index=ts,
            )
            daily_rets[asset] = rets.resample("D").sum()

        if len(daily_rets) >= 3:
            ret_df = pd.DataFrame(daily_rets).fillna(0)
            corr = ret_df.corr().values
            names = list(ret_df.columns)
            exp_r = np.array([
                float(
                    (sdf[sdf["asset"] == a]["direction"]
                     * sdf[sdf["asset"] == a]["fwd_return"]).mean()
                )
                for a in names
            ])
            nco_weights = compute_nco_weights(corr, exp_r, names, n_clusters=N_CLUSTERS)
            print(f"  NCO weights for {len(nco_weights)} assets")
        else:
            print("  Insufficient assets for NCO, using uniform weights")

    # Phase 5: Save artifacts
    print("\n[5] Saving artifacts...", flush=True)

    model_path = output_dir / "model.joblib"
    scaler_path = output_dir / "scaler.joblib"
    meta_path = output_dir / "metadata.json"
    nco_path = output_dir.parent / "nco_weights.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "C": C,
        "lookback_months": LOOKBACK_MONTHS,
        "confidence_threshold": CONF_THR,
        "n_features": X_train.shape[1],
        "n_samples": len(X_train),
        "n_assets": len(selected),
        "nonzero_coefs": int((np.abs(model.coef_[0]) > 1e-6).sum()),
        "train_start": str(train_end - pd.DateOffset(months=LOOKBACK_MONTHS)),
        "train_end": str(train_end),
        "train_bars": len(X_train),
        "feature_list": cfg.features.feature_list,
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    nco_data = {
        "weights": nco_weights,
        "n_clusters": N_CLUSTERS,
        "computed": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_assets": len(nco_weights),
    }
    nco_path.parent.mkdir(parents=True, exist_ok=True)
    with open(nco_path, "w") as f:
        json.dump(nco_data, f, indent=2)

    print(f"\n  model.joblib:     {model_path}")
    print(f"  scaler.joblib:    {scaler_path}")
    print(f"  metadata.json:    {meta_path}")
    print(f"  nco_weights.json: {nco_path}")
    print(f"\n  Total: {time.time() - t0:.0f}s")
    print("\nDone. Model ready for live trading.")


if __name__ == "__main__":
    main()
