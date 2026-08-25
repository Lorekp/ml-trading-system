"""EXP-15: Meta-Labeling — LogReg primary + GBM meta-labeler."""

import sys
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def run_meta_labeling():
    cfg = load_config(Path("configs/default.yaml"))
    core = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
            "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}
    data_dirs = [Path(d) for d in cfg.data.data_dirs]

    print("=== EXP-15: Meta-Labeling (LogReg primary + GBM meta) ===\n")

    all_meta_signals = []
    all_nometa_signals = []

    for asset_name, symbol in core.items():
        print(f"Processing {asset_name}...", flush=True)
        ohlcv_1m = load_binance_1m(symbol, data_dirs)
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        features = compute_features(ohlcv_15m, cfg.features)

        close = ohlcv_15m["close"]
        fwd_ret = close.pct_change(4).shift(-4)
        vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
        feature_cols = cfg.features.feature_list
        feat = features[feature_cols].copy()

        start_date = ohlcv_15m.index.min()
        end_date = ohlcv_15m.index.max()
        first_test = start_date + pd.DateOffset(months=6)
        all_months = pd.period_range(
            start=first_test.to_period("M"), end=end_date.to_period("M"), freq="M"
        )
        median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
        embargo_td = median_gap * 4

        cached_primary = cached_scaler = cached_meta = cached_meta_scaler = None
        meta_rows = []
        nometa_rows = []

        for mi, month in enumerate(all_months):
            ms = pd.Timestamp(month.start_time, tz="UTC")
            me = pd.Timestamp(month.end_time, tz="UTC")
            if feat.index.tz is None:
                ms = ms.tz_localize(None)
                me = me.tz_localize(None)
            train_cutoff = ms - embargo_td

            fc = feat.dropna()
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            tr = valid[valid < train_cutoff]
            te = valid[(valid >= ms) & (valid <= me)]
            if len(tr) < 500 or len(te) < 10:
                continue

            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 20:
                continue
            vt = tv.quantile(0.50)
            vol_al = vol.reindex(te)
            hv = (vol_al > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5:
                continue

            X_train = fc.loc[tr]
            y_train_dir = (y_all.loc[tr] > 0).astype(int)
            X_test = fc.loc[te]

            need_retrain = mi % 2 == 0 or cached_primary is None

            try:
                if need_retrain:
                    ts = X_train.index.values.astype(np.int64)
                    sw = compute_sample_weights(ts, fwd_bars=4)

                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train)
                    primary = make_logreg()
                    primary.fit(X_tr_s, y_train_dir, sample_weight=sw)

                    # Primary predictions on training data
                    train_probs = primary.predict_proba(X_tr_s)[:, 1]
                    train_dirs = np.where(train_probs > 0.5, 1, -1)
                    actual_dirs = np.where(y_all.loc[tr].values > 0, 1, -1)
                    train_correct = (train_dirs == actual_dirs).astype(int)

                    # Meta features: original + primary prob
                    meta_X = np.column_stack([X_tr_s, train_probs])
                    meta_scaler = StandardScaler()
                    meta_X_s = meta_scaler.fit_transform(meta_X)

                    meta = GradientBoostingClassifier(
                        n_estimators=100, max_depth=3, min_samples_leaf=20,
                        learning_rate=0.05, random_state=42, subsample=0.8,
                    )
                    meta.fit(meta_X_s, train_correct, sample_weight=sw)

                    cached_primary = primary
                    cached_scaler = scaler
                    cached_meta = meta
                    cached_meta_scaler = meta_scaler

                X_te_s = cached_scaler.transform(X_test)
                p_probs = cached_primary.predict_proba(X_te_s)[:, 1]

                meta_X_te = np.column_stack([X_te_s, p_probs])
                meta_X_te_s = cached_meta_scaler.transform(meta_X_te)
                meta_probs = cached_meta.predict_proba(meta_X_te_s)[:, 1]
            except Exception:
                continue

            directions = np.where(p_probs > 0.5, 1.0, -1.0)
            conf = np.abs(p_probs - 0.5)
            sigs = get_signal(p_probs, num_classes=2)
            sigs_disc = discretize_signal(sigs, step_size=0.1)
            sig_str = np.abs(sigs_disc) if np.ndim(sigs_disc) > 0 else np.full(len(p_probs), abs(float(sigs_disc)))

            fwd_vals = fwd_ret.reindex(te).values
            finite = np.isfinite(fwd_vals)

            # All signals (no meta filter) for comparison
            if finite.any():
                batch_all = pd.DataFrame({
                    "timestamp": te[finite], "direction": directions[finite],
                    "fwd_return": fwd_vals[finite], "confidence": conf[finite],
                    "probability": p_probs[finite], "signal_strength": sig_str[finite],
                })
                batch_all["asset"] = asset_name
                nometa_rows.append(batch_all)

            # Meta-filtered: only keep meta_prob >= 0.55
            meta_pass = meta_probs >= 0.55
            combined = finite & meta_pass
            if combined.any():
                batch = pd.DataFrame({
                    "timestamp": te[combined], "direction": directions[combined],
                    "fwd_return": fwd_vals[combined], "confidence": conf[combined],
                    "probability": p_probs[combined], "signal_strength": sig_str[combined],
                })
                batch["asset"] = asset_name
                meta_rows.append(batch)

        if meta_rows:
            asset_meta = pd.concat(meta_rows)
            all_meta_signals.append(asset_meta)
            print(f"  {asset_name}: {len(asset_meta)} meta-filtered signals")
        if nometa_rows:
            asset_nometa = pd.concat(nometa_rows)
            all_nometa_signals.append(asset_nometa)

    # Results
    print("\n" + "=" * 60)

    for label, signals_list in [("NO meta (baseline)", all_nometa_signals),
                                 ("WITH meta (>0.55)", all_meta_signals)]:
        if not signals_list:
            print(f"{label}: no signals")
            continue
        combined = pd.concat(signals_list).sort_values("timestamp")
        sim = simulate_portfolio(combined, cfg.portfolio)
        if sim and sim["n_trades"] >= 10:
            dpnl = compute_daily_pnl_series(sim)
            if dpnl is not None:
                stats = compute_full_stats(dpnl, cfg.portfolio.starting_capital, sim["n_trades"], sim["n_trades"] / len(dpnl))
                print(f"{label}:")
                print(f"  Sharpe={stats['sharpe']:.3f}  Return={stats['total_return_pct']:.2f}%  "
                      f"MaxDD={stats['max_dd_pct']:.2f}%  Trades={sim['n_trades']}")
        else:
            print(f"{label}: insufficient trades")


if __name__ == "__main__":
    t0 = time.time()
    run_meta_labeling()
    print(f"\nTotal time: {time.time() - t0:.0f}s")
