"""A/B test: 4 BERT integration approaches vs baseline (ETH only).

Leak-free: PCA/scaling fitted per walk-forward fold on training data only.

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/bert_features_eth.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig
from signal_engine.weights import compute_sample_weights
from signal_engine.bert_features import extract_bar_features
from models.bbo_bert.model import BBOBert
from models.bbo_bert.config import BBOBertConfig

# ============================================================
DATA_DIRS = [Path("data/binance-1m")]
PARQUET_DIR = Path("data/bookTicker_parquet/ETHUSDT")
BERT_CKPT = Path("data/bbo_bert_eth_ft/finetuned.pt")
EMBED_CACHE = Path("data/bbo_bert_eth_ft/embeddings_cache.npz")

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

FWD_BARS = 4
C_REG = 0.10
# ============================================================


def load_bert_model(device):
    config = BBOBertConfig()
    model = BBOBert(config).to(device)
    ckpt = torch.load(BERT_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded BERT: {model.count_parameters():,} params, d_model={config.d_model}")
    return model, config


def make_logreg_l1():
    return LogisticRegression(C=C_REG, penalty="l1", solver="saga",
                              max_iter=1000, random_state=42)


def make_elastic_net():
    return LogisticRegression(C=C_REG, penalty="elasticnet", solver="saga",
                              l1_ratio=0.5, max_iter=1000, random_state=42)


def make_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_child_weight=50, subsample=0.7, colsample_bytree=0.5,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0)


# ============================================================
# Approach functions: (ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx) → (X_tr, X_te, info)
# ============================================================

def approach_baseline(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    return ohlcv_arr[tr_idx], ohlcv_arr[te_idx], {}


def approach_a_pca_logreg(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    """Per-fold PCA(20) on BERT embeddings → merge with OHLCV."""
    bert_tr = bert_arr[tr_idx]
    # Only fit PCA on bars with actual BERT data (non-zero)
    mask = (bert_tr != 0).any(axis=1)
    if mask.sum() < 40:
        return ohlcv_arr[tr_idx], ohlcv_arr[te_idx], {"pca_var": 0}
    pca = PCA(n_components=20)
    pca.fit(bert_tr[mask])
    bert_tr_pca = pca.transform(bert_tr)
    bert_te_pca = pca.transform(bert_arr[te_idx])
    X_tr = np.hstack([ohlcv_arr[tr_idx], bert_tr_pca])
    X_te = np.hstack([ohlcv_arr[te_idx], bert_te_pca])
    return X_tr, X_te, {"pca_var": pca.explained_variance_ratio_.sum()}


def approach_b_elastic(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    """Raw 128-dim BERT + OHLCV → Elastic Net (no PCA)."""
    X_tr = np.hstack([ohlcv_arr[tr_idx], bert_arr[tr_idx]])
    X_te = np.hstack([ohlcv_arr[te_idx], bert_arr[te_idx]])
    return X_tr, X_te, {}


def approach_c_xgboost(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    """Raw 128-dim BERT + OHLCV → XGBoost (no PCA)."""
    X_tr = np.hstack([ohlcv_arr[tr_idx], bert_arr[tr_idx]])
    X_te = np.hstack([ohlcv_arr[te_idx], bert_arr[te_idx]])
    return X_tr, X_te, {}


def approach_d_stats(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    """Temporal stats (mean+std+slope) → per-fold PCA(15) → LogReg."""
    stats_tr = stats_arr[tr_idx]
    mask = (stats_tr != 0).any(axis=1)
    if mask.sum() < 30:
        return ohlcv_arr[tr_idx], ohlcv_arr[te_idx], {"pca_var": 0}
    pca = PCA(n_components=15)
    pca.fit(stats_tr[mask])
    stats_tr_pca = pca.transform(stats_tr)
    stats_te_pca = pca.transform(stats_arr[te_idx])
    X_tr = np.hstack([ohlcv_arr[tr_idx], stats_tr_pca])
    X_te = np.hstack([ohlcv_arr[te_idx], stats_te_pca])
    return X_tr, X_te, {"pca_var": pca.explained_variance_ratio_.sum()}


APPROACHES = {
    "Baseline":       (approach_baseline,       make_logreg_l1),
    "A: PCA+LogReg":  (approach_a_pca_logreg,   make_logreg_l1),
    "B: Elastic Net": (approach_b_elastic,       make_elastic_net),
    "C: XGBoost":     (approach_c_xgboost,      make_xgboost),
    "D: Stats+PCA":   (approach_d_stats,         make_logreg_l1),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    t_start = time.time()

    # ============================================================
    # 1. Load OHLCV
    # ============================================================
    print("[1/4] Loading OHLCV data...")
    ohlcv_1m = load_binance_1m("ETHUSDT", DATA_DIRS)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    del ohlcv_1m
    print(f"  15m bars: {len(ohlcv_15m):,}", flush=True)

    # ============================================================
    # 2. Compute OHLCV features
    # ============================================================
    print("[2/4] Computing OHLCV features...")
    feat_config = FeatureConfig(feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                                vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20)
    ohlcv_feat = compute_features(ohlcv_15m, feat_config)
    print(f"  {len(OHLCV_FEATURES)} features", flush=True)

    # ============================================================
    # 3. Extract BERT embeddings
    # ============================================================
    print("[3/4] Loading BERT embeddings...")
    bert_model, bert_config = load_bert_model(device)
    d_model = bert_config.d_model

    bt_dates = sorted([f.stem for f in PARQUET_DIR.glob("*.parquet")])
    print(f"  bookTicker: {len(bt_dates)} dates ({bt_dates[0]} to {bt_dates[-1]})")

    # Restrict to overlap period
    bt_start = pd.Timestamp(bt_dates[0], tz="UTC")
    bt_end = pd.Timestamp(bt_dates[-1], tz="UTC") + pd.Timedelta(days=1)
    overlap_mask = (ohlcv_15m.index >= bt_start) & (ohlcv_15m.index < bt_end)
    overlap_bars = ohlcv_15m.index[overlap_mask]

    # Extract + aggregate to bars in one memory-efficient pass
    bar_bert, bar_stats = extract_bar_features(
        bert_model, bert_config, PARQUET_DIR, bt_dates, overlap_bars,
        device=str(device), cache_path=EMBED_CACHE)

    del bert_model
    torch.cuda.empty_cache()

    ohlcv_overlap = ohlcv_15m.loc[overlap_mask].copy()
    feat_overlap = ohlcv_feat.loc[overlap_mask].copy()

    print(f"  Overlap: {len(overlap_bars):,} bars ({overlap_bars.min().date()} to {overlap_bars.max().date()})")

    # ============================================================
    # 4. Walk-forward A/B test
    # ============================================================
    print(f"\n[4/4] Walk-forward test (per-fold PCA/scaling)...", flush=True)

    # Forward return
    close = ohlcv_overlap["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)

    # Vol for vol filter
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    # Embargo
    median_gap = ohlcv_overlap.index.to_series().diff().dropna().median()
    embargo_td = median_gap * (FWD_BARS + 1)

    # Aligned arrays (drop NaN rows)
    valid_mask = feat_overlap[OHLCV_FEATURES].notna().all(axis=1) & fwd_ret.notna()
    valid_idx = feat_overlap.index[valid_mask]

    ohlcv_arr = feat_overlap.loc[valid_idx, OHLCV_FEATURES].values.astype(np.float32)
    bert_arr = bar_bert.loc[valid_idx].values.astype(np.float32)
    stats_arr = bar_stats.loc[valid_idx].values.astype(np.float32)
    y_all = (fwd_ret.loc[valid_idx] > 0).astype(int).values
    fwd_ret_all = fwd_ret.loc[valid_idx].values
    vol_all = vol.reindex(valid_idx)

    # Generate monthly test periods (start 6 months in)
    start_date = valid_idx.min()
    end_date = valid_idx.max()
    first_test = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(first_test, end_date, freq="M")

    print(f"  Valid bars: {len(valid_idx):,}")
    print(f"  Test months: {len(all_months)} ({all_months[0]} to {all_months[-1]})")

    # Results storage
    results = {name: [] for name in APPROACHES}

    for mi, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        train_cutoff = ms - embargo_td

        # Train/test indices
        tr_idx = np.where(valid_idx < train_cutoff)[0]
        te_idx = np.where((valid_idx >= ms) & (valid_idx <= me))[0]

        if len(tr_idx) < 200 or len(te_idx) < 10:
            continue

        # Vol filter on test set (top 50% by training vol)
        tv = vol_all.iloc[tr_idx].dropna()
        if len(tv) < 20:
            continue
        vt = tv.quantile(0.50)
        vol_te = vol_all.iloc[te_idx]
        high_vol = (vol_te > vt).fillna(False).values
        te_idx = te_idx[high_vol]
        if len(te_idx) < 5:
            continue

        # Labels and sample weights
        y_train = y_all[tr_idx]
        y_test_fwd = fwd_ret_all[te_idx]
        train_ts = valid_idx[tr_idx].values.astype(np.int64)
        sample_w = compute_sample_weights(train_ts, fwd_bars=FWD_BARS)

        # Per-window winsorization on returns
        ohlcv_tr = ohlcv_arr.copy()
        ohlcv_te_full = ohlcv_arr.copy()
        for ci, col in enumerate(OHLCV_FEATURES):
            if col.startswith("ret_"):
                train_vals = ohlcv_arr[tr_idx, ci]
                lo = np.nanpercentile(train_vals, 1)
                hi = np.nanpercentile(train_vals, 99)
                ohlcv_tr[:, ci] = np.clip(ohlcv_arr[:, ci], lo, hi)
                ohlcv_te_full[:, ci] = np.clip(ohlcv_arr[:, ci], lo, hi)

        print(f"\n  Month {month}: train={len(tr_idx):,} test={len(te_idx):,}", flush=True)

        for name, (approach_fn, model_fn) in APPROACHES.items():
            try:
                X_train, X_test, info = approach_fn(
                    ohlcv_tr, bert_arr, stats_arr, tr_idx, te_idx)

                # Scale per fold
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                # Fit
                clf = model_fn()
                if name == "C: XGBoost":
                    clf.fit(X_train_s, y_train, sample_weight=sample_w)
                else:
                    clf.fit(X_train_s, y_train, sample_weight=sample_w)

                # Predict
                proba = clf.predict_proba(X_test_s)[:, 1]
                direction = np.where(proba > 0.5, 1.0, -1.0)
                confidence = np.abs(proba - 0.5)

                # Per-window confidence threshold (70th pct of training conf)
                train_proba = clf.predict_proba(X_train_s)[:, 1]
                train_conf = np.abs(train_proba - 0.5)
                conf_threshold = np.percentile(train_conf, 70.0)
                conf_mask = confidence >= conf_threshold

                # Record filtered signals
                pnl = direction[conf_mask] * y_test_fwd[conf_mask]
                results[name].append({
                    "month": str(month),
                    "n_signals": int(conf_mask.sum()),
                    "pnl": float(pnl.sum()),
                    "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0,
                    "mean_conf": float(confidence[conf_mask].mean()) if conf_mask.any() else 0,
                    **info,
                })

                pca_str = f" pca_var={info['pca_var']:.1%}" if 'pca_var' in info else ""
                print(f"    {name:18s}: {conf_mask.sum():4d} sigs, "
                      f"PnL={pnl.sum():+.4f}, WR={((pnl>0).mean() if len(pnl)>0 else 0):.1%}{pca_str}",
                      flush=True)

            except Exception as e:
                print(f"    {name:18s}: ERROR - {e}", flush=True)
                results[name].append({"month": str(month), "n_signals": 0,
                                      "pnl": 0, "win_rate": 0, "mean_conf": 0})

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print(f"  LEAK-FREE A/B TEST: ETH 15m Walk-Forward")
    print(f"  Period: {all_months[0]} to {all_months[-1]}")
    print("=" * 70)

    for name, rows in results.items():
        if not rows:
            print(f"\n  {name}: No results")
            continue
        total_sigs = sum(r["n_signals"] for r in rows)
        total_pnl = sum(r["pnl"] for r in rows)
        all_pnl = []
        for r in rows:
            if r["n_signals"] > 0:
                all_pnl.append(r["pnl"])
        avg_wr = np.mean([r["win_rate"] for r in rows if r["n_signals"] > 0]) if any(r["n_signals"] > 0 for r in rows) else 0

        # Rough annualized Sharpe from monthly PnL
        if len(all_pnl) > 1:
            monthly_arr = np.array(all_pnl)
            sharpe = monthly_arr.mean() / (monthly_arr.std() + 1e-10) * np.sqrt(12)
        else:
            sharpe = 0

        print(f"\n  === {name} ===")
        print(f"  Signals: {total_sigs}")
        print(f"  Total PnL: {total_pnl:+.4f}")
        print(f"  Avg Win Rate: {avg_wr:.1%}")
        print(f"  Sharpe (annualized): {sharpe:.2f}")
        print(f"  Per-month:")
        for r in rows:
            pca_str = f" [PCA var={r['pca_var']:.0%}]" if 'pca_var' in r else ""
            print(f"    {r['month']}: PnL={r['pnl']:+.4f} ({r['n_signals']} sigs, WR={r['win_rate']:.0%}){pca_str}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
