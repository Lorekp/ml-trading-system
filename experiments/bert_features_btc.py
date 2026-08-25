"""Out-of-sample test: ETH-trained BERT on BTC data (never seen).

Tests whether BERT learned generalizable microstructure patterns.
Only runs most promising approaches: Baseline, Elastic Net, Stats+PCA.

Usage:
    PYTHONPATH=src PYTHONUNBUFFERED=1 python experiments/bert_features_btc.py
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
PARQUET_DIR = Path("data/bookTicker_parquet/BTCUSDT")
BERT_CKPT = Path("data/bbo_bert_eth_ft/finetuned.pt")
EMBED_CACHE = Path("data/bbo_bert_eth_ft/btc_embeddings_cache.npz")

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
    print(f"Loaded ETH-trained BERT: {model.count_parameters():,} params")
    return model, config


def make_logreg_l1():
    return LogisticRegression(C=C_REG, penalty="l1", solver="saga",
                              max_iter=1000, random_state=42)


def make_elastic_net():
    return LogisticRegression(C=C_REG, penalty="elasticnet", solver="saga",
                              l1_ratio=0.5, max_iter=1000, random_state=42)


def approach_baseline(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    return ohlcv_arr[tr_idx], ohlcv_arr[te_idx], {}


def approach_elastic(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
    X_tr = np.hstack([ohlcv_arr[tr_idx], bert_arr[tr_idx]])
    X_te = np.hstack([ohlcv_arr[te_idx], bert_arr[te_idx]])
    return X_tr, X_te, {}


def approach_stats_pca(ohlcv_arr, bert_arr, stats_arr, tr_idx, te_idx):
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
    "Baseline":       (approach_baseline,   make_logreg_l1),
    "B: Elastic Net": (approach_elastic,    make_elastic_net),
    "D: Stats+PCA":   (approach_stats_pca,  make_logreg_l1),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("  OUT-OF-SAMPLE: ETH-trained BERT applied to BTC")
    print("  BERT has NEVER seen BTC tick data")
    print("=" * 60)
    t_start = time.time()

    # 1. Load BTC OHLCV
    print("\n[1/4] Loading BTC OHLCV...")
    ohlcv_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    del ohlcv_1m
    print(f"  15m bars: {len(ohlcv_15m):,}", flush=True)

    # 2. OHLCV features
    print("[2/4] Computing OHLCV features...")
    feat_config = FeatureConfig(feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
                                vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20)
    ohlcv_feat = compute_features(ohlcv_15m, feat_config)

    # 3. BERT embeddings on BTC (using ETH-trained model)
    print("[3/4] Extracting BERT embeddings on BTC...")
    bert_model, bert_config = load_bert_model(device)
    d_model = bert_config.d_model

    bt_dates = sorted([f.stem for f in PARQUET_DIR.glob("*.parquet")])
    print(f"  BTC bookTicker: {len(bt_dates)} dates ({bt_dates[0]} to {bt_dates[-1]})")

    bt_start = pd.Timestamp(bt_dates[0], tz="UTC")
    bt_end = pd.Timestamp(bt_dates[-1], tz="UTC") + pd.Timedelta(days=1)
    overlap_mask = (ohlcv_15m.index >= bt_start) & (ohlcv_15m.index < bt_end)
    overlap_bars = ohlcv_15m.index[overlap_mask]

    bar_bert, bar_stats = extract_bar_features(
        bert_model, bert_config, PARQUET_DIR, bt_dates, overlap_bars,
        device=str(device), cache_path=EMBED_CACHE)

    del bert_model
    torch.cuda.empty_cache()

    ohlcv_overlap = ohlcv_15m.loc[overlap_mask].copy()
    feat_overlap = ohlcv_feat.loc[overlap_mask].copy()
    print(f"  Overlap: {len(overlap_bars):,} bars", flush=True)

    # 4. Walk-forward
    print(f"\n[4/4] Walk-forward (leak-free)...", flush=True)

    close = ohlcv_overlap["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
    median_gap = ohlcv_overlap.index.to_series().diff().dropna().median()
    embargo_td = median_gap * (FWD_BARS + 1)

    valid_mask = feat_overlap[OHLCV_FEATURES].notna().all(axis=1) & fwd_ret.notna()
    valid_idx = feat_overlap.index[valid_mask]

    ohlcv_arr = feat_overlap.loc[valid_idx, OHLCV_FEATURES].values.astype(np.float32)
    bert_arr = bar_bert.loc[valid_idx].values.astype(np.float32)
    stats_arr = bar_stats.loc[valid_idx].values.astype(np.float32)
    y_all = (fwd_ret.loc[valid_idx] > 0).astype(int).values
    fwd_ret_all = fwd_ret.loc[valid_idx].values
    vol_all = vol.reindex(valid_idx)

    start_date = valid_idx.min()
    end_date = valid_idx.max()
    first_test = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(first_test, end_date, freq="M")

    print(f"  Valid bars: {len(valid_idx):,}")
    print(f"  Test months: {len(all_months)} ({all_months[0]} to {all_months[-1]})")

    results = {name: [] for name in APPROACHES}

    for mi, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        train_cutoff = ms - embargo_td

        tr_idx = np.where(valid_idx < train_cutoff)[0]
        te_idx = np.where((valid_idx >= ms) & (valid_idx <= me))[0]

        if len(tr_idx) < 200 or len(te_idx) < 10:
            continue

        tv = vol_all.iloc[tr_idx].dropna()
        if len(tv) < 20:
            continue
        vt = tv.quantile(0.50)
        vol_te = vol_all.iloc[te_idx]
        high_vol = (vol_te > vt).fillna(False).values
        te_idx = te_idx[high_vol]
        if len(te_idx) < 5:
            continue

        y_train = y_all[tr_idx]
        y_test_fwd = fwd_ret_all[te_idx]
        train_ts = valid_idx[tr_idx].values.astype(np.int64)
        sample_w = compute_sample_weights(train_ts, fwd_bars=FWD_BARS)

        # Per-window winsorization
        ohlcv_tr = ohlcv_arr.copy()
        for ci, col in enumerate(OHLCV_FEATURES):
            if col.startswith("ret_"):
                train_vals = ohlcv_arr[tr_idx, ci]
                lo = np.nanpercentile(train_vals, 1)
                hi = np.nanpercentile(train_vals, 99)
                ohlcv_tr[:, ci] = np.clip(ohlcv_arr[:, ci], lo, hi)

        print(f"\n  Month {month}: train={len(tr_idx):,} test={len(te_idx):,}", flush=True)

        for name, (approach_fn, model_fn) in APPROACHES.items():
            try:
                X_train, X_test, info = approach_fn(
                    ohlcv_tr, bert_arr, stats_arr, tr_idx, te_idx)

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = model_fn()
                clf.fit(X_train_s, y_train, sample_weight=sample_w)

                proba = clf.predict_proba(X_test_s)[:, 1]
                direction = np.where(proba > 0.5, 1.0, -1.0)
                confidence = np.abs(proba - 0.5)

                train_proba = clf.predict_proba(X_train_s)[:, 1]
                train_conf = np.abs(train_proba - 0.5)
                conf_threshold = np.percentile(train_conf, 70.0)
                conf_mask = confidence >= conf_threshold

                pnl = direction[conf_mask] * y_test_fwd[conf_mask]
                results[name].append({
                    "month": str(month),
                    "n_signals": int(conf_mask.sum()),
                    "pnl": float(pnl.sum()),
                    "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0,
                })

                print(f"    {name:18s}: {conf_mask.sum():4d} sigs, "
                      f"PnL={pnl.sum():+.4f}, WR={((pnl>0).mean() if len(pnl)>0 else 0):.1%}",
                      flush=True)

            except Exception as e:
                print(f"    {name:18s}: ERROR - {e}", flush=True)
                results[name].append({"month": str(month), "n_signals": 0,
                                      "pnl": 0, "win_rate": 0})

    # Summary
    print("\n" + "=" * 60)
    print("  OUT-OF-SAMPLE RESULTS: ETH BERT on BTC")
    print("=" * 60)

    for name, rows in results.items():
        if not rows:
            continue
        total_sigs = sum(r["n_signals"] for r in rows)
        total_pnl = sum(r["pnl"] for r in rows)
        active = [r for r in rows if r["n_signals"] > 0]
        avg_wr = np.mean([r["win_rate"] for r in active]) if active else 0

        monthly_pnl = np.array([r["pnl"] for r in active]) if active else np.array([0])
        sharpe = monthly_pnl.mean() / (monthly_pnl.std() + 1e-10) * np.sqrt(12) if len(monthly_pnl) > 1 else 0

        print(f"\n  === {name} ===")
        print(f"  Signals: {total_sigs}")
        print(f"  Total PnL: {total_pnl:+.4f}")
        print(f"  Avg Win Rate: {avg_wr:.1%}")
        print(f"  Sharpe (annualized): {sharpe:.2f}")
        print(f"  Per-month:")
        for r in rows:
            print(f"    {r['month']}: PnL={r['pnl']:+.4f} ({r['n_signals']} sigs, WR={r['win_rate']:.0%})")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
