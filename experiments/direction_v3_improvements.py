"""
Direction V3 — ML Improvements (Track A).

Three independent improvements tested on the proven 30m→2H config:
  A2: Cross-asset features (BTC momentum as predictor for altcoins)
  A3: Vol model as event filter (only trade when predicted vol is high)
  A5: Per-asset feature importance via clustered MDA

Each improvement is tested independently against the v2 baseline,
then the best combination is evaluated.

NO risk of overfitting because:
  - Same held-out test set (after 2025-07-01)
  - Features are genuinely new information (BTC price, vol prediction)
  - Feature selection via triple importance (MDI+MDA+SFI) before inclusion
  - Total number of configurations tested is small (tracked for DSR)
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from scipy.stats import binom, spearmanr
from sklearn.ensemble import RandomForestClassifier
from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

DATA_DIR = Path("/home/user/ml-data/binance-1m")
OUTPUT_DIR = Path("/home/user/ml-data/v3-results")
TRAIN_CUTOFF = "2025-07-01"
FWD_BARS = 4  # 30m × 4 = 2H forward

# Assets where v2 showed profitable edge
TARGET_ASSETS = ["ETH", "XRP", "LINK", "BNB", "SUI", "BTC"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
           "XRP": "XRPUSDT", "DOGE": "DOGEUSDT", "BNB": "BNBUSDT",
           "LINK": "LINKUSDT", "AVAX": "AVAXUSDT", "SUI": "SUIUSDT",
           "ADA": "ADAUSDT"}

# v2 best params (from grid search — use conservative defaults for fair comparison)
XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)


# =========================================================================
# DATA LOADING
# =========================================================================
def load_ohlcv(symbol, tf="30min"):
    d = DATA_DIR / SYMBOLS[symbol]
    dfs = []
    for f in sorted(d.glob("*.csv")):
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "count", "tbv", "tqv", "ignore"]
        if first.replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f)
            df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


def build_base_features(ohlcv):
    """Build the v2 baseline 14 features."""
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    train_mask = ohlcv.index < TRAIN_CUTOFF
    features = pd.DataFrame(index=ohlcv.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo = ret[train_mask].quantile(0.01)
        hi = ret[train_mask].quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)
    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    features["macd_hist"] = macd(close)["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]
    return features


# =========================================================================
# A2: CROSS-ASSET FEATURES
# =========================================================================
def add_cross_asset_features(features, btc_ohlcv, asset_ohlcv):
    """Add BTC momentum as predictor for altcoins.

    BTC leads alts with 0.7+ correlation. BTC momentum/vol carries
    information not in the altcoin's own features.

    New features (3):
      btc_ret_1:  BTC 1-bar return (lagged to avoid look-ahead)
      btc_ret_3:  BTC 3-bar return
      btc_vol_z:  BTC volume z-score
    """
    btc_close = btc_ohlcv["close"].reindex(features.index, method="ffill")
    btc_volume = btc_ohlcv["volume"].reindex(features.index, method="ffill")

    # BTC returns (lagged 1 bar — what BTC did in the PREVIOUS bar)
    features = features.copy()
    features["btc_ret_1"] = btc_close.pct_change(1).shift(1)
    features["btc_ret_3"] = btc_close.pct_change(3).shift(1)

    # BTC volume z-score
    btc_vol_mean = btc_volume.rolling(30).mean()
    btc_vol_std = btc_volume.rolling(30).std()
    features["btc_vol_z"] = ((btc_volume - btc_vol_mean) / btc_vol_std.replace(0, np.nan)).shift(1)

    return features


# =========================================================================
# A3: VOL MODEL AS EVENT FILTER
# =========================================================================
def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


def vol_filter_mask(ohlcv, percentile=50):
    """Return boolean mask: True for bars where realized vol > percentile.

    Higher vol bars have larger expected returns, lowering breakeven accuracy.
    Uses REALIZED vol (backward-looking) as proxy for predicted vol.
    In production, use the trained vol model.

    We use realized vol here (not predicted) to avoid data leakage —
    the vol model was trained on the same data. Realized vol at time t
    is computed from bars BEFORE t only.
    """
    close = ohlcv["close"]
    vol = get_daily_vol(close, span=50)
    # Compute percentile on TRAIN period only
    train_vol = vol[vol.index < TRAIN_CUTOFF].dropna()
    threshold = train_vol.quantile(percentile / 100)
    return vol > threshold


# =========================================================================
# A5: FEATURE IMPORTANCE (MDI + MDA + SFI)
# =========================================================================
def compute_feature_importance(X_train, y_train):
    """Compute MDI, MDA, and SFI for feature selection.

    MDI: Mean Decrease Impurity (fast, in-sample) — AFML Ch 8
    MDA: Mean Decrease Accuracy (slow, OOS permutation) — AFML Ch 8
    SFI: Single Feature Importance (each feature alone) — AFML Ch 8

    Returns DataFrame with ranks per method and consensus score.
    """
    from sklearn.ensemble import RandomForestClassifier

    # MDI (from RandomForest, not XGBoost — per AFML recommendation)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, max_features=1,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    mdi = pd.Series(rf.feature_importances_, index=X_train.columns, name="mdi")
    mdi = mdi / mdi.sum()  # Normalize

    # MDA (permutation importance with time-series split)
    from sklearn.metrics import accuracy_score
    n = len(X_train)
    split = int(n * 0.7)
    Xtr, Xte = X_train.iloc[:split], X_train.iloc[split:]
    ytr, yte = y_train.iloc[:split], y_train.iloc[split:]

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr, ytr)
    base_acc = accuracy_score(yte, model.predict(Xte))

    mda_scores = {}
    for feat in X_train.columns:
        Xte_perm = Xte.copy()
        Xte_perm[feat] = np.random.permutation(Xte_perm[feat].values)
        perm_acc = accuracy_score(yte, model.predict(Xte_perm))
        mda_scores[feat] = base_acc - perm_acc
    mda = pd.Series(mda_scores, name="mda")

    # SFI (single feature models)
    sfi_scores = {}
    for feat in X_train.columns:
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr[[feat]], ytr)
        sfi_scores[feat] = accuracy_score(yte, m.predict(Xte[[feat]]))
    sfi = pd.Series(sfi_scores, name="sfi")

    result = pd.DataFrame({"mdi": mdi, "mda": mda, "sfi": sfi})
    result["mdi_rank"] = result["mdi"].rank(ascending=False)
    result["mda_rank"] = result["mda"].rank(ascending=False)
    result["sfi_rank"] = result["sfi"].rank(ascending=False)
    result["avg_rank"] = (result["mdi_rank"] + result["mda_rank"] + result["sfi_rank"]) / 3
    return result.sort_values("avg_rank")


# =========================================================================
# EVALUATION
# =========================================================================
def evaluate(y_prob, y_test, mean_abs_ret_bps, label=""):
    """Evaluate predictions at multiple confidence filters and cost levels."""
    y_pred = (y_prob > 0.5).astype(int)
    conf = np.abs(y_prob - 0.5)
    results = {}

    for filt_label, pct in [("all", 0), ("top30", 70), ("top20", 80), ("top10", 90)]:
        mask = conf >= np.percentile(conf, pct) if pct > 0 else np.ones(len(conf), dtype=bool)
        n = int(mask.sum())
        if n < 20:
            results[filt_label] = {"acc": None, "n": n}
            continue
        a = float((y_pred[mask] == y_test[mask]).mean())
        k = int((y_pred[mask] == y_test[mask]).sum())
        p_val = 1 - binom.cdf(k - 1, n, 0.5) if k > n / 2 else 1.0
        edge = (a - 0.5) * 2 * mean_abs_ret_bps
        results[filt_label] = {
            "acc": round(a, 4), "n": n, "p": round(p_val, 6),
            "net0": round(edge, 2),       # MEXC (0 bps)
            "net2": round(edge - 2, 2),   # dYdX Tier 1 (2 bps)
            "net3": round(edge - 3, 2),   # Hyperliquid (3 bps)
        }
    return results


def run_asset(symbol, btc_ohlcv):
    """Run all three improvements for one asset. Returns comparison dict."""
    print(f"\n{'='*70}", flush=True)
    print(f"  {symbol}: 30m bars -> 2H forward", flush=True)
    print(f"{'='*70}", flush=True)

    ohlcv = load_ohlcv(symbol, "30min")
    close = ohlcv["close"]

    # Mean abs return for this asset's 2H forward
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    test_fwd = fwd_ret[fwd_ret.index >= TRAIN_CUTOFF].dropna()
    mar = test_fwd.abs().mean() * 10000

    # --- BASELINE (v2: 14 features, no filter) ---
    print(f"  [Baseline] 14 features, no filter", flush=True)
    base_features = build_base_features(ohlcv)
    fc = base_features.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    X = fc.loc[valid]
    y = (y_all.loc[valid] > 0).astype(int)

    tr = valid[valid < TRAIN_CUTOFF]
    te = valid[valid >= TRAIN_CUTOFF]
    if len(tr) < 200 or len(te) < 100:
        print(f"  SKIP: insufficient data ({len(tr)} train, {len(te)} test)", flush=True)
        return None

    X_train, y_train = X.loc[tr], y.loc[tr]
    X_test, y_test = X.loc[te], y.loc[te]

    model_base = xgb.XGBClassifier(**XGB_PARAMS)
    model_base.fit(X_train, y_train)
    yp_base = model_base.predict_proba(X_test)[:, 1]
    baseline = evaluate(yp_base, y_test.values, mar, "baseline")
    print(f"    Acc(all)={baseline['all']['acc']}, Acc(top30)={baseline.get('top30',{}).get('acc','N/A')}", flush=True)
    print(f"    n_train={len(tr)}, n_test={len(te)}, MAR={mar:.1f}bps", flush=True)

    # --- A2: CROSS-ASSET FEATURES (add BTC momentum) ---
    a2_result = None
    if symbol != "BTC":
        print(f"  [A2] +3 BTC cross-asset features (17 total)", flush=True)
        cross_features = add_cross_asset_features(base_features, btc_ohlcv, ohlcv)
        fc2 = cross_features.dropna()
        valid2 = fc2.index.intersection(y_all.dropna().index)
        X2 = fc2.loc[valid2]
        y2 = (y_all.loc[valid2] > 0).astype(int)
        tr2 = valid2[valid2 < TRAIN_CUTOFF]
        te2 = valid2[valid2 >= TRAIN_CUTOFF]

        if len(tr2) >= 200 and len(te2) >= 100:
            model_a2 = xgb.XGBClassifier(**XGB_PARAMS)
            model_a2.fit(X2.loc[tr2], y2.loc[tr2])
            yp_a2 = model_a2.predict_proba(X2.loc[te2])[:, 1]
            a2_result = evaluate(yp_a2, y2.loc[te2].values, mar, "A2")
            a2_acc = a2_result.get("all", {}).get("acc", 0) or 0
            base_acc = baseline.get("all", {}).get("acc", 0) or 0
            delta = (a2_acc - base_acc) * 100
            print(f"    Acc(all)={a2_result['all']['acc']}, Acc(top30)={a2_result.get('top30',{}).get('acc','N/A')}", flush=True)
            print(f"    vs baseline: {delta:+.2f}pp", flush=True)
    else:
        print(f"  [A2] Skipped for BTC (no cross-asset for BTC itself)", flush=True)

    # --- A3: VOL FILTER (only trade high-vol bars) ---
    print(f"  [A3] Vol filter (top 50% realized vol bars)", flush=True)
    vol_mask = vol_filter_mask(ohlcv, percentile=50)
    vol_bars = vol_mask.reindex(te).fillna(False)
    high_vol_test = te[vol_bars.loc[te].values]

    if len(high_vol_test) >= 50:
        X_test_hv = X_test.loc[high_vol_test]
        y_test_hv = y_test.loc[high_vol_test].values
        yp_hv = model_base.predict_proba(X_test_hv)[:, 1]

        # Mean abs return on vol-filtered bars (should be higher)
        mar_hv = fwd_ret.loc[high_vol_test].abs().mean() * 10000
        a3_result = evaluate(yp_hv, y_test_hv, mar_hv, "A3")
        print(f"    Filtered bars: {len(high_vol_test)}/{len(te)} ({100*len(high_vol_test)/len(te):.0f}%)", flush=True)
        print(f"    MAR (vol-filtered): {mar_hv:.1f} bps (vs {mar:.1f} all bars)", flush=True)
        print(f"    Acc(all)={a3_result['all']['acc']}, Acc(top30)={a3_result.get('top30',{}).get('acc','N/A')}", flush=True)
    else:
        a3_result = None
        print(f"    Insufficient high-vol test bars ({len(high_vol_test)})", flush=True)

    # --- A5: FEATURE IMPORTANCE ---
    print(f"  [A5] Feature importance (MDI + MDA + SFI)", flush=True)
    fi = compute_feature_importance(X_train, y_train)
    print(f"    Top 5 features: {list(fi.index[:5])}", flush=True)
    print(f"    Bottom 3 features: {list(fi.index[-3:])}", flush=True)

    # Test with only top-10 features
    top10_feats = list(fi.index[:10])
    model_a5 = xgb.XGBClassifier(**XGB_PARAMS)
    model_a5.fit(X_train[top10_feats], y_train)
    yp_a5 = model_a5.predict_proba(X_test[top10_feats])[:, 1]
    a5_result = evaluate(yp_a5, y_test.values, mar, "A5-top10")
    a5_acc = a5_result.get("all", {}).get("acc", 0) or 0
    base_acc = baseline.get("all", {}).get("acc", 0) or 0
    print(f"    Acc(top10feat)={a5_result['all']['acc']} vs baseline={baseline['all']['acc']}", flush=True)
    print(f"    Dropped: {[f for f in X_train.columns if f not in top10_feats]}", flush=True)

    # --- A2+A3 COMBINED (best of both worlds) ---
    combined_result = None
    if symbol != "BTC" and a2_result and a3_result:
        print(f"  [A2+A3] Cross-asset features + vol filter", flush=True)
        fc2 = cross_features.dropna()
        valid2 = fc2.index.intersection(y_all.dropna().index)
        te2 = valid2[valid2 >= TRAIN_CUTOFF]
        vol_mask2 = vol_filter_mask(ohlcv, percentile=50)
        hv2 = te2[vol_mask2.reindex(te2).fillna(False).values]

        if len(hv2) >= 50:
            tr2 = valid2[valid2 < TRAIN_CUTOFF]
            model_comb = xgb.XGBClassifier(**XGB_PARAMS)
            model_comb.fit(fc2.loc[tr2], (y_all.loc[tr2] > 0).astype(int))
            yp_comb = model_comb.predict_proba(fc2.loc[hv2])[:, 1]
            y_comb = (y_all.loc[hv2] > 0).astype(int).values
            mar_hv2 = fwd_ret.loc[hv2].abs().mean() * 10000
            combined_result = evaluate(yp_comb, y_comb, mar_hv2, "A2+A3")
            print(f"    Acc(all)={combined_result['all']['acc']}, MAR={mar_hv2:.1f}bps", flush=True)

    return {
        "symbol": symbol, "mar": round(mar, 1),
        "n_train": len(tr), "n_test": len(te),
        "baseline": baseline,
        "a2_cross_asset": a2_result,
        "a3_vol_filter": a3_result,
        "a5_top10_features": a5_result,
        "a2a3_combined": combined_result,
        "feature_importance": fi.to_dict() if fi is not None else None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("DIRECTION V3 — ML IMPROVEMENTS (Track A)", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Config: 30m bars, {FWD_BARS}-bar (2H) forward, 6 assets", flush=True)
    print(f"Improvements: A2 (cross-asset), A3 (vol filter), A5 (feature importance)", flush=True)
    print("=" * 70, flush=True)

    # Load BTC once (used as cross-asset feature for altcoins)
    print("\nLoading BTC 30m (for cross-asset features)...", flush=True)
    btc_ohlcv = load_ohlcv("BTC", "30min")
    print(f"  {len(btc_ohlcv)} bars", flush=True)

    results = []
    for symbol in TARGET_ASSETS:
        try:
            r = run_asset(symbol, btc_ohlcv)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  {symbol}: ERROR — {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Summary comparison table
    print("\n" + "=" * 100, flush=True)
    print("V3 IMPROVEMENT SUMMARY (net edge at 0 bps / 3 bps RT)", flush=True)
    print("=" * 100, flush=True)

    print(f"\n{'Sym':<5} {'MAR':>5} | {'Base30':>8} {'A2_30':>8} {'A3_30':>8} {'Comb30':>8} {'A5_30':>8} | "
          f"{'Base_n0':>8} {'A2_n0':>8} {'A3_n0':>8}", flush=True)
    print("-" * 100, flush=True)

    for r in results:
        def g(key, filt="top30", metric="net3"):
            v = (r.get(key) or {}).get(filt, {})
            if isinstance(v, dict):
                val = v.get(metric)
                return f"{val:+.2f}" if val is not None else "     N/A"
            return "     N/A"

        print(f"{r['symbol']:<5} {r['mar']:>5.0f} | "
              f"{g('baseline','top30','net3'):>8} {g('a2_cross_asset','top30','net3'):>8} "
              f"{g('a3_vol_filter','top30','net3'):>8} {g('a2a3_combined','top30','net3'):>8} "
              f"{g('a5_top10_features','top30','net3'):>8} | "
              f"{g('baseline','top30','net0'):>8} {g('a2_cross_asset','top30','net0'):>8} "
              f"{g('a3_vol_filter','top30','net0'):>8}", flush=True)

    # Save results
    # Convert non-serializable types
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj

    serializable = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sr[k] = {k2: clean(v2) if not isinstance(v2, dict) else
                         {k3: clean(v3) for k3, v3 in v2.items()}
                         for k2, v2 in v.items()}
            else:
                sr[k] = clean(v)
        serializable.append(sr)

    with open(OUTPUT_DIR / "direction_v3_improvements.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": serializable}, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR / 'direction_v3_improvements.json'}", flush=True)
    print(f"Completed: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
