"""
Final audit checks:
- Feature importance analysis (which features drive the edge?)
- OHLCV-only baseline (does derivatives data actually help?)
- Check if model is just exploiting return autocorrelation
- Prediction bias analysis
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
import lz4.frame, csv, io
from pathlib import Path
from scipy.stats import binom
from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from derivatives import funding_features, oi_features, premium_features, impact_features

# ===== DATA LOADING =====
print("Loading data...", flush=True)
btc_dir = Path("/home/user/ml-data/binance-1m/BTCUSDT")
csvs = sorted(btc_dir.glob("*.csv"))
dfs = []
for f in csvs:
    with open(f) as fh:
        first = fh.readline().strip().split(",")[0]
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qv","count","tbv","tqv","ignore"]
    if first.replace(".","").replace("-","").isdigit():
        df = pd.read_csv(f, header=None, names=cols)
    else:
        df = pd.read_csv(f)
        df.columns = cols
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open","high","low","close","volume"]]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    dfs.append(df)
combined = pd.concat(dfs).sort_index()
combined = combined[~combined.index.duplicated(keep="first")]
btc_15m = combined.resample("15min").agg({
    "open":"first","high":"max","low":"min","close":"last","volume":"sum",
}).dropna(subset=["open"])

ctx_dir = Path("/home/user/ml-data/hl-asset-ctxs")
ohlcv_start = btc_15m.index[0].strftime("%Y%m%d")
ohlcv_end = btc_15m.index[-1].strftime("%Y%m%d")
ctx_dfs = []
for fpath in sorted(ctx_dir.glob("*.lz4")):
    date_str = fpath.stem.replace(".csv", "")
    if date_str < ohlcv_start or date_str > ohlcv_end:
        continue
    try:
        raw = lz4.frame.decompress(open(fpath, "rb").read()).decode("utf-8")
        rows = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("coin") == "BTC"]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        num_cols = ["funding","open_interest","premium","oracle_px","mark_px","impact_bid_px","impact_ask_px"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        ctx_dfs.append(df[num_cols])
    except Exception:
        pass
deriv = pd.concat(ctx_dfs).sort_index()
deriv = deriv[~deriv.index.duplicated(keep="first")]
deriv_15m = deriv.resample("15min").last().reindex(btc_15m.index, method="ffill")

close = btc_15m["close"]
high = btc_15m["high"]
low = btc_15m["low"]
volume = btc_15m["volume"]
features = pd.DataFrame(index=btc_15m.index)

for lb in [1, 3, 7, 14]:
    ret = close.pct_change(lb)
    lo_q, hi_q = ret.quantile(0.01), ret.quantile(0.99)
    features[f"ret_{lb}"] = ret.clip(lo_q, hi_q)

features["bar_range"] = (high - low) / close
features["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
features["rsi_14"] = rsi(close, 14)
features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
macd_df = macd(close)
features["macd_hist"] = macd_df["macd_hist"]
features["bb_pct_b"] = bollinger_pct_b(close, 20)
features["atr_14"] = atr(high, low, close, 14)
sw = swing_levels(high, low, lookback=20)
features["dist_swing_high"] = sw["dist_swing_high"]
features["dist_swing_low"] = sw["dist_swing_low"]
features["dist_nearest_level"] = sw["dist_nearest_level"]

if len(deriv_15m) > 0 and "funding" in deriv_15m.columns:
    fund_f = funding_features(deriv_15m["funding"])
    for col in fund_f.columns:
        features[col] = fund_f[col]
    oi_f = oi_features(deriv_15m["open_interest"], close)
    for col in oi_f.columns:
        features[col] = oi_f[col]
    prem_f = premium_features(deriv_15m["mark_px"], deriv_15m["oracle_px"])
    for col in prem_f.columns:
        features[col] = prem_f[col]
    impact_f = impact_features(deriv_15m["impact_bid_px"], deriv_15m["impact_ask_px"], deriv_15m["mark_px"])
    for col in impact_f.columns:
        features[col] = impact_f[col]

features_clean = features.dropna()
train_mask = features_clean.index < "2025-07-01"
test_mask = features_clean.index >= "2025-07-01"
X_train = features_clean[train_mask]
X_test = features_clean[test_mask]

h = 1
fwd_ret = close.pct_change(h).shift(-h)
y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
common_train = X_train.index.intersection(y_all.index)
common_test = X_test.index.intersection(y_all.index)
y_train = (y_all.loc[common_train] > 0).astype(int)
y_test = (y_all.loc[common_test] > 0).astype(int)

# ===== TEST A: OHLCV-ONLY MODEL =====
print("\n=== TEST A: OHLCV-ONLY MODEL (no derivatives) ===", flush=True)
ohlcv_cols = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore"]
model_ohlcv = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model_ohlcv.fit(X_train.loc[common_train][ohlcv_cols], y_train)
y_prob_ohlcv = model_ohlcv.predict_proba(X_test.loc[common_test][ohlcv_cols])[:, 1]
y_pred_ohlcv = (y_prob_ohlcv > 0.5).astype(int)
acc_ohlcv = (y_pred_ohlcv == y_test.values).mean()
print(f"OHLCV-only accuracy: {acc_ohlcv:.4f}")

# ===== TEST B: TECHNICAL-ONLY MODEL =====
print("\n=== TEST B: TECHNICAL-ONLY MODEL ===", flush=True)
tech_cols = ["rsi_14","stoch_rsi","macd_hist","bb_pct_b","atr_14",
             "dist_swing_high","dist_swing_low","dist_nearest_level"]
model_tech = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model_tech.fit(X_train.loc[common_train][tech_cols], y_train)
y_prob_tech = model_tech.predict_proba(X_test.loc[common_test][tech_cols])[:, 1]
y_pred_tech = (y_prob_tech > 0.5).astype(int)
acc_tech = (y_pred_tech == y_test.values).mean()
print(f"Technical-only accuracy: {acc_tech:.4f}")

# ===== TEST C: DERIVATIVES-ONLY MODEL =====
print("\n=== TEST C: DERIVATIVES-ONLY MODEL ===", flush=True)
deriv_cols = [c for c in features_clean.columns if c not in ohlcv_cols and c not in tech_cols]
model_deriv = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model_deriv.fit(X_train.loc[common_train][deriv_cols], y_train)
y_prob_deriv = model_deriv.predict_proba(X_test.loc[common_test][deriv_cols])[:, 1]
y_pred_deriv = (y_prob_deriv > 0.5).astype(int)
acc_deriv = (y_pred_deriv == y_test.values).mean()
print(f"Derivatives-only accuracy: {acc_deriv:.4f}")

print(f"\nFull model: 0.5213")
print(f"OHLCV-only: {acc_ohlcv:.4f}")
print(f"Tech-only:  {acc_tech:.4f}")
print(f"Deriv-only: {acc_deriv:.4f}")

# ===== TEST D: FEATURE IMPORTANCE =====
print("\n=== TEST D: FEATURE IMPORTANCE (full model) ===", flush=True)
model_full = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model_full.fit(X_train.loc[common_train], y_train)
importances = pd.Series(model_full.feature_importances_, index=features_clean.columns)
importances = importances.sort_values(ascending=False)
print("All features ranked by importance:")
for feat, imp in importances.items():
    cat = "OHLCV" if feat in ohlcv_cols else "TECH" if feat in tech_cols else "DERIV"
    print(f"  {feat:>25}: {imp:.4f} [{cat}]")

# ===== TEST E: ret_1 AUTOCORRELATION CHECK =====
print("\n=== TEST E: IS THE MODEL JUST EXPLOITING RETURN AUTOCORRELATION? ===", flush=True)
# BTC 15m returns have slight negative autocorrelation (mean-reversion)
ret1 = close.pct_change(1)
ret1_test = ret1[ret1.index >= "2025-07-01"].dropna()
ac = ret1_test.autocorr(lag=1)
print(f"BTC 15m return autocorrelation (lag-1): {ac:.4f}")

# Simple strategy: predict direction = -sign(ret_1) (mean-reversion)
mr_pred = (-np.sign(ret1_test.shift(1))).dropna()
mr_pred = (mr_pred > 0).astype(int)
# Align with test labels
fwd = close.pct_change(1).shift(-1)
y_mr = (np.sign(fwd).reindex(mr_pred.index) > 0).astype(int).dropna()
common_mr = mr_pred.index.intersection(y_mr.index)
acc_mr = (mr_pred.loc[common_mr] == y_mr.loc[common_mr]).mean()
print(f"Simple mean-reversion accuracy: {acc_mr:.4f}")

# Simple strategy: predict direction = sign(ret_1) (momentum)
mom_pred = (np.sign(ret1_test.shift(1))).dropna()
mom_pred = (mom_pred > 0).astype(int)
common_mom = mom_pred.index.intersection(y_mr.index)
acc_mom = (mom_pred.loc[common_mom] == y_mr.loc[common_mom]).mean()
print(f"Simple momentum accuracy: {acc_mom:.4f}")

# ===== TEST F: PREDICTION BIAS ANALYSIS =====
print("\n=== TEST F: PREDICTION BIAS ANALYSIS ===", flush=True)
y_prob_full = model_full.predict_proba(X_test.loc[common_test])[:, 1]
y_pred_full = (y_prob_full > 0.5).astype(int)

# Accuracy when predicting up vs down
pred_up_mask = y_pred_full == 1
pred_down_mask = y_pred_full == 0
if pred_up_mask.sum() > 0:
    acc_when_up = (y_test.values[pred_up_mask] == 1).mean()
    print(f"Accuracy when predicting UP: {acc_when_up:.4f} (n={pred_up_mask.sum()})")
if pred_down_mask.sum() > 0:
    acc_when_down = (y_test.values[pred_down_mask] == 0).mean()
    print(f"Accuracy when predicting DOWN: {acc_when_down:.4f} (n={pred_down_mask.sum()})")

# Accuracy by actual direction
actual_up = y_test.values == 1
actual_down = y_test.values == 0
print(f"Precision on UP: {(y_pred_full[actual_up] == 1).mean():.4f} (recall)")
print(f"Precision on DOWN: {(y_pred_full[actual_down] == 0).mean():.4f} (recall)")

# ===== TEST G: WALK-FORWARD STABILITY =====
print("\n=== TEST G: WALK-FORWARD CHUNKS (2-month windows) ===", flush=True)
# Split test into 2-month chunks and retrain on expanding window
test_start = pd.Timestamp("2025-07-01", tz="UTC")
chunk_size_days = 60
results_wf = []

chunk_start = test_start
while chunk_start < features_clean.index[-1]:
    chunk_end = chunk_start + pd.Timedelta(days=chunk_size_days)
    chunk_mask = (features_clean.index >= chunk_start) & (features_clean.index < chunk_end)
    X_chunk = features_clean[chunk_mask]

    y_chunk_all = y_all.reindex(X_chunk.index).dropna()
    if len(y_chunk_all) < 100:
        chunk_start = chunk_end
        continue

    # Train on all data before this chunk
    train_before = features_clean.index < chunk_start
    X_tr = features_clean[train_before]
    common_tr = X_tr.index.intersection(y_all.index)
    y_tr = (y_all.loc[common_tr] > 0).astype(int)

    common_ch = X_chunk.index.intersection(y_chunk_all.index)
    y_ch = (y_chunk_all.loc[common_ch] > 0).astype(int)

    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    m.fit(X_tr.loc[common_tr], y_tr)
    prob = m.predict_proba(X_chunk.loc[common_ch])[:, 1]
    pred = (prob > 0.5).astype(int)
    a = (pred == y_ch.values).mean()

    k = int(a * len(y_ch))
    p = 1 - binom.cdf(k - 1, len(y_ch), 0.5) if k > len(y_ch) / 2 else 1.0

    results_wf.append({
        "start": chunk_start.strftime("%Y-%m-%d"),
        "end": chunk_end.strftime("%Y-%m-%d"),
        "acc": a,
        "n": len(y_ch),
        "p": p
    })
    print(f"  {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: acc={a:.4f}, n={len(y_ch)}, p={p:.4f}")

    chunk_start = chunk_end

if results_wf:
    accs = [r["acc"] for r in results_wf]
    print(f"\nWalk-forward mean: {np.mean(accs):.4f}")
    print(f"Walk-forward min:  {np.min(accs):.4f}")
    print(f"Walk-forward max:  {np.max(accs):.4f}")
    below = sum(1 for a in accs if a < 0.50)
    print(f"Chunks below 50%: {below}/{len(accs)}")

print("\nAll final checks complete.", flush=True)
