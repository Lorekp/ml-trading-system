"""
Adversarial tests for direction prediction audit.
Tests: shuffle labels, shift features, reverse split, monthly stability.
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

# ===== DATA LOADING (same as experiment) =====
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

# Load derivatives
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

# Build features (same as experiment)
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
print(f"Features: {len(features_clean.columns)} cols, {len(features_clean)} rows", flush=True)

train_mask = features_clean.index < "2025-07-01"
test_mask = features_clean.index >= "2025-07-01"
X_train = features_clean[train_mask]
X_test = features_clean[test_mask]

# Labels for h=1
h = 1
fwd_ret = close.pct_change(h).shift(-h)
y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
common_train = X_train.index.intersection(y_all.index)
common_test = X_test.index.intersection(y_all.index)
y_train = (y_all.loc[common_train] > 0).astype(int)
y_test = (y_all.loc[common_test] > 0).astype(int)

print(f"Train: {len(common_train)}, Test: {len(common_test)}", flush=True)

# ===== BASELINE: Reproduce claimed result =====
print("\n=== TEST 0: REPRODUCE BASELINE ===", flush=True)
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model.fit(X_train.loc[common_train], y_train)
y_prob = model.predict_proba(X_test.loc[common_test])[:, 1]
y_pred = (y_prob > 0.5).astype(int)
acc_baseline = (y_pred == y_test.values).mean()
print(f"Baseline accuracy: {acc_baseline:.4f} (claimed: 0.5213)", flush=True)

# ===== TEST 1: SHUFFLED LABELS =====
print("\n=== TEST 1: SHUFFLED LABELS (should get ~50%) ===", flush=True)
np.random.seed(42)
y_train_shuffled = y_train.sample(frac=1.0, random_state=42).values
model_shuf = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model_shuf.fit(X_train.loc[common_train], y_train_shuffled)
y_prob_shuf = model_shuf.predict_proba(X_test.loc[common_test])[:, 1]
y_pred_shuf = (y_prob_shuf > 0.5).astype(int)
acc_shuffled = (y_pred_shuf == y_test.values).mean()
print(f"Shuffled accuracy: {acc_shuffled:.4f}", flush=True)
if acc_shuffled > 0.51:
    print("WARNING: Shuffled labels still gives >51% -- possible data leak!", flush=True)
else:
    print("PASS: Shuffled labels gives ~50% as expected.", flush=True)

# ===== TEST 2: SHIFT FEATURES +1 BAR (timing leak test) =====
print("\n=== TEST 2: FEATURES SHIFTED +1 BAR (should drop to ~50%) ===", flush=True)
X_test_shifted = X_test.loc[common_test].shift(1).dropna()
y_test_shifted = y_test.reindex(X_test_shifted.index)
y_prob_shift = model.predict_proba(X_test_shifted)[:, 1]
y_pred_shift = (y_prob_shift > 0.5).astype(int)
acc_shifted = (y_pred_shift == y_test_shifted.values).mean()
print(f"Shifted accuracy: {acc_shifted:.4f}", flush=True)
if acc_shifted > 0.515:
    print("CONCERN: Shifted features still gives >51.5%", flush=True)
elif acc_shifted < 0.505:
    print("PASS: Shifted features drops to ~50% -- no timing leak.", flush=True)
else:
    print("MARGINAL: Shifted features gives modest accuracy -- some autocorrelation possible.", flush=True)

# ===== TEST 3: REVERSE SPLIT (train on test, test on train) =====
print("\n=== TEST 3: REVERSE SPLIT ===", flush=True)
model_rev = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
model_rev.fit(X_test.loc[common_test], y_test)
y_prob_rev = model_rev.predict_proba(X_train.loc[common_train])[:, 1]
y_pred_rev = (y_prob_rev > 0.5).astype(int)
acc_reversed = (y_pred_rev == y_train.values).mean()
print(f"Reversed accuracy: {acc_reversed:.4f} (original: {acc_baseline:.4f})", flush=True)
if acc_reversed > acc_baseline + 0.01:
    print("CONCERN: Reversed split gives higher accuracy -- possible memorization.", flush=True)
elif abs(acc_reversed - acc_baseline) < 0.005:
    print("OK: Similar accuracy in both directions -- model generalizes.", flush=True)
else:
    print(f"NOTABLE: Reversed accuracy differs by {abs(acc_reversed - acc_baseline):.4f}", flush=True)

# ===== TEST 4: MONTHLY STABILITY =====
print("\n=== TEST 4: MONTHLY ACCURACY STABILITY ===", flush=True)
test_dates = X_test.loc[common_test].index
y_pred_series = pd.Series(y_pred, index=test_dates)
y_test_series = pd.Series(y_test.values, index=test_dates)
correct = (y_pred_series == y_test_series).astype(int)

monthly_acc = correct.resample("ME").mean()
monthly_n = correct.resample("ME").count()
print(f"{'Month':>12} {'Acc':>7} {'N':>6} {'p_val':>8}")
print("-" * 40)
for month, acc in monthly_acc.items():
    n = monthly_n.loc[month]
    if n > 0:
        k = int(acc * n)
        p = 1 - binom.cdf(k - 1, n, 0.5) if k > n / 2 else 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{month.strftime('%Y-%m'):>12} {acc:>7.4f} {n:>6} {p:>8.4f} {sig}")

# Check if accuracy is concentrated in specific months
acc_values = monthly_acc.dropna().values
if len(acc_values) > 2:
    print(f"\nMean monthly acc: {np.mean(acc_values):.4f}")
    print(f"Std monthly acc:  {np.std(acc_values):.4f}")
    print(f"Min monthly acc:  {np.min(acc_values):.4f}")
    print(f"Max monthly acc:  {np.max(acc_values):.4f}")
    below_50 = sum(1 for a in acc_values if a < 0.50)
    print(f"Months below 50%: {below_50}/{len(acc_values)}")

# ===== TEST 5: CHECK BASE RATE =====
print("\n=== TEST 5: BASE RATE CHECK ===", flush=True)
up_pct_train = y_train.mean()
up_pct_test = y_test.mean()
print(f"Train up%: {up_pct_train:.4f}")
print(f"Test up%:  {up_pct_test:.4f}")
print(f"If always predict majority class: {max(up_pct_test, 1-up_pct_test):.4f}")
if abs(up_pct_test - 0.5) > 0.01:
    print("CONCERN: Test set is imbalanced -- accuracy may be inflated by bias.")
else:
    print("OK: Test set is roughly balanced.")

# ===== TEST 6: PREDICTION AUTOCORRELATION =====
print("\n=== TEST 6: PREDICTION AUTOCORRELATION ===", flush=True)
pred_series = pd.Series(y_pred, index=test_dates)
for lag in [1, 5, 10, 50]:
    ac = pred_series.autocorr(lag=lag)
    print(f"Prediction autocorrelation lag-{lag}: {ac:.4f}")
# Also check if correct predictions are autocorrelated
correct_series = correct
for lag in [1, 5, 10, 50]:
    ac = correct_series.autocorr(lag=lag)
    print(f"Correct-prediction autocorrelation lag-{lag}: {ac:.4f}")

# ===== TEST 7: WHAT MODEL ACTUALLY PREDICTS =====
print("\n=== TEST 7: PREDICTION DISTRIBUTION ===", flush=True)
print(f"Mean predicted probability: {y_prob.mean():.4f}")
print(f"Std predicted probability:  {y_prob.std():.4f}")
print(f"Min predicted probability:  {y_prob.min():.4f}")
print(f"Max predicted probability:  {y_prob.max():.4f}")
print(f"% predictions > 0.5: {(y_prob > 0.5).mean():.4f}")
print(f"% predictions > 0.55: {(y_prob > 0.55).mean():.4f}")
print(f"% predictions < 0.45: {(y_prob < 0.45).mean():.4f}")

# Check if model is just predicting the majority class
unique_preds = np.unique(y_pred)
print(f"Unique predictions: {unique_preds}")
pred_up_pct = y_pred.mean()
print(f"Model predicts up: {pred_up_pct:.4f}")

print("\nDone.", flush=True)
