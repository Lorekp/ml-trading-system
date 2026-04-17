"""
Audit checks: stationarity, multiple testing, effective sample size, label verification.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import lz4.frame, csv, io
from pathlib import Path
from scipy.stats import binom, norm
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

# Build features
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
test_mask = features_clean.index >= "2025-07-01"

# ===== CHECK 8: STATIONARITY (ADF proxy via autocorrelation decay) =====
print("\n=== CHECK 8: FEATURE STATIONARITY ===", flush=True)
print("Testing each feature for stationarity using first-order autocorrelation.", flush=True)
print("Stationary features: autocorrelation decays. Non-stationary: autocorrelation near 1.", flush=True)
print()
print(f"{'Feature':>25} {'AC(1)':>7} {'AC(10)':>7} {'AC(50)':>7} {'Status':>12}")
print("-" * 70)

test_data = features_clean[test_mask]
for col in features_clean.columns:
    series = test_data[col].dropna()
    ac1 = series.autocorr(lag=1)
    ac10 = series.autocorr(lag=10)
    ac50 = series.autocorr(lag=50)
    if abs(ac1) > 0.99:
        status = "NON-STAT"
    elif abs(ac1) > 0.95:
        status = "CONCERN"
    elif abs(ac1) > 0.8:
        status = "SLOW-DECAY"
    else:
        status = "OK"
    print(f"{col:>25} {ac1:>7.4f} {ac10:>7.4f} {ac50:>7.4f} {status:>12}")

# Also check raw derivatives before normalization
print("\n=== RAW DERIVATIVES STATIONARITY ===", flush=True)
raw_deriv = deriv_15m[deriv_15m.index >= "2025-07-01"].dropna()
for col in ["funding", "open_interest", "premium", "oracle_px", "mark_px"]:
    if col in raw_deriv.columns:
        series = raw_deriv[col].dropna()
        if len(series) > 100:
            ac1 = series.autocorr(lag=1)
            ac10 = series.autocorr(lag=10)
            ac50 = series.autocorr(lag=50)
            print(f"{col:>25} {ac1:>7.4f} {ac10:>7.4f} {ac50:>7.4f}")

# ===== CHECK 9: MULTIPLE TESTING / BONFERRONI =====
print("\n=== CHECK 9: MULTIPLE TESTING CORRECTION ===", flush=True)
# We tested 5 horizons. Best was h=1 with p<0.0001
# Binomial test: n=22704, k=0.5213*22704 = 11843.5
n_test = 22704
acc = 0.5213
k = int(acc * n_test)
p_raw = 1 - binom.cdf(k - 1, n_test, 0.5)
print(f"Raw p-value for h=1: {p_raw:.2e}")
print(f"Bonferroni correction (K=5 horizons): {min(p_raw * 5, 1.0):.2e}")
print(f"Still significant at 0.05? {p_raw * 5 < 0.05}")
print(f"Still significant at 0.01? {p_raw * 5 < 0.01}")

# Broader correction: 77 strategies tested before this
K_total = 77
p_broad = min(p_raw * K_total, 1.0)
print(f"\nBroad Bonferroni (K=77 total strategies): {p_broad:.2e}")
print(f"Still significant at 0.05? {p_broad < 0.05}")

# Deflated Sharpe Ratio (AFML Ch 14)
print("\n=== DEFLATED SHARPE RATIO ===", flush=True)
# Convert accuracy to pseudo-Sharpe
# For a direction bet: E[r] = acc - (1-acc) = 2*acc - 1 per trade
# SR = E[r] / std(r). For binary outcome with prob p:
# E[r] = 2p-1, Var(r) = 4p(1-p), std(r) = 2*sqrt(p(1-p))
# SR = (2p-1) / (2*sqrt(p*(1-p)))
p_acc = acc
sr = (2 * p_acc - 1) / (2 * np.sqrt(p_acc * (1 - p_acc)))
print(f"Pseudo-Sharpe (per trade): {sr:.4f}")

# Annualized SR (96 bars/day * 365 days)
bars_per_year = 96 * 365
sr_annual = sr * np.sqrt(bars_per_year)
print(f"Annualized SR (96 bars/day * 365): {sr_annual:.4f}")

# DSR: Expected max SR from K independent trials with SR=0
# E[max(SR)] ~ sqrt(2 * ln(K)) for K trials
K = 77
sr_expected_max = np.sqrt(2 * np.log(K))
print(f"Expected max SR from {K} independent trials (SR=0): {sr_expected_max:.4f}")
print(f"Our annualized SR: {sr_annual:.4f}")
print(f"SR > E[max]? {sr_annual > sr_expected_max}")

# More precise DSR
# Under H0: SR ~ N(0, 1/sqrt(T)) where T is number of observations
T = n_test
sr_se = 1 / np.sqrt(T)
# Prob that max of K N(0, sr_se) > observed sr
# P(max > sr) = 1 - Phi(sr/sr_se)^K... simplified
z = sr / sr_se  # this is already the z-score
p_single = 1 - norm.cdf(z)
p_dsr = 1 - (1 - p_single) ** K
print(f"\nPer-trade SR: {sr:.6f}")
print(f"Z-score: {z:.4f}")
print(f"P(single trial > observed): {p_single:.2e}")
print(f"DSR p-value (K={K}): {p_dsr:.2e}")
print(f"DSR significant at 0.05? {p_dsr < 0.05}")

# ===== CHECK 10: EFFECTIVE SAMPLE SIZE =====
print("\n=== CHECK 10: EFFECTIVE SAMPLE SIZE ===", flush=True)
# Predictions have autocorrelation of 0.2952 at lag 1
# This reduces effective sample size
# N_eff = N * (1 - rho) / (1 + rho) for AR(1) with autocorrelation rho
rho = 0.2952
N = 22704
N_eff = N * (1 - rho) / (1 + rho)
print(f"Nominal N: {N}")
print(f"Prediction autocorrelation (lag-1): {rho:.4f}")
print(f"Effective N (AR(1) correction): {N_eff:.0f}")

# Re-test with effective N
k_eff = int(acc * N_eff)
p_eff = 1 - binom.cdf(k_eff - 1, int(N_eff), 0.5)
print(f"Accuracy with effective N: p-value = {p_eff:.2e}")
print(f"Still significant at 0.05? {p_eff < 0.05}")

# Required N for 52% to have power 0.80 at significance 0.05
# For detecting p=0.52 vs p=0.50:
# z_alpha = 1.645 (one-sided), z_beta = 0.842
# N = ((z_alpha + z_beta) / (p - 0.5))^2 * p * (1-p)
p_alt = 0.52
z_alpha = 1.645  # one-sided
z_beta = 0.842
N_required = ((z_alpha + z_beta) * np.sqrt(p_alt * (1 - p_alt)) / (p_alt - 0.5)) ** 2
print(f"\nRequired N for 52% detection (power=0.80, alpha=0.05): {N_required:.0f}")
print(f"We have: {N} nominal, {N_eff:.0f} effective")
print(f"Adequate? Nominal: {N > N_required}, Effective: {N_eff > N_required}")

# ===== CHECK 11: LABEL VERIFICATION =====
print("\n=== CHECK 11: LABEL CONSTRUCTION VERIFICATION ===", flush=True)
h = 1
fwd_ret = close.pct_change(h).shift(-h)
# Manual verification on a few points
for i in [100, 500, 1000, 5000, 10000]:
    if i + h < len(close):
        c_now = close.iloc[i]
        c_next = close.iloc[i + h]
        manual_ret = (c_next - c_now) / c_now
        computed = fwd_ret.iloc[i]
        match = abs(manual_ret - computed) < 1e-10 if not pd.isna(computed) else "NaN"
        print(f"  idx={i}: close={c_now:.2f}, close+1={c_next:.2f}, manual={manual_ret:.8f}, computed={computed:.8f}, match={match}")

# Check for zeros in labels (exactly flat bars)
y_all = np.sign(fwd_ret).reindex(features_clean.index).dropna()
zero_count = (y_all == 0).sum()
print(f"\nZero (flat) labels: {zero_count} ({zero_count/len(y_all)*100:.2f}%)")
print("Note: sign(0) = 0, but y = (sign > 0) maps 0 to class 0 (down).")
print("This is a slight bias against up predictions -- flat bars are treated as down.")

# ===== CHECK 12: TRAIN/TEST CONTAMINATION =====
print("\n=== CHECK 12: TRAIN/TEST BOUNDARY ===", flush=True)
train_end = features_clean[features_clean.index < "2025-07-01"].index[-1]
test_start = features_clean[features_clean.index >= "2025-07-01"].index[0]
print(f"Last train bar: {train_end}")
print(f"First test bar: {test_start}")
gap = (test_start - train_end).total_seconds() / 60
print(f"Gap: {gap:.0f} minutes ({gap/60:.1f} hours)")
print(f"No label overlap possible for 1-bar horizon with chronological split.")

# Check if any feature at test_start depends on future data
print(f"\nFeatures at first test bar use rolling windows that extend into training period.")
print(f"This is ACCEPTABLE for features (standard practice). Labels are properly separated.")

print("\nAll checks complete.", flush=True)
