"""Generate ML training data for cointegration pair manager.

For each pair × month: compute 15 features + profit label.
Uses top-30 universe by volume per month, all 435 pairs screened.
Output: results/coint_ml/training_data.parquet (~16K-33K rows)

Parallelized per-month for speed.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")
OUT_DIR = Path("results/coint_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBARGO = 96 * 7; MIN_AGE_DAYS = 180; MIN_CORR = 0.40
ENTRY_Z = 2.0; MIN_HOLD = 96; MAX_HOLD = 96 * 30

SECTORS = {
    "L1": ["BTC","ETH","SOL","ADA","AVAX","NEAR","DOT","APT","SUI"],
    "DeFi": ["UNI","AAVE","LINK"], "AI": ["FET","TAO"],
    "Storage": ["FIL","ANKR"], "Gaming": ["ENJ"],
    "Exchange": ["BNB","FTM","TRX"],
    "Payment": ["XRP","XLM","LTC","BCH"],
}
ASSET_SECTOR = {}
for s, assets in SECTORS.items():
    for a in assets: ASSET_SECTOR[a] = s


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


def half_life_bars(resid):
    if len(resid) < 50: return float("inf")
    lag = resid[:-1]; delta = np.diff(resid)
    d = np.dot(lag - lag.mean(), lag - lag.mean())
    if d <= 0: return float("inf")
    phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
    return -math.log(2) / phi if phi < 0 else float("inf")


def hurst_exponent(series, max_lag=64):
    if len(series) < max_lag * 3: return 0.5
    lags = np.arange(2, min(max_lag + 1, len(series) // 3))
    tau = [np.std(series[l:] - series[:-l]) for l in lags]
    tau_arr = np.array(tau); keep = np.isfinite(tau_arr) & (tau_arr > 0)
    if keep.sum() < 5: return 0.5
    return float(np.clip(np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)[0], 0, 1))


def simulate_pair_month(y_close, x_close, beta, intercept, mu, sigma, month_start, month_end):
    """Simulate z-score spread trading for one pair for one month. Returns PnL."""
    mask = (y_close.index >= month_start) & (y_close.index <= month_end)
    y = y_close[mask].values.astype(float); x = x_close[mask].values.astype(float)
    if len(y) < 96 * 10: return 0.0

    log_y = np.log(y); log_x = np.log(x)
    y_ret = np.zeros(len(y)); x_ret = np.zeros(len(x))
    y_ret[1:] = y[1:] / y[:-1] - 1; x_ret[1:] = x[1:] / x[:-1] - 1

    pos = 0; entry_i = 0; pnl = 0.0
    for i in range(1, len(y)):
        spread = log_y[i] - beta * log_x[i] - intercept
        z = (spread - mu) / sigma if sigma > 0 else 0

        if pos != 0:
            pnl += pos * (y_ret[i] - beta * x_ret[i])
            bh = i - entry_i
            should_exit = False
            if bh >= MIN_HOLD:
                if pos == 1 and z >= 0: should_exit = True
                elif pos == -1 and z <= 0: should_exit = True
            if bh >= MAX_HOLD: should_exit = True
            if should_exit: pos = 0

        if pos == 0:
            if z <= -ENTRY_Z: pos = 1; entry_i = i
            elif z >= ENTRY_Z: pos = -1; entry_i = i

    return pnl


def process_month(args):
    """Worker: process one month, compute features for all pairs."""
    month_str, close_dict, volume_dict, first_dates, prev_features = args
    from statsmodels.tsa.stattools import coint as _coint

    month_start = pd.Timestamp(f"{month_str}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthEnd(0)
    next_month_start = month_end + pd.Timedelta(days=1)
    next_month_end = next_month_start + pd.offsets.MonthEnd(0)

    # Select universe (top 30 by volume)
    lookback_end = month_start - pd.Timedelta(days=1)
    candidates = {}
    for name, vol_s in volume_dict.items():
        avail = vol_s.loc[:lookback_end]
        if len(avail) < 30: continue
        age = (month_start - first_dates.get(name, month_start)).days
        if age < MIN_AGE_DAYS: continue
        latest = avail.iloc[-1]
        if np.isnan(latest) or latest <= 0: continue
        candidates[name] = float(latest)
    universe = [a for a, _ in sorted(candidates.items(), key=lambda x: -x[1])[:30]]

    if len(universe) < 6: return []

    # Build close panel for screening
    screen_end = month_start - pd.Timedelta(minutes=15 * EMBARGO)
    screen_start = screen_end - pd.DateOffset(months=6)

    rows = []
    for a_i, a_j in combinations(universe, 2):
        if a_i not in close_dict or a_j not in close_dict: continue
        y_s = close_dict[a_i]; x_s = close_dict[a_j]
        common = y_s.index.intersection(x_s.index)
        y_full = y_s.reindex(common).astype(float)
        x_full = x_s.reindex(common).astype(float)

        # Training window
        tmask = (y_full.index >= screen_start) & (y_full.index < screen_end)
        y_train = y_full[tmask].values; x_train = x_full[tmask].values
        if len(y_train) < 3000: continue

        # Correlation prescreen
        ret_y = np.diff(y_train) / y_train[:-1]
        ret_x = np.diff(x_train) / x_train[:-1]
        valid = np.isfinite(ret_y) & np.isfinite(ret_x)
        if valid.sum() < 2000: continue
        corr = float(np.corrcoef(ret_y[valid], ret_x[valid])[0, 1])
        if not np.isfinite(corr) or corr < MIN_CORR: continue

        # EG test
        try:
            _, p_val, _ = _coint(y_train, x_train, maxlag=10)
        except: p_val = 1.0

        # OLS
        log_y = np.log(y_train); log_x = np.log(x_train)
        beta, intercept = ols_fit(log_y, log_x)
        if not np.isfinite(beta) or beta <= 0: continue
        resid = log_y - beta * log_x - intercept
        mu = float(np.mean(resid)); sigma = float(np.std(resid))
        if sigma <= 1e-12: continue

        # Features
        hl = half_life_bars(resid) / 96  # days
        hurst = hurst_exponent(resid)

        # Beta stability
        betas = []
        for cy, cx in zip(np.array_split(log_y, 4), np.array_split(log_x, 4)):
            b, _ = ols_fit(cy, cx)
            if np.isfinite(b) and b > 0: betas.append(b)
        beta_stab = max(betas) / min(betas) if len(betas) == 4 and min(betas) > 0 else 99

        # Spread vol and skewness
        spread_vol = float(np.std(resid))
        spread_skew = float(pd.Series(np.diff(resid)).skew()) if len(resid) > 50 else 0

        # Mean reversion speed (AR1 phi)
        if len(resid) >= 50:
            lag = resid[:-1]; delta = np.diff(resid)
            d = np.dot(lag - lag.mean(), lag - lag.mean())
            mr_speed = float(np.dot(lag - lag.mean(), delta - delta.mean()) / d) if d > 0 else 0
        else:
            mr_speed = 0

        # Volume ratio
        y_vol = candidates.get(a_i, 1); x_vol = candidates.get(a_j, 1)
        vol_ratio = min(y_vol, x_vol) / max(y_vol, x_vol) if max(y_vol, x_vol) > 0 else 0

        # Sector match
        sector_match = 1 if ASSET_SECTOR.get(a_i) == ASSET_SECTOR.get(a_j) else 0

        # Temporal features from previous months
        pair_key = f"{a_i}/{a_j}"
        prev = prev_features.get(pair_key, {})
        eg_3mo = [prev.get(f"p_{k}", 1.0) for k in range(1, 4)]
        hl_3mo = [prev.get(f"hl_{k}", 99) for k in range(1, 4)]

        # P-value trend (slope)
        if len(eg_3mo) >= 2 and all(np.isfinite(eg_3mo)):
            p_trend = float(np.polyfit(range(len(eg_3mo)), eg_3mo, 1)[0])
        else:
            p_trend = 0

        # HL trend
        if len(hl_3mo) >= 2 and all(v < 99 for v in hl_3mo):
            hl_trend = float(np.polyfit(range(len(hl_3mo)), hl_3mo, 1)[0])
        else:
            hl_trend = 0

        # Consecutive passes/fails
        consec_passes = prev.get("consec_passes", 0)
        consec_fails = prev.get("consec_fails", 0)
        if p_val < 0.05:
            consec_passes += 1; consec_fails = 0
        else:
            consec_fails += 1; consec_passes = 0

        months_since_pass = prev.get("months_since_pass", 99)
        if p_val < 0.05: months_since_pass = 0
        else: months_since_pass += 1

        # LABEL: is pair profitable next month?
        label_pnl = simulate_pair_month(y_full, x_full, beta, intercept, mu, sigma,
                                         next_month_start, next_month_end)
        label = 1 if label_pnl > 0 else 0

        rows.append({
            "pair": pair_key, "y": a_i, "x": a_j, "month": month_str,
            "eg_pvalue": p_val, "half_life_days": min(hl, 100),
            "hurst": hurst, "beta_stability": min(beta_stab, 10),
            "return_corr": corr, "spread_vol": spread_vol,
            "eg_pvalue_3mo_trend": p_trend, "half_life_trend": hl_trend,
            "consecutive_passes": consec_passes, "consecutive_fails": consec_fails,
            "months_since_last_pass": min(months_since_pass, 24),
            "spread_mr_speed": mr_speed, "volume_ratio": vol_ratio,
            "sector_match": sector_match, "spread_skewness": spread_skew,
            "label": label, "label_pnl": label_pnl,
            "beta": beta,
        })

    return rows


def main():
    t0 = time.time()
    print("=" * 100)
    print("ML TRAINING DATA GENERATION — Cointegration Pair Features + Labels")
    print("=" * 100)

    # Load all data
    print("\nLoading from parquet cache...", end=" ", flush=True)
    close_dict = {}; volume_dict = {}; first_dates = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000:
                close_dict[name] = df["close"]
                dv = (df["close"] * df["volume"]).resample("1D").sum()
                volume_dict[name] = dv.rolling(30, min_periods=20).median()
                first_dates[name] = df.index.min()
        except: pass
    print(f"{len(close_dict)} assets in {time.time()-t0:.1f}s")

    # Generate data month by month (sequential — needs prev_features state)
    months = pd.period_range("2023-06", "2025-12", freq="M")
    all_rows = []
    prev_features = {}  # pair_key -> {p_1, p_2, p_3, hl_1, ..., consec_passes, ...}

    for month in months:
        mt0 = time.time()
        month_str = str(month)
        rows = process_month((month_str, close_dict, volume_dict, first_dates, prev_features))

        # Update prev_features for next month
        for r in rows:
            pk = r["pair"]
            old = prev_features.get(pk, {})
            prev_features[pk] = {
                "p_1": r["eg_pvalue"], "p_2": old.get("p_1", 1.0), "p_3": old.get("p_2", 1.0),
                "hl_1": r["half_life_days"], "hl_2": old.get("hl_1", 99), "hl_3": old.get("hl_2", 99),
                "consec_passes": r["consecutive_passes"],
                "consec_fails": r["consecutive_fails"],
                "months_since_pass": r["months_since_last_pass"],
            }

        all_rows.extend(rows)
        elapsed = time.time() - mt0
        n_pos = sum(1 for r in rows if r["label"] == 1)
        print(f"  {month_str}: {len(rows)} pairs, {n_pos} profitable ({n_pos/max(len(rows),1)*100:.0f}%), "
              f"{elapsed:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_DIR / "training_data.parquet", index=False)

    print(f"\n{'='*100}")
    print(f"DATASET SUMMARY")
    print(f"{'='*100}")
    print(f"Total observations: {len(df):,}")
    print(f"Unique pairs: {df['pair'].nunique()}")
    print(f"Months: {df['month'].nunique()}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    print(f"Positive rate: {df['label'].mean()*100:.1f}%")
    print(f"\nFeature statistics:")
    features = ["eg_pvalue", "half_life_days", "hurst", "beta_stability", "return_corr",
                "spread_vol", "consecutive_passes", "consecutive_fails", "spread_mr_speed",
                "volume_ratio", "sector_match"]
    for f in features:
        if f in df.columns:
            print(f"  {f:<30s} mean={df[f].mean():>8.3f}  std={df[f].std():>8.3f}  "
                  f"min={df[f].min():>8.3f}  max={df[f].max():>8.3f}")

    print(f"\nSaved to: {OUT_DIR / 'training_data.parquet'}")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
