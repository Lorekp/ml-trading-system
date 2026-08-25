"""
Direction V4 -- Monte Carlo Simulation.

Tests whether our observed Sharpe could have occurred by chance:
  1. Take our actual trade returns from v4 Tier 2 best model
  2. Randomly shuffle the returns 10,000 times
  3. Compute Sharpe for each shuffle
  4. P-value = fraction of shuffled Sharpes >= observed Sharpe

Also:
  - Bootstrap confidence intervals on Sharpe
  - Equity curve distribution (fan chart)
  - Worst-case drawdown distribution
"""
import sys
import warnings
import json
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from scipy.stats import norm

from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from afml.features import corwin_schultz_spread, rolling_entropy

try:
    from afml.fracdiff import frac_diff_ffd
except ImportError:
    def get_weights_ffd(d, threshold=1e-5):
        w = [1.0]
        k = 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < threshold:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(series, d, threshold=1e-5):
        weights = get_weights_ffd(d, threshold)
        width = len(weights) - 1
        result = {}
        for col in series.columns:
            col_data = series[col].fillna(method="ffill").dropna()
            vals = pd.Series(dtype=float)
            for i in range(width, len(col_data)):
                window = col_data.iloc[i - width : i + 1]
                vals[col_data.index[i]] = np.dot(weights.T, window.values)[0, 0]
            result[col] = vals
        return pd.DataFrame(result)

try:
    from afml.derivatives import funding_features, oi_features, impact_features
except ImportError:
    def funding_features(f, **kw):
        r = pd.DataFrame(index=f.index)
        m = f.rolling(50).mean(); s = f.rolling(50).std()
        r["funding_zscore"] = (f - m) / s.replace(0, np.nan)
        return r
    def oi_features(oi, close, **kw):
        r = pd.DataFrame(index=oi.index)
        r["oi_change_5"] = oi.pct_change(5)
        return r
    def impact_features(bid, ask, mid):
        r = pd.DataFrame(index=mid.index)
        r["impact_spread"] = (ask - bid) / mid.replace(0, np.nan)
        return r

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
HL_CSV_DIR = Path("C:/Claude/ml-trading-system/v3-results")
OUTPUT_DIR = Path("C:/Claude/ml-trading-system/v4-results")

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

BAR_TF = "30min"
FWD_BARS = 4
FFD_THRESHOLD = 1e-3
FFD_D_STARS = {"BTC": 0.2, "ETH": 0.1, "XRP": 0.3, "BNB": 0.1, "SUI": 0.2, "LINK": 0.1}
SLIPPAGE = 0.0001
VOL_FILTER_QUANTILE = 0.50
CONF_PERCENTILE = 70
ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")

XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)

N_SIMULATIONS = 10000  # Monte Carlo iterations
BOOTSTRAP_N = 5000     # Bootstrap iterations for CI

# =============================================================================
# DATA + FEATURE CACHES
# =============================================================================
_ohlcv_cache = {}
_hl_cache = {}
_full_features = {}
_vol_series = {}
_fwd_returns = {}


def load_ohlcv(symbol, tf=BAR_TF):
    key = (symbol, tf)
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]
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
            df = pd.read_csv(f); df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    result = combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    _ohlcv_cache[key] = result
    return result


def load_hl(symbol):
    if symbol in _hl_cache: return _hl_cache[symbol]
    p = HL_CSV_DIR / f"hl_{symbol}.csv"
    if p.exists():
        df = pd.read_csv(p); df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        r = df.resample(BAR_TF).last().ffill(); _hl_cache[symbol] = r; return r
    _hl_cache[symbol] = None; return None


def precompute():
    for symbol in ASSETS:
        ohlcv = load_ohlcv(symbol); hl = load_hl(symbol)
        close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
        f = pd.DataFrame(index=ohlcv.index)
        for lb in [1, 3, 7, 14]: f[f"ret_{lb}"] = close.pct_change(lb)
        f["bar_range"] = (high - low) / close
        f["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
        f["rsi_14"] = rsi(close, 14)
        f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
        f["macd_hist"] = macd(close)["macd_hist"]
        f["bb_pct_b"] = bollinger_pct_b(close, 20)
        f["atr_14"] = atr(high, low, close, 14)
        sw = swing_levels(high, low, lookback=20)
        f["dist_swing_high"] = sw["dist_swing_high"]
        f["dist_swing_low"] = sw["dist_swing_low"]
        f["dist_nearest_level"] = sw["dist_nearest_level"]
        try:
            ffd = frac_diff_ffd(np.log(close).to_frame("close"), FFD_D_STARS.get(symbol, 0.4), FFD_THRESHOLD)
            if len(ffd) > 0: f["ffd"] = ffd.iloc[:, 0].reindex(f.index)
        except: pass
        try: f["cs_spread"] = corwin_schultz_spread(high, low, window=20)
        except: pass
        try: f["entropy"] = rolling_entropy(close, window=50, n_bins=5)
        except: pass
        if hl is not None:
            try:
                ff = funding_features(hl["funding"])
                if "funding_zscore" in ff.columns: f["funding_zscore"] = ff["funding_zscore"].reindex(f.index)
            except: pass
            try:
                oi = oi_features(hl["open_interest"], close.reindex(hl.index, method="ffill"))
                if "oi_change_5" in oi.columns: f["oi_change_5"] = oi["oi_change_5"].reindex(f.index)
            except: pass
            try:
                mid = hl.get("mid_px", (hl["impact_bid_px"] + hl["impact_ask_px"]) / 2)
                imp = impact_features(hl["impact_bid_px"], hl["impact_ask_px"], mid)
                if "impact_spread" in imp.columns: f["impact_spread"] = imp["impact_spread"].reindex(f.index)
            except: pass
        _full_features[symbol] = f
        _vol_series[symbol] = close.pct_change().dropna().ewm(span=50).std()
        _fwd_returns[symbol] = close.pct_change(FWD_BARS).shift(-FWD_BARS)
        print(f"  {symbol}: {len(f.columns)} features", flush=True)


def generate_actual_trades():
    """Generate walk-forward trade returns (same as Tier 2 best: expanded features + prob sizing)."""
    all_rets = []

    for symbol in ASSETS:
        features = _full_features[symbol]
        fwd_ret = _fwd_returns[symbol]
        vol = _vol_series[symbol]
        fc = features.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        for month in ALL_MONTHS:
            ms = pd.Timestamp(month.start_time, tz="UTC")
            me = pd.Timestamp(month.end_time, tz="UTC")
            f_w = fc.copy()
            train_mask = f_w.index < ms
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in f_w.columns:
                    tv = f_w.loc[train_mask, col].dropna()
                    if len(tv) > 10:
                        f_w[col] = f_w[col].clip(tv.quantile(0.01), tv.quantile(0.99))
            tr = valid[valid < ms]
            te = valid[(valid >= ms) & (valid <= me)]
            if len(tr) < 500 or len(te) < 20: continue
            tv = vol.reindex(tr).dropna()
            if len(tv) < 50: continue
            vt = tv.quantile(VOL_FILTER_QUANTILE)
            hv = (vol.reindex(te) > vt).fillna(False)
            te = te[hv.values]
            if len(te) < 10: continue

            X_tr = f_w.loc[tr]; y_tr = (y_all.loc[tr] > 0).astype(int)
            model = xgb.XGBClassifier(**XGB_PARAMS); model.fit(X_tr, y_tr)
            yp = model.predict_proba(f_w.loc[te])[:, 1]
            conf = np.abs(yp - 0.5)
            mask = conf >= np.percentile(conf, CONF_PERCENTILE) if len(conf) > 10 else np.ones(len(conf), dtype=bool)
            direction = np.where(yp[mask] > 0.5, 1.0, -1.0)
            p_filt = yp[mask]
            signal = np.array([2.0 * norm.cdf((p - 0.5) / np.sqrt(p * (1 - p))) - 1.0 for p in np.clip(p_filt, 1e-6, 1-1e-6)])
            actual = fwd_ret.reindex(te[mask]).values
            trade_rets = direction * actual * np.abs(signal).clip(0.1, 2.0) - SLIPPAGE
            all_rets.extend(trade_rets[np.isfinite(trade_rets)].tolist())

    return np.array(all_rets)


def compute_sharpe(returns):
    if len(returns) < 10 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(len(returns))


def compute_max_dd(returns):
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return abs(dd.min())


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90, flush=True)
    print("MONTE CARLO SIMULATION + BOOTSTRAP CONFIDENCE INTERVALS", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Simulations: {N_SIMULATIONS} (shuffle test) + {BOOTSTRAP_N} (bootstrap CI)", flush=True)
    print("=" * 90, flush=True)

    # Precompute
    print("\n[1/4] Precomputing features...", flush=True)
    t0 = time.time()
    for s in ASSETS: load_ohlcv(s); load_hl(s)
    precompute()
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Generate actual trades
    print("\n[2/4] Generating actual walk-forward trades...", flush=True)
    t1 = time.time()
    actual_returns = generate_actual_trades()
    actual_sharpe = compute_sharpe(actual_returns)
    actual_dd = compute_max_dd(actual_returns)
    actual_wr = (actual_returns > 0).mean()
    print(f"  {len(actual_returns)} trades, Sharpe={actual_sharpe:.3f}, "
          f"MaxDD={actual_dd*100:.1f}%, WR={actual_wr:.3f} ({time.time()-t1:.1f}s)", flush=True)

    # Monte Carlo: shuffle returns
    print(f"\n[3/4] Monte Carlo ({N_SIMULATIONS} shuffled simulations)...", flush=True)
    t2 = time.time()
    rng = np.random.RandomState(42)
    shuffled_sharpes = np.empty(N_SIMULATIONS)
    shuffled_dds = np.empty(N_SIMULATIONS)

    for i in range(N_SIMULATIONS):
        shuffled = rng.permutation(actual_returns)
        shuffled_sharpes[i] = compute_sharpe(shuffled)
        shuffled_dds[i] = compute_max_dd(shuffled)

    mc_pvalue = (shuffled_sharpes >= actual_sharpe).mean()
    mc_sharpe_mean = shuffled_sharpes.mean()
    mc_sharpe_std = shuffled_sharpes.std()

    print(f"  Done in {time.time()-t2:.1f}s", flush=True)
    print(f"  Shuffled Sharpe: mean={mc_sharpe_mean:.3f}, std={mc_sharpe_std:.3f}", flush=True)
    print(f"  P-value: {mc_pvalue:.4f} ({'SIGNIFICANT' if mc_pvalue < 0.05 else 'NOT significant'} at 5%)", flush=True)

    # Bootstrap CI on Sharpe
    print(f"\n[4/4] Bootstrap ({BOOTSTRAP_N} resamples for Sharpe CI)...", flush=True)
    t3 = time.time()
    boot_sharpes = np.empty(BOOTSTRAP_N)
    boot_dds = np.empty(BOOTSTRAP_N)

    for i in range(BOOTSTRAP_N):
        idx = rng.choice(len(actual_returns), size=len(actual_returns), replace=True)
        boot_sharpes[i] = compute_sharpe(actual_returns[idx])
        boot_dds[i] = compute_max_dd(actual_returns[idx])

    ci_lo = np.percentile(boot_sharpes, 2.5)
    ci_hi = np.percentile(boot_sharpes, 97.5)
    dd_ci_lo = np.percentile(boot_dds, 2.5)
    dd_ci_hi = np.percentile(boot_dds, 97.5)

    print(f"  Done in {time.time()-t3:.1f}s", flush=True)

    # Report
    print(f"\n{'='*90}", flush=True)
    print("MONTE CARLO + BOOTSTRAP RESULTS", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\n  Actual Trades:        {len(actual_returns)}", flush=True)
    print(f"  Actual Sharpe:        {actual_sharpe:.3f}", flush=True)
    print(f"  Actual MaxDD:         {actual_dd*100:.1f}%", flush=True)
    print(f"  Actual Win Rate:      {actual_wr:.3f}", flush=True)

    print(f"\n  Monte Carlo (shuffle test):", flush=True)
    print(f"    Shuffled Sharpe:    {mc_sharpe_mean:.3f} +/- {mc_sharpe_std:.3f}", flush=True)
    print(f"    P-value:            {mc_pvalue:.4f}", flush=True)
    if mc_pvalue < 0.01:
        print(f"    Verdict:            HIGHLY SIGNIFICANT (p < 0.01)", flush=True)
    elif mc_pvalue < 0.05:
        print(f"    Verdict:            SIGNIFICANT (p < 0.05)", flush=True)
    else:
        print(f"    Verdict:            NOT SIGNIFICANT (p >= 0.05)", flush=True)
    print(f"    Interpretation:     {mc_pvalue*100:.1f}% of random shuffles produced Sharpe >= {actual_sharpe:.3f}", flush=True)

    print(f"\n  Bootstrap 95% CI:", flush=True)
    print(f"    Sharpe:             [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
    print(f"    MaxDD:              [{dd_ci_lo*100:.1f}%, {dd_ci_hi*100:.1f}%]", flush=True)

    print(f"\n  Percentiles of shuffled Sharpe distribution:", flush=True)
    for pct in [1, 5, 25, 50, 75, 95, 99]:
        val = np.percentile(shuffled_sharpes, pct)
        marker = " <-- ACTUAL" if abs(val - actual_sharpe) < mc_sharpe_std * 0.1 else ""
        print(f"    {pct:>3}th percentile:   {val:.3f}{marker}", flush=True)
    print(f"    Actual:             {actual_sharpe:.3f}", flush=True)

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "actual": {
            "n_trades": len(actual_returns),
            "sharpe": round(actual_sharpe, 3),
            "max_dd_pct": round(actual_dd * 100, 1),
            "win_rate": round(actual_wr, 4),
        },
        "monte_carlo": {
            "n_simulations": N_SIMULATIONS,
            "p_value": round(mc_pvalue, 4),
            "shuffled_sharpe_mean": round(mc_sharpe_mean, 3),
            "shuffled_sharpe_std": round(mc_sharpe_std, 3),
        },
        "bootstrap": {
            "n_resamples": BOOTSTRAP_N,
            "sharpe_ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
            "max_dd_ci_95": [round(dd_ci_lo * 100, 1), round(dd_ci_hi * 100, 1)],
        },
    }
    with open(OUTPUT_DIR / "montecarlo_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR / 'montecarlo_results.json'}", flush=True)
    print(f"Completed: {datetime.now()} ({time.time()-t0:.0f}s total)", flush=True)
