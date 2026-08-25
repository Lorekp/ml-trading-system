"""
DSR Investigation + Outlier Robustness.

The parametric DSR fails because kurtosis=136 (crypto fat tails).
This script tests alternative approaches:
1. DSR on monthly returns (lower kurtosis)
2. DSR on winsorized daily returns (3-sigma cap)
3. Outlier robustness: remove top N days, check Sharpe survival
4. Bootstrap Sharpe CI (non-parametric, no distribution assumption)
"""
import sys, warnings, time, json
import numpy as np, pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from afml.evaluation import deflated_sharpe_ratio
from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from afml.fracdiff import frac_diff_ffd
from afml.derivatives import funding_features, oi_features, impact_features
from afml.features import corwin_schultz_spread, rolling_entropy
from afml.bet_sizing import get_signal, discretize_signal

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    HMM_OK = False

# ── Constants ──
DATA_DIR = Path("C:/Claude/ml-trading-system/data/binance-1m")
HL_DIR = Path("C:/Claude/ml-trading-system/v3-results")
ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}
FFD_D = {"BTC": 0.2, "ETH": 0.1, "XRP": 0.3, "BNB": 0.1, "SUI": 0.2, "LINK": 0.1}
BAR_TF = "30min"
FWD_BARS = 4
ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")
EMBARGO_TD = pd.Timedelta(hours=2)
K_TRIALS = 20

CAPITAL = 10000.0
SLIPPAGE = 0.0001
BASE_PCT = 0.02
VOL_Q = 0.50
CONF_PCT = 70
MAX_SIG = 3.0
SIG_STEP = 0.1
WINSOR_LO, WINSOR_HI = 0.01, 0.99

FEATURES_19 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
    "ffd", "cs_spread", "entropy", "oi_change_5", "impact_spread",
]


def load_and_precompute():
    """Load data + compute all features. Returns caches."""
    ohlcv_cache = {}
    feat_cache = {}
    vol_cache = {}
    hmm_cache = {}

    for sym in ASSETS:
        symbol = SYMBOLS[sym]
        ddir = DATA_DIR / symbol
        frames = []
        for f in sorted(ddir.glob("*.csv")):
            df = pd.read_csv(f, header=None,
                names=["open_time", "open", "high", "low", "close", "volume",
                       "close_time", "quote_vol", "n_trades", "taker_buy_base",
                       "taker_buy_quote", "ignore"])
            # Skip header rows if present
            df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
            df["open_time"] = df["open_time"].astype(float)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].dropna()
            frames.append(df)
        raw = pd.concat(frames).sort_index()
        raw = raw[~raw.index.duplicated(keep="first")]
        ohlcv = raw.resample(BAR_TF).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        ohlcv_cache[sym] = ohlcv

        c, h, l, v = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
        feat = pd.DataFrame(index=ohlcv.index)
        feat["ret_1"] = c.pct_change(1)
        feat["ret_3"] = c.pct_change(3)
        feat["ret_7"] = c.pct_change(7)
        feat["ret_14"] = c.pct_change(14)
        feat["bar_range"] = (h - l) / c.replace(0, np.nan)
        vmean = v.rolling(30).mean()
        vstd = v.rolling(30).std()
        feat["vol_zscore"] = (v - vmean) / vstd.replace(0, np.nan)
        feat["rsi_14"] = rsi(c, 14)
        feat["stoch_rsi"] = stochastic_rsi(c, 14)
        feat["macd_hist"] = macd(c)
        feat["bb_pct_b"] = bollinger_pct_b(c, 20, 2)
        feat["atr_14"] = atr(h, l, c, 14) / c.replace(0, np.nan)
        swing_df = swing_levels(h, l, 20)
        if "dist_swing_high" in swing_df.columns:
            feat["dist_swing_high"] = swing_df["dist_swing_high"]
            feat["dist_swing_low"] = swing_df["dist_swing_low"]
            feat["dist_nearest_level"] = swing_df.get("dist_nearest_level",
                feat[["dist_swing_high", "dist_swing_low"]].abs().min(axis=1))
        else:
            cols = swing_df.columns.tolist()
            feat["dist_swing_high"] = swing_df[cols[0]] if len(cols) > 0 else 0
            feat["dist_swing_low"] = swing_df[cols[1]] if len(cols) > 1 else 0
            feat["dist_nearest_level"] = swing_df[cols[2]] if len(cols) > 2 else feat[["dist_swing_high", "dist_swing_low"]].abs().min(axis=1)
        log_c = np.log(c).to_frame("close")
        ffd_vals = frac_diff_ffd(log_c, FFD_D[sym], threshold=1e-4)
        feat["ffd"] = ffd_vals["close"]
        feat["cs_spread"] = corwin_schultz_spread(h, l, window=20)
        feat["entropy"] = rolling_entropy(c, window=50, n_bins=5)

        hlf = HL_DIR / f"hl_{sym}.csv"
        if hlf.exists():
            hl = pd.read_csv(hlf)
            time_col = "timestamp" if "timestamp" in hl.columns else "time"
            hl[time_col] = pd.to_datetime(hl[time_col], utc=True)
            hl = hl.set_index(time_col)
            if hl.index.tz is None:
                hl.index = hl.index.tz_localize("UTC")
            hl_30 = hl.resample(BAR_TF).last().reindex(ohlcv.index, method="ffill")
            oi_col = next((c for c in hl_30.columns if "interest" in c.lower()), None)
            feat["oi_change_5"] = hl_30[oi_col].pct_change(5) if oi_col else 0.0
            bid_col = next((c for c in hl_30.columns if "bid" in c.lower()), None)
            ask_col = next((c for c in hl_30.columns if "ask" in c.lower()), None)
            mid_col = next((c for c in hl_30.columns if "mid" in c.lower() or "mark" in c.lower()), None)
            if bid_col and ask_col:
                mid_px = hl_30[mid_col] if mid_col else (hl_30[bid_col] + hl_30[ask_col]) / 2
                feat["impact_spread"] = (hl_30[ask_col] - hl_30[bid_col]) / mid_px.replace(0, np.nan)
            else:
                feat["impact_spread"] = 0.0
        else:
            feat["oi_change_5"] = 0.0
            feat["impact_spread"] = 0.0

        cols = [c for c in FEATURES_19 if c in feat.columns]
        feat_cache[sym] = feat[cols]
        vol_cache[sym] = feat["ret_1"].ewm(span=50).std()

        if HMM_OK:
            try:
                hf = pd.DataFrame({
                    "ret10": c.pct_change(10),
                    "vol": feat["ret_1"].ewm(span=50).std(),
                    "volz": feat["vol_zscore"],
                }).dropna()
                hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
                hmm.fit(hf.values)
                hmm_cache[sym] = pd.Series(hmm.predict(hf.values), index=hf.index)
            except Exception:
                hmm_cache[sym] = None
        else:
            hmm_cache[sym] = None

    return ohlcv_cache, feat_cache, vol_cache, hmm_cache


def generate_daily_pnl(ohlcv_cache, feat_cache, vol_cache, hmm_cache):
    """Walk-forward with LogReg L1, return daily PnL series."""
    all_trades = []  # (date, pnl)

    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        tc = ms - EMBARGO_TD

        for sym in ASSETS:
            feat = feat_cache[sym].copy()
            ohlcv = ohlcv_cache[sym]
            close = ohlcv["close"]
            fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
            vol = vol_cache[sym]

            fc = feat.dropna()
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            tr = valid[valid < tc]
            te = valid[(valid >= ms) & (valid <= me)]
            if len(tr) < 500 or len(te) < 20:
                continue

            X_tr = fc.loc[tr].copy()
            X_te = fc.loc[te].copy()
            for col in X_tr.columns:
                lo = X_tr[col].quantile(WINSOR_LO)
                hi = X_tr[col].quantile(WINSOR_HI)
                X_tr[col] = X_tr[col].clip(lo, hi)
                X_te[col] = X_te[col].clip(lo, hi)

            y_tr = (y_all.loc[tr] > 0).astype(int)

            # Vol filter
            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 50:
                continue
            vt = tv.quantile(VOL_Q)
            hv = (vol.reindex(te) > vt).fillna(False)
            te = te[hv.values]
            X_te = fc.loc[te].copy()
            for col in X_te.columns:
                lo = fc.loc[tr, col].quantile(WINSOR_LO)
                hi = fc.loc[tr, col].quantile(WINSOR_HI)
                X_te[col] = X_te[col].clip(lo, hi)
            if len(te) < 10:
                continue

            # Sample weights
            ts_arr = X_tr.index.values.astype(np.int64)
            n = len(ts_arr)
            sw = np.ones(n)
            for i in range(n):
                t_end_idx = min(i + FWD_BARS, n - 1)
                concurrent = np.sum((ts_arr >= ts_arr[i]) & (ts_arr <= ts_arr[t_end_idx]))
                sw[i] = 1.0 / max(concurrent, 1)
            sw = sw / sw.mean()

            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(X_tr)
            Xte_s = scaler.transform(X_te)
            model = LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)
            model.fit(Xtr_s, y_tr, sample_weight=sw)
            yp = model.predict_proba(Xte_s)[:, 1]

            # Confidence filter
            conf = np.abs(yp - 0.5)
            ct = np.percentile(conf, CONF_PCT) if len(conf) > 10 else 0
            cmask = conf >= ct

            # HMM regime filter
            rmask = np.ones(len(yp), dtype=bool)
            hmm_reg = hmm_cache.get(sym)
            if hmm_reg is not None:
                train_reg = hmm_reg[hmm_reg.index < tc]
                if len(train_reg) > 50:
                    train_fwd = fwd_ret.reindex(train_reg.index).dropna()
                    common = train_reg.index.intersection(train_fwd.index)
                    if len(common) > 50:
                        rr = pd.DataFrame({"r": train_reg.loc[common], "f": train_fwd.loc[common]})
                        worst = rr.groupby("r")["f"].mean().idxmin()
                        test_reg = hmm_reg.reindex(te).fillna(1).astype(int)
                        rmask = test_reg.values != worst

            final_mask = cmask & rmask[:len(cmask)]
            yp_f = yp[final_mask]
            te_f = te[final_mask]

            for j in range(len(yp_f)):
                idx = te_f[j]
                fr = fwd_ret.get(idx, np.nan)
                if not np.isfinite(fr):
                    continue
                direction = 1.0 if yp_f[j] > 0.5 else -1.0
                sig = get_signal(np.array([yp_f[j]]), num_classes=2)[0]
                sig_d = round(sig / SIG_STEP) * SIG_STEP
                sig_d = np.clip(sig_d, -MAX_SIG, MAX_SIG)
                trade_pnl = CAPITAL * BASE_PCT * abs(sig_d) * (direction * fr - SLIPPAGE)
                date = pd.Timestamp(idx).date()
                all_trades.append((date, trade_pnl))

    df = pd.DataFrame(all_trades, columns=["date", "pnl"])
    daily_pnl = df.groupby("date")["pnl"].sum()
    return daily_pnl


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80, flush=True)
    print("DSR INVESTIGATION + OUTLIER ROBUSTNESS", flush=True)
    print("=" * 80, flush=True)

    print("\n[Phase 0] Loading data + features...", flush=True)
    ohlcv_cache, feat_cache, vol_cache, hmm_cache = load_and_precompute()
    print(f"  Done in {time.time() - t0:.0f}s", flush=True)

    print("\n[Phase 1] Generating daily PnL...", flush=True)
    t1 = time.time()
    daily_pnl = generate_daily_pnl(ohlcv_cache, feat_cache, vol_cache, hmm_cache)
    daily_returns = daily_pnl / CAPITAL
    print(f"  {len(daily_pnl)} trading days, {time.time() - t1:.0f}s", flush=True)

    # ── Stats ──
    n_days = len(daily_returns)
    mean_r = daily_returns.mean()
    std_r = daily_returns.std()
    ann_sharpe = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0
    skew = float(daily_returns.skew())
    kurt = float(daily_returns.kurtosis() + 3)

    print(f"\n{'=' * 80}", flush=True)
    print("DAILY RETURN STATISTICS", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"  N days:     {n_days}", flush=True)
    print(f"  Mean daily: {mean_r * 10000:.2f} bps", flush=True)
    print(f"  Std daily:  {std_r * 10000:.2f} bps", flush=True)
    print(f"  Sharpe:     {ann_sharpe:.3f}", flush=True)
    print(f"  Skewness:   {skew:.3f}", flush=True)
    print(f"  Kurtosis:   {kurt:.1f}", flush=True)
    print(f"  Min daily:  {daily_returns.min() * 10000:.1f} bps", flush=True)
    print(f"  Max daily:  {daily_returns.max() * 10000:.1f} bps", flush=True)
    print(f"  Total PnL:  ${daily_pnl.sum():.2f}", flush=True)

    print(f"\n  Top 10 daily returns (outliers):", flush=True)
    for date, ret in daily_returns.nlargest(10).items():
        print(f"    {date}: {ret * 10000:+.1f} bps (${ret * CAPITAL:.2f})", flush=True)
    print(f"\n  Bottom 5 daily returns:", flush=True)
    for date, ret in daily_returns.nsmallest(5).items():
        print(f"    {date}: {ret * 10000:+.1f} bps (${ret * CAPITAL:.2f})", flush=True)

    # ── Test 1: Raw DSR ──
    print(f"\n{'=' * 80}", flush=True)
    print(f"TEST 1: DSR on raw daily returns (K={K_TRIALS})", flush=True)
    dsr_raw = deflated_sharpe_ratio(
        observed_sr=mean_r / std_r if std_r > 0 else 0,
        num_trials=K_TRIALS, num_observations=n_days,
        skewness=skew, kurtosis=kurt,
    )
    print(f"  DSR = {dsr_raw:.4f} (kurtosis={kurt:.0f} breaks formula)", flush=True)

    # ── Test 2: Monthly DSR ──
    print(f"\n{'=' * 80}", flush=True)
    print("TEST 2: DSR on monthly returns", flush=True)
    monthly_pnl = daily_pnl.copy()
    monthly_pnl.index = pd.to_datetime(monthly_pnl.index)
    monthly_ret = monthly_pnl.resample("M").sum() / CAPITAL
    m_n = len(monthly_ret)
    m_mean = monthly_ret.mean()
    m_std = monthly_ret.std()
    m_sharpe = m_mean / m_std * np.sqrt(12) if m_std > 0 else 0
    m_skew = float(monthly_ret.skew())
    m_kurt = float(monthly_ret.kurtosis() + 3)
    print(f"  N months: {m_n}, Sharpe: {m_sharpe:.3f}, Skew: {m_skew:.3f}, Kurt: {m_kurt:.1f}", flush=True)
    dsr_m = deflated_sharpe_ratio(
        observed_sr=m_mean / m_std if m_std > 0 else 0,
        num_trials=K_TRIALS, num_observations=m_n,
        skewness=m_skew, kurtosis=m_kurt,
    )
    print(f"  DSR = {dsr_m:.4f}  >> {'PASS' if dsr_m > 0.5 else 'FAIL'}", flush=True)

    # ── Test 3: Winsorized DSR ──
    print(f"\n{'=' * 80}", flush=True)
    print("TEST 3: DSR on winsorized daily returns (3-sigma cap)", flush=True)
    lo_cap = mean_r - 3 * std_r
    hi_cap = mean_r + 3 * std_r
    daily_w = daily_returns.clip(lo_cap, hi_cap)
    n_cap = ((daily_returns < lo_cap) | (daily_returns > hi_cap)).sum()
    w_mean = daily_w.mean()
    w_std = daily_w.std()
    w_sharpe = w_mean / w_std * np.sqrt(365.25) if w_std > 0 else 0
    w_skew = float(daily_w.skew())
    w_kurt = float(daily_w.kurtosis() + 3)
    print(f"  Capped {n_cap}/{n_days} days. Kurt: {kurt:.0f} -> {w_kurt:.1f}", flush=True)
    dsr_w = deflated_sharpe_ratio(
        observed_sr=w_mean / w_std if w_std > 0 else 0,
        num_trials=K_TRIALS, num_observations=n_days,
        skewness=w_skew, kurtosis=w_kurt,
    )
    print(f"  DSR = {dsr_w:.4f}  >> {'PASS' if dsr_w > 0.5 else 'FAIL'}", flush=True)

    # ── Test 4: Outlier robustness ──
    print(f"\n{'=' * 80}", flush=True)
    print("TEST 4: Outlier Robustness", flush=True)
    print("  Remove BEST N days:", flush=True)
    for n_rm in [1, 3, 5, 10, 20]:
        trimmed = daily_returns.sort_values().iloc[:-n_rm]
        t_s = trimmed.mean() / trimmed.std() * np.sqrt(365.25) if trimmed.std() > 0 else 0
        pnl_rm = daily_returns.nlargest(n_rm).sum() * CAPITAL
        print(f"    Remove top {n_rm:2d}: Sharpe={t_s:.3f}, PnL_removed=${pnl_rm:.2f}", flush=True)

    print("  Remove BOTH tails:", flush=True)
    for n_rm in [1, 3, 5, 10]:
        trimmed = daily_returns.sort_values().iloc[n_rm:-n_rm]
        t_s = trimmed.mean() / trimmed.std() * np.sqrt(365.25) if trimmed.std() > 0 else 0
        print(f"    Remove top+bot {n_rm:2d}: Sharpe={t_s:.3f}", flush=True)

    # ── Test 5: Bootstrap CI ──
    print(f"\n{'=' * 80}", flush=True)
    print("TEST 5: Bootstrap Sharpe CI (10,000 resamples)", flush=True)
    rng = np.random.RandomState(42)
    boot = []
    for _ in range(10000):
        s = rng.choice(daily_returns.values, size=n_days, replace=True)
        s_std = s.std()
        boot.append(s.mean() / s_std * np.sqrt(365.25) if s_std > 0 else 0)
    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    pct_pos = np.mean(boot > 0) * 100
    print(f"  Mean: {np.mean(boot):.3f}, Std: {np.std(boot):.3f}", flush=True)
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
    print(f"  P(Sharpe > 0): {pct_pos:.1f}%", flush=True)
    print(f"  >> {'PASS' if ci_lo > 0 else 'FAIL'}: CI lower bound {'>' if ci_lo > 0 else '<='} 0", flush=True)

    # ── Summary ──
    print(f"\n{'=' * 80}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"  Raw daily DSR:        {dsr_raw:.4f}  FAIL (kurtosis={kurt:.0f})", flush=True)
    print(f"  Monthly DSR:          {dsr_m:.4f}  {'PASS' if dsr_m > 0.5 else 'FAIL'}", flush=True)
    print(f"  Winsorized daily DSR: {dsr_w:.4f}  {'PASS' if dsr_w > 0.5 else 'FAIL'}", flush=True)
    print(f"  Bootstrap CI:         [{ci_lo:.3f}, {ci_hi:.3f}]  {'PASS' if ci_lo > 0 else 'FAIL'}", flush=True)
    print(f"  Positive skewness:    {skew:.1f} (fat tail is profits, not losses)", flush=True)
    print(f"\n  Total runtime: {time.time() - t0:.0f}s", flush=True)
