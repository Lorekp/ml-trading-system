"""
Volatility Targeting — principled dynamic position sizing.

Reference: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
Also: AFML Ch 10 (bet sizing), Chan Ch 6 (Kelly + risk management)

Concept: scale position sizes so that daily portfolio risk is approximately
constant regardless of market volatility. On high-vol days, positions shrink.
On low-vol days, positions grow (up to a cap).

    position_size = base_size * (target_vol / realized_vol)

where:
    target_vol  = median realized vol from training period (no look-ahead)
    realized_vol = trailing N-bar exponential moving average of vol

This has ZERO fitted parameters beyond target_vol, which comes from training
data. It's continuous (no binary threshold) and well-established.

Test approach: compare 4 variants through the same portfolio simulator:
    1. BASELINE: current system (flat 2% sizing + probability scaling)
    2. VOL_TARGET: add volatility targeting on top of probability sizing
    3. THRESHOLD: add the 90th-percentile intraday-DD threshold rule
    4. VOL_TARGET + THRESHOLD: both combined

Split-sample: design target_vol on first 7 months, test on last 7.
"""
import sys, warnings, time, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, "C:/Claude/ml-trading-system/src")

from afml.technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels
from afml.fracdiff import frac_diff_ffd
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
OUTPUT_FILE = Path("C:/Claude/ml-trading-system/v4-results/vol_targeting_results.json")

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}
FFD_D = {"BTC": 0.2, "ETH": 0.1, "XRP": 0.3, "BNB": 0.1, "SUI": 0.2, "LINK": 0.1}
BAR_TF = "30min"
FWD_BARS = 4
ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")
EMBARGO_TD = pd.Timedelta(hours=2)

CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001
MAX_EXP = 0.60
MAX_ASSET = 0.20
BASE_PCT = 0.02
VOL_Q = 0.50
CONF_PCT = 70
MAX_SIG = 3.0
SIG_STEP = 0.1
WINSOR_LO, WINSOR_HI = 0.01, 0.99

# Vol targeting parameters
VOL_LOOKBACK = 50  # bars for realized vol EWM
VOL_TARGET_CAP = 2.0  # max scaling multiplier (don't go above 2x base)
VOL_TARGET_FLOOR = 0.25  # min scaling multiplier (don't go below 0.25x)

# Intraday DD threshold (from design half 90th percentile)
DD_THRESHOLD_PCT = 6.4

FEATURES_19 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
    "ffd", "cs_spread", "entropy", "oi_change_5", "impact_spread",
]


def load_data():
    """Load OHLCV + HL data for all assets."""
    ohlcv_cache = {}
    hl_cache = {}

    for sym in ASSETS:
        symbol = SYMBOLS[sym]
        ddir = DATA_DIR / symbol
        frames = []
        for f in sorted(ddir.glob("*.csv")):
            df = pd.read_csv(f, header=None,
                names=["ot", "open", "high", "low", "close", "volume",
                       "ct", "qv", "nt", "tbb", "tbq", "ig"])
            df = df[pd.to_numeric(df["ot"], errors="coerce").notna()]
            df["ot"] = df["ot"].astype(float)
            df["timestamp"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].dropna()
            frames.append(df)
        raw = pd.concat(frames).sort_index()
        raw = raw[~raw.index.duplicated(keep="first")]
        ohlcv_cache[sym] = raw.resample(BAR_TF).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        hlf = HL_DIR / f"hl_{sym}.csv"
        if hlf.exists():
            hl = pd.read_csv(hlf)
            tc = "timestamp" if "timestamp" in hl.columns else "time"
            hl[tc] = pd.to_datetime(hl[tc], utc=True)
            hl = hl.set_index(tc)
            hl_cache[sym] = hl
        else:
            hl_cache[sym] = None

    return ohlcv_cache, hl_cache


def precompute_features(ohlcv_cache, hl_cache):
    """Compute all 19 features + vol + HMM for each asset."""
    feat_cache = {}
    vol_cache = {}
    hmm_cache = {}
    realized_vol_cache = {}  # for vol targeting

    for sym in ASSETS:
        ohlcv = ohlcv_cache[sym]
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
            feat["dist_nearest_level"] = swing_df.get(
                "dist_nearest_level",
                feat[["dist_swing_high", "dist_swing_low"]].abs().min(axis=1),
            )
        else:
            cols = swing_df.columns.tolist()
            feat["dist_swing_high"] = swing_df[cols[0]] if len(cols) > 0 else 0
            feat["dist_swing_low"] = swing_df[cols[1]] if len(cols) > 1 else 0
            feat["dist_nearest_level"] = swing_df[cols[2]] if len(cols) > 2 else 0

        log_c = np.log(c).to_frame("close")
        ffd_vals = frac_diff_ffd(log_c, FFD_D[sym], threshold=1e-4)
        feat["ffd"] = ffd_vals["close"]
        feat["cs_spread"] = corwin_schultz_spread(h, l, window=20)
        feat["entropy"] = rolling_entropy(c, window=50, n_bins=5)

        hl = hl_cache[sym]
        if hl is not None:
            hl_30 = hl.resample(BAR_TF).last().reindex(ohlcv.index, method="ffill")
            oi_col = next((x for x in hl_30.columns if "interest" in x.lower()), None)
            feat["oi_change_5"] = hl_30[oi_col].pct_change(5) if oi_col else 0.0
            bid_col = next((x for x in hl_30.columns if "bid" in x.lower()), None)
            ask_col = next((x for x in hl_30.columns if "ask" in x.lower()), None)
            mid_col = next((x for x in hl_30.columns if "mid" in x.lower() or "mark" in x.lower()), None)
            if bid_col and ask_col:
                mid_px = hl_30[mid_col] if mid_col else (hl_30[bid_col] + hl_30[ask_col]) / 2
                feat["impact_spread"] = (hl_30[ask_col] - hl_30[bid_col]) / mid_px.replace(0, np.nan)
            else:
                feat["impact_spread"] = 0.0
        else:
            feat["oi_change_5"] = 0.0
            feat["impact_spread"] = 0.0

        cols = [x for x in FEATURES_19 if x in feat.columns]
        feat_cache[sym] = feat[cols]
        vol_cache[sym] = feat["ret_1"].ewm(span=50).std()

        # Realized vol for vol targeting: EWM of absolute returns
        realized_vol_cache[sym] = feat["ret_1"].abs().ewm(span=VOL_LOOKBACK).mean()

        # HMM
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

    return feat_cache, vol_cache, hmm_cache, realized_vol_cache


def compute_btc_intraday_dd(ohlcv_cache):
    """Compute BTC intraday drawdown at each 30-min bar (for threshold filter)."""
    btc = ohlcv_cache["BTC"]
    # Running daily high at each bar
    daily_high = btc["high"].groupby(btc.index.date).cummax()
    # Intraday DD = (daily_high_so_far - current_close) / daily_high_so_far
    intraday_dd = (daily_high.values - btc["close"].values) / daily_high.values * 100
    return pd.Series(intraday_dd, index=btc.index)


def generate_signals(feat_cache, vol_cache, hmm_cache, realized_vol_cache, ohlcv_cache):
    """Walk-forward signal generation with LogReg L1 C=0.1.
    Returns DataFrame with per-signal metadata needed for all 4 variants."""
    rows = []

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
            rvol = realized_vol_cache[sym]

            fc = feat.dropna()
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            tr = valid[valid < tc]
            te = valid[(valid >= ms) & (valid <= me)]
            if len(tr) < 500 or len(te) < 20:
                continue

            # Winsorize
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
                t_end = min(i + FWD_BARS, n - 1)
                concurrent = np.sum((ts_arr >= ts_arr[i]) & (ts_arr <= ts_arr[t_end]))
                sw[i] = 1.0 / max(concurrent, 1)
            sw = sw / sw.mean()

            # Fit model
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

            # Compute target_vol from training data
            train_rvol = rvol.loc[rvol.index.intersection(tr)].dropna()
            target_vol = train_rvol.median() if len(train_rvol) > 30 else None

            for j in range(len(yp_f)):
                idx = te_f[j]
                fr = fwd_ret.get(idx, np.nan)
                if not np.isfinite(fr):
                    continue
                direction = 1.0 if yp_f[j] > 0.5 else -1.0
                sig = get_signal(np.array([yp_f[j]]), num_classes=2)[0]
                sig_d = round(sig / SIG_STEP) * SIG_STEP
                sig_d = np.clip(sig_d, -MAX_SIG, MAX_SIG)

                # Current realized vol at this bar
                cur_rvol = rvol.get(idx, np.nan)
                # Vol target ratio
                if target_vol and cur_rvol and cur_rvol > 0 and np.isfinite(cur_rvol):
                    vt_ratio = float(np.clip(target_vol / cur_rvol, VOL_TARGET_FLOOR, VOL_TARGET_CAP))
                else:
                    vt_ratio = 1.0

                rows.append({
                    "timestamp": idx,
                    "month": str(month),
                    "asset": sym,
                    "direction": direction,
                    "fwd_return": float(fr),
                    "signal_strength": float(abs(sig_d)),
                    "probability": float(yp_f[j]),
                    "vol_target_ratio": vt_ratio,
                })

    return pd.DataFrame(rows)


def simulate(signals_df, use_vol_target=False, use_dd_threshold=False, btc_intraday_dd=None):
    """Portfolio simulation with optional vol targeting and DD threshold."""
    if len(signals_df) < 20:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    equity = CAPITAL
    peak_equity = equity
    halted = False
    current_exposure = 0.0
    asset_exposure = {}

    daily_pnl = {}

    for i in range(n):
        if halted:
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= MAX_DD:
            halted = True
            continue

        # Deleveraging
        if dd > DELEVER_START:
            dd_ratio = (dd - DELEVER_START) / (MAX_DD - DELEVER_START)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        # Base position size
        sig_abs = sub.iloc[i]["signal_strength"]
        raw_size = equity * BASE_PCT * sig_abs * size_mult

        # Vol targeting
        if use_vol_target:
            vt_ratio = sub.iloc[i]["vol_target_ratio"]
            raw_size *= vt_ratio

        # DD threshold
        if use_dd_threshold and btc_intraday_dd is not None:
            ts = sub.iloc[i]["timestamp"]
            dd_val = btc_intraday_dd.asof(ts) if hasattr(btc_intraday_dd, "asof") else 0
            if not np.isnan(dd_val) and dd_val > DD_THRESHOLD_PCT:
                raw_size *= 0.5

        # Exposure limits
        asset = sub.iloc[i]["asset"]
        asset_exp = asset_exposure.get(asset, 0.0)
        if asset_exp + raw_size > equity * MAX_ASSET:
            raw_size = max(equity * MAX_ASSET - asset_exp, 0.0)
        if current_exposure + raw_size > equity * MAX_EXP:
            raw_size = max(equity * MAX_EXP - current_exposure, 0.0)
        if raw_size < 1.0:
            continue

        # Execute
        trade_ret = sub.iloc[i]["direction"] * sub.iloc[i]["fwd_return"] - SLIPPAGE
        pnl = raw_size * trade_ret
        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        current_exposure = min(current_exposure + raw_size * 0.1, equity * MAX_EXP)
        ts = sub.iloc[i]["timestamp"]
        if i > 0 and ts != sub.iloc[i - 1]["timestamp"]:
            current_exposure *= 0.9
            asset_exposure = {k: v * 0.9 for k, v in asset_exposure.items()}
        asset_exposure[asset] = asset_exposure.get(asset, 0.0) + raw_size * 0.1

        date = pd.Timestamp(ts).strftime("%Y-%m-%d")
        daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl

    # Compute metrics
    if not daily_pnl:
        return None
    daily_ret = pd.Series(daily_pnl) / CAPITAL
    n_days = len(daily_ret)
    if n_days < 5 or daily_ret.std() == 0:
        return None

    total_ret = equity / CAPITAL - 1
    ann_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)
    sortino_denom = daily_ret[daily_ret < 0].std()
    ann_sortino = daily_ret.mean() / sortino_denom * np.sqrt(365.25) if sortino_denom > 0 else 0

    # Max DD from daily equity curve
    equity_curve = (1 + daily_ret).cumprod()
    running_max = equity_curve.cummax()
    max_dd = ((equity_curve - running_max) / running_max).min()

    n_trades = len(sub[sub.index.isin(range(n))])  # approximate
    skew = float(daily_ret.skew())
    kurt = float(daily_ret.kurtosis() + 3)

    return {
        "sharpe": round(ann_sharpe, 3),
        "sortino": round(ann_sortino, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "max_dd_pct": round(abs(max_dd) * 100, 2),
        "n_days": n_days,
        "n_trades": len(sub),
        "final_equity": round(equity, 2),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 1),
        "halted": halted,
        "mean_daily_pnl_bps": round(daily_ret.mean() * 10000, 2),
        "std_daily_pnl_bps": round(daily_ret.std() * 10000, 2),
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80, flush=True)
    print("VOLATILITY TARGETING — Principled Dynamic Position Sizing", flush=True)
    print("=" * 80, flush=True)
    print("Reference: Moskowitz, Ooi & Pedersen (2012)", flush=True)
    print(f"Vol lookback: {VOL_LOOKBACK} bars, Cap: {VOL_TARGET_CAP}x, Floor: {VOL_TARGET_FLOOR}x", flush=True)
    print(f"DD threshold: {DD_THRESHOLD_PCT}% (90th pctile from design half)", flush=True)
    print(flush=True)

    print("[Phase 0] Loading data...", flush=True)
    ohlcv_cache, hl_cache = load_data()
    print(f"  Done in {time.time() - t0:.0f}s", flush=True)

    print("[Phase 1] Precomputing features...", flush=True)
    t1 = time.time()
    feat_cache, vol_cache, hmm_cache, rvol_cache = precompute_features(ohlcv_cache, hl_cache)
    print(f"  Done in {time.time() - t1:.0f}s", flush=True)

    print("[Phase 2] Computing BTC intraday DD...", flush=True)
    btc_dd = compute_btc_intraday_dd(ohlcv_cache)
    print(f"  Done", flush=True)

    print("[Phase 3] Generating signals...", flush=True)
    t2 = time.time()
    signals = generate_signals(feat_cache, vol_cache, hmm_cache, rvol_cache, ohlcv_cache)
    print(f"  {len(signals)} signals in {time.time() - t2:.0f}s", flush=True)

    # Vol target ratio statistics
    vtr = signals["vol_target_ratio"]
    print(f"\n  Vol target ratio stats:", flush=True)
    print(f"    Mean:   {vtr.mean():.3f}", flush=True)
    print(f"    Median: {vtr.median():.3f}", flush=True)
    print(f"    Std:    {vtr.std():.3f}", flush=True)
    print(f"    Min:    {vtr.min():.3f} (high-vol bars, sizing reduced)", flush=True)
    print(f"    Max:    {vtr.max():.3f} (low-vol bars, sizing increased)", flush=True)
    print(f"    <0.5:   {(vtr < 0.5).sum()} signals ({(vtr < 0.5).mean()*100:.1f}%)", flush=True)
    print(f"    >1.5:   {(vtr > 1.5).sum()} signals ({(vtr > 1.5).mean()*100:.1f}%)", flush=True)

    # Phase 4: Simulate 4 variants
    print(f"\n[Phase 4] Simulating 4 variants...", flush=True)

    variants = [
        ("BASELINE", False, False),
        ("VOL_TARGET", True, False),
        ("DD_THRESHOLD", False, True),
        ("VOL_TARGET + DD_THRESHOLD", True, True),
    ]

    results = {}
    for name, use_vt, use_dd in variants:
        sim = simulate(signals, use_vol_target=use_vt, use_dd_threshold=use_dd, btc_intraday_dd=btc_dd)
        results[name] = sim
        if sim:
            print(f"  {name:30s}: Sharpe={sim['sharpe']:6.3f}, Return={sim['total_return_pct']:+6.2f}%, "
                  f"MaxDD={sim['max_dd_pct']:.2f}%, Std={sim['std_daily_pnl_bps']:.1f}bps", flush=True)
        else:
            print(f"  {name:30s}: FAILED", flush=True)

    # Phase 5: Split-sample analysis
    print(f"\n[Phase 5] Split-sample validation...", flush=True)
    split_date = pd.Timestamp("2025-08-01", tz="UTC")

    for half_name, mask_fn in [("DESIGN (Jan-Jul 2025)", lambda s: s["timestamp"] < split_date),
                                ("TEST (Aug 2025-Feb 2026)", lambda s: s["timestamp"] >= split_date)]:
        half_signals = signals[mask_fn(signals)].copy()
        if len(half_signals) < 20:
            print(f"\n  {half_name}: too few signals", flush=True)
            continue

        print(f"\n  {half_name} ({len(half_signals)} signals):", flush=True)
        for name, use_vt, use_dd in variants:
            sim = simulate(half_signals, use_vol_target=use_vt, use_dd_threshold=use_dd, btc_intraday_dd=btc_dd)
            if sim:
                print(f"    {name:30s}: Sharpe={sim['sharpe']:6.3f}, Return={sim['total_return_pct']:+6.2f}%, "
                      f"MaxDD={sim['max_dd_pct']:.2f}%", flush=True)

    # Summary
    print(f"\n{'=' * 80}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'=' * 80}", flush=True)

    base = results.get("BASELINE", {})
    vt = results.get("VOL_TARGET", {})
    dd = results.get("DD_THRESHOLD", {})
    both = results.get("VOL_TARGET + DD_THRESHOLD", {})

    if base and vt:
        vt_delta = vt["sharpe"] - base["sharpe"]
        std_reduction = (1 - vt["std_daily_pnl_bps"] / base["std_daily_pnl_bps"]) * 100 if base["std_daily_pnl_bps"] > 0 else 0
        print(f"  Vol targeting Sharpe delta: {vt_delta:+.3f}", flush=True)
        print(f"  Vol targeting std reduction: {std_reduction:+.1f}%", flush=True)

    if base and dd:
        dd_delta = dd["sharpe"] - base["sharpe"]
        print(f"  DD threshold Sharpe delta:  {dd_delta:+.3f}", flush=True)

    if base and both:
        both_delta = both["sharpe"] - base["sharpe"]
        print(f"  Combined Sharpe delta:      {both_delta:+.3f}", flush=True)

    # Save
    all_results = {
        "timestamp": str(pd.Timestamp.now()),
        "variants": results,
        "config": {
            "vol_lookback": VOL_LOOKBACK,
            "vol_target_cap": VOL_TARGET_CAP,
            "vol_target_floor": VOL_TARGET_FLOOR,
            "dd_threshold_pct": DD_THRESHOLD_PCT,
        },
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_FILE}", flush=True)
    print(f"Total runtime: {time.time() - t0:.0f}s ({(time.time()-t0)/60:.1f}m)", flush=True)
