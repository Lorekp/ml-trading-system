"""
Re-run ONLY Test 2 (Rolling vs Expanding Window) with the tz bug fixed.
Uses the same infrastructure as validation_honest.py / model_comparison.py.
"""
import sys, warnings, time, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075
SLIPPAGE = 0.0001
MAX_EXP = 0.60
MAX_ASSET = 0.20
BASE_PCT = 0.02
VOL_Q = 0.50
CONF_PCT = 70
SIG_STEP = 0.1
MAX_SIG = 3.0
WINSOR_LO, WINSOR_HI = 0.01, 0.99

FEATURES_19 = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low", "dist_nearest_level",
    "ffd", "cs_spread", "entropy", "oi_change_5", "impact_spread",
]

# Reuse vol_targeting.py's load_data and precompute_features (proven working)
from vol_targeting import load_data, precompute_features


def generate_signals(feat_cache, vol_cache, hmm_cache, ohlcv_cache, rolling_months=None):
    """Walk-forward with optional rolling window."""
    rows = []
    for month in ALL_MONTHS:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        tc = ms - EMBARGO_TD

        # Rolling window: restrict training start
        if rolling_months is not None:
            train_start = ms - pd.DateOffset(months=rolling_months)
            if train_start.tzinfo is None:
                train_start = train_start.tz_localize("UTC")
        else:
            train_start = None

        for sym in ASSETS:
            feat = feat_cache[sym].copy()
            ohlcv = ohlcv_cache[sym]
            close = ohlcv["close"]
            fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
            vol = vol_cache[sym]

            fc = feat.dropna()
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)

            # Train/test split with optional rolling start
            if train_start is not None:
                tr = valid[(valid >= train_start) & (valid < tc)]
            else:
                tr = valid[valid < tc]
            te = valid[(valid >= ms) & (valid <= me)]

            if len(tr) < 200 or len(te) < 20:
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

            # Fit
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

            # HMM regime
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
                rows.append({
                    "timestamp": idx,
                    "asset": sym,
                    "direction": direction,
                    "fwd_return": float(fr),
                    "signal_strength": float(abs(sig_d)),
                })

    return pd.DataFrame(rows)


def simulate(signals_df):
    """Simplified portfolio sim — same as other scripts."""
    if len(signals_df) < 20:
        return {"sharpe": 0.0, "n_trades": 0}

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    equity = CAPITAL
    peak_equity = equity
    halted = False
    current_exposure = 0.0
    asset_exposure = {}
    daily_pnl = {}

    for i in range(len(sub)):
        if halted:
            continue
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= MAX_DD:
            halted = True
            continue
        size_mult = max(1.0 - (dd - DELEVER_START) / (MAX_DD - DELEVER_START), 0.0) if dd > DELEVER_START else 1.0
        sig_abs = sub.iloc[i]["signal_strength"]
        raw_size = equity * BASE_PCT * sig_abs * size_mult

        asset = sub.iloc[i]["asset"]
        a_exp = asset_exposure.get(asset, 0.0)
        if a_exp + raw_size > equity * MAX_ASSET:
            raw_size = max(equity * MAX_ASSET - a_exp, 0.0)
        if current_exposure + raw_size > equity * MAX_EXP:
            raw_size = max(equity * MAX_EXP - current_exposure, 0.0)
        if raw_size < 1.0:
            continue

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

    if not daily_pnl:
        return {"sharpe": 0.0, "n_trades": 0}
    daily_ret = pd.Series(daily_pnl) / CAPITAL
    if daily_ret.std() == 0:
        return {"sharpe": 0.0, "n_trades": len(sub)}
    ann_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365.25)
    return {
        "sharpe": round(ann_sharpe, 3),
        "n_trades": len(sub),
        "total_return_pct": round((equity / CAPITAL - 1) * 100, 2),
    }


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70, flush=True)
    print("TEST 2 RE-RUN: Rolling (6mo) vs Expanding Window", flush=True)
    print("=" * 70, flush=True)

    print("\n[Phase 0] Loading data + features...", flush=True)
    sys.path.insert(0, "C:/Claude/ml-trading-system/experiments")
    ohlcv_cache, hl_cache = load_data()
    feat_cache, vol_cache, hmm_cache, _ = precompute_features(ohlcv_cache, hl_cache)
    print(f"  Done in {time.time() - t0:.0f}s", flush=True)

    print("\n[Phase 1] Generating EXPANDING window signals...", flush=True)
    t1 = time.time()
    sig_expanding = generate_signals(feat_cache, vol_cache, hmm_cache, ohlcv_cache, rolling_months=None)
    sim_expanding = simulate(sig_expanding)
    print(f"  {len(sig_expanding)} signals, Sharpe={sim_expanding['sharpe']:.3f}, "
          f"Return={sim_expanding['total_return_pct']:+.2f}% ({time.time()-t1:.0f}s)", flush=True)

    print("\n[Phase 2] Generating ROLLING 6-month window signals...", flush=True)
    t2 = time.time()
    sig_rolling = generate_signals(feat_cache, vol_cache, hmm_cache, ohlcv_cache, rolling_months=6)
    sim_rolling = simulate(sig_rolling)
    print(f"  {len(sig_rolling)} signals, Sharpe={sim_rolling['sharpe']:.3f}, "
          f"Return={sim_rolling['total_return_pct']:+.2f}% ({time.time()-t2:.0f}s)", flush=True)

    # Also test other window sizes
    print("\n[Phase 3] Testing multiple rolling window sizes...", flush=True)
    for months in [3, 6, 9, 12]:
        t3 = time.time()
        sig = generate_signals(feat_cache, vol_cache, hmm_cache, ohlcv_cache, rolling_months=months)
        sim = simulate(sig)
        print(f"  Rolling {months:2d}mo: {len(sig):5d} signals, Sharpe={sim['sharpe']:.3f}, "
              f"Return={sim['total_return_pct']:+.2f}% ({time.time()-t3:.0f}s)", flush=True)

    delta = abs(sim_expanding["sharpe"] - sim_rolling["sharpe"])
    passed = delta < 0.3

    print(f"\n{'='*70}", flush=True)
    print(f"RESULT:", flush=True)
    print(f"  Expanding: Sharpe={sim_expanding['sharpe']:.3f}", flush=True)
    print(f"  Rolling 6mo: Sharpe={sim_rolling['sharpe']:.3f}", flush=True)
    print(f"  Delta: {delta:.3f}", flush=True)
    print(f"  >> {'[PASS]' if passed else '[FAIL]'} |delta| < 0.3", flush=True)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s", flush=True)
