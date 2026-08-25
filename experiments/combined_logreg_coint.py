"""Combined Portfolio: LogReg Direction + Cointegration Spread

Runs BOTH strategies on the same test period (2025-03 to 2026-02):
1. LogReg: per-asset 15m direction prediction (simplified walk-forward)
2. Coint: spread mean-reversion with rolling health check

Measures: individual PnL, actual daily correlation, combined portfolio metrics.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")
DATA_DIR = Path("data/binance-1m")

TEST_START = "2025-03-01"; TEST_END = "2026-02-28"; CAL_DAYS = 366
CAPITAL = 100000.0  # $100K prop account

# LogReg config (T1 from previous session)
LR_ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "AVAX", "LINK",
             "NEAR", "FIL", "DOT", "LTC", "UNI", "AAVE", "FET", "APT"]
LR_C = 0.02; LR_LOOKBACK = 2; LR_THRESHOLD = 0.10; LR_HOLD = 6
LR_SIZE_PCT = 0.06; LR_LEVERAGE = 5.0

# Coint config (optimal from sweep)
COINT_SIZE = 0.03; COINT_ENTRY_Z = 2.0; COINT_EXIT_Z = 0.5
COINT_MIN_HOLD = 96; COINT_MAX_HOLD = 96*30; COINT_EMBARGO = 96*7
COINT_MIN_CORR = 0.40; COINT_DROP_MONTHS = 3; COINT_ADD_MONTHS = 2
COINT_MAX_PAIRS = 8; COINT_BREAKDOWN_Z = 5.0


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


# ============================================================
# LogReg Strategy (simplified walk-forward)
# ============================================================

def run_logreg(test_start, test_end):
    """Run simplified LogReg on 15m bars. Returns daily PnL Series."""
    feat_cfg = FeatureConfig()
    daily_pnl = {}; n_trades = 0

    for asset in LR_ASSETS:
        symbol = f"{asset}USDT"
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        except:
            continue

        feats = compute_features(ohlcv_15m, feat_cfg)
        close = ohlcv_15m["close"]

        # Forward return (label)
        fwd = close.pct_change(LR_HOLD).shift(-LR_HOLD)
        label = (fwd > 0).astype(int)

        # Align
        common = feats.dropna().index.intersection(label.dropna().index)
        X_all = feats.reindex(common).values
        y_all = label.reindex(common).values
        close_all = close.reindex(common)

        ts = pd.Timestamp(test_start, tz="UTC")
        te = pd.Timestamp(test_end, tz="UTC")

        # Walk-forward: train on prior LR_LOOKBACK months, predict current month
        test_mask = (common >= ts) & (common <= te)
        test_idx = common[test_mask]
        if len(test_idx) < 96 * 30: continue

        # Train on everything before test start
        train_end = ts - pd.Timedelta(days=7)
        train_start = train_end - pd.DateOffset(months=LR_LOOKBACK)
        train_mask = (common >= train_start) & (common < train_end)

        X_train = X_all[train_mask]; y_train = y_all[train_mask]
        valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[valid]; y_train = y_train[valid]

        if len(X_train) < 500: continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = LogisticRegression(C=LR_C, penalty="l1", solver="saga", max_iter=1000, random_state=42)
        model.fit(X_train_s, y_train)

        # Predict on test
        X_test = X_all[test_mask]
        valid_test = np.isfinite(X_test).all(axis=1)
        X_test_clean = np.nan_to_num(X_test, nan=0)
        X_test_s = scaler.transform(X_test_clean)

        probs = model.predict_proba(X_test_s)[:, 1]

        # Generate signals and PnL
        pos = 0; hold_count = 0
        for i, idx in enumerate(test_idx):
            if not valid_test[i]: continue
            p = probs[i]
            ret_1bar = float(close_all.pct_change().reindex([idx]).iloc[0]) if idx in close_all.index else 0
            if np.isnan(ret_1bar): ret_1bar = 0

            bar_pnl = 0.0
            notional = CAPITAL * LR_SIZE_PCT * LR_LEVERAGE

            if pos != 0:
                bar_pnl = pos * ret_1bar * notional
                hold_count += 1
                if hold_count >= LR_HOLD:
                    pos = 0; hold_count = 0

            if pos == 0:
                confidence = abs(p - 0.5)
                if confidence > LR_THRESHOLD:
                    direction = 1 if p > 0.5 else -1
                    pos = direction; hold_count = 0; n_trades += 1

            day = idx.normalize()
            daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    print(f"  LogReg: {n_trades} trades across {len(LR_ASSETS)} assets")
    return pd.Series(daily_pnl).sort_index()


# ============================================================
# Coint Strategy (optimal config)
# ============================================================

def run_coint(test_start, test_end):
    """Run optimal coint config. Returns daily PnL Series."""
    # Load close data
    ohlcv_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        n = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000: ohlcv_data[n] = df
        except: pass
    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()

    months = pd.period_range(test_start[:7], test_end[:7], freq="M")

    def select_universe(ms):
        le = ms - pd.Timedelta(days=1); c = {}
        for n, o in ohlcv_data.items():
            if (ms - o.index.min()).days < 180: continue
            a = o.loc[:le]
            if len(a) < 96*45: continue
            dv = (a["close"]*a["volume"]).resample("1D").sum()
            r = dv.iloc[-30:] if len(dv) >= 30 else dv; v = r.median()
            if np.isnan(v) or v <= 0: continue
            if len(dv) >= 90 and dv.iloc[-90:].median() > 0 and v/dv.iloc[-90:].median() > 5: continue
            c[n] = float(v)
        return [a for a, _ in sorted(c.items(), key=lambda x: -x[1])[:30]]

    book = {}; candidates = {}; daily_pnl = {}; n_trades = 0

    for month in months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)
        universe = select_universe(ms)
        if len(universe) < 6: continue

        # Screen pairs
        se = ms - pd.Timedelta(minutes=15*COINT_EMBARGO); ss = se - pd.DateOffset(months=6)
        avail = [a for a in universe if a in close_panel.columns]
        passing = set()
        pairs_info = {}

        for ai, aj in combinations(avail, 2):
            m = (close_panel.index >= ss) & (close_panel.index < se)
            ys = close_panel.loc[m, ai].dropna(); xs = close_panel.loc[m, aj].dropna()
            co = ys.index.intersection(xs.index)
            if len(co) < 3000: continue
            ya = ys.reindex(co).values.astype(float); xa = xs.reindex(co).values.astype(float)
            ry = np.diff(ya)/ya[:-1]; rx = np.diff(xa)/xa[:-1]
            v = np.isfinite(ry) & np.isfinite(rx)
            if v.sum() < 2000: continue
            corr = float(np.corrcoef(ry[v], rx[v])[0, 1])
            if not np.isfinite(corr) or corr < COINT_MIN_CORR: continue
            try: _, p, _ = coint(ya, xa, maxlag=10)
            except: p = 1.0
            ly = np.log(ya); lx = np.log(xa)
            beta, intercept = ols_fit(ly, lx)
            if not np.isfinite(beta) or beta <= 0: continue
            resid = ly - beta*lx - intercept; hl = half_life_bars(resid)/96
            if p < 0.05 and 1 < hl < 20:
                passing.add((ai, aj))
                pairs_info[(ai, aj)] = {"beta": beta, "intercept": intercept,
                    "mu": float(np.mean(resid)), "sigma": float(np.std(resid))}

        # Book management
        for key in list(book.keys()):
            if key in passing or (key[1], key[0]) in passing:
                book[key]["fc"] = 0
            else:
                book[key]["fc"] = book[key].get("fc", 0) + 1
                if book[key]["fc"] >= COINT_DROP_MONTHS:
                    if book[key].get("_pos", 0) != 0:
                        ct = COST_BPS/10000*(1+abs(book[key].get("beta",1)))*CAPITAL*COINT_SIZE
                        daily_pnl[ms.normalize()] = daily_pnl.get(ms.normalize(), 0) - ct
                    book.pop(key); continue
            pi = pairs_info.get(key) or pairs_info.get((key[1], key[0]))
            if pi: book[key].update(pi)

        COST_BPS = 5.0
        for key in passing:
            if key in book or (key[1], key[0]) in book: continue
            candidates[key] = candidates.get(key, 0) + 1
            if candidates[key] >= COINT_ADD_MONTHS and len(book) < COINT_MAX_PAIRS:
                pi = pairs_info.get(key)
                if pi:
                    book[key] = {**pi, "fc": 0, "_pos": 0, "_ets": None}
                    candidates.pop(key, None)
        for key in list(candidates.keys()):
            if key not in passing and (key[1], key[0]) not in passing:
                candidates.pop(key, None)

        # Trade
        month_bars = close_panel.index[(close_panel.index >= ms) & (close_panel.index <= me)]
        for key, pair in book.items():
            yn, xn = key; beta = pair["beta"]; mu = pair["mu"]; sigma = pair["sigma"]
            if not np.isfinite(beta) or sigma <= 0: continue
            if yn not in close_panel.columns or xn not in close_panel.columns: continue
            pos = pair.get("_pos", 0); ets = pair.get("_ets", None)
            notional = CAPITAL * COINT_SIZE

            for ts in month_bars:
                yp = close_panel.at[ts, yn]; xp = close_panel.at[ts, xn]
                if np.isnan(yp) or np.isnan(xp) or yp <= 0 or xp <= 0: continue
                spread = math.log(yp) - beta*math.log(xp) - pair.get("intercept", 0)
                z = (spread - mu)/sigma if sigma > 0 else 0
                yr = ret_panel.at[ts, yn] if ts in ret_panel.index and yn in ret_panel.columns else 0
                xr = ret_panel.at[ts, xn] if ts in ret_panel.index and xn in ret_panel.columns else 0
                if np.isnan(yr): yr = 0
                if np.isnan(xr): xr = 0
                bar_pnl = 0.0; cost = 5.0/10000*(1+abs(beta))*notional
                if pos != 0:
                    bar_pnl = pos*(yr - beta*xr)*notional
                    bh = int((ts-ets).total_seconds()/(15*60)) if isinstance(ets, pd.Timestamp) else COINT_MIN_HOLD+1
                    ex = False
                    if bh >= COINT_MIN_HOLD:
                        if pos == 1 and z >= COINT_EXIT_Z: ex = True
                        elif pos == -1 and z <= -COINT_EXIT_Z: ex = True
                    if bh >= COINT_MAX_HOLD: ex = True
                    if abs(z) > COINT_BREAKDOWN_Z and bh > 48: ex = True
                    if ex: bar_pnl -= cost; pos = 0; ets = None
                if pos == 0:
                    if z <= -COINT_ENTRY_Z: pos = 1; ets = ts; bar_pnl -= cost; n_trades += 1
                    elif z >= COINT_ENTRY_Z: pos = -1; ets = ts; bar_pnl -= cost; n_trades += 1
                day = ts.normalize()
                daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl
            pair["_pos"] = pos; pair["_ets"] = ets

    print(f"  Coint: {n_trades} trades across {len(book)} pairs")
    return pd.Series(daily_pnl).sort_index()


# ============================================================
# Main
# ============================================================

def compute_metrics(dpnl, label, capital=CAPITAL):
    dr = dpnl / capital
    if len(dr) < 10 or dr.std() == 0: return
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25)
    cum = dpnl.cumsum()
    peak = (cum + capital).cummax()
    dd = abs(((cum + capital - peak) / peak).min())
    wd = abs(dr.min())
    pnl_cal = cum.iloc[-1] / CAL_DAYS
    monthly = dpnl.resample("ME").sum()
    pm = (monthly > 0).sum()
    print(f"  {label:<25s} Sharpe={sharpe:>5.2f}  ${pnl_cal:>7.2f}/cal  "
          f"DD={dd*100:>5.1f}%  WD={wd*100:>5.2f}%  +Mo={pm}/{len(monthly)} ({pm/len(monthly)*100:.0f}%)")
    return {"sharpe": sharpe, "pnl_cal": pnl_cal, "dd": dd, "wd": wd, "dpnl": dpnl}


def main():
    t0 = time.time()
    print("=" * 100)
    print("COMBINED PORTFOLIO: LogReg Direction + Cointegration Spread")
    print(f"Capital: ${CAPITAL:,.0f}  |  Test: {TEST_START} to {TEST_END}")
    print("=" * 100)

    # Run LogReg
    print("\n--- Running LogReg ---")
    lr_pnl = run_logreg(TEST_START, TEST_END)

    # Run Coint
    print("\n--- Running Coint ---")
    coint_pnl = run_coint(TEST_START, TEST_END)

    # Align dates
    common_dates = lr_pnl.index.intersection(coint_pnl.index)
    lr_aligned = lr_pnl.reindex(common_dates).fillna(0)
    coint_aligned = coint_pnl.reindex(common_dates).fillna(0)
    combined = lr_aligned + coint_aligned

    # Also: combined with all dates (not just intersection)
    all_dates = lr_pnl.index.union(coint_pnl.index).sort_values()
    combined_full = lr_pnl.reindex(all_dates).fillna(0) + coint_pnl.reindex(all_dates).fillna(0)

    print(f"\n{'='*100}")
    print("INDIVIDUAL & COMBINED METRICS")
    print(f"{'='*100}\n")

    lr_m = compute_metrics(lr_pnl, "LogReg (direction)")
    coint_m = compute_metrics(coint_pnl, "Coint (spread)")
    combined_m = compute_metrics(combined_full, "COMBINED")

    # Correlation
    if len(common_dates) > 20:
        rho = float(lr_aligned.corr(coint_aligned))
        print(f"\n  Daily PnL correlation: {rho:.3f}")
        if abs(rho) < 0.1:
            print("  -> Near-zero correlation: GENUINE diversification")
        elif abs(rho) < 0.3:
            print("  -> Low correlation: good diversification")
        else:
            print("  -> Moderate correlation: some overlap")

    # Monthly breakdown
    print(f"\n{'='*100}")
    print("MONTHLY PNL COMPARISON")
    print(f"{'='*100}")

    lr_monthly = lr_pnl.resample("ME").sum()
    coint_monthly = coint_pnl.resample("ME").sum()
    combined_monthly = combined_full.resample("ME").sum()

    print(f"\n{'Month':<10s} {'LogReg':>10s} {'Coint':>10s} {'Combined':>10s}")
    print("-" * 45)
    all_months = sorted(set(lr_monthly.index) | set(coint_monthly.index))
    for dt in all_months:
        lr_v = lr_monthly.get(dt, 0); c_v = coint_monthly.get(dt, 0)
        combo = lr_v + c_v
        flag = " <<<" if combo < 0 else ""
        print(f"{dt.strftime('%Y-%m'):<10s} ${lr_v:>9.0f} ${c_v:>9.0f} ${combo:>9.0f}{flag}")

    # Key question: does coint help in LogReg's bad months?
    if lr_m and coint_m:
        print(f"\n{'='*100}")
        print("DIVERSIFICATION ANALYSIS")
        print(f"{'='*100}")

        # Months where LogReg lost money
        lr_bad = lr_monthly[lr_monthly < 0]
        if len(lr_bad) > 0:
            print(f"\nLogReg negative months ({len(lr_bad)}):")
            for dt in lr_bad.index:
                lr_v = float(lr_bad.loc[dt])
                c_v = float(coint_monthly.get(dt, 0))
                helped = "HELPED" if c_v > 0 else "HURT"
                print(f"  {dt.strftime('%Y-%m')}: LogReg ${lr_v:.0f} + Coint ${c_v:.0f} = ${lr_v+c_v:.0f} ({helped})")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
