"""LogReg T1 Regime Analysis — When does it lose money?

Overlay daily PnL with market regimes:
- BTC trend (up/down/sideways)
- Volatility regime (low/medium/high)
- Market momentum
- Drawdown periods
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SCACHE = Path("data/optuna/signal_cache")
PCACHE = Path("data/cache_15m")
CAPITAL = 10000.0

W = [("W1", 245), ("W2", 244), ("W3", 122)]


def load_logreg_dpnl():
    from signal_engine.config import PortfolioConfig
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
    import test_original_pipeline as orig

    all_dpnl = {}
    for wl, cd in W:
        p = SCACHE / f"C0.02000_lb2_{wl}.pkl"
        if not p.exists(): continue
        with open(p, "rb") as f: raw = pickle.load(f)
        fi = raw[raw["confidence"] > 0.10].copy()
        if len(fi) < 10: continue
        fi["fwd_return"] = fi["ret_6"].values
        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.30, max_signal_mult=3.0)
        sim = simulate_portfolio(combined, pcfg)
        if sim is None: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None: continue
        for dt, val in dpnl.items():
            dt2 = dt.tz_localize(None) if hasattr(dt, "tz") and dt.tz is not None else dt
            all_dpnl[dt2] = all_dpnl.get(dt2, 0) + val
    return pd.Series(all_dpnl).sort_index()


def main():
    t0 = time.time()
    print("=" * 110)
    print("LOGREG T1 REGIME ANALYSIS")
    print("=" * 110)

    # Load PnL
    dpnl = load_logreg_dpnl()
    print(f"Daily PnL: {len(dpnl)} days, {dpnl.index[0].strftime('%Y-%m-%d')} to {dpnl.index[-1].strftime('%Y-%m-%d')}")

    # Load BTC daily
    btc = pd.read_parquet(PCACHE / "BTCUSDT.parquet", columns=["close", "volume"])
    btc_daily = btc["close"].resample("1D").last().dropna()
    btc_daily.index = btc_daily.index.tz_localize(None)

    # Align
    common = dpnl.index.intersection(btc_daily.index)
    dpnl = dpnl.reindex(common)
    btc_d = btc_daily.reindex(common)

    # ---- REGIME INDICATORS ----

    # 1. BTC trend (20-day return)
    btc_ret_20d = btc_d.pct_change(20)
    # 2. BTC 7-day return
    btc_ret_7d = btc_d.pct_change(7)
    # 3. BTC 1-day return
    btc_ret_1d = btc_d.pct_change(1)
    # 4. BTC realized vol (20-day)
    btc_vol_20d = btc_d.pct_change().rolling(20).std() * np.sqrt(365)
    # 5. BTC vol regime (percentile-based)
    btc_vol_pct = btc_vol_20d.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    # 6. BTC distance from 20d high
    btc_20d_high = btc_d.rolling(20).max()
    btc_dd_from_high = (btc_d - btc_20d_high) / btc_20d_high
    # 7. BTC SMA cross (close vs 20d SMA)
    btc_sma20 = btc_d.rolling(20).mean()
    btc_above_sma = (btc_d > btc_sma20).astype(int)

    df = pd.DataFrame({
        "pnl": dpnl, "btc_close": btc_d,
        "btc_ret_1d": btc_ret_1d, "btc_ret_7d": btc_ret_7d,
        "btc_ret_20d": btc_ret_20d, "btc_vol_20d": btc_vol_20d,
        "btc_vol_pct": btc_vol_pct, "btc_dd_from_high": btc_dd_from_high,
        "btc_above_sma": btc_above_sma,
    }).dropna()

    print(f"Analysis period: {len(df)} days\n")

    # ---- ANALYSIS BY REGIME ----

    def regime_stats(mask, label):
        sub = df.loc[mask, "pnl"]
        if len(sub) < 5: return
        sr = sub.mean() / sub.std() * np.sqrt(365.25) if sub.std() > 0 else 0
        wr = (sub > 0).sum() / len(sub)
        avg = sub.mean()
        total = sub.sum()
        print(f"  {label:<35s} N={len(sub):>4d}  avg=${avg:>7.2f}  SR={sr:>5.2f}  "
              f"WR={wr*100:>4.0f}%  total=${total:>8.0f}")

    # A. BTC 20-day trend
    print("--- A. BTC 20-day Trend ---")
    regime_stats(df["btc_ret_20d"] < -0.15, "Strong downtrend (<-15%)")
    regime_stats((df["btc_ret_20d"] >= -0.15) & (df["btc_ret_20d"] < -0.05), "Mild downtrend (-15% to -5%)")
    regime_stats((df["btc_ret_20d"] >= -0.05) & (df["btc_ret_20d"] < 0.05), "Sideways (-5% to +5%)")
    regime_stats((df["btc_ret_20d"] >= 0.05) & (df["btc_ret_20d"] < 0.15), "Mild uptrend (+5% to +15%)")
    regime_stats(df["btc_ret_20d"] >= 0.15, "Strong uptrend (>+15%)")

    # B. BTC 7-day trend
    print("\n--- B. BTC 7-day Trend ---")
    regime_stats(df["btc_ret_7d"] < -0.10, "Sharp drop (<-10%)")
    regime_stats((df["btc_ret_7d"] >= -0.10) & (df["btc_ret_7d"] < -0.03), "Down (-10% to -3%)")
    regime_stats((df["btc_ret_7d"] >= -0.03) & (df["btc_ret_7d"] < 0.03), "Flat (-3% to +3%)")
    regime_stats((df["btc_ret_7d"] >= 0.03) & (df["btc_ret_7d"] < 0.10), "Up (+3% to +10%)")
    regime_stats(df["btc_ret_7d"] >= 0.10, "Sharp rally (>+10%)")

    # C. Volatility regime
    print("\n--- C. BTC Annualized Volatility ---")
    regime_stats(df["btc_vol_20d"] < 0.30, "Very low vol (<30%)")
    regime_stats((df["btc_vol_20d"] >= 0.30) & (df["btc_vol_20d"] < 0.50), "Low vol (30-50%)")
    regime_stats((df["btc_vol_20d"] >= 0.50) & (df["btc_vol_20d"] < 0.80), "Medium vol (50-80%)")
    regime_stats((df["btc_vol_20d"] >= 0.80) & (df["btc_vol_20d"] < 1.20), "High vol (80-120%)")
    regime_stats(df["btc_vol_20d"] >= 1.20, "Extreme vol (>120%)")

    # D. Volatility percentile
    print("\n--- D. BTC Vol Percentile (60-day rolling) ---")
    regime_stats(df["btc_vol_pct"] < 0.20, "Bottom 20% vol")
    regime_stats((df["btc_vol_pct"] >= 0.20) & (df["btc_vol_pct"] < 0.40), "20-40% vol")
    regime_stats((df["btc_vol_pct"] >= 0.40) & (df["btc_vol_pct"] < 0.60), "40-60% vol")
    regime_stats((df["btc_vol_pct"] >= 0.60) & (df["btc_vol_pct"] < 0.80), "60-80% vol")
    regime_stats(df["btc_vol_pct"] >= 0.80, "Top 20% vol")

    # E. Drawdown from 20d high
    print("\n--- E. BTC Distance from 20-day High ---")
    regime_stats(df["btc_dd_from_high"] > -0.02, "Near high (0-2% off)")
    regime_stats((df["btc_dd_from_high"] <= -0.02) & (df["btc_dd_from_high"] > -0.05), "Small dip (2-5% off)")
    regime_stats((df["btc_dd_from_high"] <= -0.05) & (df["btc_dd_from_high"] > -0.10), "Pullback (5-10% off)")
    regime_stats((df["btc_dd_from_high"] <= -0.10) & (df["btc_dd_from_high"] > -0.20), "Correction (10-20% off)")
    regime_stats(df["btc_dd_from_high"] <= -0.20, "Crash (>20% off)")

    # F. SMA regime
    print("\n--- F. BTC vs 20-day SMA ---")
    regime_stats(df["btc_above_sma"] == 1, "Above SMA (bullish)")
    regime_stats(df["btc_above_sma"] == 0, "Below SMA (bearish)")

    # G. BTC daily move on same day
    print("\n--- G. BTC Same-Day Move (concurrent) ---")
    regime_stats(df["btc_ret_1d"] < -0.05, "BTC crashes >5%")
    regime_stats((df["btc_ret_1d"] >= -0.05) & (df["btc_ret_1d"] < -0.02), "BTC down 2-5%")
    regime_stats((df["btc_ret_1d"] >= -0.02) & (df["btc_ret_1d"] < 0), "BTC down 0-2%")
    regime_stats((df["btc_ret_1d"] >= 0) & (df["btc_ret_1d"] < 0.02), "BTC up 0-2%")
    regime_stats((df["btc_ret_1d"] >= 0.02) & (df["btc_ret_1d"] < 0.05), "BTC up 2-5%")
    regime_stats(df["btc_ret_1d"] >= 0.05, "BTC rallies >5%")

    # ---- WORST DAYS ----
    print("\n--- TOP 10 WORST DAYS ---")
    worst = df.nsmallest(10, "pnl")
    for dt, row in worst.iterrows():
        print(f"  {dt.strftime('%Y-%m-%d')}  PnL=${row['pnl']:>8.0f}  "
              f"BTC=${row['btc_close']:>8.0f}  "
              f"BTC_1d={row['btc_ret_1d']*100:>+5.1f}%  "
              f"BTC_7d={row['btc_ret_7d']*100:>+5.1f}%  "
              f"Vol={row['btc_vol_20d']*100:>5.0f}%")

    # ---- WORST WEEKS ----
    print("\n--- TOP 10 WORST WEEKS ---")
    weekly = dpnl.reindex(df.index).resample("W").sum()
    btc_weekly = btc_d.reindex(df.index).resample("W").last()
    btc_wret = btc_weekly.pct_change()
    worst_w = weekly.nsmallest(10)
    for dt in worst_w.index:
        br = btc_wret.get(dt, 0)
        bv = btc_vol_20d.reindex(df.index).resample("W").last().get(dt, 0)
        print(f"  {dt.strftime('%Y-%m-%d')}  PnL=${worst_w[dt]:>8.0f}  "
              f"BTC_week={br*100:>+5.1f}%  Vol={bv*100:>5.0f}%")

    # ---- MONTHLY BREAKDOWN ----
    print("\n--- MONTHLY PnL + BTC RETURN ---")
    monthly_pnl = dpnl.reindex(df.index).resample("ME").sum()
    btc_monthly = btc_d.reindex(df.index).resample("ME").last().pct_change()
    btc_vol_monthly = btc_vol_20d.reindex(df.index).resample("ME").mean()
    print(f"  {'Month':<10s} {'PnL':>8s} {'BTC%':>7s} {'Vol%':>6s}")
    for dt in monthly_pnl.index:
        pnl_v = monthly_pnl[dt]
        br = btc_monthly.get(dt, 0)
        bv = btc_vol_monthly.get(dt, 0)
        flag = " <<<" if pnl_v < 0 else ""
        print(f"  {dt.strftime('%Y-%m'):<10s} ${pnl_v:>7.0f} {br*100:>+6.1f}% {bv*100:>5.0f}%{flag}")

    # ---- CORRELATION ----
    print("\n--- CORRELATION: PnL vs Market Factors ---")
    for col in ["btc_ret_1d", "btc_ret_7d", "btc_ret_20d", "btc_vol_20d", "btc_dd_from_high"]:
        r = df["pnl"].corr(df[col])
        print(f"  PnL vs {col:<20s} rho={r:>+.3f}")

    # ---- CONDITIONAL EXPECTATION ----
    print("\n--- KEY INSIGHT: When does LogReg CONSISTENTLY lose? ---")
    # Find regimes where average PnL is negative AND Sharpe < -0.5
    conditions = [
        ("BTC crash day (>5% down)", df["btc_ret_1d"] < -0.05),
        ("BTC strong rally (>5% up)", df["btc_ret_1d"] >= 0.05),
        ("BTC 20d strong downtrend", df["btc_ret_20d"] < -0.15),
        ("BTC 20d strong uptrend", df["btc_ret_20d"] >= 0.15),
        ("Extreme vol (>120% ann)", df["btc_vol_20d"] >= 1.20),
        ("Very low vol (<30% ann)", df["btc_vol_20d"] < 0.30),
        ("Crash regime (>20% off high)", df["btc_dd_from_high"] <= -0.20),
        ("Below SMA20", df["btc_above_sma"] == 0),
        ("Above SMA20", df["btc_above_sma"] == 1),
        ("Vol expanding (top 20%ile)", df["btc_vol_pct"] >= 0.80),
        ("Vol compressing (bottom 20%ile)", df["btc_vol_pct"] < 0.20),
    ]
    print(f"  {'Regime':<40s} {'N':>4s} {'Avg$':>7s} {'SR':>6s} {'WR%':>5s} {'Verdict':>10s}")
    print("  " + "-" * 80)
    for label, mask in conditions:
        sub = df.loc[mask, "pnl"]
        if len(sub) < 3: continue
        sr = sub.mean() / sub.std() * np.sqrt(365.25) if sub.std() > 0 else 0
        wr = (sub > 0).sum() / len(sub)
        avg = sub.mean()
        verdict = "LOSES" if avg < 0 and sr < -0.5 else "WEAK" if avg < 0 else "OK" if sr < 1 else "STRONG"
        print(f"  {label:<40s} {len(sub):>4d} ${avg:>6.1f} {sr:>6.2f} {wr*100:>4.0f}%  {verdict:>10s}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
