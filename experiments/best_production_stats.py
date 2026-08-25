#!/usr/bin/env python
"""Full stats for the best production config:
15m+1h+4h+cs_momentum, per-asset vol, aggressive linear depth scaling, sz=4.5%

simulate_portfolio + Bybit VIP0 costs.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.config import *

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
L2_DIR = Path("data/l2-features")
ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}
OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

STRAT_CONFIGS = {
    "15m": ("15min", 4, 0.10, 0.08),
    "1h":  ("1h",    4, 0.10, 0.08),
    "4h":  ("4h",    6, 0.20, 0.08),
}


def _gen(args):
    name, sym, tf, fwd, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="l1", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf, regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def compute_per_asset_vol():
    vol_series = {}
    for name, sym in ASSETS.items():
        try:
            ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
            if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
            vol_series[name] = ohlcv.pct_change().rolling(20).std()
        except Exception:
            pass
    return vol_series


def apply_per_asset_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series: continue
        vs = vol_series[asset]
        t = ts.iloc[i]
        idx = vs.index.searchsorted(t)
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


def apply_depth_scale(signals, ofi_data):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        sym = asset + "USDT" if not asset.endswith("USDT") else asset
        if sym not in ofi_data or ofi_data[sym] is None: continue
        ofi_df = ofi_data[sym]
        t = ts.iloc[i]
        idx = ofi_df.index.searchsorted(t)
        if idx == 0 or idx > len(ofi_df): continue
        depth_val = ofi_df.iloc[idx - 1]["ms_depth_imb_2pct"]
        if pd.isna(depth_val): continue
        direction = out.iloc[i]["direction"]
        agreement = depth_val * direction
        # Aggressive linear scaling (best from sweep)
        mult = np.clip(0.3 + agreement * 8, 0.2, 2.0)
        out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
    return out


def main():
    T0 = time.time()
    print("=" * 80)
    print("  BEST PRODUCTION CONFIG — FULL STATS")
    print("  15m+1h+4h+cs, per-asset vol, aggressive depth scale, sz=4.5%")
    print("=" * 80)

    # Generate strategies
    print("\nGenerating strategies...")
    strat_sigs = {}
    for sname, (tf, fwd, C, conf) in STRAT_CONFIGS.items():
        tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for f in as_completed({pool.submit(_gen, t): t for t in tasks}):
                r = f.result()
                if r is not None and len(r) > 0: all_sigs.append(r)
        if all_sigs:
            m = pd.concat(all_sigs, ignore_index=True)
            m["strategy"] = sname; m["hold_bars"] = fwd
            if "confidence" in m.columns:
                m = m[m["confidence"] >= conf].reset_index(drop=True)
            strat_sigs[sname] = m
            print(f"  {sname}: {len(m):,} signals")

    scs = pd.read_parquet("results/signals_cache/cs_momentum.parquet")
    print(f"  cs_momentum: {len(scs):,} signals")

    combined = pd.concat(list(strat_sigs.values()) + [scs], ignore_index=True)

    # Load OFI data for depth scaling
    ofi_data = {}
    for name, sym in ASSETS.items():
        p = L2_DIR / f"{sym}_micro_15m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            ofi_data[sym] = df[["ms_trade_imbalance", "ms_depth_imb_2pct"]]

    vol_series = compute_per_asset_vol()
    print(f"  OFI: {len(ofi_data)} assets, Vol: {len(vol_series)} assets")

    # Apply per-asset vol + depth scaling
    print("\nApplying per-asset vol + aggressive depth scaling...")
    combined = apply_per_asset_vol(combined, vol_series)
    combined = apply_depth_scale(combined, ofi_data)

    # Prepare for portfolio sim
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in combined.columns: combined[c] = 1.0
        combined[c] = combined[c].fillna(1.0)
    combined["probability"] = combined["probability"].astype(float)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    port = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    # Simulate
    print("\nRunning simulate_portfolio...")
    sim = simulate_portfolio(combined, port)
    dpnl = compute_daily_pnl_series(sim)
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Bybit costs
    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    daily_cost = total_extra / max(len(dpnl), 1)
    dpnl = dpnl - daily_cost

    dr = dpnl / CAPITAL
    eq = (1 + dr).cumprod()
    dd_series = (eq.cummax() - eq) / eq.cummax()

    # === COMPREHENSIVE STATS ===
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    calmar = float(dr.mean() * 365.25 / dd_series.max()) if dd_series.max() > 0 else 0
    max_dd = float(dd_series.max() * 100)
    cagr = float((eq.iloc[-1]) ** (365.25 / len(dr)) - 1) * 100

    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    cvar1 = float(-dr[dr <= dr.quantile(0.01)].mean() * 10000) if len(dr) > 100 else 0
    var5 = float(-dr.quantile(0.05) * 10000)
    var1 = float(-dr.quantile(0.01) * 10000)
    skew = float(dr.skew())
    kurt = float(dr.kurtosis() + 3)

    win_days = int((dpnl > 0).sum())
    lose_days = int((dpnl < 0).sum())
    win_rate = win_days / max(win_days + lose_days, 1) * 100
    avg_win = float(dpnl[dpnl > 0].mean())
    avg_loss = float(dpnl[dpnl < 0].mean())
    pf = float(dpnl[dpnl > 0].sum() / abs(dpnl[dpnl < 0].sum())) if (dpnl < 0).any() else 99

    mx_lose = mx_win = cur_l = cur_w = 0
    for v in dpnl.values:
        if v < 0: cur_l += 1; mx_lose = max(mx_lose, cur_l); cur_w = 0
        else: cur_w += 1; mx_win = max(mx_win, cur_w); cur_l = 0

    monthly = dpnl.resample("MS").sum()
    weekly = dpnl.resample("W").sum()
    monthly_sr = float(monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() > 0 else 0

    dd_idx = dd_series.idxmax()
    post = eq.loc[dd_idx:]
    recovery = post[post >= eq.cummax().loc[dd_idx]]
    dd_recovery = (recovery.index[0] - dd_idx).days if len(recovery) > 0 else -1

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(5000)
    for b in range(5000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    # PSR
    sr_unann = sr / np.sqrt(365.25)
    denom_inner = (1 - skew * sr_unann + ((kurt - 1) / 4) * sr_unann ** 2) / max(len(dr) - 1, 1)
    psr = norm.cdf(sr_unann / np.sqrt(max(denom_inner, 1e-10))) * 100

    n_cal = (dpnl.index.max() - dpnl.index.min()).days
    years = n_cal / 365.25
    pos_m = int((monthly > 0).sum())
    neg_m = int((monthly < 0).sum())

    print(f"""
{'='*80}
  BEST PRODUCTION CONFIG — DETAILED STATS
  15m+1h+4h+cs | per-asset vol | aggressive depth scale | sz=4.5% | Bybit VIP0
{'='*80}

  PERIOD
    Calendar days:         {n_cal:,} ({years:.1f} years)
    Active trading days:   {len(dpnl):,}
    Trades executed:       {n_trades:,} ({n_trades/n_cal:.1f}/day)

  RETURNS
    Total PnL:             ${dpnl.sum():,.0f}
    CAGR:                  {cagr:.1f}%
    PnL / day:             ${dpnl.mean():,.1f}
    PnL / month:           ${monthly.mean():,.0f}
    Final equity:          ${CAPITAL + dpnl.sum():,.0f}

  RISK-ADJUSTED
    Sharpe Ratio:          {sr:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}] 95% CI
    Sortino Ratio:         {sortino:.3f}
    Calmar Ratio:          {calmar:.2f}
    Monthly Sharpe:        {monthly_sr:.3f}
    PSR (SR > 0):          {psr:.1f}%

  DRAWDOWN
    Max Drawdown:          {max_dd:.2f}%  (${CAPITAL * max_dd / 100:,.0f})
    DD Recovery:           {dd_recovery} days
    Current DD:            {dd_series.iloc[-1]*100:.2f}%

  TAIL RISK
    VaR 5% (daily):        {var5:.1f} bps
    CVaR 5% (daily):       {cvar5:.1f} bps
    VaR 1% (daily):        {var1:.1f} bps
    CVaR 1% (daily):       {cvar1:.1f} bps
    Skewness:              {skew:.3f}
    Kurtosis:              {kurt:.1f}

  DAILY
    Win rate:              {win_rate:.1f}%
    Avg winning day:       ${avg_win:,.0f}
    Avg losing day:        ${avg_loss:,.0f}
    Profit factor:         {pf:.2f}
    Best day:              ${dpnl.max():,.0f}
    Worst day:             ${dpnl.min():,.0f}
    Max win streak:        {mx_win} days
    Max lose streak:       {mx_lose} days

  WEEKLY
    Best week:             ${weekly.max():,.0f}
    Worst week:            ${weekly.min():,.0f}

  MONTHLY
    Positive months:       {pos_m}/{pos_m+neg_m} ({pos_m/(pos_m+neg_m)*100:.0f}%)
    Best month:            ${monthly.max():,.0f}
    Worst month:           ${monthly.min():,.0f}
    Avg month:             ${monthly.mean():,.0f}

  COSTS
    Total Bybit fees:      ${total_extra:,.0f} (${daily_cost:.0f}/day)
    Avg trade size:        ${avg_sz:,.0f}
    Fee drag:              {total_extra/(dpnl.sum()+total_extra)*100:.1f}% of gross PnL
""")

    # Year-by-year
    print("  YEAR-BY-YEAR")
    print(f"  {'Year':<6} {'SR':>7} {'Sortino':>8} {'DD':>6} {'PnL':>11} {'Days':>5}")
    print("  " + "-" * 44)
    for year in sorted(dpnl.index.year.unique()):
        y = dpnl[dpnl.index.year == year]
        if len(y) > 10:
            dr_y = y / CAPITAL
            sr_y = float(dr_y.mean() / dr_y.std() * np.sqrt(365.25)) if dr_y.std() > 0 else 0
            eq_y = (1 + dr_y).cumprod()
            dd_y = float(((eq_y.cummax() - eq_y) / eq_y.cummax()).max() * 100)
            down_y = dr_y[dr_y < 0]
            sort_y = float(dr_y.mean() / down_y.std() * np.sqrt(365.25)) if len(down_y) > 0 and down_y.std() > 0 else 0
            print(f"  {year:<6} {sr_y:>7.3f} {sort_y:>8.3f} {dd_y:>5.1f}% ${y.sum():>9,.0f} {len(y):>5}")

    # Per-strategy attribution
    print(f"\n  PER-STRATEGY")
    active = ~sim["skipped"]
    strats = combined["strategy"].values
    strat_pnl = {}
    strat_n = {}
    for i in range(len(combined)):
        if active[i]:
            s = strats[i]
            strat_pnl[s] = strat_pnl.get(s, 0) + sim["pnl"][i]
            strat_n[s] = strat_n.get(s, 0) + 1
    for s, p in sorted(strat_pnl.items(), key=lambda x: -x[1]):
        print(f"  {s:<15} PnL=${p:>10,.0f}  trades={strat_n.get(s,0):>7,}")

    # Per-asset top/bottom
    print(f"\n  PER-ASSET (top 10 / bottom 5)")
    assets_arr = combined["asset"].values if "asset" in combined.columns else np.full(len(combined), "UNK")
    asset_pnl = {}
    for i in range(len(combined)):
        if active[i]:
            a = assets_arr[i]
            asset_pnl[a] = asset_pnl.get(a, 0) + sim["pnl"][i]
    sorted_assets = sorted(asset_pnl.items(), key=lambda x: -x[1])
    print("  Top 10:")
    for a, p in sorted_assets[:10]:
        print(f"    {a:<8} ${p:>9,.0f}")
    print("  Bottom 5:")
    for a, p in sorted_assets[-5:]:
        print(f"    {a:<8} ${p:>9,.0f}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
