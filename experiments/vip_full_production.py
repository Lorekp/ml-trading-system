#!/usr/bin/env python
"""Full production sim with VIP tier progression.

Uses simulate_portfolio (exposure limits, per-asset caps, deleveraging) +
per-asset vol + aggressive depth scaling + dynamic Bybit VIP fees.

Tests: $100K, $250K, $500K, $1M starting capital.
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

VIP_TIERS = [
    ("VIP0",    0,           0,             2.00, 5.50),
    ("VIP1",    100_000,     10_000_000,    1.80, 4.00),
    ("VIP2",    250_000,     25_000_000,    1.60, 3.75),
    ("VIP3",    500_000,     50_000_000,    1.40, 3.50),
    ("VIP4",    1_000_000,   100_000_000,   1.20, 3.20),
    ("VIP5",    2_000_000,   250_000_000,   1.00, 3.20),
    ("Supreme", 5_000_000,   500_000_000,   0.00, 3.00),
]
SPREAD_BPS = 0.5


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


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
        mult = np.clip(0.3 + agreement * 8, 0.2, 2.0)
        out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
    return out


def run_with_vip(signals_prepared, starting_capital, label):
    """Run simulate_portfolio then apply dynamic VIP fees on daily PnL."""
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001,  # 1 bps idealized slippage in portfolio sim
        max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    merged = signals_prepared.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_portfolio(merged, port)
    if sim is None:
        print(f"  {label}: sim failed")
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print(f"  {label}: no PnL")
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # Compute daily trade volume from sim
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None:
        ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date

    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            d = dates_arr[i]
            daily_vol[d] = daily_vol.get(d, 0) + abs(sizes[i]) * 2  # roundtrip

    vol_series = pd.Series(daily_vol).sort_index()
    vol_30d = vol_series.rolling(30, min_periods=1).sum()

    # Compute equity curve (from portfolio sim, before extra Bybit fees)
    eq_curve = starting_capital + dpnl.cumsum()

    # Apply VIP-tiered fees day by day
    # Portfolio sim already applied 1 bps slippage. VIP extra = RT_bps - 1 bps (already in sim)
    # Actually, the portfolio sim applies config.slippage=0.0001 (1 bps) per trade.
    # Real Bybit cost = RT_bps. Extra cost = RT_bps - 1 bps (already accounted).
    # For VIP0: extra = 4.5 - 1.0 = 3.5 bps
    # For VIP3: extra = 3.3 - 1.0 = 2.3 bps

    dpnl_adj = dpnl.copy()
    tier_log = []
    prev_tier = "VIP0"

    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        tier_name, maker_bps, taker_bps = get_vip_tier(eq, v30)
        rt_bps = 2 * maker_bps + SPREAD_BPS
        extra_bps = rt_bps - 1.0  # subtract the 1 bps already in portfolio sim

        # Daily extra fee = n_trades_today * avg_size * extra_bps / 10000
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz_day = daily_vol.get(d_date, 0) / max(n_day * 2, 1)  # per side
        daily_extra = n_day * avg_sz_day * extra_bps / 10000
        dpnl_adj[d] -= daily_extra

        if tier_name != prev_tier:
            tier_log.append((d_date, tier_name, rt_bps))
            prev_tier = tier_name

    # Stats
    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30:
        return None

    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    worst_day = float(dr.min() * 100)
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    pnl = float(dpnl_adj.sum())
    final_eq = starting_capital + pnl
    cagr = float((eq_adj.iloc[-1]) ** (365.25 / len(dr)) - 1) * 100
    monthly = dpnl_adj.resample("MS").sum()
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    n_cal = (dpnl_adj.index.max() - dpnl_adj.index.min()).days

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    # PSR
    skew = float(dr.skew())
    kurt = float(dr.kurtosis() + 3)
    sr_unann = sr / np.sqrt(365.25)
    denom_inner = (1 - skew * sr_unann + ((kurt - 1) / 4) * sr_unann ** 2) / max(len(dr) - 1, 1)
    psr = norm.cdf(sr_unann / np.sqrt(max(denom_inner, 1e-10))) * 100

    pos_m = int((monthly > 0).sum())
    tot_m = len(monthly)

    print(f"""
{'='*80}
  {label} — FULL PRODUCTION STATS
  15m+1h+4h+cs | per-asset vol | depth scale | VIP progression | Bybit
{'='*80}

  Starting Capital:      ${starting_capital:,.0f}
  Final Equity:          ${final_eq:,.0f}
  Total PnL:             ${pnl:,.0f}
  CAGR:                  {cagr:.1f}%
  PnL / month:           ${monthly.mean():,.0f}

  Sharpe Ratio:          {sr:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}] 95% CI
  Sortino Ratio:         {sortino:.3f}
  Calmar Ratio:          {calmar:.2f}
  PSR:                   {psr:.1f}%
  Profit Factor:         {pf:.2f}

  Max Drawdown:          {dd:.2f}%
  Worst Day:             {worst_day:.2f}%
  CVaR 5%:               {cvar5:.0f} bps
  Positive Months:       {pos_m}/{tot_m} ({pos_m/tot_m*100:.0f}%)

  Trades:                {sim['n_trades']:,}""")

    # Tier progression
    if tier_log:
        print(f"\n  VIP Tier Progression:")
        for d, t, rt in tier_log:
            print(f"    {d}: {t} (RT={rt:.1f} bps)")

    # Year-by-year
    print(f"\n  Year-by-Year:")
    print(f"  {'Year':<6} {'SR':>7} {'Sortino':>8} {'DD':>6} {'PnL':>12} {'Tier':>8}")
    print("  " + "-" * 48)
    for year in sorted(dpnl_adj.index.year.unique()):
        y = dpnl_adj[dpnl_adj.index.year == year]
        if len(y) > 10:
            dr_y = y / starting_capital
            sr_y = float(dr_y.mean() / dr_y.std() * np.sqrt(365.25)) if dr_y.std() > 0 else 0
            eq_y = (1 + dr_y).cumprod()
            dd_y = float(((eq_y.cummax() - eq_y) / eq_y.cummax()).max() * 100)
            down_y = dr_y[dr_y < 0]
            sort_y = float(dr_y.mean() / down_y.std() * np.sqrt(365.25)) if len(down_y) > 0 and down_y.std() > 0 else 0
            # Find tier at end of year
            year_end = y.index[-1].date()
            end_tier = "VIP0"
            for d, t, rt in tier_log:
                if d <= year_end:
                    end_tier = t
            print(f"  {year:<6} {sr_y:>7.3f} {sort_y:>8.3f} {dd_y:>5.1f}% ${y.sum():>10,.0f} {end_tier:>8}")

    return {"sr": sr, "ci": ci, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "final_eq": final_eq, "cagr": cagr, "psr": psr}


def main():
    T0 = time.time()
    print("=" * 80)
    print("  FULL PRODUCTION SIM + VIP TIER PROGRESSION")
    print("=" * 80)

    # Generate strategies
    print("\nGenerating strategies...")
    strat_sigs = {}
    for sname, (tf, fwd, C, conf) in [("15m", ("15min", 4, 0.10, 0.08)),
                                         ("1h", ("1h", 4, 0.10, 0.08)),
                                         ("4h", ("4h", 6, 0.20, 0.08))]:
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
            print(f"  {sname}: {len(m):,}")

    scs = pd.read_parquet("results/signals_cache/cs_momentum.parquet")
    combined = pd.concat(list(strat_sigs.values()) + [scs], ignore_index=True)

    # Apply per-asset vol + depth scaling
    print("\nApplying per-asset vol + depth scaling...")
    ofi_data = {}
    for name, sym in ASSETS.items():
        p = L2_DIR / f"{sym}_micro_15m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            ofi_data[sym] = df[["ms_trade_imbalance", "ms_depth_imb_2pct"]]
    vol_series = compute_per_asset_vol()
    combined = apply_per_asset_vol(combined, vol_series)
    combined = apply_depth_scale(combined, ofi_data)
    print(f"  Prepared: {len(combined):,} signals")

    # Run at different capitals
    for cap in [100_000, 250_000, 500_000, 1_000_000]:
        run_with_vip(combined, cap, f"${cap//1000}K")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
