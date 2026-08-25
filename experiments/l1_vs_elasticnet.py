#!/usr/bin/env python
"""L1 vs Elastic Net comparison on OHLCV-only features.

Runs the full production pipeline (30 assets, 15m+1h+4h+cs, per-asset vol,
depth scaling, VIP fees) with only the penalty changed: L1 vs Elastic Net.

Usage:
    PYTHONPATH=src python experiments/l1_vs_elasticnet.py
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm, wilcoxon

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
    name, sym, tf, fwd, C, penalty = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty=penalty, solver="saga", max_iter=1000, seed=42,
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


def run_sim(signals_prepared, starting_capital, label):
    """Run simulate_portfolio with VIP fees, return daily PnL series + stats."""
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
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
    if sim is None: return None, None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None, None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # VIP fee adjustment
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            d = dates_arr[i]
            daily_vol[d] = daily_vol.get(d, 0) + abs(sizes[i]) * 2
    vol_series = pd.Series(daily_vol).sort_index()
    vol_30d = vol_series.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()

    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, _ = get_vip_tier(eq, v30)
        rt_bps = 2 * maker_bps + SPREAD_BPS
        extra_bps = rt_bps - 1.0
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz_day = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        daily_extra = n_day * avg_sz_day * extra_bps / 10000
        dpnl_adj[d] -= daily_extra

    # Stats
    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30: return None, None

    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    monthly = dpnl_adj.resample("MS").sum()
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    pos_m = int((monthly > 0).sum())
    tot_m = len(monthly)

    # Bootstrap CI on Sharpe
    rng = np.random.RandomState(42)
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    stats = {
        "sr": sr, "ci": ci, "sortino": sortino, "dd": dd, "calmar": calmar,
        "pnl": pnl, "pf": pf, "pos_months": f"{pos_m}/{tot_m}",
        "n_trades": sim["n_trades"], "monthly": monthly,
    }
    return dpnl_adj, stats


def generate_all_signals(penalty_name, penalty_value):
    """Generate signals for all assets/timeframes with given penalty."""
    print(f"\n  Generating signals [{penalty_name}]...")
    strat_sigs = {}
    for sname, (tf, fwd, C) in [("15m", ("15min", 4, 0.10)),
                                   ("1h", ("1h", 4, 0.10)),
                                   ("4h", ("4h", 6, 0.20))]:
        tasks = [(n, s, tf, fwd, C, penalty_value) for n, s in ASSETS.items()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_gen, t): t for t in tasks}
            for f in as_completed(futs):
                r = f.result()
                if r is not None and len(r) > 0: all_sigs.append(r)
        if all_sigs:
            m = pd.concat(all_sigs, ignore_index=True)
            m["strategy"] = sname; m["hold_bars"] = fwd
            if "confidence" in m.columns:
                m = m[m["confidence"] >= 0.08].reset_index(drop=True)
            strat_sigs[sname] = m
            print(f"    {sname}: {len(m):,} signals")
    return strat_sigs


def main():
    T0 = time.time()
    print("=" * 80)
    print("  L1 (LASSO) vs ELASTIC NET — FULL PRODUCTION COMPARISON")
    print("  30 assets | 15m+1h+4h+cs | per-asset vol | depth | VIP fees")
    print("=" * 80)

    # Pre-compute shared data
    print("\nPre-computing vol + depth + cs signals...")
    vol_series = compute_per_asset_vol()
    ofi_data = {}
    for name, sym in ASSETS.items():
        p = L2_DIR / f"{sym}_micro_15m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            ofi_data[sym] = df[["ms_trade_imbalance", "ms_depth_imb_2pct"]]
    cs_path = Path("results/signals_cache/cs_momentum.parquet")
    scs = pd.read_parquet(cs_path) if cs_path.exists() else pd.DataFrame()

    # Generate signals for both variants
    variants = {
        "L1 (Lasso)": "l1",
        "Elastic Net": "elasticnet",
    }
    all_results = {}
    CAP = 250_000  # single capital level for clean comparison

    for vname, penalty in variants.items():
        strat_sigs = generate_all_signals(vname, penalty)
        parts = list(strat_sigs.values())
        if len(scs) > 0: parts.append(scs)
        combined = pd.concat(parts, ignore_index=True)
        combined = apply_per_asset_vol(combined, vol_series)
        combined = apply_depth_scale(combined, ofi_data)
        print(f"    Total: {len(combined):,} signals")

        dpnl, stats = run_sim(combined, CAP, vname)
        if stats is not None:
            all_results[vname] = {"dpnl": dpnl, "stats": stats}

    # ============================================================
    # Comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("  RESULTS COMPARISON — $250K Capital")
    print("=" * 80)

    header = f"  {'Metric':<22} {'L1 (Lasso)':>16} {'Elastic Net':>16} {'Diff':>12}"
    print(header)
    print("  " + "-" * 66)

    s1 = all_results.get("L1 (Lasso)", {}).get("stats")
    s2 = all_results.get("Elastic Net", {}).get("stats")

    if s1 and s2:
        rows = [
            ("Sharpe", f"{s1['sr']:.3f} {s1['ci']}", f"{s2['sr']:.3f} {s2['ci']}", f"{s2['sr']-s1['sr']:+.3f}"),
            ("Sortino", f"{s1['sortino']:.3f}", f"{s2['sortino']:.3f}", f"{s2['sortino']-s1['sortino']:+.3f}"),
            ("Total PnL", f"${s1['pnl']:,.0f}", f"${s2['pnl']:,.0f}", f"${s2['pnl']-s1['pnl']:+,.0f}"),
            ("Max Drawdown", f"{s1['dd']:.2f}%", f"{s2['dd']:.2f}%", f"{s2['dd']-s1['dd']:+.2f}%"),
            ("Calmar", f"{s1['calmar']:.2f}", f"{s2['calmar']:.2f}", f"{s2['calmar']-s1['calmar']:+.2f}"),
            ("Profit Factor", f"{s1['pf']:.2f}", f"{s2['pf']:.2f}", f"{s2['pf']-s1['pf']:+.2f}"),
            ("Pos Months", s1["pos_months"], s2["pos_months"], ""),
            ("Trades", f"{s1['n_trades']:,}", f"{s2['n_trades']:,}", f"{s2['n_trades']-s1['n_trades']:+,}"),
        ]
        for label, v1, v2, diff in rows:
            print(f"  {label:<22} {v1:>16} {v2:>16} {diff:>12}")

        # Monthly breakdown
        m1 = s1["monthly"]
        m2 = s2["monthly"]
        common = m1.index.intersection(m2.index)
        if len(common) > 0:
            print(f"\n  Monthly PnL Comparison:")
            print(f"  {'Month':<10} {'L1':>12} {'Elastic':>12} {'Diff':>12} {'Winner':>8}")
            print("  " + "-" * 54)
            l1_wins = 0; en_wins = 0
            for m in common:
                v1 = m1[m]; v2 = m2[m]
                diff = v2 - v1
                winner = "EN" if diff > 0 else "L1" if diff < 0 else "TIE"
                if diff > 0: en_wins += 1
                elif diff < 0: l1_wins += 1
                print(f"  {str(m.date()):<10} ${v1:>10,.0f} ${v2:>10,.0f} ${diff:>10,.0f} {winner:>8}")
            print(f"\n  Score: L1={l1_wins} vs EN={en_wins} (out of {len(common)} months)")

            # Wilcoxon signed-rank test on monthly PnL differences
            diffs = (m2[common] - m1[common]).values
            if len(diffs) >= 10 and np.any(diffs != 0):
                stat, p_val = wilcoxon(diffs)
                print(f"  Wilcoxon signed-rank test: p={p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'} at 5%)")

            # Year-by-year Sharpe
            d1 = all_results["L1 (Lasso)"]["dpnl"]
            d2 = all_results["Elastic Net"]["dpnl"]
            print(f"\n  Year-by-Year Sharpe:")
            print(f"  {'Year':<6} {'L1':>8} {'EN':>8} {'Diff':>8}")
            print("  " + "-" * 30)
            for year in sorted(d1.index.year.unique()):
                y1 = d1[d1.index.year == year] / CAP
                y2 = d2[d2.index.year == year] / CAP
                sr1 = float(y1.mean() / y1.std() * np.sqrt(365.25)) if y1.std() > 0 and len(y1) > 10 else 0
                sr2 = float(y2.mean() / y2.std() * np.sqrt(365.25)) if y2.std() > 0 and len(y2) > 10 else 0
                print(f"  {year:<6} {sr1:>8.3f} {sr2:>8.3f} {sr2-sr1:>+8.3f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
