#!/usr/bin/env python
"""Close Model Experiment — Parametric, ML, and Hybrid Exit Strategies.

Tests whether dynamic exits beat the static T+4 hold. Uses the production
signal pipeline (30 assets, 15m+1h+4h, EN, vol scaling, MkrTkr60 costs).

Variants:
  a. Baseline T+4 (static hold)
  b. Triple Barrier (SL+TP+max hold, ATR-scaled)
  c. Trailing Stop (ATR-based)
  d. Time Decay (cut losers at T+2, let winners run to T+6)
  e. Adaptive Hold (confidence-scaled 2-8 bars)
  f. LogReg Close Model (ML, 15m only)
  g. XGBoost Close Model (ML, 15m only)
  h. Partial Ladder (50% at TP, trail rest)
  i. ML + Safety Stop (XGBoost + hard 2-ATR SL)

Usage:
    PYTHONPATH=src python experiments/close_model_experiment.py [--smoke] [--skip-ml]
"""
import sys, os, time, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.exits import compute_path_exits, compute_atr
from signal_engine.config import *

SMOKE = "--smoke" in sys.argv
SKIP_ML = "--skip-ml" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

ALL_ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}
ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"} if SMOKE else ALL_ASSETS

OHLCV_FEATS = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
               "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

VIP_TIERS = [
    ("VIP0",0,0,2.00,5.50),("VIP1",100_000,10_000_000,1.80,4.00),
    ("VIP2",250_000,25_000_000,1.60,3.75),("VIP3",500_000,50_000_000,1.40,3.50),
    ("VIP4",1_000_000,100_000_000,1.20,3.20),("VIP5",2_000_000,250_000_000,1.00,3.20),
    ("Supreme",5_000_000,500_000_000,0.00,3.00),
]
SPREAD_BPS = 0.5
CAP = 250_000

TF_MAP = {"15m": "15min", "1h": "1h", "4h": "4h"}


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]: result = tier
    return result[0], result[3], result[4]


# ── Signal generation (identical to execution_strategy_ranking) ──

def _gen(args):
    name, sym, tf, fwd, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATS), winsor_lo=0.01, winsor_hi=0.99,
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


def generate_all_signals():
    strat_sigs = {}
    tfs = [("15m","15min",4,0.10)] if SMOKE else [
        ("15m","15min",4,0.10),("1h","1h",4,0.10),("4h","4h",6,0.20)]
    for sname, tf, fwd, C in tfs:
        tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
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
            print(f"    {sname}: {len(m):,} signals", flush=True)
    return strat_sigs


# ── Vol scaling ──

def _vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series: continue
        vs = vol_series[asset]
        idx = vs.index.searchsorted(ts.iloc[i])
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


# ── Portfolio sim with MkrTkr60 VIP fees ──

def run_sim(signals_prepared, starting_capital=CAP):
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
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    # MkrTkr60 VIP fee overlay (90% maker, 10% taker)
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()
    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, taker_bps = get_vip_tier(eq, v30)
        rt_bps = 0.90 * 2 * maker_bps + 0.10 * 2 * taker_bps + SPREAD_BPS
        extra = rt_bps - 1.0
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        dpnl_adj[d] -= n_day * avg_sz * extra / 10000

    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30: return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    monthly = dpnl_adj.resample("MS").sum()
    pos_m = int((monthly > 0).sum()); tot_m = len(monthly)
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"],
            "monthly": monthly, "pos_months": f"{pos_m}/{tot_m}"}


# ── OHLCV loading (per-asset, per-timeframe) ──

def load_ohlcv_dict(assets, timeframe):
    """Load OHLCV for all assets at a given timeframe. Returns {asset_name: ohlcv_df}."""
    result = {}
    for name, sym in assets.items():
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            ohlcv_tf = resample_ohlcv(ohlcv_1m, timeframe)
            if ohlcv_tf.index.tz is not None:
                ohlcv_tf.index = ohlcv_tf.index.tz_localize(None)
            result[name] = ohlcv_tf
        except Exception:
            pass
    return result


# ── Custom parametric exits (TimeDecay + PartialLadder) ──

def apply_time_decay(signals_df, ohlcv_dict, cut_bar=2, run_bar=6):
    """Cut losers at cut_bar, let winners run to run_bar."""
    out = signals_df.copy()
    new_ret = out["fwd_return"].values.copy()
    new_hold = out["hold_bars"].values.copy()
    ts_arr = out["timestamp"].values
    dirs = out["direction"].values.astype(float)
    assets = out["asset"].values

    for asset_name in out["asset"].unique():
        if asset_name not in ohlcv_dict: continue
        ohlcv = ohlcv_dict[asset_name]
        ohlcv_idx = ohlcv.index
        ts_map = {t: i for i, t in enumerate(ohlcv_idx)}
        mask = assets == asset_name

        for i in np.where(mask)[0]:
            ts = pd.Timestamp(ts_arr[i])
            if ts.tzinfo is not None: ts = ts.tz_localize(None)
            idx = ts_map.get(ts)
            if idx is None:
                idx = ohlcv_idx.searchsorted(ts)
                if idx >= len(ohlcv): continue
            entry_px = float(ohlcv["close"].iat[idx])
            if entry_px <= 0: continue

            # Check at cut_bar
            check_idx = idx + cut_bar
            if check_idx >= len(ohlcv): continue
            check_px = float(ohlcv["close"].iat[check_idx])
            unreal = dirs[i] * (check_px - entry_px) / entry_px

            if unreal < 0:
                # Losing → cut early
                new_ret[i] = (check_px - entry_px) / entry_px
                new_hold[i] = cut_bar
            else:
                # Winning → let run
                run_idx = min(idx + run_bar, len(ohlcv) - 1)
                run_px = float(ohlcv["close"].iat[run_idx])
                new_ret[i] = (run_px - entry_px) / entry_px
                new_hold[i] = min(run_bar, run_idx - idx)

    out["fwd_return"] = new_ret
    out["hold_bars"] = new_hold
    valid = np.isfinite(out["fwd_return"].values)
    return out[valid].reset_index(drop=True)


def apply_partial_ladder(signals_df, ohlcv_dict, tp_atr_mult=1.5, trail_atr_mult=1.0,
                         max_bars=4):
    """50% exit at TP, trail rest with trailing stop."""
    out = signals_df.copy()
    new_ret = out["fwd_return"].values.copy()
    new_hold = out["hold_bars"].values.copy()
    ts_arr = out["timestamp"].values
    dirs = out["direction"].values.astype(float)
    assets = out["asset"].values

    atr_cache = {}
    for name in out["asset"].unique():
        if name in ohlcv_dict:
            atr_cache[name] = compute_atr(ohlcv_dict[name])

    for asset_name in out["asset"].unique():
        if asset_name not in ohlcv_dict: continue
        ohlcv = ohlcv_dict[asset_name]
        ohlcv_idx = ohlcv.index
        ts_map = {t: i for i, t in enumerate(ohlcv_idx)}
        atr_s = atr_cache.get(asset_name)
        mask = assets == asset_name

        for i in np.where(mask)[0]:
            ts = pd.Timestamp(ts_arr[i])
            if ts.tzinfo is not None: ts = ts.tz_localize(None)
            idx = ts_map.get(ts)
            if idx is None:
                idx = ohlcv_idx.searchsorted(ts)
                if idx >= len(ohlcv): continue
            entry_px = float(ohlcv["close"].iat[idx])
            if entry_px <= 0: continue
            atr_val = float(atr_s.iat[idx]) if atr_s is not None and idx < len(atr_s) else np.nan
            if not np.isfinite(atr_val) or atr_val <= 0: continue
            d = dirs[i]

            tp_dist = tp_atr_mult * atr_val
            tp_px = entry_px + tp_dist * d

            # Phase 1: walk bars looking for TP
            tp_bar = None
            for j in range(1, max_bars + 1):
                bi = idx + j
                if bi >= len(ohlcv): break
                if d > 0 and float(ohlcv["high"].iat[bi]) >= tp_px:
                    tp_bar = j; break
                if d < 0 and float(ohlcv["low"].iat[bi]) <= tp_px:
                    tp_bar = j; break

            if tp_bar is not None:
                tp_ret = (tp_px - entry_px) / entry_px  # 50% exits here

                # Phase 2: trail rest from tp_bar to max_bars
                trail_dist = trail_atr_mult * atr_val
                peak = tp_px
                trail_exit_px = None
                final_bar = tp_bar
                for j in range(tp_bar + 1, max_bars + 1):
                    bi = idx + j
                    if bi >= len(ohlcv): break
                    if d > 0:
                        peak = max(peak, float(ohlcv["high"].iat[bi]))
                        trail_stop = peak - trail_dist
                        if float(ohlcv["low"].iat[bi]) <= trail_stop:
                            trail_exit_px = trail_stop; final_bar = j; break
                    else:
                        peak = min(peak, float(ohlcv["low"].iat[bi]))
                        trail_stop = peak + trail_dist
                        if float(ohlcv["high"].iat[bi]) >= trail_stop:
                            trail_exit_px = trail_stop; final_bar = j; break
                    final_bar = j

                if trail_exit_px is None:
                    # Trail didn't trigger → exit at close of last bar
                    end_idx = min(idx + max_bars, len(ohlcv) - 1)
                    trail_exit_px = float(ohlcv["close"].iat[end_idx])
                trail_ret = (trail_exit_px - entry_px) / entry_px

                new_ret[i] = 0.5 * tp_ret + 0.5 * trail_ret
                new_hold[i] = final_bar
            else:
                # No TP hit → standard hold to max_bars
                end_idx = min(idx + max_bars, len(ohlcv) - 1)
                new_ret[i] = (float(ohlcv["close"].iat[end_idx]) - entry_px) / entry_px
                new_hold[i] = min(max_bars, end_idx - idx)

    out["fwd_return"] = new_ret
    out["hold_bars"] = new_hold
    valid = np.isfinite(out["fwd_return"].values) & (out["hold_bars"].values > 0)
    return out[valid].reset_index(drop=True)


# ── Main ──

def main():
    T0 = time.time()
    print("=" * 80)
    print("  CLOSE MODEL EXPERIMENT — Parametric, ML, Hybrid Exits")
    print("  Production pipeline, MkrTkr60 costs, VIP progression")
    print("=" * 80)

    # Phase 1: Generate signals
    print("\n  Generating signals...", flush=True)
    t0 = time.time()
    strat_sigs = generate_all_signals()
    combined = pd.concat(list(strat_sigs.values()), ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    print(f"  Total: {len(combined):,} signals ({time.time()-t0:.0f}s)")

    # Vol scaling
    print("  Computing vol scaling...", flush=True)
    vol_series = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_vol_one, list(ASSETS.items())):
            if vs is not None: vol_series[name] = vs
    combined = apply_vol(combined, vol_series)

    # Phase 2: Load OHLCV for exit computation
    print("  Loading OHLCV for exit computation...", flush=True)
    ohlcv_by_tf = {}
    for strat_name, tf_str in TF_MAP.items():
        if strat_name in [s["strategy"].iloc[0] for _, s in strat_sigs.items() if len(s) > 0]:
            continue
    # Load all timeframes used
    for tf_label in strat_sigs.keys():
        tf_str = TF_MAP[tf_label]
        print(f"    Loading {tf_str} OHLCV...", flush=True)
        ohlcv_by_tf[tf_label] = load_ohlcv_dict(ASSETS, tf_str)

    # Phase 3: Apply close strategies
    print(f"\n{'='*80}")
    print("  APPLYING CLOSE STRATEGIES")
    print(f"{'='*80}")

    variants = {}

    # a. Baseline T+4
    variants["Baseline T+4"] = combined.copy()
    print("  [a] Baseline T+4: using original signals", flush=True)

    # Helper: apply exits per-timeframe then recombine
    def _apply_exits_per_tf(exit_cfg, label):
        parts = []
        for tf_label, sigs in strat_sigs.items():
            if tf_label not in ohlcv_by_tf: continue
            sigs_copy = apply_vol(sigs.copy(), vol_series)
            result = compute_path_exits(sigs_copy, ohlcv_by_tf[tf_label], exit_cfg)
            if "bars_held" in result.columns:
                result["hold_bars"] = result["bars_held"]
            parts.append(result)
        if parts:
            out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            print(f"  [{label}] {len(out):,} signals, avg hold={out['hold_bars'].mean():.1f} bars",
                  flush=True)
            return out
        return None

    # b1. Triple Barrier (SL=1.0, TP=1.5 ATR)
    r = _apply_exits_per_tf(ExitConfig(max_hold_bars=4, stop_loss_atr_mult=1.0,
                                        take_profit_atr_mult=1.5), "b1")
    if r is not None: variants["TriBarrier(1.0/1.5)"] = r

    # b2. Triple Barrier (SL=1.5, TP=2.0 ATR)
    r = _apply_exits_per_tf(ExitConfig(max_hold_bars=4, stop_loss_atr_mult=1.5,
                                        take_profit_atr_mult=2.0), "b2")
    if r is not None: variants["TriBarrier(1.5/2.0)"] = r

    # c. Trailing Stop (1.0 ATR)
    r = _apply_exits_per_tf(ExitConfig(max_hold_bars=4, trail_atr_mult=1.0,
                                        trail_min_profit_pct=0.001), "c")
    if r is not None: variants["TrailStop(1.0)"] = r

    # d. Time Decay (cut T+2, run T+6)
    print("  [d] Time Decay...", flush=True)
    parts_d = []
    for tf_label, sigs in strat_sigs.items():
        if tf_label not in ohlcv_by_tf: continue
        sigs_copy = apply_vol(sigs.copy(), vol_series)
        fwd = 4 if tf_label != "4h" else 6
        r = apply_time_decay(sigs_copy, ohlcv_by_tf[tf_label], cut_bar=2,
                             run_bar=fwd + 2)
        parts_d.append(r)
    if parts_d:
        td = pd.concat(parts_d, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        print(f"  [d] {len(td):,} signals, avg hold={td['hold_bars'].mean():.1f} bars", flush=True)
        variants["TimeDecay(2/6)"] = td

    # e. Adaptive Hold (confidence-scaled 2-8 bars)
    r = _apply_exits_per_tf(ExitConfig(max_hold_bars=4, adaptive_hold=True,
                                        adaptive_hold_extra=4), "e")
    if r is not None: variants["AdaptiveHold(2-8)"] = r

    # h. Partial Ladder (50% at 1.5 ATR TP, trail rest)
    print("  [h] Partial Ladder...", flush=True)
    parts_h = []
    for tf_label, sigs in strat_sigs.items():
        if tf_label not in ohlcv_by_tf: continue
        sigs_copy = apply_vol(sigs.copy(), vol_series)
        fwd = 4 if tf_label != "4h" else 6
        r = apply_partial_ladder(sigs_copy, ohlcv_by_tf[tf_label],
                                 tp_atr_mult=1.5, trail_atr_mult=1.0, max_bars=fwd)
        parts_h.append(r)
    if parts_h:
        pl = pd.concat(parts_h, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        print(f"  [h] {len(pl):,} signals, avg hold={pl['hold_bars'].mean():.1f} bars", flush=True)
        variants["PartialLadder"] = pl

    # ML variants (skip in smoke mode or with --skip-ml)
    if not SMOKE and not SKIP_ML:
        print("\n  ML close models (15m only, may take 10-20 min each)...", flush=True)
        try:
            from signal_engine.close_integration import compute_close_adjusted_signals
            feature_config = FeatureConfig(feature_list=list(OHLCV_FEATS), winsor_lo=0.01,
                                           winsor_hi=0.99, vol_filter_quantile=0.50,
                                           vol_zscore_window=30, swing_lookback=20)
            port_config = PortfolioConfig(
                starting_capital=CAP, max_dd=0.10, delever_start=0.05,
                slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
                max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
                sizing_mode="probability_edge", use_regime_exposure_multiplier=True,
                regime_exposure_floor=0.50)

            for model_type, label in [("logistic", "f"), ("xgboost", "g")]:
                print(f"  [{label}] {model_type} close model...", flush=True)
                close_cfg = CloseModelConfig(
                    enabled=True, model_type=model_type,
                    label_mode="residual_triple_barrier",
                    de_risk_mode="binary_exit",
                    pt_mult=1.5, sl_mult=1.0, min_improvement_bps=2.0)

                # Apply ML close to 15m signals only
                ml_15m = compute_close_adjusted_signals(
                    combined[combined["strategy"] == "15m"].copy(),
                    "logreg_15m", ASSETS, feature_config, close_cfg, port_config)
                ml_15m = apply_vol(ml_15m, vol_series)

                # Combine with unmodified 1h/4h signals
                other = combined[combined["strategy"] != "15m"].copy()
                ml_combined = pd.concat([ml_15m, other], ignore_index=True)
                ml_combined = ml_combined.sort_values("timestamp").reset_index(drop=True)

                name = f"{'LogReg' if model_type == 'logistic' else 'XGBoost'}Close"
                print(f"  [{label}] {name}: {len(ml_combined):,} signals", flush=True)
                variants[name] = ml_combined

                # i. ML + Safety Stop (XGBoost only)
                if model_type == "xgboost":
                    print("  [i] ML + Safety Stop (2 ATR)...", flush=True)
                    # Post-process: enforce 2-ATR hard SL on ML-adjusted 15m signals
                    safety_cfg = ExitConfig(max_hold_bars=8, stop_loss_atr_mult=2.0)
                    ml_safe = compute_path_exits(ml_15m.copy(), ohlcv_by_tf.get("15m", {}),
                                                 safety_cfg)
                    if "bars_held" in ml_safe.columns:
                        ml_safe["hold_bars"] = ml_safe["bars_held"]
                    ml_safe = apply_vol(ml_safe, vol_series)
                    safe_combined = pd.concat([ml_safe, other], ignore_index=True)
                    safe_combined = safe_combined.sort_values("timestamp").reset_index(drop=True)
                    variants["XGB+SafetyStop"] = safe_combined
                    print(f"  [i] XGB+SafetyStop: {len(safe_combined):,} signals", flush=True)

        except Exception as e:
            print(f"  ML close model failed: {e}", flush=True)
    else:
        print("  Skipping ML variants (--smoke or --skip-ml)", flush=True)

    # Phase 4: Evaluate all variants
    print(f"\n{'='*80}")
    print("  RESULTS")
    print(f"{'='*80}")

    results = {}
    for name, sigs in variants.items():
        s = run_sim(sigs)
        if s is not None:
            # Compute avg hold bars
            avg_hold = sigs["hold_bars"].mean() if "hold_bars" in sigs.columns else 4.0
            s["avg_hold"] = round(avg_hold, 1)
            results[name] = s

    # Print comparison table
    baseline = results.get("Baseline T+4")
    ranked = sorted(results.items(), key=lambda x: x[1]["sr"], reverse=True)

    print(f"\n  {'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>7}"
          f" {'Calmar':>7} {'PF':>5} {'AvgH':>5} {'Pos Mo':>7}")
    print("  " + "-" * 82)
    for name, s in ranked:
        tag = " *" if name == "Baseline T+4" else ""
        print(f"  {name:<22} {s['sr']:>7.3f} {s['sortino']:>8.3f} ${s['pnl']:>9,.0f}"
              f" {s['dd']:>6.2f}% {s['calmar']:>7.2f} {s['pf']:>5.2f}"
              f" {s['avg_hold']:>5.1f} {s['pos_months']:>7}{tag}")

    # Wilcoxon vs baseline
    if baseline:
        base_monthly = baseline["monthly"]
        print(f"\n  vs Baseline T+4 (monthly Wilcoxon):")
        for name, s in ranked:
            if name == "Baseline T+4": continue
            monthly = s["monthly"]
            common = base_monthly.index.intersection(monthly.index)
            if len(common) < 10: continue
            diffs = (monthly[common] - base_monthly[common]).values
            if np.all(diffs == 0):
                print(f"    {name:<22}: identical"); continue
            try:
                stat, p = wilcoxon(diffs)
                wins = (diffs > 0).sum()
                sig = "SIG" if p < 0.05 else "ns"
                delta_sr = s["sr"] - baseline["sr"]
                print(f"    {name:<22}: dSR={delta_sr:>+.3f} W={wins:>2}L={len(common)-wins:>2}/{len(common)}"
                      f"  p={p:.4f} ({sig})")
            except Exception:
                print(f"    {name:<22}: test failed")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
