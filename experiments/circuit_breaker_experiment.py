#!/usr/bin/env python
"""Per-Asset Circuit Breaker Experiment.

Tests 3 breaker approaches + retrain-skip variants on the Full5 combo:
  A. Exponential decay on losing streaks
  B. Drawdown velocity (rolling PnL window)
  C. Win rate z-score
  D. B/C + 24h skip (simulated retrain)

Usage:
    PYTHONPATH=src python experiments/circuit_breaker_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import *

SMOKE = "--smoke" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
} if not SMOKE else {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"}
OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
    "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
VIP_TIERS = [
    ("VIP0",0,0,2.00,5.50),("VIP1",100_000,10_000_000,1.80,4.00),
    ("VIP2",250_000,25_000_000,1.60,3.75),("VIP3",500_000,50_000_000,1.40,3.50),
    ("VIP4",1_000_000,100_000_000,1.20,3.20),("VIP5",2_000_000,250_000_000,1.00,3.20),
    ("Supreme",5_000_000,500_000_000,0.00,3.00),
]
SPREAD_BPS = 0.5; CAP = 250_000
NEEDED_TFS = [("15m","15min"),("1h","1h"),("3h","3h"),("4h","4h"),("24h","1D")]
FULL5 = [("15m",2),("1h",4),("3h",16),("4h",6),("24h",12)]
BASELINE_WR = 0.65  # observed ~65% win rate across timeframes

# Breaker variants: (name, type, params)
VARIANTS = [
    ("Baseline",        "none",  {}),
    # A: Exponential decay
    ("A1: decay50@3",   "streak", {"threshold": 3, "decay": 0.50}),
    ("A2: decay70@3",   "streak", {"threshold": 3, "decay": 0.70}),
    ("A3: decay50@5",   "streak", {"threshold": 5, "decay": 0.50}),
    ("A4: pause@3",     "streak", {"threshold": 3, "decay": 0.00}),
    # B: Drawdown velocity
    ("B1: 5t/-2%@0.25", "ddvel",  {"window": 5,  "threshold": 0.02, "reduction": 0.25}),
    ("B2: 5t/-1%@0.50", "ddvel",  {"window": 5,  "threshold": 0.01, "reduction": 0.50}),
    ("B3: 10t/-3%@0.25","ddvel",  {"window": 10, "threshold": 0.03, "reduction": 0.25}),
    ("B4: 3t/-1%@pause","ddvel",  {"window": 3,  "threshold": 0.01, "reduction": 0.00}),
    # C: Win rate z-score
    ("C1: 20t/z-2@0.25","zscore", {"lookback": 20, "z_thresh": -2.0, "reduction": 0.25}),
    ("C2: 20t/z-1.5@.5","zscore", {"lookback": 20, "z_thresh": -1.5, "reduction": 0.50}),
    ("C3: 10t/z-2@0.25","zscore", {"lookback": 10, "z_thresh": -2.0, "reduction": 0.25}),
    ("C4: 20t/z-2@paus","zscore", {"lookback": 20, "z_thresh": -2.0, "reduction": 0.00}),
    # D: Retrain-skip
    ("D1: B2+skip24h",  "ddvel_skip",  {"window": 5, "threshold": 0.01, "reduction": 0.50, "skip_ns": 86_400_000_000_000}),
    ("D2: C1+skip24h",  "zscore_skip", {"lookback": 20, "z_thresh": -2.0, "reduction": 0.25, "skip_ns": 86_400_000_000_000}),
]
if SMOKE:
    VARIANTS = [VARIANTS[0], VARIANTS[1], VARIANTS[5], VARIANTS[9], VARIANTS[13]]


def get_vip_tier(eq, v30):
    r = VIP_TIERS[0]
    for t in VIP_TIERS:
        if eq >= t[1] or v30 >= t[2]: r = t
    return r[0], r[3], r[4]


def _edge_from_probability(prob, sig):
    if np.isfinite(prob):
        return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0))
    return float(np.clip(abs(sig), 0.0, 1.0))


# ── Modified portfolio sim with circuit breaker ──

def simulate_with_breaker(signals_df, config, breaker_type, breaker_params):
    """simulate_portfolio clone with per-asset circuit breaker injection."""
    if signals_df is None or len(signals_df) < 10:
        return None

    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)

    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values if "asset" in sub.columns else np.full(n, "SINGLE")
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = sub["signal_strength"].values.astype(np.float64) if "signal_strength" in sub.columns else None
    probabilities = sub["probability"].values.astype(np.float64) if "probability" in sub.columns else np.full(n, np.nan)
    regime_mults = sub["regime_exposure_mult"].values.astype(np.float64) if "regime_exposure_mult" in sub.columns else np.ones(n)
    sizing_mults = sub["sizing_mult"].values.astype(np.float64) if "sizing_mult" in sub.columns else np.ones(n)
    hold_bars_arr = sub["hold_bars"].values.astype(np.int64) if "hold_bars" in sub.columns else np.full(n, 4, dtype=np.int64)
    strategies_arr = sub["strategy"].values if "strategy" in sub.columns else np.full(n, "default")

    equity_arr = np.empty(n); pnl_arr = np.zeros(n); dd_arr = np.empty(n)
    size_arr = np.zeros(n); skipped = np.ones(n, dtype=bool)
    equity = config.starting_capital; peak_equity = equity; halted = False
    active_positions = []
    unique_ts = pd.Series(timestamps).unique()
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}
    use_prob = signal_strengths is not None

    # Circuit breaker state (per-asset)
    asset_streak = defaultdict(int)
    asset_recent_pnl = defaultdict(lambda: deque(maxlen=breaker_params.get("window", 10)))
    asset_outcomes = defaultdict(lambda: deque(maxlen=breaker_params.get("lookback", 20)))
    asset_skip_until = defaultdict(int)  # skip until this ts index
    cb_triggers = 0  # count total triggers

    # Convert timestamp to ns for skip logic
    ts_ns = pd.to_datetime(pd.Series(timestamps)).values.astype(np.int64) if "skip_ns" in breaker_params else None

    for i in range(n):
        current_ts = timestamps[i]
        current_ts_idx = ts_to_idx.get(current_ts, 0)
        active_positions = [(e, sz, a, st) for e, sz, a, st in active_positions if e > current_ts_idx]
        current_exposure = sum(sz for _, sz, _, _ in active_positions)
        asset_exposure = {}; strategy_exposure = {}
        for _, sz, a, st in active_positions:
            asset_exposure[a] = asset_exposure.get(a, 0.0) + sz
            strategy_exposure[st] = strategy_exposure.get(st, 0.0) + sz

        if halted:
            equity_arr[i] = equity; dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            continue

        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= config.max_dd:
            halted = True; equity_arr[i] = equity; dd_arr[i] = dd; continue

        if dd > config.delever_start:
            size_mult = max(1.0 - (dd - config.delever_start) / (config.max_dd - config.delever_start), 0.0)
        else:
            size_mult = 1.0

        if use_prob:
            sig_abs = abs(signal_strengths[i])
            if config.sizing_mode == "probability_edge":
                sig_abs = _edge_from_probability(probabilities[i], sig_abs)
            if config.use_regime_exposure_multiplier:
                sig_abs *= float(np.clip(regime_mults[i], config.regime_exposure_floor, 1.0))
            raw_size = equity * config.base_sizing_pct * sig_abs * size_mult * sizing_mults[i]
            raw_size = min(raw_size, equity * config.base_sizing_pct * config.max_signal_mult * size_mult)
        else:
            raw_size = equity * config.base_sizing_pct * size_mult

        # ═══ CIRCUIT BREAKER ═══
        asset = assets_arr[i]

        # Skip-until check (D variants)
        if asset_skip_until[asset] > current_ts_idx:
            equity_arr[i] = equity; dd_arr[i] = dd
            continue

        cb_mult = 1.0
        if breaker_type == "streak":
            streak = asset_streak[asset]
            thr = breaker_params["threshold"]; decay = breaker_params["decay"]
            if streak >= thr:
                cb_mult = decay ** (streak - thr + 1)
                cb_triggers += 1
        elif breaker_type in ("ddvel", "ddvel_skip"):
            recent = asset_recent_pnl[asset]
            if len(recent) >= breaker_params.get("window", 5):
                pnl_sum = sum(recent)
                if pnl_sum / equity < -breaker_params["threshold"]:
                    cb_mult = breaker_params["reduction"]
                    cb_triggers += 1
                    if breaker_type == "ddvel_skip" and ts_ns is not None:
                        skip_end_ns = ts_ns[i] + breaker_params["skip_ns"]
                        # Find ts_idx for skip end
                        for k in range(i + 1, min(i + 500, n)):
                            if ts_ns[k] >= skip_end_ns:
                                asset_skip_until[asset] = ts_to_idx.get(timestamps[k], current_ts_idx)
                                break
        elif breaker_type in ("zscore", "zscore_skip"):
            outcomes = asset_outcomes[asset]
            lb = breaker_params["lookback"]
            if len(outcomes) >= lb:
                obs_wr = sum(outcomes) / len(outcomes)
                se = math.sqrt(BASELINE_WR * (1 - BASELINE_WR) / len(outcomes))
                z = (obs_wr - BASELINE_WR) / se if se > 0 else 0
                if z < breaker_params["z_thresh"]:
                    cb_mult = breaker_params["reduction"]
                    cb_triggers += 1
                    if breaker_type == "zscore_skip" and ts_ns is not None:
                        skip_end_ns = ts_ns[i] + breaker_params["skip_ns"]
                        for k in range(i + 1, min(i + 500, n)):
                            if ts_ns[k] >= skip_end_ns:
                                asset_skip_until[asset] = ts_to_idx.get(timestamps[k], current_ts_idx)
                                break

        raw_size *= cb_mult
        # ═══ END CIRCUIT BREAKER ═══

        # Exposure limits
        asset_exp = asset_exposure.get(asset, 0.0)
        if asset_exp + raw_size > equity * config.max_per_asset:
            raw_size = max(equity * config.max_per_asset - asset_exp, 0.0)
        strategy = strategies_arr[i]
        mps = getattr(config, "max_per_strategy", 1.0)
        strat_exp = strategy_exposure.get(strategy, 0.0)
        if mps < 1.0 and strat_exp + raw_size > equity * mps:
            raw_size = max(equity * mps - strat_exp, 0.0)
        if current_exposure + raw_size > equity * config.max_exposure:
            raw_size = max(equity * config.max_exposure - current_exposure, 0.0)

        if raw_size < 1.0:
            equity_arr[i] = equity; dd_arr[i] = dd; continue

        # Execute trade
        trade_return = directions[i] * fwd_returns[i] - config.slippage
        pnl = raw_size * trade_return
        equity += pnl
        if equity > peak_equity: peak_equity = equity

        expiry_ts_idx = current_ts_idx + int(hold_bars_arr[i])
        active_positions.append((expiry_ts_idx, raw_size, asset, strategy))
        equity_arr[i] = equity; pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        size_arr[i] = raw_size; skipped[i] = False

        # Update breaker state AFTER trade
        asset_streak[asset] = asset_streak[asset] + 1 if pnl < 0 else 0
        asset_recent_pnl[asset].append(pnl)
        asset_outcomes[asset].append(pnl >= 0)

    return {
        "equity": equity_arr, "pnl": pnl_arr, "dd": dd_arr,
        "sizes": size_arr, "skipped": skipped, "timestamps": timestamps,
        "halted": halted, "final_equity": equity,
        "n_trades": int((~skipped).sum()), "cb_triggers": cb_triggers,
    }


# ── Signal generation (same as other experiments) ──

def _gen_one(args):
    name, sym, tf_resample = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=0.10, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=4, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf_resample, regime_timeframe="4h"),
        assets=ASSETS)
    try: ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError: return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def generate_signals(label, tf_resample):
    tasks = [(n, s, tf_resample) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0: all_sigs.append(r)
    if not all_sigs: return None
    m = pd.concat(all_sigs, ignore_index=True)
    m["strategy"] = label; m["hold_bars"] = 4
    if "confidence" in m.columns: m = m[m["confidence"] >= 0.08].reset_index(drop=True)
    return m if len(m) > 0 else None


def recompute_returns(signals, tf_resample, hold_bars):
    if hold_bars == 4: return signals.copy()
    close_cache = {}
    for a in signals["asset"].unique():
        sym = ASSETS.get(a)
        if not sym: continue
        try:
            c = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), tf_resample)["close"]
            if c.index.tz is not None: c.index = c.index.tz_localize(None)
            close_cache[a] = c
        except: continue
    sig_ts = pd.to_datetime(signals["timestamp"])
    if sig_ts.dt.tz is not None: sig_ts = sig_ts.dt.tz_localize(None)
    out = signals.copy(); fwd = np.full(len(signals), np.nan)
    for a, close in close_cache.items():
        mask = (signals["asset"] == a).values; pos = np.where(mask)[0]
        idx = close.index.searchsorted(sig_ts.iloc[pos].values)
        for k, p in enumerate(pos):
            bi, ei = idx[k], idx[k] + hold_bars
            if 0 <= bi and ei < len(close):
                fwd[p] = (close.values[ei] - close.values[bi]) / close.values[bi]
    out["fwd_return"] = fwd; out["hold_bars"] = hold_bars
    return out.dropna(subset=["fwd_return"]).reset_index(drop=True)


def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        o = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if o.index.tz is not None: o.index = o.index.tz_localize(None)
        return name, o.pct_change().rolling(20).std()
    except: return name, None


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        a = out.iloc[i].get("asset", "")
        if a not in vol_series: continue
        vs = vol_series[a]; idx = vs.index.searchsorted(ts.iloc[i])
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


def run_variant(merged, variant_name, breaker_type, breaker_params):
    """Run one circuit breaker variant and return stats."""
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_with_breaker(prepared, port, breaker_type, breaker_params)
    if sim is None: return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    # MkrTkr60 VIP fee overlay
    mfp = 0.90; active = ~sim["skipped"]; sizes = sim["sizes"]
    ts_arr = pd.to_datetime(prepared["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(prepared)):
        if active[i]: daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = CAP + dpnl.cumsum()
    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        dd = d.date(); eq = eq_curve.get(d, CAP); v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = get_vip_tier(eq, v30)
        rt = mfp * 2 * mbps + (1 - mfp) * 2 * tbps + SPREAD_BPS
        extra = rt - 1.0
        nd = sum(1 for i in range(len(prepared)) if active[i] and dates_arr[i] == dd)
        avg = daily_vol.get(dd, 0) / max(nd * 2, 1)
        dpnl_adj[d] -= nd * avg * extra / 10000

    dr = dpnl_adj / CAP
    if dr.std() == 0 or len(dr) < 30: return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd_s = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()
    max_dd = float(dd_s.max() * 100)
    calmar = float(dr.mean() * 365.25 / (max_dd / 100)) if max_dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99

    eq_full = CAP + dpnl_adj.cumsum()
    daily_dd_pct = dpnl_adj / eq_full.shift(1).fillna(CAP) * 100
    max_daily_dd = float(daily_dd_pct.min())
    worst_day = daily_dd_pct.idxmin()
    worst_day_pnl = float(dpnl_adj[worst_day])

    monthly = dpnl_adj.resample("MS").sum()
    pos_months = int((monthly > 0).sum()); tot_months = len(monthly)

    return {
        "sr": sr, "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
        "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"],
        "cb_triggers": sim["cb_triggers"],
        "max_daily_dd": max_daily_dd, "worst_day": str(worst_day.date()),
        "worst_day_pnl": worst_day_pnl,
        "pnl_mo": pnl / max(tot_months, 1),
        "pos_months": pos_months, "tot_months": tot_months,
        "daily_pnl": dpnl_adj,
    }


def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 120)
    print(f"  PER-ASSET CIRCUIT BREAKER EXPERIMENT ({mode})")
    print(f"  {len(ASSETS)} assets, Full5 combo, {len(VARIANTS)} variants, MkrTkr60")
    print("=" * 120)

    # Signal gen
    vol_series = {n: v for n, v in (ProcessPoolExecutor(max_workers=8).map(_compute_vol_one, list(ASSETS.items()))) if v is not None}
    print(f"  Vol computed", flush=True)

    print(f"  Generating signals...", flush=True)
    signal_cache = {}
    for label, resample in NEEDED_TFS:
        t1 = time.time()
        sigs = generate_signals(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"    {label:<6} {len(sigs):>8,} ({time.time()-t1:.0f}s)", flush=True)

    # Build Full5 merged signals
    print(f"\n  Building Full5 combo...", flush=True)
    hold_cache = {}
    for tf, h in FULL5:
        if tf not in signal_cache: continue
        resample = dict(NEEDED_TFS)[tf]
        sigs_h = recompute_returns(signal_cache[tf], resample, h)
        hold_cache[(tf, h)] = apply_vol(sigs_h, vol_series)
    parts = [hold_cache[(tf, h)] for tf, h in FULL5 if (tf, h) in hold_cache]
    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"  Full5: {len(merged):,} signals")

    # Run all variants
    print(f"\n{'='*120}")
    print("  RESULTS")
    print(f"{'='*120}")
    print(f"\n  {'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'WorstDay$':>10} {'Calmar':>7} {'PF':>6} {'Trades':>7} {'CB Trg':>7} {'$/mo':>8}")
    print("  " + "-" * 115)

    all_results = {}
    for name, btype, bparams in VARIANTS:
        r = run_variant(merged, name, btype, bparams)
        if r is None:
            print(f"  {name:<22} FAIL"); continue
        all_results[name] = r
        base = " <-- BASE" if btype == "none" else ""
        print(f"  {name:<22} {r['sr']:>7.3f} {r['sortino']:>8.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% ${r['worst_day_pnl']:>8,.0f} {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} {r['n_trades']:>7,} {r['cb_triggers']:>7,} ${r['pnl_mo']:>7,.0f}{base}")

    if "Baseline" not in all_results:
        print("  ERROR: Baseline failed"); return

    # Ranking by Sharpe
    print(f"\n{'='*120}")
    print("  RANKING BY SHARPE (vs Baseline)")
    print(f"{'='*120}")
    base = all_results["Baseline"]
    ranked = sorted(all_results.items(), key=lambda x: x[1]["sr"], reverse=True)
    print(f"\n  {'#':>3} {'Variant':<22} {'Sharpe':>7} {'dSharpe':>8} {'PnL':>11} {'dPnL':>10}"
          f" {'MaxDD':>6} {'DayDD':>7} {'$/mo':>8}")
    print("  " + "-" * 90)
    for i, (name, r) in enumerate(ranked, 1):
        ds = r["sr"] - base["sr"]; dp = r["pnl"] - base["pnl"]
        mark = " <--" if name == "Baseline" else ""
        print(f"  {i:>3} {name:<22} {r['sr']:>7.3f} {ds:>+7.3f} ${r['pnl']:>9,.0f} ${dp:>+8,.0f}"
              f" {r['max_dd']:>5.1f}% {r['max_daily_dd']:>6.2f}% ${r['pnl_mo']:>7,.0f}{mark}")

    # Best variant deep dive
    best_name, best = ranked[0]
    if best_name != "Baseline":
        print(f"\n{'='*120}")
        print(f"  BEST VARIANT: {best_name}")
        print(f"{'='*120}")

        print(f"\n  {'Metric':<20} {'Baseline':>14} {best_name:>14} {'Delta':>10}")
        print("  " + "-" * 60)
        metrics = [
            ("Sharpe", base["sr"], best["sr"]),
            ("Sortino", base["sortino"], best["sortino"]),
            ("PnL", base["pnl"], best["pnl"]),
            ("MaxDD %", base["max_dd"], best["max_dd"]),
            ("Max Daily DD %", base["max_daily_dd"], best["max_daily_dd"]),
            ("Calmar", base["calmar"], best["calmar"]),
            ("PF", base["pf"], best["pf"]),
            ("Trades", base["n_trades"], best["n_trades"]),
            ("$/month", base["pnl_mo"], best["pnl_mo"]),
        ]
        for label, bv, rv in metrics:
            d = rv - bv
            print(f"  {label:<20} {bv:>14,.3f} {rv:>14,.3f} {d:>+10,.3f}")

        # Monthly comparison (last 12 months)
        print(f"\n  Monthly PnL (last 12 months):")
        bm = base["daily_pnl"].resample("MS").sum()
        rm = best["daily_pnl"].resample("MS").sum()
        for m in bm.index[-12:]:
            bv = bm.get(m, 0); rv = rm.get(m, 0)
            print(f"    {m.strftime('%Y-%m')}: ${bv:>10,.0f} -> ${rv:>10,.0f} (${rv-bv:>+8,.0f})")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
