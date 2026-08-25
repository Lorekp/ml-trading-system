#!/usr/bin/env python
"""Circuit Breaker V2 — Dynamic breakers + 24h early exit.

New approaches:
  E. EMA win rate sizing (no hard reset)
  F. PnL-weighted tracking (rolling PnL / rolling notional)
  G. 24h early exit on 4h forecast (check hourly, exit if strongly unfavorable)
  H. 24h early exit on unrealized loss (check hourly, exit if down X%)

All tested on Full5 combo. Signals loaded from disk cache.

Usage:
    PYTHONPATH=src python experiments/circuit_breaker_v2_experiment.py [--smoke]
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
OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
    "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
VIP_TIERS = [
    ("VIP0",0,0,2.00,5.50),("VIP1",100_000,10_000_000,1.80,4.00),
    ("VIP2",250_000,25_000_000,1.60,3.75),("VIP3",500_000,50_000_000,1.40,3.50),
    ("VIP4",1_000_000,100_000_000,1.20,3.20),("VIP5",2_000_000,250_000_000,1.00,3.20),
    ("Supreme",5_000_000,500_000_000,0.00,3.00),
]
SPREAD_BPS = 0.5; CAP = 250_000; BASELINE_WR = 0.65
NEEDED_TFS = [("15m","15min"),("1h","1h"),("3h","3h"),("4h","4h"),("24h","1D")]
FULL5 = [("15m",2),("1h",4),("3h",16),("4h",6),("24h",12)]
CACHE_DIR = Path("experiments/.cache")

VARIANTS = [
    ("Baseline",              "none",      {}),
    ("A1: decay50@3",         "streak",    {"threshold": 3, "decay": 0.50}),
    # E: EMA win rate
    ("E1: ema a=0.90",       "ema",       {"alpha": 0.90, "floor": 0.10}),
    ("E2: ema a=0.85",       "ema",       {"alpha": 0.85, "floor": 0.10}),
    ("E3: ema a=0.95",       "ema",       {"alpha": 0.95, "floor": 0.10}),
    ("E4: ema a=0.90 fl=0",  "ema",       {"alpha": 0.90, "floor": 0.00}),
    # F: PnL-weighted
    ("F1: pnl10 t=-1%",      "pnl_ratio", {"window": 10, "threshold": -0.01, "floor": 0.10}),
    ("F2: pnl20 t=-1%",      "pnl_ratio", {"window": 20, "threshold": -0.01, "floor": 0.10}),
    ("F3: pnl10 t=-0.5%",    "pnl_ratio", {"window": 10, "threshold": -0.005, "floor": 0.10}),
    ("F4: pnl10 t=-1% fl=0", "pnl_ratio", {"window": 10, "threshold": -0.01, "floor": 0.00}),
]
if SMOKE:
    VARIANTS = [VARIANTS[0], VARIANTS[1], VARIANTS[2], VARIANTS[6]]


def get_vip_tier(eq, v30):
    r = VIP_TIERS[0]
    for t in VIP_TIERS:
        if eq >= t[1] or v30 >= t[2]: r = t
    return r[0], r[3], r[4]


def _edge_from_probability(prob, sig):
    return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0)) if np.isfinite(prob) else float(np.clip(abs(sig), 0.0, 1.0))


# ── Signal gen with disk cache ──

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


def generate_signals_cached(label, tf_resample):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"sigs_{label}_fwd4.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Filter to current asset set (smoke vs full)
        if "asset" in df.columns:
            df = df[df["asset"].isin(ASSETS.keys())].reset_index(drop=True)
        return df if len(df) > 0 else None
    tasks = [(n, s, tf_resample) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0: all_sigs.append(r)
    if not all_sigs: return None
    m = pd.concat(all_sigs, ignore_index=True)
    # NEW (B-058): derive hold_bars from actual strategy timeframe, not
    # hardcoded 4. The cache key is now strategy-aware so 15m signals get
    # hold_bars=2 (per FULL5 spec), 1h=4, etc.
    if "hold_bars" not in m.columns:
        _hold_map = {"15m": 2, "1h": 4, "3h": 12, "4h": 16, "24h": 96}
        _strat = m.get("strategy", label)
        if hasattr(_strat, "iloc"):
            m["hold_bars"] = _strat.map(_hold_map).fillna(4).astype(int)
        else:
            m["hold_bars"] = _hold_map.get(str(label), 4)
    else:
        m["hold_bars"] = m["hold_bars"].astype(int)
    m["strategy"] = label
    if "confidence" in m.columns: m = m[m["confidence"] >= 0.08].reset_index(drop=True)
    if len(m) > 0: m.to_parquet(cache_path)
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
            if 0 <= bi and ei < len(close): fwd[p] = (close.values[ei] - close.values[bi]) / close.values[bi]
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


def apply_vol(signals, vol_series, target_vol=0.025, floor=0.10, cap=2.0):
    """Per-asset inverse-vol sizing overlay — mirrors the live
    S8Sym15Strategy.compute_vol_mult: sizing_mult *= clip(target_vol/v, floor, cap).

    FAILS SAFE: when an asset has no vol series, or the point-in-time lookup
    is out of range, or the vol is non-finite, the trade is scaled by
    `floor` (smallest size) — NOT left at 1.0. Leaving warmup / unknown-vol
    trades at full notional was the live sizing bug (DENT/PIPPIN = 97% of
    the live paper loss). `floor` was also raised from 0.3 — far too high
    for the live universe's ~27x volatility dispersion.

    The `idx-1` searchsorted lookup keeps the vol point-in-time (no
    lookahead): v is the most recent daily vol bar strictly before the
    signal timestamp.
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    col = out.columns.get_loc("sizing_mult")
    for i in range(len(out)):
        a = out.iloc[i].get("asset", "")
        vs = vol_series.get(a)
        mult = floor  # fail-safe: unknown vol -> smallest size, never 1.0
        if vs is not None:
            idx = vs.index.searchsorted(ts.iloc[i])
            if 0 < idx <= len(vs):
                v = vs.iloc[idx - 1]
                if pd.notna(v) and v > 0:
                    mult = float(np.clip(target_vol / v, floor, cap))
        out.iat[i, col] *= mult
    return out


# ── 24h early exit: recompute fwd_return with hourly checkpoints ──

def apply_24h_early_exit(signals_24h, mode, params):
    """Modify 24h signals' fwd_return by checking hourly for early exit.

    mode='forecast': exit if 4h model direction flips against position
    mode='stoploss': exit if unrealized PnL drops below -X%

    Returns modified signals with adjusted fwd_return and hold_bars.
    """
    out = signals_24h.copy()
    if len(out) == 0: return out

    # Load 1h close prices per asset for hourly checkpoints
    close_1h_cache = {}
    # Load 4h close + direction cache for forecast mode
    close_4h_cache = {}

    for asset_name in out["asset"].unique():
        sym = ASSETS.get(asset_name)
        if not sym: continue
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            c1h = resample_ohlcv(ohlcv_1m, "1h")["close"]
            if c1h.index.tz is not None: c1h.index = c1h.index.tz_localize(None)
            close_1h_cache[asset_name] = c1h

            if mode == "forecast":
                ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
                if ohlcv_4h.index.tz is not None: ohlcv_4h.index = ohlcv_4h.index.tz_localize(None)
                close_4h_cache[asset_name] = ohlcv_4h
            del ohlcv_1m
        except: continue

    sig_ts = pd.to_datetime(out["timestamp"])
    if sig_ts.dt.tz is not None: sig_ts = sig_ts.dt.tz_localize(None)

    # NEW (B-021): pre-compute pct_change(2) for every 4h asset ONCE (perf).
    pct_4h_cache = {a: df["close"].pct_change(2)
                    for a, df in close_4h_cache.items()}

    new_fwd = out["fwd_return"].values.copy()
    new_hold = out["hold_bars"].values.copy()
    exit_count = 0

    for i in range(len(out)):
        asset = out.iloc[i]["asset"]
        direction = out.iloc[i]["direction"]
        entry_ts = sig_ts.iloc[i]

        if asset not in close_1h_cache: continue
        c1h = close_1h_cache[asset]

        # NEW (B-016): use side='right' and subtract 1 to get the LAST bar at
        # or BEFORE entry_ts. The previous searchsorted(entry_ts) returns the
        # FIRST bar strictly GREATER than entry_ts if the entry falls inside
        # an incomplete bar — that's a 1h lookahead bias.
        entry_idx = c1h.index.searchsorted(entry_ts, side="right") - 1
        if entry_idx < 0 or entry_idx >= len(c1h): continue
        entry_price = c1h.values[entry_idx]

        # Original hold = 12 bars of 24h = 12 days = ~288 hourly bars
        max_1h_bars = 12 * 24  # 12 days in hourly bars

        # Check every hour
        exited = False
        for h in range(1, max_1h_bars):
            check_idx = entry_idx + h
            if check_idx >= len(c1h): break
            current_price = c1h.values[check_idx]
            unrealized_ret = (current_price - entry_price) / entry_price

            if mode == "stoploss":
                threshold = params.get("stop_pct", 0.02)
                # Check if unrealized loss exceeds threshold
                dir_ret = direction * unrealized_ret
                if dir_ret < -threshold:
                    new_fwd[i] = unrealized_ret  # exit at this price
                    new_hold[i] = max(1, h // 24)  # approximate hold in 24h bars
                    exit_count += 1; exited = True; break

            elif mode == "forecast":
                # Check every 4 hours (at 4h bar boundaries)
                if h % 4 != 0: continue
                # Get the 4h model's latest direction for this asset
                check_ts = c1h.index[check_idx]
                if asset not in close_4h_cache: continue
                ohlcv_4h = close_4h_cache[asset]
                bar_idx = ohlcv_4h.index.searchsorted(check_ts)
                if bar_idx < 2 or bar_idx >= len(ohlcv_4h): continue

                # NEW (B-021): pre-computed pct_change(2) per asset (perf).
                # Falls back to 0 if pre-cache missing (e.g., asset-specific path).
                pct_cache_local = pct_4h_cache.get(asset)
                if pct_cache_local is None:
                    recent_4h_ret = 0
                else:
                    recent_4h_ret = pct_cache_local.iloc[bar_idx]
                strongly_against = (direction > 0 and recent_4h_ret < -params.get("threshold", 0.03)) or \
                                   (direction < 0 and recent_4h_ret > params.get("threshold", 0.03))

                if strongly_against:
                    new_fwd[i] = unrealized_ret
                    new_hold[i] = max(1, h // 24)
                    exit_count += 1; exited = True; break

    out["fwd_return"] = new_fwd
    out["hold_bars"] = new_hold
    return out, exit_count


# ── Modified sim ──

def simulate_with_breaker(signals_df, config, breaker_type, breaker_params,
                          compound=True):
    if signals_df is None or len(signals_df) < 10: return None
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

    # Per-asset tracking
    asset_ema_wr = defaultdict(lambda: BASELINE_WR)  # EMA win rate
    asset_recent_pnl = defaultdict(lambda: deque(maxlen=breaker_params.get("window", 10)))
    asset_recent_notional = defaultdict(lambda: deque(maxlen=breaker_params.get("window", 10)))
    asset_streak = defaultdict(int)
    cb_triggers = 0

    for i in range(n):
        current_ts = timestamps[i]
        # NEW (B-017): default to -1 (unknown) and log; v1 used 0 which caused
        # positions to pile up indefinitely for NaT/duplicate-dropped timestamps.
        current_ts_idx = ts_to_idx.get(current_ts, -1)
        if current_ts_idx == -1:
            logger.warning("simulate_with_breaker: unknown timestamp %s at i=%d; "
                           "skipping", current_ts, i)
            continue
        # NEW (B-055): strict > became >= to expire positions exactly at expiry.
        active_positions = [(e, sz, a, st) for e, sz, a, st in active_positions if e >= current_ts_idx]
        current_exposure = sum(sz for _, sz, _, _ in active_positions)
        asset_exposure = {}; strategy_exposure = {}
        for _, sz, a, st in active_positions:
            asset_exposure[a] = asset_exposure.get(a, 0.0) + sz
            strategy_exposure[st] = strategy_exposure.get(st, 0.0) + sz
        if halted: equity_arr[i] = equity; dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0; continue
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= config.max_dd: halted = True; equity_arr[i] = equity; dd_arr[i] = dd; continue
        size_mult = max(1.0 - (dd - config.delever_start) / (config.max_dd - config.delever_start), 0.0) if dd > config.delever_start else 1.0
        # compound=False -> size off FIXED starting capital (non-compound):
        # PnL still accumulates into `equity`, but notional does not grow.
        # NC is the honest planning metric (compounding inflates magnitude
        # ~9x and is dominated by the late high-equity period).
        size_base = equity if compound else config.starting_capital
        if use_prob:
            sig_abs = abs(signal_strengths[i])
            if config.sizing_mode == "probability_edge": sig_abs = _edge_from_probability(probabilities[i], sig_abs)
            if config.use_regime_exposure_multiplier: sig_abs *= float(np.clip(regime_mults[i], config.regime_exposure_floor, 1.0))
            raw_size = size_base * config.base_sizing_pct * sig_abs * size_mult * sizing_mults[i]
            raw_size = min(raw_size, size_base * config.base_sizing_pct * config.max_signal_mult * size_mult)
        else:
            raw_size = size_base * config.base_sizing_pct * size_mult

        # ═══ CIRCUIT BREAKER ═══
        asset = assets_arr[i]
        cb_mult = 1.0

        cb_fired = False  # NEW (B-060): track fired state; only count after trade executes

        if breaker_type == "streak":
            streak = asset_streak[asset]; thr = breaker_params["threshold"]; decay = breaker_params["decay"]
            # NEW (B-019): floor at 0.10 to match EMA variant. Without floor,
            # cb_mult can reach ~0.004 at streak=10 (decay=0.5) -> ~0.4% size,
            # which is below the raw_size<1.0 skip threshold (no trade).
            if streak >= thr:
                cb_mult = max(decay ** (streak - thr + 1),
                              breaker_params.get("floor", 0.10))
                cb_fired = True

        elif breaker_type == "ema":
            ema = asset_ema_wr[asset]
            # Proportional sizing: scale by ema/baseline, with floor
            ratio = ema / BASELINE_WR
            cb_mult = max(ratio, breaker_params.get("floor", 0.10))
            if cb_mult < 0.95: cb_fired = True  # count meaningful reductions

        elif breaker_type in ("pnl_ratio", "pnl_ratio_symmetric"):
            recent_pnl = asset_recent_pnl[asset]
            recent_not = asset_recent_notional[asset]
            # NEW (B-018): activate with as few as 3 trades (not 10) so new
            # assets get immediate protection. The 10-trade requirement left
            # new assets with cb_mult=1.0 regardless of recent losing streak.
            min_trades = max(1, breaker_params.get("min_trades_for_cb", 3))
            if len(recent_pnl) >= min_trades:
                total_pnl = sum(recent_pnl)
                total_not = sum(recent_not)
                if total_not > 0:
                    pnl_ratio = total_pnl / total_not
                    if pnl_ratio < breaker_params["threshold"]:
                        # Scale DOWN proportionally
                        t = breaker_params["threshold"]
                        # REVERTED 2026-06-25 (B-056): the "smooth" linear map
                        # scale = 1.0 + (pnl_ratio - t) * 0.90 / |t| with t<0
                        # gives scale=1.9 at pnl_ratio=0 (the natural intercept
                        # when t=-0.01) -- but pnl_ratio=0 is NOT in the if-branch
                        # (the if is pnl_ratio<t, not pnl_ratio<=t). The real
                        # regression: the formula produces values > 1.0 for
                        # pnl_ratio in (-0.01, 0) (i.e. the if-branch, just
                        # barely losing) because the line crosses 1.0 at
                        # pnl_ratio=t=-0.01 and goes UP as pnl_ratio increases
                        # toward 0. So a trade that is "barely losing" gets
                        # scale>1.0 (LEVERAGED) and the next day when the
                        # streak extends the loss, it gets even more leveraged
                        # in the wrong direction. The honest old formula was
                        # a binary "floor or full" which is less elegant but
                        # monotonic in pnl_ratio. Reverting B-056 to the
                        # binary version until a corrected smooth formula is
                        # designed.
                        scale = breaker_params.get("floor", 0.10) if t != 0 else 1.0
                        cb_mult = scale
                        cb_fired = True
                    elif (breaker_type == "pnl_ratio_symmetric"
                          and pnl_ratio > breaker_params.get("up_threshold", 0.01)):
                        # Scale UP proportionally, capped
                        up_t = breaker_params["up_threshold"]
                        cap = breaker_params.get("cap", 1.5)
                        scale = min(1.0 + (pnl_ratio - up_t) / up_t, cap)
                        cb_mult = scale
                        cb_fired = True  # scale-up also counts as a breaker event

        raw_size *= cb_mult
        # ═══ END CIRCUIT BREAKER ═══

        # Exposure limits (off size_base — fixed capital in non-compound mode)
        asset_exp = asset_exposure.get(asset, 0.0)
        if asset_exp + raw_size > size_base * config.max_per_asset: raw_size = max(size_base * config.max_per_asset - asset_exp, 0.0)
        strategy = strategies_arr[i]; mps = getattr(config, "max_per_strategy", 1.0)
        strat_exp = strategy_exposure.get(strategy, 0.0)
        if mps < 1.0 and strat_exp + raw_size > size_base * mps: raw_size = max(size_base * mps - strat_exp, 0.0)
        if current_exposure + raw_size > size_base * config.max_exposure: raw_size = max(size_base * config.max_exposure - current_exposure, 0.0)
        # NEW (B-060): only commit cb_triggers AFTER the skip guard so a
        # CB fire that drops raw_size below 1.0 doesn't pollute the audit
        # count with phantom triggers.
        if raw_size < 1.0: equity_arr[i] = equity; dd_arr[i] = dd; continue
        if cb_fired: cb_triggers += 1

        trade_return = directions[i] * fwd_returns[i] - config.slippage
        pnl = raw_size * trade_return; equity += pnl
        if equity > peak_equity: peak_equity = equity
        active_positions.append((current_ts_idx + int(hold_bars_arr[i]), raw_size, asset, strategy))
        equity_arr[i] = equity; pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        size_arr[i] = raw_size; skipped[i] = False

        # Update tracking AFTER trade
        win = pnl >= 0
        alpha = breaker_params.get("alpha", 0.90)
        asset_ema_wr[asset] = asset_ema_wr[asset] * alpha + (1.0 if win else 0.0) * (1 - alpha)
        asset_streak[asset] = asset_streak[asset] + 1 if pnl < 0 else 0
        asset_recent_pnl[asset].append(pnl)
        asset_recent_notional[asset].append(raw_size)

    return {"equity": equity_arr, "pnl": pnl_arr, "dd": dd_arr, "sizes": size_arr,
            "skipped": skipped, "timestamps": timestamps, "halted": halted,
            "final_equity": equity, "n_trades": int((~skipped).sum()), "cb_triggers": cb_triggers}


def run_variant(merged, breaker_type, breaker_params, port=None, compound=True):
    """Run one breaker variant. ``port`` overrides the default research
    PortfolioConfig — pass a live-config PortfolioConfig to re-backtest the
    deployed system (base_sizing_pct=0.30, max_per_asset=0.25, ...).
    ``compound=False`` sizes off fixed capital — the honest non-compound
    Sharpe/PnL (compounding inflates magnitude ~9x; see feedback_noncompound_truth)."""
    if port is None:
        port = PortfolioConfig(
            starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
            base_sizing_pct=0.045, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    sim = simulate_with_breaker(prepared, port, breaker_type, breaker_params,
                                compound=compound)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    mfp = 0.90; active = ~sim["skipped"]; sizes = sim["sizes"]
    ts_arr = pd.to_datetime(prepared["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date; daily_vol = {}
    for i in range(len(prepared)):
        if active[i]: daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    # NEW (B-059): min_periods=30 (was 1) so first 29 days are NaN, not
    # partial sums. With min_periods=1 a day-1 burst falsely qualified
    # the asset for VIP3+ tier for the first 29 days of every fold.
    # REVERTED 2026-06-25: min_periods=30 left v30=NaN for the first 29 days
    # (because vol_30d.get(dd, 0) returns NaN, not 0, when the rolling sum is
    # still NaN at that date). That NaN propagated through get_vip_tier -> the
    # extra-bps adjustment -> the equity-curve -> the floor-aggregate Sharpe,
    # which crashed from Sharpe 3.40 to -14.86. The per-year aggregation
    # survived (groupby on per-trade pnl) but the headline Sharpe is on the
    # equity curve. min_periods=1 was the honest behavior; if we want stricter
    # warmup, do it at the TIER CALCULATION, not at the rolling sum.
    vol_s = pd.Series(daily_vol).sort_index(); vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = CAP + dpnl.cumsum(); dpnl_adj = dpnl.copy()
    # NEW (B-057): subtract the configured slippage in bps, not the
    # hardcoded 1.0 magic constant. The engine ALREADY charges
    # `port.slippage` (typically 1 bp = 0.0001) per trade at line 457;
    # the v1 `extra = rt - 1.0` was double-charging the same 1 bp.
    slippage_bps = port.slippage * 10000
    for d in dpnl.index:
        dd = d.date(); eq = eq_curve.get(d, CAP); v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = get_vip_tier(eq, v30)
        rt = mfp * 2 * mbps + (1 - mfp) * 2 * tbps + SPREAD_BPS
        extra = rt - slippage_bps
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
    daily_dd_pct = dpnl_adj / (CAP + dpnl_adj.cumsum()).shift(1).fillna(CAP) * 100
    max_daily_dd = float(daily_dd_pct.min())
    worst_day = daily_dd_pct.idxmin()
    monthly = dpnl_adj.resample("MS").sum()
    return {"sr": sr, "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"], "cb_triggers": sim["cb_triggers"],
            "max_daily_dd": max_daily_dd, "worst_day": str(worst_day.date()),
            "worst_day_pnl": float(dpnl_adj[worst_day]),
            "pnl_mo": pnl / max(len(monthly), 1)}


def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 120)
    print(f"  CIRCUIT BREAKER V2 ({mode})")
    print(f"  {len(ASSETS)} assets, {len(VARIANTS)} breaker variants + 24h early exit, MkrTkr60")
    print("=" * 120)

    # Load signals
    vol_series = {n: v for n, v in (ProcessPoolExecutor(max_workers=8).map(
        _compute_vol_one, list(ASSETS.items()))) if v is not None}

    signal_cache = {}
    for label, resample in NEEDED_TFS:
        t1 = time.time()
        sigs = generate_signals_cached(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"  {label}: {len(sigs):,} ({time.time()-t1:.0f}s)", flush=True)

    # Build Full5 parts (separate 24h for early exit variants)
    hold_cache = {}
    for tf, h in FULL5:
        if tf not in signal_cache: continue
        resample = dict(NEEDED_TFS)[tf]
        hold_cache[(tf, h)] = apply_vol(recompute_returns(signal_cache[tf], resample, h), vol_series)

    non_24h = [hold_cache[(tf, h)] for tf, h in FULL5 if (tf, h) in hold_cache and tf != "24h"]
    sigs_24h = hold_cache.get(("24h", 12))
    merged_base = pd.concat(non_24h + ([sigs_24h] if sigs_24h is not None else []),
                            ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"  Full5: {len(merged_base):,} signals")

    # ── Part 1: Sizing breakers (E, F) + reference (Baseline, A1) ──
    print(f"\n{'='*120}")
    print("  PART 1: SIZING BREAKERS")
    print(f"{'='*120}")
    print(f"\n  {'Variant':<24} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'WorstDay$':>10} {'Calmar':>7} {'PF':>6} {'Trades':>7} {'CB':>7} {'$/mo':>8}")
    print("  " + "-" * 115)

    sizing_results = {}
    for name, btype, bparams in VARIANTS:
        r = run_variant(merged_base, btype, bparams)
        if r is None: print(f"  {name:<24} FAIL"); continue
        sizing_results[name] = r
        base = " <-- BASE" if btype == "none" else ""
        print(f"  {name:<24} {r['sr']:>7.3f} {r['sortino']:>8.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% ${r['worst_day_pnl']:>8,.0f} {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} {r['n_trades']:>7,} {r['cb_triggers']:>7,} ${r['pnl_mo']:>7,.0f}{base}")

    # ── Part 2: 24h early exit (G, H) ──
    if sigs_24h is not None and len(sigs_24h) > 0:
        print(f"\n{'='*120}")
        print("  PART 2: 24h EARLY EXIT")
        print(f"{'='*120}")

        exit_variants = [
            ("G1: 4h forecast 3%",  "forecast", {"threshold": 0.03}),
            ("G2: 4h forecast 2%",  "forecast", {"threshold": 0.02}),
            ("G3: 4h forecast 5%",  "forecast", {"threshold": 0.05}),
            ("H1: stoploss 2%",     "stoploss", {"stop_pct": 0.02}),
            ("H2: stoploss 3%",     "stoploss", {"stop_pct": 0.03}),
            ("H3: stoploss 1%",     "stoploss", {"stop_pct": 0.01}),
            ("H4: stoploss 5%",     "stoploss", {"stop_pct": 0.05}),
        ]
        if SMOKE:
            exit_variants = [exit_variants[0], exit_variants[3], exit_variants[5]]

        print(f"\n  {'Variant':<24} {'24h Exits':>10} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6}"
              f" {'DayDD':>7} {'Calmar':>7} {'PF':>6} {'$/mo':>8}")
        print("  " + "-" * 100)

        # Baseline (no early exit on 24h)
        r_base = run_variant(merged_base, "none", {})
        if r_base:
            print(f"  {'Baseline (no exit)':<24} {'0':>10} {r_base['sr']:>7.3f} {r_base['sortino']:>8.3f}"
                  f" ${r_base['pnl']:>9,.0f} {r_base['max_dd']:>5.1f}% {r_base['max_daily_dd']:>6.2f}%"
                  f" {r_base['calmar']:>7.2f} {r_base['pf']:>5.2f} ${r_base['pnl_mo']:>7,.0f}")

        for ex_name, ex_mode, ex_params in exit_variants:
            t1 = time.time()
            modified_24h, exit_count = apply_24h_early_exit(sigs_24h, ex_mode, ex_params)
            # Rebuild Full5 with modified 24h
            merged_mod = pd.concat(non_24h + [modified_24h], ignore_index=True)\
                .sort_values("timestamp").reset_index(drop=True)
            r = run_variant(merged_mod, "none", {})
            if r is None: print(f"  {ex_name:<24} FAIL"); continue
            print(f"  {ex_name:<24} {exit_count:>10,} {r['sr']:>7.3f} {r['sortino']:>8.3f}"
                  f" ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}% {r['max_daily_dd']:>6.2f}%"
                  f" {r['calmar']:>7.2f} {r['pf']:>5.2f} ${r['pnl_mo']:>7,.0f}")

    # ── Part 3: Best combo (best sizing + best 24h exit) ──
    print(f"\n{'='*120}")
    print("  PART 3: COMBINED (best sizing breaker + best 24h exit)")
    print(f"{'='*120}")

    # Find best sizing breaker (excluding baseline)
    best_sizing = max(((n, r) for n, r in sizing_results.items() if n != "Baseline"),
                      key=lambda x: x[1]["sr"], default=None)
    if best_sizing:
        best_name, _ = best_sizing
        best_btype = [bt for n, bt, _ in VARIANTS if n == best_name][0]
        best_bparams = [bp for n, _, bp in VARIANTS if n == best_name][0]
        print(f"  Best sizing breaker: {best_name}")

        # Run best sizing on baseline merged
        r_sizing = run_variant(merged_base, best_btype, best_bparams)

        # Run best sizing on best 24h exit variant (H2: 3% stoploss as likely candidate)
        if sigs_24h is not None:
            for stop_pct in [0.02, 0.03]:
                mod_24h, ec = apply_24h_early_exit(sigs_24h, "stoploss", {"stop_pct": stop_pct})
                merged_combo = pd.concat(non_24h + [mod_24h], ignore_index=True)\
                    .sort_values("timestamp").reset_index(drop=True)
                r_combo = run_variant(merged_combo, best_btype, best_bparams)
                if r_combo:
                    label = f"{best_name} + H({stop_pct*100:.0f}%SL)"
                    print(f"  {label:<40} Sharpe={r_combo['sr']:.3f}  PnL=${r_combo['pnl']:,.0f}"
                          f"  MaxDD={r_combo['max_dd']:.1f}%  DayDD={r_combo['max_daily_dd']:.2f}%"
                          f"  $/mo=${r_combo['pnl_mo']:,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
