#!/usr/bin/env python
"""HFT Market Making Baseline via hftbacktest.

Avellaneda-Stoikov market making with optional ML alpha signal.
Tests whether the production LogReg directional forecast improves
market making performance when used to skew quotes.

Phases:
  1. Pure MM baseline (symmetric quotes + inventory skew)
  2. MM + LogReg alpha (skewed quotes from 15m walk-forward signal)
  3. Quote update frequency sweep

Data: Binance bookTicker BBO (320 days, 37 symbols)
Alpha: Pre-computed 15m LogReg probabilities from walk-forward pipeline

Usage:
    cd C:/Claude/ml-trading-system
    PYTHONPATH=src python experiments/hft_mm_baseline.py [--smoke] [--phase N]
"""
import sys, os, time, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit, uint64, float64
from numba.typed import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from hftbacktest import (
    BacktestAsset, HashMapMarketDepthBacktest, Recorder,
    BUY, SELL, GTX, LIMIT,
)

SMOKE = "--smoke" in sys.argv
PHASE_ONLY = None
for a in sys.argv:
    if a.startswith("--phase"):
        PHASE_ONLY = int(sys.argv[sys.argv.index(a) + 1])

# ============================================================
# Constants
# ============================================================
PARQUET_DIR = Path("data/bookTicker_parquet")
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

DEPTH_EVENT = 1
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28
EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30

EVENT_DTYPE = np.dtype([
    ("ev", "u8"), ("exch_ts", "i8"), ("local_ts", "i8"),
    ("px", "f8"), ("qty", "f8"), ("order_id", "u8"),
    ("ival", "i8"), ("fval", "f8"),
], align=True)

TICK_SIZES = {
    "BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.001,
    "XRPUSDT": 0.0001, "BNBUSDT": 0.01, "ADAUSDT": 0.0001,
    "AVAXUSDT": 0.001, "DOTUSDT": 0.001, "LINKUSDT": 0.001,
    "NEARUSDT": 0.001, "UNIUSDT": 0.001, "AAVEUSDT": 0.01,
    "LTCUSDT": 0.01, "BCHUSDT": 0.01, "FILUSDT": 0.001,
    "FTMUSDT": 0.0001, "HBARUSDT": 0.00001, "TRXUSDT": 0.0001,
    "SUIUSDT": 0.0001, "APTUSDT": 0.001, "ENJUSDT": 0.0001,
    "SXPUSDT": 0.0001, "ZECUSDT": 0.01, "XLMUSDT": 0.00001,
    "FETUSDT": 0.0001, "WLDUSDT": 0.001, "BNXUSDT": 0.001,
    "ANKRUSDT": 0.00001, "TAOUSDT": 0.01, "ENAUSDT": 0.0001,
    "ARBUSDT": 0.0001, "ATOMUSDT": 0.001, "DOGEUSDT": 0.00001,
    "INJUSDT": 0.001, "MATICUSDT": 0.0001, "MKRUSDT": 0.1,
    "OPUSDT": 0.0001, "SUSHIUSDT": 0.001, "TIAUSDT": 0.001,
}
LOT_SIZES = {
    "BTCUSDT": 0.001, "ETHUSDT": 0.01, "SOLUSDT": 0.01,
    "XRPUSDT": 1.0, "BNBUSDT": 0.01, "ADAUSDT": 1.0,
    "AVAXUSDT": 0.1, "DOTUSDT": 0.1, "LINKUSDT": 0.1,
    "NEARUSDT": 0.1, "UNIUSDT": 0.1, "AAVEUSDT": 0.01,
    "LTCUSDT": 0.01, "BCHUSDT": 0.01, "FILUSDT": 0.1,
    "FTMUSDT": 1.0, "HBARUSDT": 1.0, "TRXUSDT": 1.0,
    "SUIUSDT": 1.0, "APTUSDT": 0.1, "ENJUSDT": 1.0,
    "SXPUSDT": 1.0, "ZECUSDT": 0.01, "XLMUSDT": 1.0,
    "FETUSDT": 1.0, "WLDUSDT": 0.1, "BNXUSDT": 0.1,
    "ANKRUSDT": 1.0, "TAOUSDT": 0.01, "ENAUSDT": 1.0,
    "ARBUSDT": 1.0, "ATOMUSDT": 0.1, "DOGEUSDT": 1.0,
    "INJUSDT": 0.1, "MATICUSDT": 1.0, "MKRUSDT": 0.001,
    "OPUSDT": 1.0, "SUSHIUSDT": 1.0, "TIAUSDT": 0.1,
}

# Fee scenarios (maker_fee, taker_fee)
CAPITAL = 100_000  # $100K base capital

# Fee scenarios (maker_fee, taker_fee)
FEE_SCENARIOS = {
    "binance_vip2": (0.00016, 0.000375),
    "binance_lp":   (-0.00005, 0.0007),
    "hl_std":       (0.0001, 0.00035),
    "hl_zero_mkr":  (0.0, 0.00035),
}

SMOKE_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
ALL_ASSETS = sorted(TICK_SIZES.keys())

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]


# ============================================================
# Data Loading
# ============================================================

def bookticker_to_events(pq_path):
    """Convert bookTicker parquet to hftbacktest events (vectorized).
    Uses pyarrow directly for speed (avoids pandas overhead).
    """
    try:
        import pyarrow.parquet as pq_reader
        table = pq_reader.read_table(pq_path)
        if table.num_rows == 0:
            return None
        n = table.num_rows

        events = np.zeros(2 * n, dtype=EVENT_DTYPE)
        ts_ns = table.column("transaction_time").to_numpy().astype(np.int64) * 1_000_000

        bid_flag = np.uint64(EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | BUY_EVENT)
        ask_flag = np.uint64(EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | SELL_EVENT)

        events["ev"][0::2] = bid_flag
        events["exch_ts"][0::2] = ts_ns
        events["local_ts"][0::2] = ts_ns + 10_000_000
        events["px"][0::2] = table.column("best_bid_price").to_numpy().astype(np.float64)
        events["qty"][0::2] = table.column("best_bid_qty").to_numpy().astype(np.float64)

        events["ev"][1::2] = ask_flag
        events["exch_ts"][1::2] = ts_ns
        events["local_ts"][1::2] = ts_ns + 10_000_000
        events["px"][1::2] = table.column("best_ask_price").to_numpy().astype(np.float64)
        events["qty"][1::2] = table.column("best_ask_qty").to_numpy().astype(np.float64)

        return events
    except Exception:
        return None  # skip corrupted files


def discover_dates(sym):
    """Find all available dates for a symbol."""
    pq_dir = PARQUET_DIR / sym
    if not pq_dir.exists():
        return []
    return sorted(f.stem for f in pq_dir.glob("*.parquet"))


def load_events_chunk(sym, dates):
    """Load multiple days of events, concatenated. Skips corrupted files.
    Only sorts if multi-day (single-day data is already time-sorted).
    """
    events_list = []
    for d in dates:
        pq = PARQUET_DIR / sym / f"{d}.parquet"
        if pq.exists():
            ev = bookticker_to_events(pq)
            if ev is not None and len(ev) > 0:
                events_list.append(ev)
    if not events_list:
        return None
    events = np.concatenate(events_list)
    # Only sort if multi-day concatenation (single-day is already sorted)
    if len(events_list) > 1:
        order = np.argsort(events["exch_ts"])
        events = events[order]
    return events


# ============================================================
# Alpha Signal Generation (15m LogReg)
# ============================================================

def gen_alpha_signals(name, sym):
    """Generate 15m walk-forward signals for one asset. Returns DataFrame."""
    import warnings; warnings.filterwarnings("ignore")
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.filters import apply_regime_filter
    from signal_engine.config import (
        SystemConfig, ModelConfig, FeatureConfig, RegimeFilterConfig,
        ConfidenceFilterConfig, SignalConfig, DataConfig,
    )

    config = SystemConfig(
        model=ModelConfig(C=0.10, penalty="elasticnet", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01,
                               winsor_hi=0.99, vol_filter_quantile=0.50,
                               vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS],
                        timeframe="15min", regime_timeframe="4h"),
        assets={},
    )
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, "15min")
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0:
        return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    if len(sigs) == 0:
        return None
    # Apply confidence floor
    sigs = sigs[sigs["confidence"] >= 0.08].reset_index(drop=True)
    return sigs


def prepare_alpha_arrays(sigs_df):
    """Convert signal DataFrame to sorted numpy arrays for @njit consumption.

    Returns: (timestamps_ns, forecast_values) sorted by time.
    forecast = 2*probability - 1, range [-1, 1].
    """
    if sigs_df is None or len(sigs_df) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    ts = pd.to_datetime(sigs_df["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    ts_ns = ts.values.astype(np.int64)

    forecast = (2.0 * sigs_df["probability"].values - 1.0).astype(np.float64)
    order = np.argsort(ts_ns)
    return ts_ns[order], forecast[order]


# ============================================================
# @njit Market Making Strategy
# ============================================================

@njit
def mm_strategy(hbt, recorder,
                alpha_ts_ns, alpha_vals,
                half_spread_ticks, skew_coeff, alpha_weight,
                order_qty_usd, max_position_usd,
                update_interval_ns):
    """Avellaneda-Stoikov MM with optional alpha signal.

    Posts single bid + single ask (1-level grid), updates every interval.
    Alpha signal skews the reservation price.

    Returns: (n_updates, n_bid_fills, n_ask_fills, final_position, final_mid)
    """
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size
    lot_size = hbt.depth(asset_no).lot_size

    alpha_idx = 0
    current_alpha = 0.0

    n_updates = 0
    n_bid_fills = 0
    n_ask_fills = 0

    # Monotonic order ID counter (avoids collisions across chunks)
    next_order_id = np.int64(1)
    cur_bid_id = np.int64(-1)
    cur_ask_id = np.int64(-1)
    final_mid = 0.0

    while hbt.elapse(update_interval_ns) == 0:
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)
        ts = hbt.current_timestamp

        best_bid = depth.best_bid
        best_ask = depth.best_ask
        if best_bid <= 0.0 or best_ask <= 0.0:
            continue

        mid = (best_bid + best_ask) / 2.0
        final_mid = mid
        n_updates += 1

        # Step through alpha signal
        while alpha_idx < len(alpha_ts_ns) and alpha_ts_ns[alpha_idx] <= ts:
            current_alpha = alpha_vals[alpha_idx]
            alpha_idx += 1

        # Order quantity
        order_qty = max(round((order_qty_usd / mid) / lot_size), 1) * lot_size

        # Reservation price with alpha and inventory skew
        # alpha_weight scales how much the forecast shifts the fair price
        # skew_coeff penalizes inventory (in ticks per normalized unit)
        normalized_pos = (position * mid) / max(order_qty_usd, 1.0)
        reservation = mid + alpha_weight * current_alpha * tick_size - skew_coeff * normalized_pos * tick_size

        half_spread = half_spread_ticks * tick_size
        new_bid = np.floor((reservation - half_spread) / tick_size) * tick_size
        new_ask = np.ceil((reservation + half_spread) / tick_size) * tick_size

        # Clamp: never cross the spread, post at BBO or behind
        new_bid = min(new_bid, best_bid)
        new_ask = max(new_ask, best_ask)

        new_bid_tick = np.int64(round(new_bid / tick_size))
        new_ask_tick = np.int64(round(new_ask / tick_size))

        # Position limits
        pos_usd = position * mid
        buy_ok = pos_usd < max_position_usd
        sell_ok = pos_usd > -max_position_usd

        # Cancel ALL existing orders, then submit fresh (simple cancel-replace)
        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            if order.cancellable:
                hbt.cancel(asset_no, order.order_id, False)

        # Submit new bid
        if buy_ok and new_bid_tick > 0:
            hbt.submit_buy_order(asset_no, next_order_id, new_bid, order_qty,
                                 GTX, LIMIT, False)
            cur_bid_id = next_order_id
            next_order_id += 1

        # Submit new ask
        if sell_ok and new_ask_tick > 0:
            hbt.submit_sell_order(asset_no, next_order_id, new_ask, order_qty,
                                  GTX, LIMIT, False)
            cur_ask_id = next_order_id
            next_order_id += 1

        recorder.record(hbt)

    return (n_updates, n_bid_fills, n_ask_fills, position, final_mid)


# ============================================================
# Per-Asset Runner
# ============================================================

def bps_to_ticks(bps, mid_price, tick_size):
    """Convert spread in basis points to ticks."""
    spread_usd = mid_price * bps / 10000.0
    return max(int(round(spread_usd / tick_size)), 1)


def run_mm_on_events(events, sym, params, alpha_ts_ns=None, alpha_vals=None,
                     fee_scenario="hl_std", tag=""):
    """Run MM backtest on pre-loaded events array. Avoids re-reading data.

    Returns dict with equity curve and stats.
    """
    tick_size = TICK_SIZES.get(sym, 0.01)
    lot_size = LOT_SIZES.get(sym, 0.1)
    maker_fee, taker_fee = FEE_SCENARIOS[fee_scenario]

    if events is None or len(events) < 1000:
        return None

    if alpha_ts_ns is None:
        alpha_ts_ns = np.array([], dtype=np.int64)
        alpha_vals = np.array([], dtype=np.float64)

    # Auto-compute half_spread_ticks from bps if provided
    half_spread_ticks = params.get("half_spread_ticks", 0)
    if "half_spread_bps" in params and half_spread_ticks == 0:
        # Estimate mid price from first valid events
        bid_mask = events["qty"] > 0
        mid_est = np.median(events["px"][bid_mask][:1000])
        half_spread_ticks = bps_to_ticks(params["half_spread_bps"], mid_est, tick_size)

    asset = (
        BacktestAsset()
            .data([events])
            .linear_asset(1.0)
            .constant_order_latency(10_000_000, 10_000_000)
            .power_prob_queue_model(2.0)
            .no_partial_fill_exchange()
            .trading_value_fee_model(maker_fee, taker_fee)
            .tick_size(tick_size)
            .lot_size(lot_size)
            .last_trades_capacity(0)
    )

    try:
        hbt = HashMapMarketDepthBacktest([asset])
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": f"backtest_init: {e}"}

    # Recorder capacity: 1 record per update_interval (1s default = 86400/day)
    update_ns = int(params.get("update_interval_ns", 1_000_000_000))
    time_span_ns = int(events["exch_ts"][-1] - events["exch_ts"][0])
    n_est = max(time_span_ns // update_ns + 10000, 100000)
    recorder = Recorder(1, min(n_est, 30_000_000))

    try:
        result = mm_strategy(
            hbt, recorder.recorder,
            alpha_ts_ns, alpha_vals,
            float(half_spread_ticks),
            float(params["skew"]),
            float(params.get("alpha_weight", 0.0)),
            float(params["order_qty_usd"]),
            float(params["max_position_usd"]),
            int(params["update_interval_ns"]),
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        hbt.close()
        return {"error": f"strategy: {e}"}

    n_updates = result[0]
    final_pos = result[3]
    final_mid = result[4]

    # Extract equity curve from recorder
    try:
        Path("data/cache").mkdir(parents=True, exist_ok=True)
        stats_file = f"data/cache/_mm_{sym}_{tag}_{os.getpid()}.npz"
        recorder.to_npz(stats_file)
        data = np.load(stats_file, allow_pickle=False)
        records = data["0"]
        equity = records["balance"] + records["position"] * records["price"]
        timestamps = records["timestamp"]
        data.close()
        Path(stats_file).unlink(missing_ok=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        hbt.close()
        return {"n_updates": n_updates, "error": f"recorder: {e}"}

    hbt.close()
    return {
        "n_updates": n_updates,
        "final_pos": final_pos,
        "final_mid": final_mid,
        "equity": equity,
        "timestamps": timestamps,
        "half_spread_ticks": half_spread_ticks,
    }


# Adaptive chunk sizes: BTC=1day (~5.6GB), ETH=2days, SOL=5, others=10+
CHUNK_DAYS = {
    "BTCUSDT": 1, "ETHUSDT": 2, "SOLUSDT": 5, "XRPUSDT": 5,
    "BNBUSDT": 3, "DOGEUSDT": 5, "ADAUSDT": 5,
}
DEFAULT_CHUNK_DAYS = 10  # safe for assets with <1M ticks/day


def run_mm_asset(sym, all_dates, params, alpha_df=None,
                 fee_scenario="hl_std", chunk_days=None):
    """Run MM for one asset across all dates, chunked for memory.

    Returns aggregated daily PnL and stats.
    """
    # Prepare alpha arrays
    if alpha_df is not None and len(alpha_df) > 0:
        alpha_ts_ns, alpha_vals = prepare_alpha_arrays(alpha_df)
    else:
        alpha_ts_ns = np.array([], dtype=np.int64)
        alpha_vals = np.array([], dtype=np.float64)

    # Adaptive chunk size based on asset tick density
    if chunk_days is None:
        chunk_days = CHUNK_DAYS.get(sym, DEFAULT_CHUNK_DAYS)

    # Build chunks
    chunks = []
    for i in range(0, len(all_dates), chunk_days):
        chunks.append(all_dates[i:i + chunk_days])

    all_equity = []
    all_timestamps = []
    total_updates = 0
    actual_hs_ticks = None

    t_asset = time.time()
    for ci, cdates in enumerate(chunks):
        events = load_events_chunk(sym, cdates)
        if events is None:
            continue
        r = run_mm_on_events(events, sym, params, alpha_ts_ns, alpha_vals,
                             fee_scenario, tag=f"c{ci}")
        del events; gc.collect()

        if r is None or "error" in r:
            if ci % 10 == 0:
                print(f"    [{ci+1:3d}/{len(chunks)}] {cdates[0]}: skip", flush=True)
            continue

        total_updates += r["n_updates"]
        if actual_hs_ticks is None:
            actual_hs_ticks = r.get("half_spread_ticks")

        if r.get("equity") is not None:
            all_equity.append(r["equity"])
            all_timestamps.append(r["timestamps"])

        # Progress every 10 chunks
        if (ci + 1) % 10 == 0 or ci == len(chunks) - 1:
            elapsed = time.time() - t_asset
            eta = elapsed / (ci + 1) * (len(chunks) - ci - 1)
            print(f"    [{ci+1:3d}/{len(chunks)}] {cdates[-1]}  "
                  f"pos={r.get('final_pos', 0):.3f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    if not all_equity:
        return None

    # Concatenate equity curves
    eq = np.concatenate(all_equity)
    ts = np.concatenate(all_timestamps)

    # Compute daily PnL from equity curve
    eq_series = pd.Series(eq, index=pd.to_datetime(ts, unit="ns"))
    daily_eq = eq_series.resample("1D").last().dropna()
    daily_pnl = daily_eq.diff().dropna()
    daily_pnl = daily_pnl[daily_pnl != 0]

    if len(daily_pnl) < 3:
        return None

    # Stats — compute Sharpe as daily_return / std(daily_return) using CAPITAL base
    daily_ret = daily_pnl / CAPITAL
    sr = 0.0
    if daily_ret.std() > 0:
        sr = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365.25))

    total_pnl = float(daily_pnl.sum())
    pnl_per_day = float(daily_pnl.mean())

    # Max drawdown in absolute dollars (not percentage — no capital base)
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    dd_abs = float((running_max - cum_pnl).max())

    # Also compute n_trades from recorder data
    n_trades = 0
    for eqc in all_equity:
        n_trades += len(eqc)  # approximate: 1 record per update

    return {
        "sharpe": sr,
        "total_pnl": total_pnl,
        "pnl_per_day": pnl_per_day,
        "max_dd_abs": dd_abs,
        "n_days": len(daily_pnl),
        "n_updates": total_updates,
        "daily_pnl": daily_pnl,
        "hs_ticks": actual_hs_ticks,
    }


# ============================================================
# Phase 1: Pure MM Baseline
# ============================================================

def phase1_pure_mm():
    """Run pure MM (no alpha) with parameter sweep on smoke assets."""
    print(f"\n{'='*80}", flush=True)
    print("  PHASE 1: PURE MARKET MAKING BASELINE", flush=True)
    print(f"{'='*80}", flush=True)

    assets = SMOKE_ASSETS if SMOKE else SMOKE_ASSETS  # start with 3 always
    # Capital-relative sizing: order = 5% of capital, max pos = 50% of capital
    order_qty = CAPITAL * 0.05   # $5K per side
    max_pos = CAPITAL * 0.50     # $50K max inventory

    # Spread in BPS (normalized across assets) — auto-converts to ticks
    param_grid = [
        {"half_spread_bps": bps, "half_spread_ticks": 0, "skew": sk,
         "order_qty_usd": order_qty, "max_position_usd": max_pos,
         "update_interval_ns": 1_000_000_000, "alpha_weight": 0.0}
        for bps in [5, 10, 20, 50]
        for sk in [1.0, 5.0, 10.0]
    ]
    if SMOKE:
        param_grid = [
            {"half_spread_bps": bps, "half_spread_ticks": 0, "skew": sk,
             "order_qty_usd": order_qty, "max_position_usd": max_pos,
             "update_interval_ns": 1_000_000_000, "alpha_weight": 0.0}
            for bps in [10, 20, 50]
            for sk in [5.0, 10.0]
        ]

    results = {}
    for sym in assets:
        dates = discover_dates(sym)
        if not dates:
            print(f"  {sym}: no data"); continue
        if SMOKE:
            dates = dates[:10]  # 10 days for smoke
        print(f"\n  {sym}: {len(dates)} days, {len(param_grid)} configs", flush=True)

        for pi, params in enumerate(param_grid):
            t0 = time.time()
            bps = params.get("half_spread_bps", 0) or params.get("half_spread_ticks", 0)
            sk = params["skew"]
            label = f"bps={bps}_sk={sk}"

            r = run_mm_asset(sym, dates, params, fee_scenario="hl_std")
            elapsed = time.time() - t0

            if r is None:
                print(f"    [{pi+1:2d}] {label}: FAILED ({elapsed:.0f}s)", flush=True)
                continue

            key = f"{sym}_{label}"
            results[key] = {**r, "sym": sym, "params": params}
            print(f"    [{pi+1:2d}] {label}: SR={r['sharpe']:+.3f}  "
                  f"PnL=${r['total_pnl']:+,.0f}  "
                  f"DD=${r['max_dd_abs']:,.0f}  "
                  f"days={r['n_days']}  ({elapsed:.0f}s)")

    # Summary
    if results:
        print(f"\n  {'='*60}", flush=True)
        print(f"  PHASE 1 SUMMARY", flush=True)
        print(f"  {'='*60}", flush=True)
        print(f"  {'Config':<22} {'Sym':<8} {'SR':>8} {'PnL':>12} {'DD$':>10} {'$/day':>8}", flush=True)
        print(f"  {'-'*74}", flush=True)
        for key, r in sorted(results.items(), key=lambda x: -x[1]["sharpe"]):
            bps = r["params"].get("half_spread_bps", 0) or r["params"].get("half_spread_ticks", 0)
            sk = r["params"]["skew"]
            tks = r.get("hs_ticks", "?")
            print(f"  {bps}bps(={tks}t) sk={sk:<4} {r['sym']:<8} "
                  f"{r['sharpe']:>8.3f} ${r['total_pnl']:>11,.0f} "
                  f"${r['max_dd_abs']:>9,.0f} ${r['pnl_per_day']:>7,.0f}")

    return results


# ============================================================
# Phase 2: MM + Alpha
# ============================================================

def phase2_mm_alpha(phase1_results=None):
    """Add LogReg alpha signal and sweep alpha_weight."""
    print(f"\n{'='*80}", flush=True)
    print("  PHASE 2: MM + LOGREG ALPHA", flush=True)
    print(f"{'='*80}", flush=True)

    assets_map = {s.replace("USDT", ""): s for s in SMOKE_ASSETS}

    # Find best Phase 1 config per asset (or use defaults)
    default_params = {
        "half_spread_ticks": 20, "skew": 5.0,
        "order_qty_usd": CAPITAL * 0.05, "max_position_usd": CAPITAL * 0.50,
        "update_interval_ns": 1_000_000_000,
    }

    if phase1_results:
        # Pick config with best average Sharpe
        from collections import defaultdict
        config_sharpes = defaultdict(list)
        for key, r in phase1_results.items():
            hs = r["params"]["half_spread_ticks"]
            sk = r["params"]["skew"]
            config_sharpes[f"{hs}_{sk}"].append(r["sharpe"])
        if config_sharpes:
            best_config = max(config_sharpes, key=lambda k: np.mean(config_sharpes[k]))
            hs, sk = best_config.split("_")
            default_params["half_spread_ticks"] = int(hs)
            default_params["skew"] = float(sk)
            print(f"  Using best Phase 1 config: hs={hs}, skew={sk}", flush=True)

    # Generate alpha signals
    print(f"\n  Generating 15m LogReg alpha signals...", flush=True)
    alpha_signals = {}
    with ProcessPoolExecutor(max_workers=min(4, len(assets_map))) as pool:
        futs = {pool.submit(gen_alpha_signals, n, s): (n, s)
                for n, s in assets_map.items()}
        for f in as_completed(futs):
            name, sym = futs[f]
            try:
                r = f.result()
                if r is not None and len(r) > 0:
                    alpha_signals[sym] = r
                    print(f"    {sym}: {len(r):,} signals", flush=True)
            except Exception as e:
                print(f"    {sym}: ERROR {e}", flush=True)

    if not alpha_signals:
        print("  No alpha signals generated."); return None

    # Sweep alpha weights
    alpha_weights = [0.0, 0.5, 1.0, 2.0, 5.0]
    if SMOKE:
        alpha_weights = [0.0, 1.0, 3.0]

    results = {}
    for sym in SMOKE_ASSETS:
        dates = discover_dates(sym)
        if not dates:
            continue
        if SMOKE:
            dates = dates[:10]

        alpha_df = alpha_signals.get(sym)
        print(f"\n  {sym}: {len(dates)} days, alpha={'yes' if alpha_df is not None else 'no'}", flush=True)

        for aw in alpha_weights:
            t0 = time.time()
            params = {**default_params, "alpha_weight": aw}
            r = run_mm_asset(sym, dates, params, alpha_df=alpha_df,
                             fee_scenario="hl_std")
            elapsed = time.time() - t0

            if r is None:
                print(f"    aw={aw}: FAILED ({elapsed:.0f}s)", flush=True)
                continue

            key = f"{sym}_aw={aw}"
            results[key] = {**r, "sym": sym, "alpha_weight": aw}
            print(f"    aw={aw}: SR={r['sharpe']:+.3f}  "
                  f"PnL=${r['total_pnl']:+,.0f}  "
                  f"DD=${r['max_dd_abs']:,.0f}  ({elapsed:.0f}s)")

    # Summary
    if results:
        print(f"\n  {'='*60}", flush=True)
        print(f"  PHASE 2 SUMMARY: Alpha Impact", flush=True)
        print(f"  {'='*60}", flush=True)
        print(f"  {'Sym':<8} {'Alpha':>6} {'SR':>8} {'PnL':>12} {'DD%':>6}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for key, r in sorted(results.items()):
            print(f"  {r['sym']:<8} {r['alpha_weight']:>6.1f} "
                  f"{r['sharpe']:>8.3f} ${r['total_pnl']:>11,.0f} "
                  f"${r['max_dd_abs']:>8,.0f}")

    return results


# ============================================================
# Phase 3: Frequency Sweep
# ============================================================

def phase3_frequency_sweep(phase1_results=None, phase2_results=None):
    """Sweep quote update frequency."""
    print(f"\n{'='*80}", flush=True)
    print("  PHASE 3: QUOTE UPDATE FREQUENCY SWEEP", flush=True)
    print(f"{'='*80}", flush=True)

    # Use best config from Phase 1+2
    best_params = {
        "half_spread_ticks": 20, "skew": 5.0,
        "order_qty_usd": CAPITAL * 0.05, "max_position_usd": CAPITAL * 0.50,
        "alpha_weight": 0.0,
    }

    # Determine best alpha_weight from Phase 2
    if phase2_results:
        best_aw = max(
            set(r["alpha_weight"] for r in phase2_results.values()),
            key=lambda aw: np.mean([
                r["sharpe"] for r in phase2_results.values()
                if r["alpha_weight"] == aw
            ])
        )
        best_params["alpha_weight"] = best_aw
        print(f"  Using alpha_weight={best_aw} from Phase 2", flush=True)

    frequencies_ns = [
        ("100ms", 100_000_000),
        ("500ms", 500_000_000),
        ("1s", 1_000_000_000),
        ("5s", 5_000_000_000),
        ("10s", 10_000_000_000),
        ("30s", 30_000_000_000),
        ("1m", 60_000_000_000),
    ]
    if SMOKE:
        frequencies_ns = [
            ("1s", 1_000_000_000),
            ("5s", 5_000_000_000),
            ("30s", 30_000_000_000),
        ]

    # Generate alpha if needed
    alpha_signals = {}
    if best_params["alpha_weight"] > 0:
        print("  Generating alpha signals...", flush=True)
        for sym in SMOKE_ASSETS:
            name = sym.replace("USDT", "")
            sigs = gen_alpha_signals(name, sym)
            if sigs is not None:
                alpha_signals[sym] = sigs

    results = {}
    for sym in SMOKE_ASSETS:
        dates = discover_dates(sym)
        if not dates:
            continue
        if SMOKE:
            dates = dates[:10]

        alpha_df = alpha_signals.get(sym)
        print(f"\n  {sym}: {len(dates)} days", flush=True)

        for freq_name, freq_ns in frequencies_ns:
            t0 = time.time()
            params = {**best_params, "update_interval_ns": freq_ns}
            r = run_mm_asset(sym, dates, params, alpha_df=alpha_df,
                             fee_scenario="hl_std")
            elapsed = time.time() - t0

            if r is None:
                print(f"    {freq_name}: FAILED ({elapsed:.0f}s)", flush=True)
                continue

            key = f"{sym}_{freq_name}"
            results[key] = {**r, "sym": sym, "freq": freq_name, "freq_ns": freq_ns}
            print(f"    {freq_name}: SR={r['sharpe']:+.3f}  "
                  f"PnL=${r['total_pnl']:+,.0f}  "
                  f"updates={r['n_updates']:,}  ({elapsed:.0f}s)")

    # Summary
    if results:
        print(f"\n  {'='*60}", flush=True)
        print(f"  PHASE 3 SUMMARY: Frequency Impact", flush=True)
        print(f"  {'='*60}", flush=True)
        print(f"  {'Sym':<8} {'Freq':>6} {'SR':>8} {'PnL':>12} {'Updates':>10}", flush=True)
        print(f"  {'-'*55}", flush=True)
        for key, r in sorted(results.items(), key=lambda x: x[1]["freq_ns"]):
            print(f"  {r['sym']:<8} {r['freq']:>6} "
                  f"{r['sharpe']:>8.3f} ${r['total_pnl']:>11,.0f} "
                  f"{r['n_updates']:>10,}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 80, flush=True)
    print(f"  HFT MARKET MAKING BASELINE ({mode})", flush=True)
    print(f"  Avellaneda-Stoikov MM with optional LogReg alpha", flush=True)
    print("=" * 80, flush=True)

    # Discover available data
    available = {}
    for sym in (SMOKE_ASSETS if SMOKE else ALL_ASSETS):
        dates = discover_dates(sym)
        if dates:
            available[sym] = dates
    print(f"\n  Data: {len(available)} assets", flush=True)
    for sym, dates in sorted(available.items())[:5]:
        print(f"    {sym}: {len(dates)} days ({dates[0]} to {dates[-1]})", flush=True)
    if len(available) > 5:
        print(f"    ... and {len(available)-5} more", flush=True)

    # Run phases
    p1_results = None
    p2_results = None
    p3_results = None

    if PHASE_ONLY is None or PHASE_ONLY == 1:
        p1_results = phase1_pure_mm()

    if PHASE_ONLY is None or PHASE_ONLY == 2:
        p2_results = phase2_mm_alpha(p1_results)

    if PHASE_ONLY is None or PHASE_ONLY == 3:
        p3_results = phase3_frequency_sweep(p1_results, p2_results)

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print(f"  FINAL SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    all_sharpes = []
    if p1_results:
        srs = [r["sharpe"] for r in p1_results.values()]
        print(f"\n  Phase 1 (Pure MM): {len(p1_results)} configs tested", flush=True)
        print(f"    Best SR: {max(srs):+.3f}  Worst: {min(srs):+.3f}  Mean: {np.mean(srs):+.3f}", flush=True)
        all_sharpes.extend(srs)

    if p2_results:
        # Compare alpha=0 vs best alpha
        no_alpha = [r["sharpe"] for r in p2_results.values() if r["alpha_weight"] == 0]
        with_alpha = [r["sharpe"] for r in p2_results.values() if r["alpha_weight"] > 0]
        print(f"\n  Phase 2 (Alpha): {len(p2_results)} configs tested", flush=True)
        if no_alpha and with_alpha:
            print(f"    No alpha mean SR: {np.mean(no_alpha):+.3f}", flush=True)
            print(f"    With alpha best SR: {max(with_alpha):+.3f}", flush=True)
            print(f"    Alpha improvement: {max(with_alpha) - np.mean(no_alpha):+.3f}", flush=True)

    if p3_results:
        print(f"\n  Phase 3 (Frequency): {len(p3_results)} configs tested", flush=True)
        best_freq = max(p3_results.items(), key=lambda x: x[1]["sharpe"])
        print(f"    Best frequency: {best_freq[1]['freq']} (SR={best_freq[1]['sharpe']:+.3f})", flush=True)

    print(f"\n  Total runtime: {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
