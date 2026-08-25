#!/usr/bin/env python
"""Realistic Execution Baseline via hftbacktest (Round-Trip).

Replays signals through hftbacktest with real L2 data (bookTicker BBO),
measuring BOTH entry AND exit fill quality to compute round-trip PnL.

Answers: "How much does realistic execution degrade our edge?"

Phases:
  1. Measure execution costs: entry/exit slippage from real book state
  2. Compare: same signals through idealized vs realistic execution
  3. Sensitivity: Sharpe at different cost levels

Data sources (in priority order):
  - data/bookTicker_parquet/{SYM}/{DATE}.parquet  (real Binance BBO)
  - data/hft-events/{SYM}/{DATE}.npz              (pre-converted events)

Usage:
    PYTHONPATH=src python experiments/hft_realistic_baseline.py [--smoke]
"""
import sys, os, time, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from hftbacktest import MARKET, GTC, FILLED

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.config import PortfolioConfig

SMOKE = "--smoke" in sys.argv

# ============================================================
# Paths
# ============================================================
HFT_DIR = Path("data/hft-events")
PARQUET_DIR = Path("data/bookTicker_parquet")
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

# ============================================================
# Event format (matches hft_data_converter.py)
# ============================================================
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

# ============================================================
# Trading constants
# ============================================================
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
}

TAKER_FEE = 0.00055   # 5.5 bps per leg
MAKER_FEE = 0.0002    # 2 bps (for hftbacktest fee model config)
CAPITAL = 250_000
BASE_SIZE_PCT = 0.045
HOLD_NS = 4 * 15 * 60 * 1_000_000_000   # 4 bars × 15min = 1h = 3.6T ns

OHLCV = ["ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
         "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
         "dist_swing_high", "dist_swing_low"]


# ============================================================
# Data Loading
# ============================================================

def bookticker_to_events(pq_path):
    """Convert bookTicker parquet to hftbacktest EVENT_DTYPE array (vectorized)."""
    df = pd.read_parquet(pq_path)
    n = len(df)
    events = np.zeros(2 * n, dtype=EVENT_DTYPE)

    ts_ns = df["transaction_time"].values.astype(np.int64) * 1_000_000

    bid_flag = np.uint64(EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | BUY_EVENT)
    ask_flag = np.uint64(EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | SELL_EVENT)

    events["ev"][0::2] = bid_flag
    events["exch_ts"][0::2] = ts_ns
    events["local_ts"][0::2] = ts_ns + 10_000_000
    events["px"][0::2] = df["best_bid_price"].values.astype(np.float64)
    events["qty"][0::2] = df["best_bid_qty"].values.astype(np.float64)

    events["ev"][1::2] = ask_flag
    events["exch_ts"][1::2] = ts_ns
    events["local_ts"][1::2] = ts_ns + 10_000_000
    events["px"][1::2] = df["best_ask_price"].values.astype(np.float64)
    events["qty"][1::2] = df["best_ask_qty"].values.astype(np.float64)

    return events


def load_hft_day(sym, date_str):
    """Load one day of event data. Prefers real bookTicker, falls back to .npz."""
    pq = PARQUET_DIR / sym / f"{date_str}.parquet"
    if pq.exists():
        return bookticker_to_events(pq)
    npz = HFT_DIR / sym / f"{date_str}.npz"
    if npz.exists():
        return np.load(npz)["data"]
    return None


def discover_dates(sym):
    """Find all dates with event data for a symbol."""
    dates = set()
    pq_dir = PARQUET_DIR / sym
    if pq_dir.exists():
        for f in pq_dir.glob("*.parquet"):
            dates.add(f.stem)
    npz_dir = HFT_DIR / sym
    if npz_dir.exists():
        for f in npz_dir.glob("*.npz"):
            dates.add(f.stem)
    return sorted(dates)


# ============================================================
# @njit Round-Trip Replay Strategy
# ============================================================

@njit
def replay_roundtrip(hbt, signal_times_ns, exit_times_ns, signal_dirs,
                     signal_sizes_usd, lot_size, tick_size):
    """Replay signals with entry AND exit market orders.

    Both signal_times_ns and exit_times_ns must be sorted ascending
    (guaranteed when signals are time-sorted and hold is constant).

    For each signal i:
      Entry at signal_times_ns[i] → market order in signal direction
      Exit  at exit_times_ns[i]   → market order in opposite direction

    Returns 7 arrays (all length n_signals):
      entry_fills, exit_fills     — actual fill prices
      entry_mids, exit_mids       — mid price at order time
      entry_qty                   — quantity traded
      entry_filled, exit_filled   — boolean fill status
    """
    n = len(signal_times_ns)
    entry_fills = np.full(n, np.nan)
    exit_fills = np.full(n, np.nan)
    entry_mids = np.full(n, np.nan)
    exit_mids = np.full(n, np.nan)
    entry_qty = np.zeros(n, dtype=np.float64)
    entry_filled = np.zeros(n, dtype=np.bool_)
    exit_filled = np.zeros(n, dtype=np.bool_)

    entry_ptr = 0
    exit_ptr = 0
    order_id = np.int64(1)

    while hbt.elapse(1_000_000_000) == 0:  # 1-second steps
        ts = hbt.current_timestamp
        depth = hbt.depth(0)

        if depth.best_bid <= 0.0 or depth.best_ask <= 0.0:
            continue

        mid = (depth.best_bid + depth.best_ask) / 2.0

        # === EXITS first: close positions whose hold period expired ===
        while exit_ptr < n and exit_times_ns[exit_ptr] <= ts:
            si = exit_ptr
            if entry_filled[si] and entry_qty[si] > 0.0:
                exit_mids[si] = mid
                if signal_dirs[si] > 0.0:
                    # Close long → sell at best bid
                    hbt.submit_sell_order(
                        0, order_id, depth.best_bid,
                        entry_qty[si], GTC, MARKET, True)
                else:
                    # Close short → buy at best ask
                    hbt.submit_buy_order(
                        0, order_id, depth.best_ask,
                        entry_qty[si], GTC, MARKET, True)

                order = hbt.orders(0).get(order_id)
                if order is not None and order.status == FILLED:
                    exit_fills[si] = order.exec_price_tick * tick_size
                    exit_filled[si] = True
                order_id += 1
            exit_ptr += 1

        # === ENTRIES: open new positions ===
        while entry_ptr < n and signal_times_ns[entry_ptr] <= ts:
            si = entry_ptr
            entry_mids[si] = mid

            qty = signal_sizes_usd[si] / mid
            if qty < lot_size:
                qty = lot_size
            qty = np.round(qty / lot_size) * lot_size

            if signal_dirs[si] > 0.0:
                hbt.submit_buy_order(
                    0, order_id, depth.best_ask,
                    qty, GTC, MARKET, True)
            else:
                hbt.submit_sell_order(
                    0, order_id, depth.best_bid,
                    qty, GTC, MARKET, True)

            order = hbt.orders(0).get(order_id)
            if order is not None and order.status == FILLED:
                entry_fills[si] = order.exec_price_tick * tick_size
                entry_filled[si] = True
                entry_qty[si] = qty
            order_id += 1
            entry_ptr += 1

        hbt.clear_inactive_orders(0)

        # Early exit if all entries and exits processed
        if entry_ptr >= n and exit_ptr >= n:
            break

    return (entry_fills, exit_fills, entry_mids, exit_mids,
            entry_qty, entry_filled, exit_filled)


# ============================================================
# Per-Asset Processing (chunked by day)
# ============================================================

def run_hft_asset(sym, signals_df, dates, chunk_days=7):
    """Run round-trip hftbacktest for one asset, processing data in chunks."""
    tick_size = TICK_SIZES.get(sym, 0.01)
    lot_size = LOT_SIZES.get(sym, 0.1)

    sigs = signals_df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(sigs["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    signal_times_ns = ts.values.astype(np.int64)
    exit_times_ns = signal_times_ns + HOLD_NS
    signal_dirs = sigs["direction"].values.astype(np.float64)

    edge = np.abs(2.0 * sigs["probability"].values - 1.0)
    edge = np.clip(edge, 0, 1)
    signal_sizes = (CAPITAL * BASE_SIZE_PCT * edge).astype(np.float64)

    n = len(sigs)
    r_entry_fills = np.full(n, np.nan)
    r_exit_fills = np.full(n, np.nan)
    r_entry_mids = np.full(n, np.nan)
    r_exit_mids = np.full(n, np.nan)
    r_entry_qty = np.zeros(n)
    r_entry_filled = np.zeros(n, dtype=bool)
    r_exit_filled = np.zeros(n, dtype=bool)

    # Build chunks: each chunk covers chunk_days entry-days + 1 overlap day
    if len(dates) <= chunk_days + 1:
        chunk_list = [dates]
    else:
        chunk_list = []
        for i in range(0, len(dates), chunk_days):
            end = min(i + chunk_days + 1, len(dates))
            chunk_list.append(dates[i:end])

    for ci, cdates in enumerate(chunk_list):
        # Entry days = all except last (overlap) day
        entry_dates = cdates[:-1] if len(cdates) > 1 else cdates
        first_ns = pd.Timestamp(entry_dates[0]).value
        last_ns = pd.Timestamp(entry_dates[-1]).value + 86_400_000_000_000

        mask = (signal_times_ns >= first_ns) & (signal_times_ns < last_ns)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        # Load event data
        events_list = []
        for d in cdates:
            ev = load_hft_day(sym, d)
            if ev is not None and len(ev) > 0:
                events_list.append(ev)
        if not events_list:
            continue
        chunk_events = np.concatenate(events_list)

        # Run hftbacktest on this chunk
        asset_cfg = (
            BacktestAsset()
                .data([chunk_events])
                .linear_asset(1.0)
                .constant_latency(10_000_000, 10_000_000)  # 10ms
                .risk_adverse_queue_model()
                .no_partial_fill_exchange()
                .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
                .tick_size(tick_size)
                .lot_size(lot_size)
                .last_trades_capacity(0)
        )

        try:
            hbt = HashMapMarketDepthBacktest([asset_cfg])
            ef, xf, em, xm, eq, efl, xfl = replay_roundtrip(
                hbt, signal_times_ns[idx], exit_times_ns[idx],
                signal_dirs[idx], signal_sizes[idx], lot_size, tick_size)
            hbt.close()
        except Exception as e:
            print(f"      Chunk {ci} ({entry_dates[0]}): {e}", flush=True)
            continue
        finally:
            del chunk_events
            gc.collect()

        # Map back to global arrays
        for j, gi in enumerate(idx):
            r_entry_fills[gi] = ef[j]
            r_exit_fills[gi] = xf[j]
            r_entry_mids[gi] = em[j]
            r_exit_mids[gi] = xm[j]
            r_entry_qty[gi] = eq[j]
            r_entry_filled[gi] = efl[j]
            r_exit_filled[gi] = xfl[j]

    return {
        "sym": sym, "n_signals": n,
        "entry_fills": r_entry_fills, "exit_fills": r_exit_fills,
        "entry_mids": r_entry_mids, "exit_mids": r_exit_mids,
        "entry_qty": r_entry_qty,
        "entry_filled": r_entry_filled, "exit_filled": r_exit_filled,
        "directions": signal_dirs, "sizes_usd": signal_sizes,
        "fwd_returns": sigs["fwd_return"].values.astype(np.float64),
        "timestamps": sigs["timestamp"].values,
    }


# ============================================================
# Signal Generation (15m only)
# ============================================================

def gen_signals(name, sym):
    """Generate 15m walk-forward signals for one asset."""
    import warnings; warnings.filterwarnings("ignore")
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.model import WalkForwardModel
    from signal_engine.filters import apply_regime_filter
    from signal_engine.config import (SystemConfig, ModelConfig, FeatureConfig,
                                      RegimeFilterConfig, ConfidenceFilterConfig,
                                      SignalConfig, DataConfig)

    config = SystemConfig(
        model=ModelConfig(C=0.10, penalty="elasticnet", solver="saga",
                          max_iter=1000, seed=42, fwd_bars=4,
                          signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=OHLCV, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30,
                               swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS],
                        timeframe="15min", regime_timeframe="4h"),
        assets={})
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
    return sigs if len(sigs) > 0 else None


# ============================================================
# Main
# ============================================================

def main():
    T0 = time.time()
    print("=" * 80)
    print("  REALISTIC EXECUTION BASELINE (Round-Trip)")
    print("  Real L2 data, risk-adverse queue, entry + exit market orders")
    print("=" * 80)

    # ---- 1. Discover data ----
    available = {}
    for sym in TICK_SIZES:
        dates = discover_dates(sym)
        if dates:
            available[sym] = dates

    if not available:
        print("\n  ERROR: No event data found in data/bookTicker_parquet/ or data/hft-events/")
        return

    print(f"\n  Data: {len(available)} assets")
    for sym, dates in sorted(available.items()):
        print(f"    {sym}: {len(dates)} days ({dates[0]} to {dates[-1]})")

    if SMOKE:
        top = sorted(available.keys())[:1]
        available = {k: v for k, v in available.items() if k in top}
        print(f"  SMOKE: {list(available.keys())} (dates filtered after signal gen)")

    # ---- 2. Generate signals ----
    print("\n  Generating 15m signals...", flush=True)
    ASSETS_MAP = {sym.replace("USDT", ""): sym for sym in available}

    all_signals = {}
    with ProcessPoolExecutor(max_workers=min(8, len(ASSETS_MAP))) as pool:
        futs = {pool.submit(gen_signals, n, s): (n, s) for n, s in ASSETS_MAP.items()}
        for f in as_completed(futs):
            name, sym = futs[f]
            try:
                r = f.result()
                if r is not None and len(r) > 0:
                    all_signals[sym] = r
                    print(f"    {sym}: {len(r):,} signals", flush=True)
            except Exception as e:
                print(f"    {sym}: ERROR {e}", flush=True)

    if not all_signals:
        print("  No signals generated."); return

    # ---- 3. Restrict to data overlap period ----
    for sym in list(all_signals.keys()):
        dates = available.get(sym, [])
        if not dates:
            del all_signals[sym]; continue
        sigs = all_signals[sym]
        ts = pd.to_datetime(sigs["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        start = pd.Timestamp(dates[0])
        end = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
        mask = (ts >= start) & (ts < end)
        all_signals[sym] = sigs[mask.values].reset_index(drop=True)
        if len(all_signals[sym]) < 5:
            del all_signals[sym]

    # In smoke mode, restrict to dates that actually have signals + 1 day overlap
    if SMOKE:
        for sym in list(all_signals.keys()):
            sigs = all_signals[sym]
            sig_ts = pd.to_datetime(sigs["timestamp"])
            if sig_ts.dt.tz is not None:
                sig_ts = sig_ts.dt.tz_localize(None)
            sig_dates = sorted(set(str(t.date()) for t in sig_ts))
            if len(sig_dates) > 10:
                sig_dates = sig_dates[:10]
                # Re-filter signals to only these dates
                cutoff = pd.Timestamp(sig_dates[-1]) + pd.Timedelta(days=1)
                mask2 = sig_ts < cutoff
                all_signals[sym] = sigs[mask2.values].reset_index(drop=True)
            needed = set(sig_dates)
            for d in sig_dates:
                needed.add(str((pd.Timestamp(d) + pd.Timedelta(days=1)).date()))
            available[sym] = sorted(d for d in available.get(sym, []) if d in needed)
        print(f"  SMOKE: restricted to signal dates + overlap")

    total_sigs = sum(len(s) for s in all_signals.values())
    print(f"\n  Overlap signals: {total_sigs:,} across {len(all_signals)} assets")

    # ---- 4. Round-trip hftbacktest replay ----
    print(f"\n{'='*80}")
    print("  PHASE 1: ROUND-TRIP EXECUTION REPLAY")
    print(f"{'='*80}")

    results = {}
    for sym, sigs in sorted(all_signals.items()):
        dates = available[sym]
        t0 = time.time()
        cdays = 3 if SMOKE else 7
        print(f"\n  {sym}: {len(sigs)} signals, {len(dates)} days (chunks of {cdays})...",
              flush=True)

        r = run_hft_asset(sym, sigs, dates, chunk_days=cdays)
        if r is None:
            print(f"    FAILED ({time.time()-t0:.0f}s)"); continue

        n_entry = r["entry_filled"].sum()
        n_exit = r["exit_filled"].sum()
        n_complete = int((r["entry_filled"] & r["exit_filled"]).sum())
        n_incomplete = int(r["entry_filled"].sum() - n_complete)
        results[sym] = r
        print(f"    Entries: {n_entry}/{r['n_signals']}"
              f"  Exits: {n_exit}/{r['n_signals']}"
              f"  Complete RT: {n_complete}"
              f"  Incomplete: {n_incomplete}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        print("\n  No results."); return

    # ---- 5. Execution quality metrics ----
    print(f"\n{'='*80}")
    print("  PHASE 2: EXECUTION QUALITY")
    print(f"{'='*80}")

    agg_entry_slip, agg_exit_slip = [], []
    agg_rt_pnl, agg_ideal_pnl = [], []
    daily_real, daily_ideal = {}, {}

    for sym, r in sorted(results.items()):
        complete = r["entry_filled"] & r["exit_filled"]
        nc = complete.sum()
        if nc == 0:
            print(f"\n  {sym}: 0 complete trades"); continue

        dirs = r["directions"][complete]
        ef = r["entry_fills"][complete]
        xf = r["exit_fills"][complete]
        em = r["entry_mids"][complete]
        xm = r["exit_mids"][complete]
        qty = r["entry_qty"][complete]
        sz = r["sizes_usd"][complete]
        fr = r["fwd_returns"][complete]
        ts_arr = r["timestamps"][complete]

        # Entry slippage: cost of crossing the spread (positive = worse)
        e_slip = np.where(dirs > 0,
                          (ef - em) / em * 10000,   # buy: paid above mid
                          (em - ef) / em * 10000)    # sell: received below mid

        # Exit slippage
        x_slip = np.where(dirs > 0,
                          (xm - xf) / xm * 10000,   # close long (sell): received below mid
                          (xf - xm) / xm * 10000)    # close short (buy): paid above mid

        # Round-trip PnL from actual fills
        gross = np.where(dirs > 0,
                         (xf - ef) * qty,            # long
                         (ef - xf) * qty)            # short
        fees = TAKER_FEE * (ef * qty + xf * qty)
        net = gross - fees

        # Idealized PnL (1bps slippage, no fees)
        ideal = sz * (dirs * fr - 0.0001)

        agg_entry_slip.extend(e_slip)
        agg_exit_slip.extend(x_slip)
        agg_rt_pnl.extend(net)
        agg_ideal_pnl.extend(ideal)

        for k in range(len(ts_arr)):
            d = str(pd.Timestamp(ts_arr[k]).date())
            daily_real[d] = daily_real.get(d, 0.0) + net[k]
            daily_ideal[d] = daily_ideal.get(d, 0.0) + ideal[k]

        print(f"\n  {sym}: {nc} complete round-trips")
        print(f"    Entry slip:  mean={np.mean(e_slip):.2f}  med={np.median(e_slip):.2f}"
              f"  p95={np.percentile(e_slip, 95):.2f} bps")
        print(f"    Exit slip:   mean={np.mean(x_slip):.2f}  med={np.median(x_slip):.2f}"
              f"  p95={np.percentile(x_slip, 95):.2f} bps")
        print(f"    Gross: ${gross.sum():,.0f}  Fees: ${fees.sum():,.0f}  Net: ${net.sum():,.0f}")
        print(f"    Idealized:   ${ideal.sum():,.0f}")

    if not agg_entry_slip:
        print("\n  No complete trades to analyze."); return

    # ---- 6. Aggregate ----
    e_all = np.array(agg_entry_slip)
    x_all = np.array(agg_exit_slip)
    rt_slip = e_all + x_all
    rt_pnl = np.array(agg_rt_pnl)
    id_pnl = np.array(agg_ideal_pnl)
    fee_bps = TAKER_FEE * 2 * 10000

    print(f"\n{'='*80}")
    print("  AGGREGATE")
    print(f"{'='*80}")
    print(f"\n  Trades:            {len(e_all):,}")
    print(f"  Entry Slip (bps):  mean={np.mean(e_all):.2f}  med={np.median(e_all):.2f}"
          f"  p95={np.percentile(e_all, 95):.2f}")
    print(f"  Exit Slip (bps):   mean={np.mean(x_all):.2f}  med={np.median(x_all):.2f}"
          f"  p95={np.percentile(x_all, 95):.2f}")
    print(f"  RT Slip (bps):     mean={np.mean(rt_slip):.2f}  med={np.median(rt_slip):.2f}"
          f"  p95={np.percentile(rt_slip, 95):.2f}")
    print(f"  Taker Fees (RT):   {fee_bps:.1f} bps")
    measured_cost = np.mean(rt_slip) + fee_bps
    print(f"  Total RT Cost:     {measured_cost:.2f} bps")
    print(f"\n  Idealized PnL:     ${id_pnl.sum():>12,.0f}")
    print(f"  Realistic PnL:     ${rt_pnl.sum():>12,.0f}")
    print(f"  Difference:        ${rt_pnl.sum() - id_pnl.sum():>+12,.0f}")

    # Daily Sharpe comparison (overlap period only)
    if daily_real and len(daily_real) >= 5:
        d_real = pd.Series(daily_real).sort_index()
        d_real.index = pd.to_datetime(d_real.index)
        d_ideal = pd.Series(daily_ideal).sort_index()
        d_ideal.index = pd.to_datetime(d_ideal.index)

        n_trades = len(e_all)
        tpd = n_trades / max(len(d_real), 1)

        s_real = compute_full_stats(d_real, CAPITAL, n_trades, tpd)
        s_ideal = compute_full_stats(d_ideal, CAPITAL, n_trades, tpd)

        print(f"\n  {'Metric':<22} {'Idealized':>12} {'Realistic':>12} {'Delta':>10}")
        print("  " + "-" * 56)
        for k in ["sharpe", "sortino", "max_dd_pct", "profit_factor", "win_rate_days"]:
            iv = s_ideal.get(k, 0)
            rv = s_real.get(k, 0)
            d = rv - iv
            fmt = ".2f%" if "pct" in k else (".1f%" if "rate" in k else ".3f")
            if "pct" in k or "rate" in k:
                print(f"  {k:<22} {iv*100 if 'rate' in k else iv:>11.2f}%"
                      f" {rv*100 if 'rate' in k else rv:>11.2f}%"
                      f" {d*100 if 'rate' in k else d:>+9.2f}%")
            else:
                print(f"  {k:<22} {iv:>12.3f} {rv:>12.3f} {d:>+10.3f}")

        if s_ideal.get("sharpe", 0) > 0:
            degradation = (s_ideal["sharpe"] - s_real.get("sharpe", 0)) / s_ideal["sharpe"] * 100
            print(f"\n  SR degradation: {degradation:.1f}%")
            if degradation < 20:
                print("  VERDICT: Edge survives realistic execution (< 20% degradation)")
            else:
                print("  WARNING: Significant SR degradation (>= 20%)")

    # ---- 7. Slippage sensitivity ----
    print(f"\n{'='*80}")
    print("  SLIPPAGE SENSITIVITY (overlap-period signals, different cost assumptions)")
    print(f"{'='*80}")

    combined = pd.concat(list(all_signals.values()), ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    slip_bps = [1, 3, 5, 7, 10, 13, 15, 20]
    print(f"\n  {'Cost (bps)':>12} {'Sharpe':>8} {'PnL':>12} {'Trades':>8}")
    print("  " + "-" * 42)

    measured_round = round(measured_cost)
    for bps in slip_bps:
        sl = bps / 10000.0
        cfg = PortfolioConfig(
            starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
            slippage=sl, max_exposure=0.90, max_per_asset=0.06,
            max_per_strategy=0.50, base_sizing_pct=BASE_SIZE_PCT,
            max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        sim = simulate_portfolio(combined, cfg)
        if sim is None:
            continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None:
            continue
        st = compute_full_stats(dpnl, CAPITAL, sim["n_trades"],
                                sim["n_trades"] / max(len(dpnl), 1))
        pnl = sim["final_equity"] - CAPITAL
        tag = " <-- measured" if bps == measured_round else ""
        tag = tag or (" <-- current" if bps == 1 else "")
        print(f"  {bps:>10}bps {st['sharpe']:>8.3f} ${pnl:>10,.0f} {sim['n_trades']:>8,}{tag}")

    print(f"\n  Measured RT cost: {measured_cost:.1f} bps (slip {np.mean(rt_slip):.1f} + fees {fee_bps:.0f})")

    # ---- 8. Execution strategy optimizer ----
    print(f"\n{'='*80}")
    print("  EXECUTION STRATEGY OPTIMIZER")
    print(f"{'='*80}")
    print("  Testing: taker, maker, hybrid, reprice, confidence-routed")
    print("  Fill model: BBO touch (ask crosses limit buy / bid crosses limit sell)")

    for sym, r in sorted(results.items()):
        complete = r["entry_filled"] & r["exit_filled"]
        if complete.sum() == 0:
            continue

        dirs = r["directions"][complete]
        fwd_rets = r["fwd_returns"][complete]
        sizes = r["sizes_usd"][complete]
        edges = np.abs(2.0 * r["exit_fills"][complete] / r["exit_fills"][complete] - 1.0)
        # Recompute edges from probability stored in sizes relationship
        edges = sizes / (CAPITAL * BASE_SIZE_PCT)
        edges = np.clip(edges, 0.01, 1.0)
        sig_ts = r["timestamps"][complete]
        tick = TICK_SIZES.get(sym, 0.1)

        # Load BBO data
        sig_dates = sorted(set(
            str(pd.Timestamp(t).date())
            for t in pd.to_datetime(sig_ts).tz_localize(None)
        ))
        bbo_chunks, loaded = [], set()
        for d in sig_dates:
            for dd in [d, str((pd.Timestamp(d) + pd.Timedelta(days=1)).date())]:
                if dd not in loaded:
                    pq = PARQUET_DIR / sym / f"{dd}.parquet"
                    if pq.exists():
                        bbo_chunks.append(pd.read_parquet(pq))
                        loaded.add(dd)
        if not bbo_chunks:
            continue

        bbo = pd.concat(bbo_chunks, ignore_index=True).sort_values("transaction_time")
        bts = bbo["transaction_time"].values.astype(np.int64) * 1_000_000
        bbid = bbo["best_bid_price"].values.astype(np.float64)
        bask = bbo["best_ask_price"].values.astype(np.float64)
        del bbo; gc.collect()

        sig_ns = pd.to_datetime(sig_ts).values.astype(np.int64)
        exit_ns = sig_ns + HOLD_NS
        n = len(dirs)

        print(f"\n  {sym}: {n} signals")

        # --- Helper: check maker fill at static price ---
        def _maker_fills(times_ns, timeout_s):
            """For each signal, check if limit fills within timeout. Returns (filled, limit_px)."""
            tns = timeout_s * 1_000_000_000
            filled = np.zeros(n, dtype=bool)
            limit_px = np.full(n, np.nan)
            for i in range(n):
                s = np.searchsorted(bts, times_ns[i], side="right") - 1
                if s < 0 or s >= len(bbid):
                    continue
                e = np.searchsorted(bts, times_ns[i] + tns, side="right")
                if dirs[i] > 0:
                    lp = bbid[s]
                    if np.any(bask[s:e] <= lp + tick * 0.01):
                        filled[i] = True; limit_px[i] = lp
                else:
                    lp = bask[s]
                    if np.any(bbid[s:e] >= lp - tick * 0.01):
                        filled[i] = True; limit_px[i] = lp
            return filled, limit_px

        # --- Helper: reprice maker (repost at current BBO every interval) ---
        def _reprice_fills(times_ns, interval_s, total_s):
            """Repricing maker: cancel and repost at current BBO every interval."""
            int_ns = interval_s * 1_000_000_000
            tot_ns = total_s * 1_000_000_000
            filled = np.zeros(n, dtype=bool)
            fill_px = np.full(n, np.nan)
            for i in range(n):
                t_cur = times_ns[i]
                t_end = t_cur + tot_ns
                while t_cur < t_end:
                    s = np.clip(np.searchsorted(bts, t_cur, side="right") - 1,
                                0, len(bbid) - 1)
                    seg_end = min(t_cur + int_ns, t_end)
                    e = np.searchsorted(bts, seg_end, side="right")
                    if dirs[i] > 0:
                        lp = bbid[s]
                        if np.any(bask[s:e] <= lp + tick * 0.01):
                            filled[i] = True; fill_px[i] = lp; break
                    else:
                        lp = bask[s]
                        if np.any(bbid[s:e] >= lp - tick * 0.01):
                            filled[i] = True; fill_px[i] = lp; break
                    t_cur = seg_end
            return filled, fill_px

        # --- Helper: taker prices ---
        def _taker_px(times_ns):
            idx = np.clip(np.searchsorted(bts, times_ns, side="right") - 1,
                          0, len(bbid) - 1)
            px = np.where(dirs > 0, bask[idx], bbid[idx])
            return px

        entry_taker_px = _taker_px(sig_ns)
        exit_taker_px = _taker_px(exit_ns)
        sig_idx = np.clip(np.searchsorted(bts, sig_ns, side="right") - 1,
                          0, len(bbid) - 1)
        entry_mid = (bbid[sig_idx] + bask[sig_idx]) / 2.0

        # Pre-compute maker fills at various timeouts
        e_mkr_30, e_px_30 = _maker_fills(sig_ns, 30)
        e_mkr_60, e_px_60 = _maker_fills(sig_ns, 60)
        x_mkr_30, x_px_30 = _maker_fills(exit_ns, 30)
        x_mkr_60, x_px_60 = _maker_fills(exit_ns, 60)
        e_rpr, e_rpx = _reprice_fills(sig_ns, 10, 60)
        x_rpr, x_rpx = _reprice_fills(exit_ns, 10, 60)

        ideal_pnl = (sizes * (dirs * fwd_rets - 0.0001)).sum()

        # --- Run all strategies ---
        def _compute_pnl(e_filled, e_px, e_fee_rate, x_filled, x_fee_rate,
                         x_px=None):
            """Compute PnL for a strategy. x_px defaults to taker."""
            if x_px is None:
                x_px = exit_taker_px
            both = e_filled & x_filled if x_filled is not None else e_filled
            if both.sum() == 0:
                return 0, 0.0, 0.0, 0
            d = dirs[both]; s = sizes[both]; em = entry_mid[both]
            ep = e_px[both]; xp = x_px[both]
            qty = s / em
            gross = np.where(d > 0, (xp - ep) * qty, (ep - xp) * qty)
            # Per-leg fees: entry_fee_rate applied to entry notional, exit to exit
            e_fees = np.where(e_filled[both],
                              np.full(both.sum(), e_fee_rate), np.full(both.sum(), TAKER_FEE))
            x_fees = np.where(x_filled[both] if x_filled is not None else np.ones(both.sum(), bool),
                              np.full(both.sum(), x_fee_rate), np.full(both.sum(), TAKER_FEE))
            fees = e_fees * ep * qty + x_fees * xp * qty
            net = gross.sum() - fees.sum()
            rt_cost = fees.sum() / np.maximum(s, 1).sum() * 10000
            return int(both.sum()), net, rt_cost, gross.sum()

        strategies = []

        # A: Pure taker
        all_true = np.ones(n, dtype=bool)
        cnt, net, cost, gross = _compute_pnl(
            all_true, entry_taker_px, TAKER_FEE,
            all_true, TAKER_FEE, exit_taker_px)
        strategies.append(("Taker / Taker", n, cnt, net, cost))

        # B: Pure maker 60s (skip unfilled)
        both_m60 = e_mkr_60 & x_mkr_60
        cnt, net, cost, _ = _compute_pnl(
            e_mkr_60, e_px_60, MAKER_FEE,
            x_mkr_60, MAKER_FEE, x_px_60)
        strategies.append(("Maker60 / Maker60", n, int(both_m60.sum()), net, cost))

        # C: Maker30→Taker / Maker30→Taker
        e_px_c = np.where(e_mkr_30, e_px_30, entry_taker_px)
        x_px_c = np.where(x_mkr_30, x_px_30, exit_taker_px)
        e_fee_c = np.where(e_mkr_30, MAKER_FEE, TAKER_FEE)
        x_fee_c = np.where(x_mkr_30, MAKER_FEE, TAKER_FEE)
        qty_c = sizes / entry_mid
        gross_c = np.where(dirs > 0, (x_px_c - e_px_c) * qty_c, (e_px_c - x_px_c) * qty_c)
        fees_c = e_fee_c * e_px_c * qty_c + x_fee_c * x_px_c * qty_c
        net_c = gross_c.sum() - fees_c.sum()
        cost_c = fees_c.sum() / sizes.sum() * 10000
        mkr_pct = (e_mkr_30.sum() + x_mkr_30.sum()) / (2 * n)
        strategies.append(("MkrTkr30 / MkrTkr30", n, n, net_c, cost_c))

        # D: Maker60→Taker / Maker60→Taker
        e_px_d = np.where(e_mkr_60, e_px_60, entry_taker_px)
        x_px_d = np.where(x_mkr_60, x_px_60, exit_taker_px)
        e_fee_d = np.where(e_mkr_60, MAKER_FEE, TAKER_FEE)
        x_fee_d = np.where(x_mkr_60, MAKER_FEE, TAKER_FEE)
        gross_d = np.where(dirs > 0, (x_px_d - e_px_d) * qty_c, (e_px_d - x_px_d) * qty_c)
        fees_d = e_fee_d * e_px_d * qty_c + x_fee_d * x_px_d * qty_c
        strategies.append(("MkrTkr60 / MkrTkr60", n, n,
                          gross_d.sum() - fees_d.sum(),
                          fees_d.sum() / sizes.sum() * 10000))

        # E: Reprice10s_60→Taker / Reprice10s_60→Taker
        e_px_e = np.where(e_rpr, e_rpx, entry_taker_px)
        x_px_e = np.where(x_rpr, x_rpx, exit_taker_px)
        e_fee_e = np.where(e_rpr, MAKER_FEE, TAKER_FEE)
        x_fee_e = np.where(x_rpr, MAKER_FEE, TAKER_FEE)
        gross_e = np.where(dirs > 0, (x_px_e - e_px_e) * qty_c, (e_px_e - x_px_e) * qty_c)
        fees_e = e_fee_e * e_px_e * qty_c + x_fee_e * x_px_e * qty_c
        strategies.append(("Rpr10s_60 / Rpr10s_60", n, n,
                          gross_e.sum() - fees_e.sum(),
                          fees_e.sum() / sizes.sum() * 10000))

        # F: Maker60→skip / Taker exit (only trade if maker entry fills)
        x_px_f = exit_taker_px
        qty_f = sizes / entry_mid
        gross_f = np.where(dirs > 0,
                           (x_px_f - e_px_60) * qty_f,
                           (e_px_60 - x_px_f) * qty_f)
        fees_f = MAKER_FEE * e_px_60 * qty_f + TAKER_FEE * x_px_f * qty_f
        mask_f = e_mkr_60
        strategies.append(("Maker60skip / Taker", n, int(mask_f.sum()),
                          (gross_f[mask_f] - fees_f[mask_f]).sum(),
                          fees_f[mask_f].sum() / sizes[mask_f].sum() * 10000 if mask_f.sum() > 0 else 0))

        # G: Confidence routing — high-edge taker, low-edge maker60→taker
        edge_med = np.median(edges)
        hi = edges > edge_med
        e_px_g = np.where(hi, entry_taker_px,
                          np.where(e_mkr_60, e_px_60, entry_taker_px))
        x_px_g = np.where(hi, exit_taker_px,
                          np.where(x_mkr_60, x_px_60, exit_taker_px))
        e_fee_g = np.where(hi, TAKER_FEE,
                           np.where(e_mkr_60, MAKER_FEE, TAKER_FEE))
        x_fee_g = np.where(hi, TAKER_FEE,
                           np.where(x_mkr_60, MAKER_FEE, TAKER_FEE))
        gross_g = np.where(dirs > 0, (x_px_g - e_px_g) * qty_c, (e_px_g - x_px_g) * qty_c)
        fees_g = e_fee_g * e_px_g * qty_c + x_fee_g * x_px_g * qty_c
        strategies.append(("EdgeRoute / EdgeRoute", n, n,
                          gross_g.sum() - fees_g.sum(),
                          fees_g.sum() / sizes.sum() * 10000))

        # Print table
        print(f"\n  {'Strategy':<26} {'Signals':>7} {'Traded':>7} {'Net PnL':>10} {'Fees bps':>9}")
        print("  " + "-" * 62)
        for name, nsig, ntrd, pnl, fbps in strategies:
            print(f"  {name:<26} {nsig:>7} {ntrd:>7} ${pnl:>8,.0f} {fbps:>8.1f}")
        print(f"  {'(Idealized 1bps)':<26} {n:>7} {n:>7} ${ideal_pnl:>8,.0f} {'1.0':>8}")

        # --- Adverse selection analysis ---
        print(f"\n  Adverse Selection (maker60 entry):")
        mkr_filled = e_mkr_60
        mkr_missed = ~e_mkr_60
        if mkr_filled.sum() > 0 and mkr_missed.sum() > 0:
            filled_ret = (dirs[mkr_filled] * fwd_rets[mkr_filled]).mean() * 10000
            missed_ret = (dirs[mkr_missed] * fwd_rets[mkr_missed]).mean() * 10000
            print(f"    Filled entry ({mkr_filled.sum():>3}): avg signed fwd_ret = {filled_ret:>+.2f} bps")
            print(f"    Missed entry ({mkr_missed.sum():>3}): avg signed fwd_ret = {missed_ret:>+.2f} bps")
            if filled_ret < missed_ret:
                print(f"    ADVERSE SELECTION: filled trades have LOWER returns ({filled_ret - missed_ret:.1f} bps)")
            else:
                print(f"    OK: filled trades have HIGHER returns")

        # --- Maker fill rate breakdown ---
        print(f"\n  Fill rates:")
        print(f"    Entry maker 30s: {e_mkr_30.sum()}/{n} ({e_mkr_30.mean():.0%})"
              f"   Exit maker 30s: {x_mkr_30.sum()}/{n} ({x_mkr_30.mean():.0%})")
        print(f"    Entry maker 60s: {e_mkr_60.sum()}/{n} ({e_mkr_60.mean():.0%})"
              f"   Exit maker 60s: {x_mkr_60.sum()}/{n} ({x_mkr_60.mean():.0%})")
        print(f"    Entry reprice:   {e_rpr.sum()}/{n} ({e_rpr.mean():.0%})"
              f"   Exit reprice:   {x_rpr.sum()}/{n} ({x_rpr.mean():.0%})")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
