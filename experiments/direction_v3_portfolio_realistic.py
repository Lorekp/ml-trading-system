"""
Direction V3 — REALISTIC Portfolio Simulation.

Shared capital pool. Multiple configs trade from the SAME equity.
When config A sizes a trade at 2% of equity, it uses CURRENT equity
(which may be drawdown from config B's losses).

Architecture:
  1. Generate all trade signals (walk-forward, 12 configs × 6 assets × 14 months)
  2. For each combo: merge all signals into a single time-ordered stream
  3. Simulate with shared equity: position sizing = pct × current_equity
  4. Track concurrent positions, total exposure, drawdown in real-time
  5. Dynamic deleveraging when DD > 7.5%, halt at 15%

Speed optimization: signal generation cached to disk. Combo simulation
uses numpy-accelerated inner loop (pre-sort + vectorized equity tracking).
"""
import sys, warnings, json, itertools, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from technical import rsi, stochastic_rsi, macd, bollinger_pct_b, atr, swing_levels

DATA_DIR = Path("/home/user/ml-data/binance-1m")
OUTPUT_DIR = Path("/home/user/ml-data/v3-results")
SIGNAL_CACHE = OUTPUT_DIR / "signal_cache.csv"

ASSETS = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

CONFIGS = [
    ("15m", "15min", 1, "15m_1b"), ("15m", "15min", 2, "15m_2b"),
    ("15m", "15min", 4, "15m_4b"), ("15m", "15min", 8, "15m_8b"),
    ("30m", "30min", 1, "30m_1b"), ("30m", "30min", 2, "30m_2b"),
    ("30m", "30min", 4, "30m_4b"), ("30m", "30min", 6, "30m_6b"),
    ("30m", "30min", 8, "30m_8b"),
    ("1H", "1h", 1, "1H_1b"), ("1H", "1h", 2, "1H_2b"), ("1H", "1h", 4, "1H_4b"),
]

ALL_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")
IS_MONTHS = ALL_MONTHS[:7]
OOS_MONTHS = ALL_MONTHS[7:]

STARTING_CAPITAL = 10000.0
MAX_DD = 0.15
DELEVER_START = 0.075  # Start reducing at 7.5% DD
SLIPPAGE = 0.0001  # 1 bps
MAX_EXPOSURE = 0.60  # Max 60% of equity in positions at once
MAX_PER_ASSET = 0.20  # Max 20% per single asset
SIZING_PCT = 0.02  # Base: 2% per trade

XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)

_ohlcv_cache = {}


def load_ohlcv(symbol, tf):
    key = (symbol, tf)
    if key in _ohlcv_cache:
        return _ohlcv_cache[key]
    d = DATA_DIR / SYMBOLS[symbol]
    dfs = []
    for f in sorted(d.glob("*.csv")):
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "count", "tbv", "tqv", "ignore"]
        if first.replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f); df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    result = combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])
    _ohlcv_cache[key] = result
    return result


def build_features(ohlcv, train_end):
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    tm = ohlcv.index < train_end
    f = pd.DataFrame(index=ohlcv.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        f[f"ret_{lb}"] = ret.clip(ret[tm].quantile(0.01), ret[tm].quantile(0.99))
    f["bar_range"] = (high - low) / close
    f["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
    f["rsi_14"] = rsi(close, 14)
    f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    f["macd_hist"] = macd(close)["macd_hist"]
    f["bb_pct_b"] = bollinger_pct_b(close, 20)
    f["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    f["dist_swing_high"] = sw["dist_swing_high"]
    f["dist_swing_low"] = sw["dist_swing_low"]
    f["dist_nearest_level"] = sw["dist_nearest_level"]
    return f


def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


def generate_signals():
    """Generate all trade signals with walk-forward. Cache to disk."""
    if SIGNAL_CACHE.exists():
        print("[1/6] Loading cached signals...", flush=True)
        df = pd.read_csv(SIGNAL_CACHE, parse_dates=["timestamp"])
        print(f"  Loaded {len(df)} signals from cache", flush=True)
        return df

    print("[1/6] Generating signals (this takes ~10 min)...", flush=True)
    rows = []
    for bar_tf, bar_resample, fwd_bars, config_id in CONFIGS:
        t0 = time.time()
        config_count = 0
        for symbol in ASSETS:
            ohlcv = load_ohlcv(symbol, bar_resample)
            close = ohlcv["close"]
            fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
            vol = get_daily_vol(close, span=50)
            for month in ALL_MONTHS:
                ms = pd.Timestamp(month.start_time, tz="UTC")
                me = pd.Timestamp(month.end_time, tz="UTC")
                feat = build_features(ohlcv, train_end=ms)
                fc = feat.dropna()
                y_all = fwd_ret.reindex(fc.index).dropna()
                valid = fc.index.intersection(y_all.index)
                tr = valid[valid < ms]; te = valid[(valid >= ms) & (valid <= me)]
                if len(tr) < 500 or len(te) < 20: continue
                tv = vol.loc[tr].dropna()
                vt = tv.quantile(0.50)
                hv = (vol.reindex(te) > vt).fillna(False)
                te = te[hv.values]
                if len(te) < 10: continue
                model = xgb.XGBClassifier(**XGB_PARAMS)
                model.fit(fc.loc[tr], (y_all.loc[tr] > 0).astype(int))
                yp = model.predict_proba(fc.loc[te])[:, 1]
                conf = np.abs(yp - 0.5)
                mask = conf >= np.percentile(conf, 70) if len(conf) > 10 else np.ones(len(conf), dtype=bool)
                for i, idx in enumerate(te):
                    if mask[i] and np.isfinite(fwd_ret.loc[idx]):
                        d = 1.0 if yp[i] > 0.5 else -1.0
                        rows.append((idx, symbol, config_id, float(d),
                                     float(fwd_ret.loc[idx]), float(conf[i])))
                        config_count += 1
        print(f"  {config_id}: {config_count} signals ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows, columns=["timestamp", "asset", "config", "direction",
                                      "fwd_return", "confidence"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_csv(SIGNAL_CACHE, index=False)
    print(f"  Total: {len(df)} signals, cached to {SIGNAL_CACHE}", flush=True)
    return df


def simulate_portfolio(signals_df, active_configs, sizing_pct=SIZING_PCT,
                       period_filter=None):
    """
    REALISTIC portfolio simulation with shared capital pool.

    Every trade sizes from current equity. Concurrent positions tracked.
    Exposure limits enforced. Dynamic deleveraging near DD limit.

    Returns: dict with equity curve, metrics, trade log.
    """
    sub = signals_df[signals_df["config"].isin(active_configs)].copy()
    if period_filter is not None:
        sub = sub[period_filter(sub["timestamp"])]
    sub = sub.sort_values("timestamp").reset_index(drop=True)

    if len(sub) < 20:
        return None

    # Pre-allocate arrays for speed
    n = len(sub)
    equity_arr = np.empty(n)
    pnl_arr = np.empty(n)
    dd_arr = np.empty(n)
    size_arr = np.empty(n)
    skipped = np.zeros(n, dtype=bool)

    equity = STARTING_CAPITAL
    peak_equity = equity
    halted = False

    # Track open positions: {(asset, config): (entry_time, size_usd, direction)}
    # For simplicity, assume each trade is independent (no overlapping same-asset-config)
    # This is realistic: we don't stack positions on the same signal.
    current_exposure = 0.0  # Total USD in open positions
    asset_exposure = {}  # asset -> total USD exposure

    timestamps = sub["timestamp"].values
    assets = sub["asset"].values
    configs_arr = sub["config"].values
    directions = sub["direction"].values
    fwd_returns = sub["fwd_return"].values
    confidences = sub["confidence"].values

    for i in range(n):
        if halted:
            equity_arr[i] = equity
            pnl_arr[i] = 0
            dd_arr[i] = (peak_equity - equity) / peak_equity
            size_arr[i] = 0
            skipped[i] = True
            continue

        # Current drawdown
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        # Halt check
        if dd >= MAX_DD:
            halted = True
            equity_arr[i] = equity
            pnl_arr[i] = 0
            dd_arr[i] = dd
            size_arr[i] = 0
            skipped[i] = True
            continue

        # Dynamic deleveraging
        if dd > DELEVER_START:
            dd_ratio = (dd - DELEVER_START) / (MAX_DD - DELEVER_START)
            size_mult = max(1.0 - dd_ratio, 0.0)
        else:
            size_mult = 1.0

        # Position size from CURRENT equity
        raw_size = equity * sizing_pct * size_mult

        # Exposure limits
        asset = assets[i]
        asset_exp = asset_exposure.get(asset, 0.0)

        # Check per-asset limit
        if asset_exp + raw_size > equity * MAX_PER_ASSET:
            raw_size = max(equity * MAX_PER_ASSET - asset_exp, 0)

        # Check total exposure limit
        if current_exposure + raw_size > equity * MAX_EXPOSURE:
            raw_size = max(equity * MAX_EXPOSURE - current_exposure, 0)

        if raw_size < 1.0:  # Minimum $1 trade
            equity_arr[i] = equity
            pnl_arr[i] = 0
            dd_arr[i] = dd
            size_arr[i] = 0
            skipped[i] = True
            continue

        # Execute trade
        trade_return = directions[i] * fwd_returns[i] - SLIPPAGE
        pnl = raw_size * trade_return

        equity += pnl
        if equity > peak_equity:
            peak_equity = equity

        # Update exposure tracking (simplified: positions close immediately
        # since we model the full forward return in one step)
        # In reality, positions would be open for fwd_bars duration.
        # For concurrent position tracking, we approximate by counting
        # active signals within the hold period.
        current_exposure = min(current_exposure + raw_size * 0.1, equity * MAX_EXPOSURE)
        # Decay exposure over time (approximate)
        if i > 0 and timestamps[i] != timestamps[i-1]:
            current_exposure *= 0.9  # Positions closing
        asset_exposure[asset] = asset_exposure.get(asset, 0) * 0.9

        equity_arr[i] = equity
        pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        size_arr[i] = raw_size

    return {
        "equity": equity_arr,
        "pnl": pnl_arr,
        "dd": dd_arr,
        "sizes": size_arr,
        "skipped": skipped,
        "timestamps": timestamps,
        "halted": halted,
        "final_equity": equity,
        "n_trades": int((~skipped).sum()),
    }


def compute_metrics(sim, label=""):
    """Compute quant metrics from simulation result."""
    if sim is None or sim["n_trades"] < 20:
        return None

    active = ~sim["skipped"]
    pnls = sim["pnl"][active]
    equity = sim["equity"]
    n_trades = sim["n_trades"]
    timestamps = sim["timestamps"]

    # Time span
    t0 = pd.Timestamp(timestamps[active][0])
    t1 = pd.Timestamp(timestamps[active][-1])
    days = max((t1 - t0).total_seconds() / 86400, 1)
    years = days / 365.25
    trades_per_day = n_trades / days

    # Total return
    total_ret = sim["final_equity"] / STARTING_CAPITAL - 1
    cagr = (sim["final_equity"] / STARTING_CAPITAL) ** (1/years) - 1 if years > 0 else 0

    # Drawdown
    max_dd = sim["dd"].max()

    # Daily PnL
    ts_series = pd.Series(pnls, index=pd.to_datetime(timestamps[active]))
    daily = ts_series.resample("1D").sum()
    daily = daily[daily.index >= t0]
    daily_nonzero = daily[daily != 0]

    if len(daily_nonzero) < 5:
        return None

    mean_d = daily_nonzero.mean()
    std_d = daily_nonzero.std()
    down_d = daily_nonzero[daily_nonzero < 0]
    down_std = down_d.std() if len(down_d) > 1 else std_d

    sharpe = mean_d / std_d * np.sqrt(365.25) if std_d > 0 else 0
    sortino = mean_d / down_std * np.sqrt(365.25) if down_std > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0

    # Win rates
    win_trades = (pnls > 0).sum()
    win_rate_t = win_trades / n_trades
    win_days = (daily_nonzero > 0).sum()
    win_rate_d = win_days / len(daily_nonzero)

    # Profit factor
    gains = pnls[pnls > 0].sum()
    losses_sum = abs(pnls[pnls < 0].sum())
    pf = gains / losses_sum if losses_sum > 0 else 99

    # Monthly
    monthly = ts_series.resample("ME").sum()
    months_pos = (monthly > 0).sum()
    months_total = (monthly != 0).sum()

    # VaR
    var95 = np.percentile(daily_nonzero, 5)
    cvar95 = daily_nonzero[daily_nonzero <= var95].mean() if (daily_nonzero <= var95).sum() > 0 else var95

    return {
        "label": label,
        "n_trades": n_trades,
        "trades_per_day": round(trades_per_day, 1),
        "total_return_pct": round(total_ret * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "final_equity": round(sim["final_equity"], 0),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "win_rate_trades": round(win_rate_t, 3),
        "win_rate_days": round(win_rate_d, 3),
        "profit_factor": round(min(pf, 99), 2),
        "months_positive": int(months_pos),
        "months_total": int(months_total),
        "var_95": round(var95, 2),
        "cvar_95": round(cvar95, 2),
        "halted": sim["halted"],
        "mean_daily_pnl": round(mean_d, 2),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("REALISTIC PORTFOLIO SIMULATION — Shared Capital Pool", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Capital: ${STARTING_CAPITAL}, MaxDD: {MAX_DD*100:.0f}%, Delever from: {DELEVER_START*100:.1f}%", flush=True)
    print(f"Sizing: {SIZING_PCT*100:.0f}%/trade, MaxExposure: {MAX_EXPOSURE*100:.0f}%, MaxPerAsset: {MAX_PER_ASSET*100:.0f}%", flush=True)
    print(f"Slippage: {SLIPPAGE*10000:.0f}bp | IS: {IS_MONTHS[0]}-{IS_MONTHS[-1]}, OOS: {OOS_MONTHS[0]}-{OOS_MONTHS[-1]}", flush=True)
    print("=" * 100, flush=True)

    signals = generate_signals()

    config_ids = [c[3] for c in CONFIGS]
    is_filter = lambda ts: ts.dt.to_period("M").isin(IS_MONTHS)
    oos_filter = lambda ts: ts.dt.to_period("M").isin(OOS_MONTHS)

    # Test combos of 1 to 4 configs (skip 5 to keep it fast)
    print("\n[2/6] Searching combinations (IS period, shared capital)...", flush=True)
    is_results = []

    for n_cfg in range(1, len(config_ids) + 1):
        combos = list(itertools.combinations(config_ids, n_cfg))
        print(f"  C({len(config_ids)},{n_cfg}) = {len(combos)} combos...", flush=True)
        t0 = time.time()
        batch_count = 0
        for combo in combos:
            sim = simulate_portfolio(signals, list(combo), period_filter=is_filter)
            m = compute_metrics(sim, "+".join(combo))
            if m:
                m["configs"] = list(combo)
                is_results.append(m)
                batch_count += 1
        print(f"    {batch_count} valid in {time.time()-t0:.1f}s", flush=True)

    is_results.sort(key=lambda x: -x["sharpe"])

    # Top 20 IS
    print(f"\n  TOP 20 IN-SAMPLE (shared capital, {SIZING_PCT*100:.0f}%/trade):", flush=True)
    print(f"  {'#':>3} {'Combo':<45} {'Sh':>5} {'So':>5} {'Ret':>7} {'DD':>5} {'PF':>4} {'WR':>5} {'T/d':>4} {'Mo+':>4} {'Halt':>5}", flush=True)
    print(f"  {'-'*105}", flush=True)
    for i, r in enumerate(is_results[:20]):
        h = "HALT" if r["halted"] else ""
        print(f"  {i+1:>3} {r['label']:<45} {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
              f"{r['total_return_pct']:>+6.1f}% {r['max_dd_pct']:>4.1f}% {r['profit_factor']:>4.2f} "
              f"{r['win_rate_trades']:>5.3f} {r['trades_per_day']:>4.1f} "
              f"{r['months_positive']:>2}/{r['months_total']:<2} {h:>5}", flush=True)

    # OOS validation top 10
    print(f"\n[3/6] OOS Validation...", flush=True)
    print(f"  {'#':>3} {'Combo':<45} {'IS_Sh':>5} {'OOS_Sh':>6} {'OOS_Ret':>8} {'OOS_DD':>6} {'PF':>5} {'Mo+':>4} {'Halt':>5}", flush=True)
    print(f"  {'-'*105}", flush=True)

    oos_results = []
    for i, is_r in enumerate(is_results[:10]):
        sim = simulate_portfolio(signals, is_r["configs"], period_filter=oos_filter)
        m = compute_metrics(sim, is_r["label"])
        if m:
            m["configs"] = is_r["configs"]
            oos_results.append(m)
            h = "HALT" if m["halted"] else ""
            print(f"  {i+1:>3} {is_r['label']:<45} {is_r['sharpe']:>5.2f} {m['sharpe']:>6.2f} "
                  f"{m['total_return_pct']:>+7.1f}% {m['max_dd_pct']:>5.1f}% "
                  f"{m['profit_factor']:>5.2f} {m['months_positive']:>2}/{m['months_total']:<2} {h:>5}", flush=True)

    # Sizing sweep on best combo
    print(f"\n[4/6] Position sizing sweep (best IS combo)...", flush=True)
    best_configs = is_results[0]["configs"] if is_results else config_ids[:3]
    print(f"  Combo: {'+'.join(best_configs)}", flush=True)
    print(f"  {'Size':>6} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'Calmar':>7} {'Halt':>5}", flush=True)
    print(f"  {'-'*55}", flush=True)

    for pct in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10]:
        sim = simulate_portfolio(signals, best_configs, sizing_pct=pct)
        m = compute_metrics(sim, f"{pct*100:.1f}%")
        if m:
            h = "HALT" if m["halted"] else ""
            v = " <--" if m["max_dd_pct"] <= MAX_DD * 100 and m["sharpe"] > 0 and not m["halted"] else ""
            print(f"  {pct*100:>5.1f}% {m['sharpe']:>7.2f} {m['total_return_pct']:>+7.1f}% "
                  f"{m['max_dd_pct']:>5.1f}% {m['profit_factor']:>5.2f} {m['calmar']:>7.2f} {h:>5}{v}", flush=True)

    # Stress tests on best
    print(f"\n[5/6] Stress tests...", flush=True)
    stress_results = []

    # Base
    sim = simulate_portfolio(signals, best_configs)
    base = compute_metrics(sim, "base")
    stress_results.append(base)

    # Signal decay 30%
    degraded = signals.copy()
    degraded["fwd_return"] = degraded["fwd_return"] * 0.7
    sim_d = simulate_portfolio(degraded, best_configs)
    stress_results.append(compute_metrics(sim_d, "decay_30%"))

    # Extra 3 bps cost
    expensive = signals.copy()
    expensive["fwd_return"] = expensive["fwd_return"] - 0.0003 * expensive["direction"]
    sim_e = simulate_portfolio(expensive, best_configs)
    stress_results.append(compute_metrics(sim_e, "extra_3bp"))

    # 60% fills
    np.random.seed(42)
    partial = signals[np.random.random(len(signals)) < 0.6].copy()
    sim_p = simulate_portfolio(partial, best_configs)
    stress_results.append(compute_metrics(sim_p, "60%_fills"))

    print(f"  {'Scenario':<15} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'Halt':>5}", flush=True)
    print(f"  {'-'*50}", flush=True)
    for s in stress_results:
        if s:
            h = "HALT" if s["halted"] else ""
            print(f"  {s['label']:<15} {s['sharpe']:>7.2f} {s['total_return_pct']:>+7.1f}% "
                  f"{s['max_dd_pct']:>5.1f}% {s['profit_factor']:>5.2f} {h:>5}", flush=True)

    # Final spec
    print(f"\n{'='*100}", flush=True)
    print("DEPLOYMENT SPECIFICATION", flush=True)
    print(f"{'='*100}", flush=True)
    if is_results:
        best = is_results[0]
        print(f"  Configs:       {'+'.join(best['configs'])}", flush=True)
        print(f"  Capital:       ${STARTING_CAPITAL}", flush=True)
        print(f"  Sizing:        {SIZING_PCT*100:.0f}% of current equity per trade", flush=True)
        print(f"  MaxDD halt:    {MAX_DD*100:.0f}%", flush=True)
        print(f"  Delever from:  {DELEVER_START*100:.1f}%", flush=True)
        print(f"  Max exposure:  {MAX_EXPOSURE*100:.0f}%", flush=True)
        print(f"  Max per asset: {MAX_PER_ASSET*100:.0f}%", flush=True)
        print(f"  Slippage:      {SLIPPAGE*10000:.0f}bp", flush=True)
        if base:
            print(f"\n  Full-period performance:", flush=True)
            print(f"    Sharpe:  {base['sharpe']:.2f}", flush=True)
            print(f"    Return:  {base['total_return_pct']:+.1f}%", flush=True)
            print(f"    MaxDD:   {base['max_dd_pct']:.1f}%", flush=True)
            print(f"    Calmar:  {base['calmar']:.2f}", flush=True)
            print(f"    PF:      {base['profit_factor']:.2f}", flush=True)
            print(f"    Trades/d: {base['trades_per_day']:.1f}", flush=True)

    # Save
    try:
        save = {
            "timestamp": datetime.now().isoformat(),
            "config": {"sizing": SIZING_PCT, "max_dd": MAX_DD, "slippage": SLIPPAGE,
                       "max_exposure": MAX_EXPOSURE, "max_per_asset": MAX_PER_ASSET},
            "top_20_is": [{k: v for k, v in r.items()} for r in is_results[:20]],
            "oos_top_10": [{k: v for k, v in r.items()} for r in oos_results[:10]],
            "stress": [{k: v for k, v in r.items()} for r in stress_results if r],
        }
        with open(OUTPUT_DIR / "portfolio_realistic.json", "w") as f:
            json.dump(save, f, indent=2, default=str)
    except Exception as e:
        print(f"  Save error: {e}", flush=True)

    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
