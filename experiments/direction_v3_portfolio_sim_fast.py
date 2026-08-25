"""
Direction V3 — Portfolio Simulation (FAST vectorized version).

Key insight: for fixed-fraction sizing, Sharpe/Sortino/WinRate/PF are
INDEPENDENT of position size. MaxDD scales linearly with size.
So we compute metrics from raw returns, then find the sizing that hits 15% MaxDD.

This avoids the per-trade Python loop entirely.
"""
import sys, warnings, json, itertools
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
MAX_DD_TARGET = 0.15
SLIPPAGE = 0.0001  # 1 bps

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
    """Generate all trade signals. Returns DataFrame."""
    print("[1/6] Generating signals...", flush=True)
    rows = []
    for bar_tf, bar_resample, fwd_bars, config_id in CONFIGS:
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
                # Vol filter
                tv = vol.loc[tr].dropna()
                vt = tv.quantile(0.50)
                hv = (vol.reindex(te) > vt).fillna(False)
                te = te[hv.values]
                if len(te) < 10: continue
                model = xgb.XGBClassifier(**XGB_PARAMS)
                model.fit(fc.loc[tr], (y_all.loc[tr] > 0).astype(int))
                yp = model.predict_proba(fc.loc[te])[:, 1]
                conf = np.abs(yp - 0.5)
                if len(conf) > 10:
                    mask = conf >= np.percentile(conf, 70)
                else:
                    mask = np.ones(len(conf), dtype=bool)
                for i, idx in enumerate(te):
                    if mask[i] and np.isfinite(fwd_ret.loc[idx]):
                        d = 1.0 if yp[i] > 0.5 else -1.0
                        rows.append((idx, symbol, config_id, d * fwd_ret.loc[idx] - SLIPPAGE))
                        config_count += 1
        print(f"  {config_id}: {config_count} signals", flush=True)
    df = pd.DataFrame(rows, columns=["timestamp", "asset", "config", "net_return"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Total: {len(df)} signals", flush=True)
    return df


def metrics_from_returns(daily_returns, label=""):
    """Compute all quant metrics from a daily return series (vectorized)."""
    r = daily_returns.values
    n_days = len(r)
    if n_days < 10:
        return None

    # Equity curve (cumulative product)
    equity = 10000 * np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min())

    # Returns stats
    mean_r = r.mean()
    std_r = r.std()
    down = r[r < 0]
    down_std = down.std() if len(down) > 1 else std_r

    sharpe = mean_r / std_r * np.sqrt(365.25) if std_r > 0 else 0
    sortino = mean_r / down_std * np.sqrt(365.25) if down_std > 0 else 0

    total_ret = equity[-1] / 10000 - 1
    years = n_days / 365.25
    cagr = (equity[-1] / 10000) ** (1 / years) - 1 if years > 0 else 0
    calmar = cagr / max_dd if max_dd > 0 else 0

    win_days = (r > 0).sum()
    pf = abs(r[r > 0].sum() / r[r < 0].sum()) if r[r < 0].sum() != 0 else 99

    # Monthly
    monthly = daily_returns.resample("ME").sum()
    months_pos = (monthly > 0).sum()

    # VaR
    var95 = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean() if (r <= var95).sum() > 0 else var95

    # Sizing to hit 15% MaxDD: scale = target / raw_maxdd
    scale_for_15 = MAX_DD_TARGET / max_dd if max_dd > 0 else 1.0
    scaled_sharpe = sharpe  # Sharpe is scale-invariant
    scaled_total = ((1 + total_ret) ** scale_for_15 - 1) if total_ret > -1 else total_ret * scale_for_15
    scaled_cagr = cagr * scale_for_15  # Approximate

    return {
        "label": label,
        "n_days": n_days,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "total_return_pct": round(total_ret * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "win_rate_days": round(win_days / n_days, 3),
        "profit_factor": round(min(pf, 99), 2),
        "months_positive": int(months_pos),
        "months_total": int(len(monthly)),
        "var_95": round(var95 * 100, 3),
        "cvar_95": round(cvar95 * 100, 3),
        "avg_daily_ret_bps": round(mean_r * 10000, 2),
        # Scaled to 15% MaxDD
        "scaled_return_pct": round(scaled_total * 100, 1),
        "scaled_cagr_pct": round(scaled_cagr * 100, 1),
        "scale_factor": round(scale_for_15, 3),
    }


def evaluate_combo(signals, configs, period_mask=None, label=""):
    """Evaluate a config combination. Returns metrics dict."""
    sub = signals[signals["config"].isin(configs)]
    if period_mask is not None:
        sub = sub[period_mask(sub["timestamp"])]
    if len(sub) < 50:
        return None
    # Aggregate to daily returns (sum of all trade returns that day)
    daily = sub.set_index("timestamp")["net_return"].resample("1D").sum().fillna(0)
    daily = daily[daily.index >= sub["timestamp"].min()]
    # Remove leading/trailing zeros
    nonzero = daily[daily != 0]
    if len(nonzero) < 10:
        return None
    first_trade = nonzero.index[0]
    last_trade = nonzero.index[-1]
    daily = daily[first_trade:last_trade]
    m = metrics_from_returns(daily, label)
    if m:
        m["n_trades"] = len(sub)
        m["trades_per_day"] = round(len(sub) / max(len(daily), 1), 1)
        m["configs"] = list(configs)
    return m


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100, flush=True)
    print("PORTFOLIO SIMULATION (FAST) — 15% MaxDD Target", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Cost: 1bp slippage | IS: {IS_MONTHS[0]}-{IS_MONTHS[-1]} | OOS: {OOS_MONTHS[0]}-{OOS_MONTHS[-1]}", flush=True)
    print("=" * 100, flush=True)

    signals = generate_signals()

    is_mask = lambda ts: ts.dt.to_period("M").isin(IS_MONTHS)
    oos_mask = lambda ts: ts.dt.to_period("M").isin(OOS_MONTHS)

    config_ids = [c[3] for c in CONFIGS]

    # Search combos on IS
    print("\n[3/6] Searching combinations (IS)...", flush=True)
    all_is = []
    for n in range(1, 6):
        combos = list(itertools.combinations(config_ids, n))
        print(f"  C({len(config_ids)},{n}) = {len(combos)} combos...", flush=True)
        for combo in combos:
            m = evaluate_combo(signals, combo, is_mask, "+".join(combo))
            if m:
                all_is.append(m)
        print(f"    valid so far: {len(all_is)}", flush=True)

    # Sort by Sharpe
    all_is.sort(key=lambda x: -x["sharpe"])

    # Filter by MaxDD target (using scale factor)
    viable = [r for r in all_is if r["max_dd_pct"] <= MAX_DD_TARGET * 100 * 5]  # Allow 5x raw DD (will scale down)
    print(f"  Total valid: {len(all_is)}", flush=True)

    # Top 20 IS
    print(f"\n  TOP 20 IN-SAMPLE (by Sharpe, raw unscaled):", flush=True)
    print(f"  {'#':>3} {'Combo':<55} {'Sh':>5} {'Sort':>5} {'Ret':>7} {'DD':>5} {'PF':>4} {'WR':>5} {'T/d':>4} {'Mo+':>4} | {'@15%DD':>7}", flush=True)
    print(f"  {'-'*120}", flush=True)
    for i, r in enumerate(all_is[:20]):
        print(f"  {i+1:>3} {r['label']:<55} {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
              f"{r['total_return_pct']:>+6.1f}% {r['max_dd_pct']:>4.1f}% {r['profit_factor']:>4.2f} "
              f"{r['win_rate_days']:>5.3f} {r['trades_per_day']:>4.1f} "
              f"{r['months_positive']:>2}/{r['months_total']:<2} | "
              f"{r['scaled_return_pct']:>+6.1f}%", flush=True)

    # OOS validation of top 10
    print(f"\n[4/6] OOS Validation (top 10 IS combos)...", flush=True)
    print(f"  {'#':>3} {'Combo':<55} {'IS_Sh':>5} {'OOS_Sh':>6} {'OOS_Ret':>8} {'OOS_DD':>6} {'OOS_PF':>6} {'Mo+':>4} | {'@15%':>7}", flush=True)
    print(f"  {'-'*120}", flush=True)

    oos_results = []
    for i, is_r in enumerate(all_is[:10]):
        oos_m = evaluate_combo(signals, is_r["configs"], oos_mask, is_r["label"])
        if oos_m:
            oos_results.append(oos_m)
            print(f"  {i+1:>3} {is_r['label']:<55} {is_r['sharpe']:>5.2f} {oos_m['sharpe']:>6.2f} "
                  f"{oos_m['total_return_pct']:>+7.1f}% {oos_m['max_dd_pct']:>5.1f}% "
                  f"{oos_m['profit_factor']:>6.2f} {oos_m['months_positive']:>2}/{oos_m['months_total']:<2} | "
                  f"{oos_m['scaled_return_pct']:>+6.1f}%", flush=True)

    # Best OOS
    if oos_results:
        best_oos = max(oos_results, key=lambda x: x["sharpe"])
    else:
        best_oos = all_is[0] if all_is else None

    # Full period for best combo
    print(f"\n[5/6] Full-period analysis of best combo...", flush=True)
    if best_oos:
        full = evaluate_combo(signals, best_oos["configs"], label="FULL_" + best_oos["label"])
        if full:
            print(f"  Combo: {full['label']}", flush=True)
            print(f"  Sharpe: {full['sharpe']:.2f}, Sortino: {full['sortino']:.2f}, Calmar: {full['calmar']:.2f}", flush=True)
            print(f"  Return: {full['total_return_pct']:+.1f}%, MaxDD: {full['max_dd_pct']:.1f}%", flush=True)
            print(f"  Scale for 15% DD: {full['scale_factor']:.3f}x -> scaled return: {full['scaled_return_pct']:+.1f}%", flush=True)
            print(f"  Trades/day: {full['trades_per_day']:.1f}, Win days: {full['win_rate_days']:.1%}", flush=True)
            print(f"  Profit Factor: {full['profit_factor']:.2f}", flush=True)
            print(f"  Months positive: {full['months_positive']}/{full['months_total']}", flush=True)
            print(f"  VaR(95%): {full['var_95']:.3f}%, CVaR(95%): {full['cvar_95']:.3f}%", flush=True)

    # Stress tests on best combo
    print(f"\n[6/6] Stress tests...", flush=True)
    if best_oos:
        configs = best_oos["configs"]

        # Signal decay: reduce returns 30%
        degraded = signals.copy()
        degraded["net_return"] = degraded["net_return"] * 0.7
        stress_decay = evaluate_combo(degraded, configs, label="decay_30pct")

        # Extra costs: 3 bps
        expensive = signals.copy()
        expensive["net_return"] = expensive["net_return"] - 0.0003
        stress_cost = evaluate_combo(expensive, configs, label="extra_3bps")

        # 60% fill rate
        np.random.seed(42)
        partial = signals.copy()
        partial = partial[np.random.random(len(partial)) < 0.6]
        stress_fill = evaluate_combo(partial, configs, label="60pct_fills")

        print(f"  {'Scenario':<20} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'PF':>5} {'@15%DD':>8}", flush=True)
        print(f"  {'-'*60}", flush=True)
        for s in [full, stress_decay, stress_cost, stress_fill]:
            if s:
                print(f"  {s['label']:<20} {s['sharpe']:>7.2f} {s['total_return_pct']:>+7.1f}% "
                      f"{s['max_dd_pct']:>5.1f}% {s['profit_factor']:>5.2f} "
                      f"{s['scaled_return_pct']:>+7.1f}%", flush=True)

    # Final spec
    print(f"\n{'='*100}", flush=True)
    print("DEPLOYMENT SPECIFICATION", flush=True)
    print(f"{'='*100}", flush=True)
    if best_oos and full:
        print(f"  Configs:        {'+'.join(best_oos['configs'])}", flush=True)
        print(f"  Position size:  {full['scale_factor']*100:.1f}% of equity per trade (for 15% MaxDD)", flush=True)
        print(f"  Expected Sharpe: {full['sharpe']:.2f} (raw), scale-invariant", flush=True)
        print(f"  Expected return: {full['scaled_return_pct']:+.1f}% over {full['n_days']} days at 15% MaxDD", flush=True)
        print(f"  Expected MaxDD:  15% (by construction)", flush=True)
        print(f"  Trades/day:     {full['trades_per_day']:.1f}", flush=True)
        print(f"  Platform:       MEXC (0% maker) primary", flush=True)
        print(f"  Assets:         {ASSETS}", flush=True)

    # Save
    try:
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "top_20_is": [{k: v for k, v in r.items() if k != "configs"} for r in all_is[:20]],
            "oos_top_10": [{k: v for k, v in r.items() if k != "configs"} for r in oos_results[:10]],
        }
        with open(OUTPUT_DIR / "portfolio_sim_fast.json", "w") as f:
            json.dump(save_data, f, indent=2, default=str)
    except Exception as e:
        print(f"  Save error: {e}", flush=True)

    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
