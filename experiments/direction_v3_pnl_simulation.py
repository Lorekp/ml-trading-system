"""
Direction V3 — Full PnL Simulation + Quant Metrics.

Compares ALL viable configurations side by side with proper quant metrics:
  - Multiple timeframes: 15m, 30m, 1H (bar size)
  - Multiple horizons: 1-bar through 8-bar forward
  - With/without vol filter
  - At 0 bps (MEXC) and 3 bps (HL)

Walk-forward methodology: retrain monthly, predict next month.
Simulate actual PnL with $1000 starting capital, equal weight per asset.

Metrics computed:
  - Total return, CAGR
  - Sharpe ratio (annualized)
  - Sortino ratio
  - Max drawdown, max drawdown duration
  - Win rate (trades), win rate (days)
  - Profit factor
  - Average win / average loss
  - Calmar ratio (CAGR / max DD)
  - Trade count, trades per day
  - Monthly return distribution (mean, std, min, max)
"""
import sys, warnings, json
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

# All configs to compare
CONFIGS = [
    # (bar_tf, bar_resample, fwd_bars, label)
    ("15m", "15min", 1, "15m->15m"),
    ("15m", "15min", 2, "15m->30m"),
    ("15m", "15min", 4, "15m->1H"),
    ("15m", "15min", 8, "15m->2H"),
    ("30m", "30min", 1, "30m->30m"),
    ("30m", "30min", 2, "30m->1H"),
    ("30m", "30min", 4, "30m->2H"),
    ("30m", "30min", 6, "30m->3H"),
    ("30m", "30min", 8, "30m->4H"),
    ("1H", "1h", 1, "1H->1H"),
    ("1H", "1h", 2, "1H->2H"),
    ("1H", "1h", 4, "1H->4H"),
]

TEST_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")
STARTING_CAPITAL = 1000.0
VOL_PERCENTILE = 50
CONF_PERCENTILE = 70

XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)


def load_ohlcv(symbol, tf):
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
            df = pd.read_csv(f)
            df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])


def build_features(ohlcv, train_end):
    close, high, low, volume = ohlcv["close"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]
    train_mask = ohlcv.index < train_end
    features = pd.DataFrame(index=ohlcv.index)
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo = ret[train_mask].quantile(0.01)
        hi = ret[train_mask].quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)
    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (volume - volume.rolling(30).mean()) / volume.rolling(30).std().replace(0, np.nan)
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    features["macd_hist"] = macd(close)["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)
    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]
    return features


def get_daily_vol(close, span=50):
    return close.pct_change().dropna().ewm(span=span).std()


def simulate_config(symbol, bar_resample, fwd_bars, config_label, use_vol_filter):
    """Full walk-forward PnL simulation for one config. Returns trade-level series."""
    ohlcv = load_ohlcv(symbol, bar_resample)
    close = ohlcv["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
    vol = get_daily_vol(close, span=50)

    all_trades = []  # list of (timestamp, return)

    for month in TEST_MONTHS:
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC")

        features = build_features(ohlcv, train_end=month_start)
        fc = features.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)

        train_idx = valid[valid < month_start]
        test_idx = valid[(valid >= month_start) & (valid <= month_end)]

        if len(train_idx) < 500 or len(test_idx) < 20:
            continue

        # Vol filter
        if use_vol_filter:
            train_vol = vol.loc[train_idx].dropna()
            vol_threshold = train_vol.quantile(VOL_PERCENTILE / 100)
            test_vol = vol.reindex(test_idx)
            hv_mask = (test_vol > vol_threshold).fillna(False)
            test_idx = test_idx[hv_mask.values]
            if len(test_idx) < 15:
                continue

        X_train = fc.loc[train_idx]
        y_train = (y_all.loc[train_idx] > 0).astype(int)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        X_test = fc.loc[test_idx]
        y_prob = model.predict_proba(X_test)[:, 1]

        # Confidence filter
        conf = np.abs(y_prob - 0.5)
        if len(conf) > 10:
            trade_mask = conf >= np.percentile(conf, CONF_PERCENTILE)
        else:
            trade_mask = np.ones(len(conf), dtype=bool)

        direction = np.where(y_prob[trade_mask] > 0.5, 1.0, -1.0)
        actual = fwd_ret.loc[test_idx[trade_mask]].values
        timestamps = test_idx[trade_mask]

        for i in range(len(direction)):
            if np.isfinite(actual[i]):
                all_trades.append((timestamps[i], direction[i] * actual[i]))

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades, columns=["timestamp", "return"])
    trades_df = trades_df.set_index("timestamp").sort_index()
    return trades_df


def compute_quant_metrics(trades_df, cost_per_trade=0.0, label=""):
    """Compute comprehensive quant metrics from trade-level returns."""
    if trades_df is None or len(trades_df) < 10:
        return None

    rets = trades_df["return"].values
    net_rets = rets - cost_per_trade
    n = len(net_rets)

    # Time span
    t_start = trades_df.index[0]
    t_end = trades_df.index[-1]
    days = (t_end - t_start).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1

    # Basic
    total_return = np.prod(1 + net_rets) - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    trades_per_day = n / days if days > 0 else 0

    # Win/loss
    wins = net_rets[net_rets > 0]
    losses = net_rets[net_rets <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float("inf")

    # Sharpe (annualized from per-trade)
    mean_ret = np.mean(net_rets)
    std_ret = np.std(net_rets)
    if std_ret > 0 and trades_per_day > 0:
        sharpe = mean_ret / std_ret * np.sqrt(trades_per_day * 365.25)
    else:
        sharpe = 0

    # Sortino (downside deviation only)
    downside = net_rets[net_rets < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_ret
    if downside_std > 0 and trades_per_day > 0:
        sortino = mean_ret / downside_std * np.sqrt(trades_per_day * 365.25)
    else:
        sortino = 0

    # Drawdown (on cumulative equity curve)
    equity = STARTING_CAPITAL * np.cumprod(1 + net_rets)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = abs(np.min(drawdown))

    # Max drawdown duration (in trades)
    in_dd = drawdown < 0
    dd_starts = np.where(np.diff(np.concatenate([[False], in_dd]).astype(int)) == 1)[0]
    dd_ends = np.where(np.diff(np.concatenate([in_dd, [False]]).astype(int)) == -1)[0]
    max_dd_duration = 0
    if len(dd_starts) > 0 and len(dd_ends) > 0:
        for s, e in zip(dd_starts, dd_ends[:len(dd_starts)]):
            max_dd_duration = max(max_dd_duration, e - s)

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0

    # Daily PnL
    daily = trades_df["return"].resample("1D").apply(
        lambda x: np.prod(1 + x - cost_per_trade) - 1 if len(x) > 0 else 0
    )
    daily_nonzero = daily[daily != 0]
    win_days = (daily_nonzero > 0).sum()
    total_days = len(daily_nonzero)
    day_win_rate = win_days / total_days if total_days > 0 else 0

    # Monthly returns
    monthly = trades_df["return"].resample("ME").apply(
        lambda x: np.prod(1 + x - cost_per_trade) - 1 if len(x) > 0 else 0
    )
    monthly_nonzero = monthly[monthly != 0]

    final_equity = equity[-1] if len(equity) > 0 else STARTING_CAPITAL

    return {
        "label": label,
        "n_trades": n,
        "trades_per_day": round(trades_per_day, 1),
        "days": round(days, 0),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "final_equity": round(final_equity, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_dd_duration_trades": max_dd_duration,
        "calmar": round(calmar, 2),
        "win_rate_trades": round(win_rate, 4),
        "win_rate_days": round(day_win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "avg_win_bps": round(avg_win * 10000, 2),
        "avg_loss_bps": round(avg_loss * 10000, 2),
        "mean_ret_bps": round(mean_ret * 10000, 2),
        "monthly_mean_pct": round(monthly_nonzero.mean() * 100, 2) if len(monthly_nonzero) > 0 else 0,
        "monthly_std_pct": round(monthly_nonzero.std() * 100, 2) if len(monthly_nonzero) > 0 else 0,
        "monthly_min_pct": round(monthly_nonzero.min() * 100, 2) if len(monthly_nonzero) > 0 else 0,
        "monthly_max_pct": round(monthly_nonzero.max() * 100, 2) if len(monthly_nonzero) > 0 else 0,
        "months_positive": int((monthly_nonzero > 0).sum()),
        "months_total": len(monthly_nonzero),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 130, flush=True)
    print("DIRECTION V3 — FULL PnL SIMULATION + QUANT METRICS", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Starting capital: ${STARTING_CAPITAL}", flush=True)
    print(f"Assets: {ASSETS} | Configs: {len(CONFIGS)} | Vol filter: top {100-VOL_PERCENTILE}%", flush=True)
    print(f"Confidence: top {100-CONF_PERCENTILE}% | Walk-forward monthly retrain", flush=True)
    print("=" * 130, flush=True)

    # Cache OHLCV per (symbol, tf) to avoid reloading
    ohlcv_cache = {}
    all_results = []

    for bar_tf, bar_resample, fwd_bars, config_label in CONFIGS:
        print(f"\n--- Config: {config_label} ---", flush=True)

        # Aggregate across assets for this config
        config_trades_all = []

        for symbol in ASSETS:
            cache_key = (symbol, bar_resample)
            if cache_key not in ohlcv_cache:
                try:
                    ohlcv_cache[cache_key] = load_ohlcv(symbol, bar_resample)
                except Exception:
                    continue

            # With vol filter
            trades = simulate_config(symbol, bar_resample, fwd_bars, config_label, use_vol_filter=True)
            if trades is not None and len(trades) > 20:
                config_trades_all.append(trades)

        if not config_trades_all:
            print(f"  No trades for {config_label}", flush=True)
            continue

        # Merge all asset trades into one portfolio
        portfolio_trades = pd.concat(config_trades_all).sort_index()

        # Compute metrics at different cost levels
        for cost_label, cost in [("0bp_MEXC", 0.0), ("3bp_HL", 0.0003)]:
            metrics = compute_quant_metrics(portfolio_trades, cost, f"{config_label}_{cost_label}")
            if metrics:
                metrics["config"] = config_label
                metrics["cost"] = cost_label
                all_results.append(metrics)
                if cost_label == "0bp_MEXC":
                    print(f"  @MEXC: {metrics['n_trades']} trades, "
                          f"ret={metrics['total_return_pct']:+.1f}%, "
                          f"Sharpe={metrics['sharpe']:.2f}, "
                          f"Sortino={metrics['sortino']:.2f}, "
                          f"MaxDD={metrics['max_drawdown_pct']:.1f}%, "
                          f"PF={metrics['profit_factor']:.2f}, "
                          f"WR={metrics['win_rate_trades']:.3f}", flush=True)

    # Big comparison table
    print(f"\n{'='*160}", flush=True)
    print("FULL COMPARISON — ALL CONFIGS (6 assets combined, vol filter, top 30% confidence)", flush=True)
    print(f"{'='*160}", flush=True)

    # Sort by Sharpe at MEXC
    mexc_results = [r for r in all_results if r["cost"] == "0bp_MEXC"]
    mexc_sorted = sorted(mexc_results, key=lambda x: -x["sharpe"])

    print(f"\n{'Config':<14} {'Trades':>6} {'T/day':>5} | {'TotRet':>8} {'CAGR':>7} {'$Final':>8} | "
          f"{'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} | {'MaxDD':>6} {'WR_T':>5} {'WR_D':>5} {'PF':>5} | "
          f"{'AvgW':>6} {'AvgL':>7} {'MeanRet':>8} | {'Mo+':>4} {'MoMean':>7} {'MoMin':>7}", flush=True)
    print("-" * 160, flush=True)

    for r in mexc_sorted:
        print(f"{r['config']:<14} {r['n_trades']:>6} {r['trades_per_day']:>5.1f} | "
              f"{r['total_return_pct']:>+7.1f}% {r['cagr_pct']:>+6.1f}% ${r['final_equity']:>7.0f} | "
              f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} | "
              f"{r['max_drawdown_pct']:>5.1f}% {r['win_rate_trades']:>5.3f} {r['win_rate_days']:>5.3f} {r['profit_factor']:>5.2f} | "
              f"{r['avg_win_bps']:>5.1f}bp {r['avg_loss_bps']:>6.1f}bp {r['mean_ret_bps']:>+7.2f}bp | "
              f"{r['months_positive']:>2}/{r['months_total']:<2} {r['monthly_mean_pct']:>+6.2f}% {r['monthly_min_pct']:>+6.2f}%", flush=True)

    # Same for HL
    print(f"\n--- AT 3 bps RT (Hyperliquid) ---", flush=True)
    hl_results = [r for r in all_results if r["cost"] == "3bp_HL"]
    hl_sorted = sorted(hl_results, key=lambda x: -x["sharpe"])

    print(f"{'Config':<14} {'Sharpe':>7} {'TotRet':>8} {'MaxDD':>6} {'PF':>5} {'MeanRet':>8} {'WR_T':>5} {'Mo+':>5}", flush=True)
    print("-" * 70, flush=True)
    for r in hl_sorted:
        print(f"{r['config']:<14} {r['sharpe']:>7.2f} {r['total_return_pct']:>+7.1f}% "
              f"{r['max_drawdown_pct']:>5.1f}% {r['profit_factor']:>5.2f} "
              f"{r['mean_ret_bps']:>+7.2f}bp {r['win_rate_trades']:>5.3f} "
              f"{r['months_positive']:>2}/{r['months_total']}", flush=True)

    # Hypothesis test: do shorter TFs produce more profit?
    print(f"\n--- TIMEFRAME COMPARISON (at MEXC 0 bps) ---", flush=True)
    tf_groups = {}
    for r in mexc_sorted:
        tf = r["config"].split("->")[0]
        tf_groups.setdefault(tf, []).append(r)

    for tf in ["15m", "30m", "1H"]:
        group = tf_groups.get(tf, [])
        if group:
            best = max(group, key=lambda x: x["sharpe"])
            avg_sharpe = np.mean([r["sharpe"] for r in group])
            avg_trades = np.mean([r["trades_per_day"] for r in group])
            print(f"  {tf}: avg Sharpe={avg_sharpe:.2f}, avg trades/day={avg_trades:.1f}, "
                  f"best={best['config']} (Sharpe={best['sharpe']:.2f}, ret={best['total_return_pct']:+.1f}%)", flush=True)

    with open(OUTPUT_DIR / "direction_v3_pnl_simulation.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)

    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
