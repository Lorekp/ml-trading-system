"""
Direction V3 — Portfolio Simulation.

Simulates trading SUI, XRP, BNB, BTC, ETH simultaneously using walk-forward
models with vol filter. Key questions:
  1. What is the PORTFOLIO Sharpe when combining assets?
  2. Do asset returns correlate (concentrated risk) or diversify?
  3. What is max drawdown? Daily PnL distribution?
  4. How many trades per day across all assets?

Uses walk-forward methodology: retrain monthly, predict next month.
Reports at 0 bps (MEXC) and 3 bps (HL).
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
FWD_BARS = 4  # 30m x 4 = 2H

PORTFOLIO_ASSETS = ["SUI", "XRP", "BNB", "BTC", "ETH"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
           "BNB": "BNBUSDT", "SUI": "SUIUSDT"}

TEST_MONTHS = pd.period_range("2025-01", "2026-02", freq="M")

XGB_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=100,
                  min_child_weight=50, subsample=0.7, colsample_bytree=0.7,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", random_state=42)

# Capital allocation: equal weight
CAPITAL_PER_ASSET = 1.0  # Normalized to $1 per asset
CONFIDENCE_THRESHOLD_PCT = 70  # Top 30% confidence


def load_ohlcv(symbol, tf="30min"):
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


def generate_monthly_trades(symbol, ohlcv, month):
    """Generate trade signals for one asset for one month. Returns per-bar PnL series."""
    close = ohlcv["close"]
    fwd_ret = close.pct_change(FWD_BARS).shift(-FWD_BARS)
    vol = get_daily_vol(close, span=50)

    month_start = pd.Timestamp(month.start_time, tz="UTC")
    month_end = pd.Timestamp(month.end_time, tz="UTC")

    features = build_features(ohlcv, train_end=month_start)
    fc = features.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)

    train_idx = valid[valid < month_start]
    test_idx = valid[(valid >= month_start) & (valid <= month_end)]

    if len(train_idx) < 500 or len(test_idx) < 30:
        return pd.Series(dtype=float)

    X_train = fc.loc[train_idx]
    y_train = (y_all.loc[train_idx] > 0).astype(int)
    X_test = fc.loc[test_idx]

    # Vol filter: train on all, predict only on high-vol bars
    train_vol = vol.loc[train_idx].dropna()
    vol_threshold = train_vol.quantile(0.50)
    test_vol = vol.reindex(test_idx)
    hv_mask = (test_vol > vol_threshold).fillna(False)
    hv_test = test_idx[hv_mask.values]

    if len(hv_test) < 20:
        # Fallback: use all test bars
        hv_test = test_idx

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    X_hv = fc.loc[hv_test]
    y_prob = model.predict_proba(X_hv)[:, 1]

    # Confidence filter: only trade top 30%
    conf = np.abs(y_prob - 0.5)
    conf_threshold = np.percentile(conf, CONFIDENCE_THRESHOLD_PCT)
    trade_mask = conf >= conf_threshold

    # Direction: +1 if prob > 0.5, -1 if prob < 0.5
    direction = np.where(y_prob > 0.5, 1.0, -1.0)

    # Per-trade return = direction * actual forward return
    actual_fwd = fwd_ret.reindex(hv_test).values

    # Only include trades that pass confidence filter
    trade_returns = pd.Series(index=hv_test, dtype=float)
    for i, idx in enumerate(hv_test):
        if trade_mask[i] and np.isfinite(actual_fwd[i]):
            trade_returns.loc[idx] = direction[i] * actual_fwd[i]

    return trade_returns.dropna()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90, flush=True)
    print("DIRECTION V3 — PORTFOLIO SIMULATION", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Assets: {PORTFOLIO_ASSETS}", flush=True)
    print(f"Config: 30m->2H, vol filter, top 30% confidence, walk-forward monthly retrain", flush=True)
    print(f"Months: {TEST_MONTHS[0]} to {TEST_MONTHS[-1]} ({len(TEST_MONTHS)} months)", flush=True)
    print("=" * 90, flush=True)

    # Load all OHLCV once
    ohlcv_data = {}
    for sym in PORTFOLIO_ASSETS:
        print(f"  Loading {sym}...", flush=True)
        ohlcv_data[sym] = load_ohlcv(sym, "30min")

    # Generate per-asset trade returns for each month
    all_asset_trades = {sym: [] for sym in PORTFOLIO_ASSETS}

    for month in TEST_MONTHS:
        print(f"\n  Month {month}:", flush=True)
        for sym in PORTFOLIO_ASSETS:
            trades = generate_monthly_trades(sym, ohlcv_data[sym], month)
            n_trades = len(trades)
            if n_trades > 0:
                mean_ret = trades.mean() * 10000
                win_rate = (trades > 0).mean()
                print(f"    {sym}: {n_trades:>4} trades, mean={mean_ret:+.1f}bp, wr={win_rate:.3f}", flush=True)
                all_asset_trades[sym].append(trades)
            else:
                print(f"    {sym}: no trades", flush=True)

    # Combine into portfolio
    print(f"\n{'='*90}", flush=True)
    print("PORTFOLIO ANALYSIS", flush=True)
    print(f"{'='*90}", flush=True)

    # Merge all trades into a single time-indexed DataFrame
    asset_pnl = {}
    for sym in PORTFOLIO_ASSETS:
        if all_asset_trades[sym]:
            combined = pd.concat(all_asset_trades[sym])
            asset_pnl[sym] = combined.sort_index()

    if not asset_pnl:
        print("  No trades generated. Exiting.", flush=True)
        return

    # Per-asset stats
    print(f"\n--- PER-ASSET STATISTICS ---", flush=True)
    print(f"{'Asset':<6} {'Trades':>7} {'MeanRet':>9} {'WinRate':>8} {'Sharpe':>7} | "
          f"{'Net@0bp':>8} {'Net@3bp':>8}", flush=True)
    print("-" * 70, flush=True)

    asset_daily_pnl = {}
    for sym, trades in asset_pnl.items():
        mean_ret = trades.mean()
        std_ret = trades.std()
        win_rate = (trades > 0).mean()
        n = len(trades)

        # Approximate annualized Sharpe
        # ~12 trades/day per asset (24h / 2h per trade), but with vol+conf filter ~3-4/day
        trades_per_day = n / ((trades.index[-1] - trades.index[0]).total_seconds() / 86400)
        trades_per_year = trades_per_day * 365.25

        sharpe_0 = (mean_ret / std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0
        net_0_bps = mean_ret * 10000
        net_3_bps = (mean_ret - 0.0003) * 10000  # 3 bps = 0.03% = 0.0003

        print(f"{sym:<6} {n:>7} {mean_ret*10000:>+8.2f}bp {win_rate:>7.3f} {sharpe_0:>7.2f} | "
              f"{net_0_bps:>+7.2f}bp {net_3_bps:>+7.2f}bp", flush=True)

        # Resample to daily PnL
        daily = trades.resample("1D").sum().fillna(0)
        asset_daily_pnl[sym] = daily

    # Portfolio daily PnL (equal weight)
    print(f"\n--- PORTFOLIO (EQUAL WEIGHT, {len(asset_pnl)} ASSETS) ---", flush=True)

    portfolio_daily = pd.DataFrame(asset_daily_pnl)
    # Fill missing days with 0
    all_dates = portfolio_daily.index
    portfolio_daily = portfolio_daily.fillna(0)
    portfolio_daily["total"] = portfolio_daily.sum(axis=1)

    total_daily = portfolio_daily["total"]
    total_daily_nonzero = total_daily[total_daily != 0]

    if len(total_daily_nonzero) > 10:
        for cost_label, cost_per_trade in [("0 bps (MEXC)", 0), ("3 bps (HL)", 0.0003)]:
            # Count trades per day across all assets
            total_trades = sum(len(t) for t in asset_pnl.values())
            trading_days = len(total_daily_nonzero)
            trades_per_day = total_trades / trading_days if trading_days > 0 else 0

            # Daily cost = trades_per_day * cost_per_trade (as fraction of position)
            daily_cost = trades_per_day * cost_per_trade
            net_daily = total_daily_nonzero - daily_cost

            mean_daily = net_daily.mean()
            std_daily = net_daily.std()
            sharpe = (mean_daily / std_daily * np.sqrt(365.25)) if std_daily > 0 else 0

            # Cumulative PnL
            cum_pnl = net_daily.cumsum()
            max_dd = 0
            peak = cum_pnl.iloc[0]
            for val in cum_pnl:
                if val > peak:
                    peak = val
                dd = (peak - val) / abs(peak) if peak != 0 else 0
                if dd > max_dd:
                    max_dd = dd

            total_return = cum_pnl.iloc[-1]
            win_days = (net_daily > 0).sum()
            total_days = len(net_daily)

            print(f"\n  At {cost_label}:", flush=True)
            print(f"    Trading days:     {trading_days}", flush=True)
            print(f"    Total trades:     {total_trades} ({trades_per_day:.1f}/day)", flush=True)
            print(f"    Mean daily PnL:   {mean_daily*10000:+.2f} bps", flush=True)
            print(f"    Daily Sharpe:     {mean_daily/std_daily:.3f}" if std_daily > 0 else "    Daily Sharpe:     N/A", flush=True)
            print(f"    Annual Sharpe:    {sharpe:.2f}", flush=True)
            print(f"    Win days:         {win_days}/{total_days} ({100*win_days/total_days:.0f}%)", flush=True)
            print(f"    Total return:     {total_return*100:.2f}% (on equal-weight portfolio)", flush=True)
            print(f"    Max drawdown:     {max_dd*100:.1f}%", flush=True)

    # Correlation between assets
    print(f"\n--- ASSET RETURN CORRELATION ---", flush=True)
    corr = portfolio_daily[PORTFOLIO_ASSETS].corr()
    for sym in PORTFOLIO_ASSETS:
        if sym in corr.columns:
            others = [f"{s}:{corr.loc[sym,s]:.2f}" for s in PORTFOLIO_ASSETS if s != sym and s in corr.columns]
            print(f"  {sym}: {', '.join(others)}", flush=True)

    # Save
    results = {
        "timestamp": datetime.now().isoformat(),
        "assets": PORTFOLIO_ASSETS,
        "months": [str(m) for m in TEST_MONTHS],
        "per_asset": {sym: {"n_trades": len(t), "mean_ret_bps": round(t.mean()*10000, 2),
                            "win_rate": round((t>0).mean(), 4)}
                      for sym, t in asset_pnl.items()},
    }
    with open(OUTPUT_DIR / "direction_v3_portfolio.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
