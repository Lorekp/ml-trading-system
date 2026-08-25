"""Close Rules Test: simple rule-based exit vs fixed 1-hour hold.

Tests on strict OOS (2024+), 30 assets, 18f_C0.01 lb=14mo thr=0.14.
For each trade, computes unrealized returns at T+1..T+4 (and T+5..T+8),
then applies exit rules to decide early close.

Rules tested:
- Baseline: fixed T+4 (1 hour)
- Trailing stop: close if drawdown from peak > X bps
- Profit target: close if return > Y bps
- Time-decay: close at T+2 or T+3 if losing
- Hold longer: T+6, T+8
- Combos
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
MAX_BARS = 8  # look up to T+8 for extended holds


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset(args):
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, []

    close = ohlcv_15m["close"].values.astype(np.float64)
    n = len(close)
    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()
    for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
        if col in feat.columns:
            feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
    fc = feat.dropna()
    fwd_ret_4 = ohlcv_15m["close"].pct_change(4).shift(-4)
    y_all = fwd_ret_4.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, []

    vol = ohlcv_15m["close"].pct_change().dropna().ewm(span=50, min_periods=20).std()
    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5
    timestamps = ohlcv_15m.index

    test_start_ts = pd.Timestamp(TEST_START, tz="UTC")
    if fc.index.tz is None:
        test_start_ts = test_start_ts.tz_localize(None)
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    all_trades = []

    for month in test_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)
        te = valid[(valid >= ms) & (valid <= me)]
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50: continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue

        train_start = ms - pd.DateOffset(months=LB)
        if train_start < start_date: continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200: continue

        try:
            sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(fc.loc[tr])
            model = make_logreg()
            model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
            Xte = scaler.transform(fc.loc[te])
            yp = model.predict_proba(Xte)[:, 1]
        except Exception: continue

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        mask = conf > CONF_THR

        for i in range(len(te)):
            if not mask[i]: continue
            bar_idx = timestamps.get_loc(te[i])
            if bar_idx + MAX_BARS >= n: continue

            entry_price = close[bar_idx]
            if entry_price <= 0: continue
            direction = dirs[i]

            # Per-bar returns (directed)
            bar_returns = []
            for k in range(1, MAX_BARS + 1):
                ret = direction * (close[bar_idx + k] - entry_price) / entry_price
                bar_returns.append(ret)

            all_trades.append({
                "timestamp": te[i],
                "direction": direction,
                "asset": asset,
                "entry_price": entry_price,
                **{f"ret_{k}": bar_returns[k-1] for k in range(1, MAX_BARS + 1)},
            })

    return asset, all_trades


def apply_rule(trades_df, rule_name, rule_fn):
    """Apply an exit rule to all trades, return the effective return per trade."""
    returns = []
    for _, trade in trades_df.iterrows():
        bar_rets = [trade[f"ret_{k}"] for k in range(1, MAX_BARS + 1)]
        exit_bar = rule_fn(bar_rets, trade)
        returns.append(bar_rets[exit_bar - 1] - 0.00001 * exit_bar)  # slippage per bar
    return np.array(returns)


def compute_stats(returns, timestamps):
    daily = pd.Series(5.0 * returns, index=timestamps).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return None
    std = daily.std()
    sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
    return {
        "sharpe": sr, "win_rate": (returns > 0).mean() * 100,
        "mean_bps": returns.mean() * 10000, "n_trades": len(returns),
        "pnl_day": daily.mean(),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("CLOSE RULES TEST: simple exit rules vs fixed T+4")
    print(f"  18f_C0.01, lb={LB}mo, thr={CONF_THR}, top-30 volume")
    print("=" * 80, flush=True)

    # Generate trades with per-bar returns
    all_trades = []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, trades = f.result()
            done += 1
            all_trades.extend(trades)
            print(f"  [{done:>2d}/30] {asset}: {len(trades)} trades", flush=True)

    df = pd.DataFrame(all_trades)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"\n  Total: {len(df)} trades in {time.time() - t0:.0f}s")

    # Define exit rules
    rules = {}

    # Baseline: fixed hold
    for bars in [2, 3, 4, 5, 6, 8]:
        rules[f"fixed_T+{bars}"] = lambda rets, t, b=bars: b

    # Trailing stop
    for stop_bps in [10, 20, 30, 50]:
        def trailing_stop(rets, t, stop=stop_bps/10000):
            peak = 0
            for k in range(4):  # check bars 1-4
                peak = max(peak, rets[k])
                if rets[k] < peak - stop:
                    return k + 1
            return 4
        rules[f"trail_{stop_bps}bps"] = trailing_stop

    # Profit target
    for target_bps in [20, 30, 50, 75, 100]:
        def profit_target(rets, t, tgt=target_bps/10000):
            for k in range(4):
                if rets[k] > tgt:
                    return k + 1
            return 4
        rules[f"target_{target_bps}bps"] = profit_target

    # Time-decay: close at T+2 if losing, else hold to T+4
    rules["decay_T2"] = lambda rets, t: 2 if rets[1] < 0 else 4
    rules["decay_T3"] = lambda rets, t: 3 if rets[2] < 0 else 4

    # Momentum reversal: close if last bar went against entry direction
    def momentum_rev(rets, t):
        for k in range(1, 4):
            bar_ret = rets[k] - rets[k-1]  # return of individual bar
            if bar_ret < -0.001:  # bar moved against direction by >10bps
                return k + 1
        return 4
    rules["mom_reversal"] = momentum_rev

    # Combo: trailing 30bps + profit target 50bps
    def combo_trail30_target50(rets, t):
        peak = 0
        for k in range(4):
            peak = max(peak, rets[k])
            if rets[k] > 0.005:  # 50bps profit target
                return k + 1
            if rets[k] < peak - 0.003:  # 30bps trailing stop
                return k + 1
        return 4
    rules["combo_t30_p50"] = combo_trail30_target50

    # Combo: decay T+2 + hold longer to T+6 for winners
    def decay_hold_longer(rets, t):
        if rets[1] < -0.001:  # losing at T+2
            return 2
        return min(6, len(rets))  # hold to T+6 for winners
    rules["decay2_hold6"] = decay_hold_longer

    # Run all rules
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"  {'Rule':<25s} {'SR':>7s} {'Win%':>6s} {'bps':>7s} {'Trades':>7s} {'PnL/d':>8s}")
    print("  " + "-" * 62)

    results = []
    for name, rule_fn in rules.items():
        rets = apply_rule(df, name, rule_fn)
        stats = compute_stats(rets, df["ts"].values)
        if stats:
            results.append((name, stats))
            print(f"  {name:<25s} {stats['sharpe']:>7.3f} {stats['win_rate']:>5.1f}% "
                  f"{stats['mean_bps']:>6.1f} {stats['n_trades']:>7d} ${stats['pnl_day']:>7.2f}")

    # Rank
    results.sort(key=lambda x: -x[1]["sharpe"])
    print(f"\n  TOP 5 BY SHARPE:")
    for i, (name, stats) in enumerate(results[:5]):
        print(f"    {i+1}. {name:<25s} SR={stats['sharpe']:.3f}  bps={stats['mean_bps']:.1f}  PnL=${stats['pnl_day']:.2f}")

    results.sort(key=lambda x: -x[1]["pnl_day"])
    print(f"\n  TOP 5 BY PnL/day:")
    for i, (name, stats) in enumerate(results[:5]):
        print(f"    {i+1}. {name:<25s} SR={stats['sharpe']:.3f}  bps={stats['mean_bps']:.1f}  PnL=${stats['pnl_day']:.2f}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
