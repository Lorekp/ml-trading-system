"""Close rules through full NCO portfolio simulation.

Top candidates from flat-bet test, now with proper position sizing.
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

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
W = max(1, os.cpu_count() - 1)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
MAX_BARS = 8


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
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
        mask = conf > CONF_THR

        for i in range(len(te)):
            if not mask[i]: continue
            bar_idx = timestamps.get_loc(te[i])
            if bar_idx + MAX_BARS >= n: continue

            entry_price = close[bar_idx]
            if entry_price <= 0: continue

            bar_rets = []
            for k in range(1, MAX_BARS + 1):
                ret = dirs[i] * (close[bar_idx + k] - entry_price) / entry_price
                bar_rets.append(ret)

            all_trades.append({
                "timestamp": te[i],
                "direction": dirs[i],
                "confidence": conf[i],
                "probability": yp[i],
                "signal_strength": float(ss[i]) if np.ndim(ss) > 0 else float(ss),
                "asset": asset,
                "bar_rets": bar_rets,
            })

    return asset, all_trades


def make_signals_df(trades, rule_fn):
    """Apply exit rule and produce signals DataFrame for portfolio sim."""
    rows = []
    for t in trades:
        exit_bar = rule_fn(t["bar_rets"])
        fwd_return = t["bar_rets"][exit_bar - 1]
        rows.append({
            "timestamp": t["timestamp"],
            "direction": t["direction"],
            "fwd_return": fwd_return,
            "confidence": t["confidence"],
            "probability": t["probability"],
            "signal_strength": t["signal_strength"],
            "asset": t["asset"],
        })
    return pd.DataFrame(rows)


def run_portfolio(signals_df, pcfg, label):
    if len(signals_df) < 50:
        print(f"  {label}: INSUFFICIENT ({len(signals_df)})")
        return
    combined = orig.apply_nco_wf(signals_df)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        print(f"  {label}: SIM FAILED")
        return
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)

    print(f"\n  {label}")
    print(f"    Sharpe:       {stats['sharpe']:.3f}")
    print(f"    Sortino:      {stats.get('sortino', 0):.3f}")
    print(f"    Calmar:       {stats.get('calmar', 0):.3f}")
    print(f"    Return:       {stats['total_return_pct']:.1f}%")
    print(f"    Max DD:       {dd.min()*100:.2f}%")
    print(f"    Worst Day:    {(dpnl/pcfg.starting_capital*100).min():.2f}%")
    print(f"    PnL/day:      ${pnl_day:.2f} ({pnl_day/pcfg.starting_capital*100:.3f}%)")
    print(f"    Win% (daily): {(dpnl>0).mean()*100:.1f}%")
    print(f"    Trades:       {sim['n_trades']}")
    print(f"    Pos Months:   {(monthly>0).sum()}/{len(monthly)}")
    print(f"    Final Equity: ${stats['final_equity']:,.2f}")


def main():
    t0 = time.time()
    print("=" * 80)
    print("CLOSE RULES — FULL NCO PORTFOLIO COMPARISON")
    print("=" * 80, flush=True)

    all_trades = []
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        done = 0
        for f in as_completed(futures):
            asset, trades = f.result()
            done += 1
            all_trades.extend(trades)
            print(f"  [{done:>2d}/30] {asset}: {len(trades)}", flush=True)

    print(f"\n  Total: {len(all_trades)} trades in {time.time() - t0:.0f}s")

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )

    rules = {
        "fixed_T+4 (baseline)": lambda rets: 4,
        "fixed_T+3": lambda rets: 3,
        "fixed_T+6": lambda rets: 6,
        "fixed_T+8": lambda rets: 8,
        "decay_T2 (close@30m if losing)": lambda rets: 2 if rets[1] < 0 else 4,
        "decay_T3 (close@45m if losing)": lambda rets: 3 if rets[2] < 0 else 4,
        "trail_50bps": lambda rets: next((k+1 for k in range(4) if rets[k] < max(rets[:k+1]) - 0.005), 4),
        "decay2_hold6 (cut losers, extend winners)": lambda rets: 2 if rets[1] < -0.001 else 6,
        "decay2_hold8": lambda rets: 2 if rets[1] < -0.001 else 8,
    }

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    for name, rule_fn in rules.items():
        signals = make_signals_df(all_trades, rule_fn)
        run_portfolio(signals, pcfg, name)

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
