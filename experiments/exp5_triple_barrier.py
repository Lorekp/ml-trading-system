"""EXP-5: Triple-Barrier Labeling — PROPER Re-Test.

Previous test was unfair: used triple-barrier labels with features designed
for fixed-horizon. This re-test:
1. Uses dynamic vol-based barrier targets
2. Adds barrier-specific features (atr_ratio, trend_strength)
3. Tests both LogReg and XGBoost (nonlinear might capture barrier dynamics)
4. Portfolio sim uses ACTUAL calendar-time forward returns (fair comparison)
"""

import sys
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_confidence_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal

import sklearn
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.1, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def triple_barrier_labels(close, vol, pt_sl_ratio=1.0, vertical_bars=16):
    """Compute triple-barrier labels.

    For each bar:
    - Upper barrier = close * (1 + pt_sl_ratio * vol)
    - Lower barrier = close * (1 - pt_sl_ratio * vol)
    - Vertical barrier = vertical_bars forward

    Label = +1 if upper hit first, -1 if lower hit first, sign(return) if vertical hit.
    Also returns the actual forward return at exit for portfolio sim.
    """
    labels = pd.Series(index=close.index, dtype=float)
    exit_returns = pd.Series(index=close.index, dtype=float)
    exit_bars = pd.Series(index=close.index, dtype=float)

    for i in range(len(close) - vertical_bars):
        entry = close.iloc[i]
        v = vol.iloc[i]
        if pd.isna(v) or v <= 0:
            labels.iloc[i] = np.nan
            continue

        upper = entry * (1 + pt_sl_ratio * v)
        lower = entry * (1 - pt_sl_ratio * v)

        # Look forward
        future = close.iloc[i + 1 : i + 1 + vertical_bars]
        hit_upper = (future >= upper)
        hit_lower = (future <= lower)

        if hit_upper.any() and hit_lower.any():
            upper_idx = hit_upper.idxmax()
            lower_idx = hit_lower.idxmax()
            if upper_idx <= lower_idx:
                labels.iloc[i] = 1
                exit_returns.iloc[i] = (close.loc[upper_idx] - entry) / entry
                exit_bars.iloc[i] = (close.index.get_loc(upper_idx) - i)
            else:
                labels.iloc[i] = -1
                exit_returns.iloc[i] = (close.loc[lower_idx] - entry) / entry
                exit_bars.iloc[i] = (close.index.get_loc(lower_idx) - i)
        elif hit_upper.any():
            labels.iloc[i] = 1
            idx = hit_upper.idxmax()
            exit_returns.iloc[i] = (close.loc[idx] - entry) / entry
            exit_bars.iloc[i] = (close.index.get_loc(idx) - i)
        elif hit_lower.any():
            labels.iloc[i] = -1
            idx = hit_lower.idxmax()
            exit_returns.iloc[i] = (close.loc[idx] - entry) / entry
            exit_bars.iloc[i] = (close.index.get_loc(idx) - i)
        else:
            # Vertical barrier hit
            exit_price = close.iloc[i + vertical_bars]
            ret = (exit_price - entry) / entry
            labels.iloc[i] = 1 if ret > 0 else -1
            exit_returns.iloc[i] = ret
            exit_bars.iloc[i] = vertical_bars

    return labels, exit_returns, exit_bars


def run_walk_forward(ohlcv, features, labels, fwd_ret, asset, cfg, model_type="logreg"):
    """Walk-forward with triple-barrier or fixed-horizon labels."""
    feature_cols = cfg.features.feature_list
    feat = features[[c for c in feature_cols if c in features.columns]].copy()
    vol = ohlcv["close"].pct_change().dropna().ewm(span=50, min_periods=20).std()

    start = ohlcv.index.min()
    end = ohlcv.index.max()
    if (end - start).days < 180:
        return pd.DataFrame()

    first_test = start + pd.DateOffset(months=6)
    months = pd.period_range(start=first_test.to_period("M"), end=end.to_period("M"), freq="M")
    if len(months) == 0:
        return pd.DataFrame()

    embargo_td = pd.Timedelta(hours=1)
    rows = []
    cached_m = cached_s = None
    last_r = -999

    for mi, month in enumerate(months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if feat.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        tc = ms - embargo_td
        fc = feat.dropna()
        y_all = labels.reindex(fc.index).dropna()
        fwd_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index).intersection(fwd_all.index)
        tr = valid[valid < tc]
        te = valid[(valid >= ms) & (valid <= me)]
        if len(tr) < 200 or len(te) < 5:
            continue

        tv = vol.loc[vol.index.intersection(tr)].dropna()
        if len(tv) < 20: continue
        vt = tv.quantile(0.50)
        va = vol.reindex(te)
        hv = (va > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 3: continue

        X_train = fc.loc[tr]
        # Convert labels to binary: +1 -> 1, -1 -> 0
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        need = (mi - last_r) >= 2 or cached_m is None
        try:
            if need:
                sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xts = scaler.fit_transform(X_train)

                if model_type == "logreg":
                    model = make_logreg()
                    model.fit(Xts, y_train, sample_weight=sw)
                else:
                    model = GradientBoostingClassifier(
                        n_estimators=100, max_depth=3, min_samples_leaf=10,
                        learning_rate=0.05, random_state=42, subsample=0.8,
                    )
                    model.fit(Xts, y_train, sample_weight=sw)
                cached_m = model
                cached_s = scaler
                last_r = mi

            Xtes = cached_s.transform(X_test)
            yp = cached_m.predict_proba(Xtes)[:, 1]
        except Exception:
            continue

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))

        # Use ACTUAL forward returns for portfolio sim (not barrier returns)
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if finite.any():
            batch = pd.DataFrame({
                "timestamp": te[finite], "direction": dirs[finite],
                "fwd_return": fv[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": ss[finite],
            })
            batch["asset"] = asset
            rows.append(batch)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(d) for d in cfg.data.data_dirs]
    assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
              "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

    print("=" * 70)
    print("EXP-5: Triple-Barrier vs Fixed-Horizon — FAIR COMPARISON")
    print("  Both use ACTUAL forward returns (1-hour calendar) for portfolio sim")
    print("  Both retrained with proper walk-forward methodology")
    print("  Tests LogReg and XGBoost for barrier labeling")
    print("=" * 70, flush=True)

    configs = [
        ("Fixed-horizon (LogReg)", "fixed", "logreg"),
        ("Triple-barrier [1,1] (LogReg)", "tb_1_1", "logreg"),
        ("Triple-barrier [2,1] (LogReg)", "tb_2_1", "logreg"),
        ("Triple-barrier [1,1] (XGBoost)", "tb_1_1", "xgb"),
    ]

    for label, barrier_type, model_type in configs:
        print(f"\n--- {label} ---", flush=True)
        all_signals = []
        t0 = time.time()

        for asset_name, symbol in assets.items():
            print(f"  {asset_name}...", end="", flush=True)
            ohlcv_1m = load_binance_1m(symbol, data_dirs)
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            features = compute_features(ohlcv_15m, cfg.features)

            close = ohlcv_15m["close"]
            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

            # Fixed 1-hour forward return (for portfolio sim — same for all)
            fwd_ret_1h = close.pct_change(4).shift(-4)

            if barrier_type == "fixed":
                # Labels = direction of 1-hour forward return
                labels = (fwd_ret_1h > 0).astype(float) * 2 - 1  # +1 or -1
            elif barrier_type == "tb_1_1":
                labels, _, _ = triple_barrier_labels(close, vol, pt_sl_ratio=1.0, vertical_bars=16)
            elif barrier_type == "tb_2_1":
                labels, _, _ = triple_barrier_labels(close, vol, pt_sl_ratio=2.0, vertical_bars=16)
            else:
                continue

            signals = run_walk_forward(
                ohlcv_15m, features, labels, fwd_ret_1h, asset_name, cfg, model_type
            )

            if len(signals) > 0:
                filtered = apply_confidence_filter(signals, cfg.confidence_filter)
                all_signals.append(filtered)
                print(f" {len(filtered)} signals", flush=True)
            else:
                print(" no signals", flush=True)

        elapsed = time.time() - t0

        if all_signals:
            combined = pd.concat(all_signals).sort_values("timestamp")
            sim = simulate_portfolio(combined, cfg.portfolio)
            if sim and sim["n_trades"] >= 10:
                dpnl = compute_daily_pnl_series(sim)
                if dpnl is not None:
                    stats = compute_full_stats(
                        dpnl, cfg.portfolio.starting_capital,
                        sim["n_trades"], sim["n_trades"] / len(dpnl),
                    )
                    print(f"\n  RESULTS ({label}):")
                    print(f"    Sharpe:  {stats['sharpe']:.3f}")
                    print(f"    Return:  {stats['total_return_pct']:.2f}%")
                    print(f"    MaxDD:   {stats['max_dd_pct']:.2f}%")
                    print(f"    Trades:  {sim['n_trades']}")
                    print(f"    Time:    {elapsed:.0f}s")


if __name__ == "__main__":
    main()
