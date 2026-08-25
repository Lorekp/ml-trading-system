"""EXP-4: Dollar Bars — Fair Comparison.

Citadel-level methodology:
1. Dollar bars constructed with adaptive threshold (rolling 30-day median)
2. Forward returns computed in CALENDAR TIME (1 hour), not bar count
3. Features retrained ON dollar bars (model sees dollar-bar statistics)
4. Compared against 15m time bars with same 1-hour forward horizon
5. Tested on multiple assets, not just BTC
6. Both use same portfolio simulation for fair PnL comparison
"""

import sys
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("exp4")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd
import time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def construct_dollar_bars(ohlcv_1m: pd.DataFrame, target_bars_per_day: int = 48) -> pd.DataFrame:
    """Construct dollar bars with adaptive daily threshold.

    Uses shift(1) on threshold to avoid look-ahead bias.
    Returns DataFrame with OHLCV + dollar_volume + duration_min columns.
    """
    dv = ohlcv_1m["close"] * ohlcv_1m["volume"]
    daily_dv = dv.resample("D").sum()
    threshold_daily = daily_dv.rolling(30, min_periods=5).median().shift(1)
    threshold_per_bar = threshold_daily / target_bars_per_day
    thresh_1m = threshold_per_bar.reindex(ohlcv_1m.index, method="ffill")

    cum_dv = 0.0
    bars = []
    bar_open = bar_high = bar_low = bar_close = bar_vol = None
    bar_start = None

    for i, (ts, row) in enumerate(ohlcv_1m.iterrows()):
        if bar_open is None:
            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_start = ts
            bar_vol = 0.0
            cum_dv = 0.0

        bar_high = max(bar_high, row["high"])
        bar_low = min(bar_low, row["low"])
        bar_close = row["close"]
        bar_vol += row["volume"]
        cum_dv += row["close"] * row["volume"]

        t = thresh_1m.iloc[i] if not pd.isna(thresh_1m.iloc[i]) else 1e18
        if cum_dv >= t and t > 0:
            bars.append({
                "timestamp": bar_start,
                "open": bar_open, "high": bar_high, "low": bar_low,
                "close": bar_close, "volume": bar_vol,
            })
            bar_open = None

    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars).set_index(pd.DatetimeIndex([b["timestamp"] for b in bars]))
    return df.drop(columns=["timestamp"])


def compute_calendar_fwd_return(ohlcv: pd.DataFrame, horizon_minutes: int = 60) -> pd.Series:
    """Compute forward return using CALENDAR TIME, not bar count.

    For each bar, find the close price ~horizon_minutes later.
    This makes dollar bars and time bars directly comparable.
    """
    close = ohlcv["close"]
    horizon_td = pd.Timedelta(minutes=horizon_minutes)
    fwd_ret = pd.Series(index=close.index, dtype=float)

    # Pre-compute for efficiency: for each bar, find the closest bar
    # at or after timestamp + horizon
    timestamps = close.index
    for i in range(len(timestamps)):
        target_time = timestamps[i] + horizon_td
        # Find the first bar at or after target_time
        future_mask = timestamps >= target_time
        if future_mask.any():
            future_idx = timestamps[future_mask][0]
            fwd_ret.iloc[i] = (close.loc[future_idx] - close.iloc[i]) / close.iloc[i]
        else:
            fwd_ret.iloc[i] = np.nan

    return fwd_ret


def compute_calendar_fwd_return_fast(ohlcv: pd.DataFrame, horizon_minutes: int = 60) -> pd.Series:
    """Fast version using searchsorted to find future close at target time."""
    close = ohlcv["close"]
    timestamps = close.index
    horizon_td = pd.Timedelta(minutes=horizon_minutes)

    target_times = timestamps + horizon_td
    # searchsorted finds the insertion point (first index >= target)
    future_indices = timestamps.searchsorted(target_times, side="left")

    fwd_ret = np.full(len(close), np.nan)
    for i in range(len(close)):
        j = future_indices[i]
        if j < len(close):
            fwd_ret[i] = (close.iloc[j] - close.iloc[i]) / close.iloc[i]

    return pd.Series(fwd_ret, index=timestamps)


def walk_forward_signals(
    ohlcv: pd.DataFrame, features: pd.DataFrame,
    fwd_ret: pd.Series, asset_name: str, cfg,
) -> pd.DataFrame:
    """Walk-forward LogReg on arbitrary bar type using CALENDAR forward returns."""
    feature_cols = cfg.features.feature_list
    feat = features[feature_cols].copy()

    vol = ohlcv["close"].pct_change().dropna().ewm(span=50, min_periods=20).std()

    start_date = ohlcv.index.min()
    end_date = ohlcv.index.max()
    if (end_date - start_date).days < 180:
        return pd.DataFrame()

    first_test = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test.to_period("M"), end=end_date.to_period("M"), freq="M"
    )
    if len(all_months) == 0:
        return pd.DataFrame()

    median_gap = ohlcv.index.to_series().diff().dropna().median()
    # Embargo in calendar time: 1 hour (matching 60-min forward horizon)
    embargo_td = pd.Timedelta(hours=1)

    rows = []
    cached_model = cached_scaler = None
    last_retrain = -999

    retrain_every = 2 if len(ohlcv) > 200000 else 1

    for mi, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if feat.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)
        train_cutoff = ms - embargo_td

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)
        tr = valid[valid < train_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]

        if len(tr) < 200 or len(te) < 5:
            continue

        # Vol filter
        tv = vol.loc[vol.index.intersection(tr)].dropna()
        if len(tv) < 20:
            continue
        vt = tv.quantile(0.50)
        vol_al = vol.reindex(te)
        hv = (vol_al > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 3:
            continue

        X_train = fc.loc[tr]
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        need_retrain = (mi - last_retrain) >= retrain_every or cached_model is None

        try:
            if need_retrain:
                sw = compute_sample_weights(X_train.index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                model = make_logreg()
                model.fit(X_tr_s, y_train, sample_weight=sw)
                cached_model = model
                cached_scaler = scaler
                last_retrain = mi

            X_te_s = cached_scaler.transform(X_test)
            yp = cached_model.predict_proba(X_te_s)[:, 1]
        except Exception:
            continue

        conf = np.abs(yp - 0.5)
        directions = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sigs_disc = discretize_signal(sigs, step_size=0.1)
        sig_str = np.abs(sigs_disc) if np.ndim(sigs_disc) > 0 else np.full(len(yp), abs(float(sigs_disc)))

        fwd_vals = fwd_ret.reindex(te).values
        finite = np.isfinite(fwd_vals)
        if finite.any():
            batch = pd.DataFrame({
                "timestamp": te[finite], "direction": directions[finite],
                "fwd_return": fwd_vals[finite], "confidence": conf[finite],
                "probability": yp[finite], "signal_strength": sig_str[finite],
            })
            batch["asset"] = asset_name
            rows.append(batch)

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main():
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(d) for d in cfg.data.data_dirs]

    # Test on 6 core assets
    assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
              "BNB": "BNBUSDT", "SUI": "SUIUSDT", "LINK": "LINKUSDT"}

    HORIZON_MIN = 60  # 1-hour forward horizon for both bar types

    print("=" * 70)
    print("EXP-4: Dollar Bars vs Time Bars — FAIR COMPARISON")
    print("  Forward horizon: 60 minutes (calendar time, same for both)")
    print("  Model retrained on each bar type separately")
    print("  6 core assets, walk-forward, same portfolio sim")
    print("=" * 70)
    print(flush=True)

    for bar_label, bar_type in [("15m Time Bars", "time"), ("Dollar Bars ~48/day", "dollar")]:
        print(f"\n--- {bar_label} ---", flush=True)
        all_signals = []
        t0 = time.time()

        for asset_name, symbol in assets.items():
            print(f"  {asset_name}...", end="", flush=True)
            ohlcv_1m = load_binance_1m(symbol, data_dirs)

            if bar_type == "time":
                ohlcv = resample_ohlcv(ohlcv_1m, "15min")
            else:
                ohlcv = construct_dollar_bars(ohlcv_1m, target_bars_per_day=48)
                if len(ohlcv) == 0:
                    print(" no bars", flush=True)
                    continue

            # Compute CALENDAR forward returns (same for both bar types)
            fwd_ret = compute_calendar_fwd_return_fast(ohlcv, horizon_minutes=HORIZON_MIN)

            # Compute features ON this bar type
            features = compute_features(ohlcv, cfg.features)

            # Walk-forward signals
            signals = walk_forward_signals(ohlcv, features, fwd_ret, asset_name, cfg)

            if len(signals) > 0:
                # Confidence filter (percentile 70)
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
                    print(f"\n  RESULTS ({bar_label}):")
                    print(f"    Sharpe:  {stats['sharpe']:.3f}")
                    print(f"    Return:  {stats['total_return_pct']:.2f}%")
                    print(f"    MaxDD:   {stats['max_dd_pct']:.2f}%")
                    print(f"    Trades:  {sim['n_trades']}")
                    print(f"    Time:    {elapsed:.0f}s")
            else:
                print(f"  Insufficient trades")
        else:
            print(f"  No signals generated")


if __name__ == "__main__":
    main()
