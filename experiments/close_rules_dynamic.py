"""Close rules with dynamic top-30 asset selection.

decay_T3 vs baseline (fixed T+4), through full NCO portfolio sim.
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import time, warnings
from pathlib import Path

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.asset_selection import compute_monthly_dollar_volume, select_assets_for_month
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
LB = 14
TEST_START = "2024-01"
CONF_THR = 0.14
TOP_N = 30
MAX_BARS = 8


def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=0.01, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=0.01, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def run_portfolio(signals_df, pcfg):
    if len(signals_df) < 50:
        return None
    combined = orig.apply_nco_wf(signals_df)
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    monthly = dpnl.resample("ME").sum()
    pnl_day = (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1)
    return {
        "sharpe": stats["sharpe"], "sortino": stats.get("sortino", 0),
        "calmar": stats.get("calmar", 0), "ret_pct": stats["total_return_pct"],
        "max_dd": dd.min() * 100, "worst_day": (dpnl / pcfg.starting_capital * 100).min(),
        "pnl_day": pnl_day, "win_daily": (dpnl > 0).mean() * 100,
        "n_trades": sim["n_trades"], "n_days": stats["n_days"],
        "n_assets": combined["asset"].nunique(),
        "pos_months": f"{(monthly > 0).sum()}/{len(monthly)}",
        "final_eq": stats["final_equity"],
    }


def print_result(label, r):
    if r is None:
        print(f"\n  {label}: NO DATA")
        return
    print(f"\n  {label}")
    print(f"    Sharpe:       {r['sharpe']:.3f}")
    print(f"    Sortino:      {r['sortino']:.3f}")
    print(f"    Calmar:       {r['calmar']:.3f}")
    print(f"    Return:       {r['ret_pct']:.1f}%")
    print(f"    Max DD:       {r['max_dd']:.2f}%")
    print(f"    Worst Day:    {r['worst_day']:.2f}%")
    print(f"    PnL/day:      ${r['pnl_day']:.2f} ({r['pnl_day']/100:.3f}%)")
    print(f"    Win% (daily): {r['win_daily']:.1f}%")
    print(f"    Trades:       {r['n_trades']}")
    print(f"    Assets:       {r['n_assets']}")
    print(f"    Pos Months:   {r['pos_months']}")
    print(f"    Final Equity: ${r['final_eq']:,.2f}")


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 80)
    print("CLOSE RULES + DYNAMIC TOP-30 SELECTION")
    print(f"  decay_T3 vs baseline (T+4), NCO portfolio, OOS 2024+")
    print("=" * 80)

    # Load all data
    print("\nLoading all assets...", flush=True)
    asset_data = {}
    asset_volumes = {}

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000:
                continue

            close = ohlcv_15m["close"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 3000:
                continue

            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
            close_arr = ohlcv_15m["close"].values.astype(np.float64)
            timestamps = ohlcv_15m.index

            asset_data[symbol] = {
                "close": close, "close_arr": close_arr, "fc": fc,
                "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "timestamps": timestamps,
                "start_date": ohlcv_15m.index.min(),
                "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception:
            pass

    print(f"  Loaded {len(asset_data)} assets in {time.time() - t0:.0f}s")

    # Walk-forward with dynamic selection
    sample_idx = next(iter(asset_volumes.values())).index
    test_start_ts = pd.Timestamp(TEST_START + "-01")
    if sample_idx.tz is not None:
        test_start_ts = test_start_ts.tz_localize(sample_idx.tz)

    end_date = max(d["end_date"] for d in asset_data.values())
    test_months = pd.period_range(start=test_start_ts.to_period("M"), end=end_date.to_period("M"), freq="M")

    # Collect trades for both rules
    signals_baseline = []  # fixed T+4
    signals_decay3 = []    # decay_T3
    signals_decay2 = []    # decay_T2
    signals_t8 = []        # fixed T+8

    print("\nGenerating signals...", flush=True)
    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)

        # Dynamic top-30 selection
        lookback_end = ms - pd.Timedelta(days=1)
        vol_candidates = {}
        for symbol, vol_s in asset_volumes.items():
            avail = vol_s.loc[:lookback_end]
            if len(avail) < 45: continue
            latest = avail.iloc[-1]
            if np.isnan(latest) or latest <= 0: continue
            if len(avail) >= 90:
                m90 = avail.iloc[-90:].median()
                if m90 > 0 and latest / m90 > 5.0: continue
            vol_candidates[symbol] = latest
        selected = [s for s, _ in sorted(vol_candidates.items(), key=lambda x: -x[1])[:TOP_N]]

        month_count = 0
        for symbol in selected:
            if symbol not in asset_data:
                continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]
            n = len(close_arr)
            fc = ad["fc"]
            y_all = ad["y_all"]
            valid = ad["valid"]
            vol = ad["vol"]
            timestamps = ad["timestamps"]

            ms_local = ms
            me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None)
                me_local = me_local.tz_localize(None)

            te = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vt = vol_train.quantile(0.50)
            test_vol = vol.reindex(te)
            hv = (test_vol > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5: continue

            train_start = ms_local - pd.DateOffset(months=LB)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
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
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te)):
                if not mask[i]: continue
                bar_idx = timestamps.get_loc(te[i])
                if bar_idx + MAX_BARS >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]

                rets = [d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, MAX_BARS + 1)]

                base = {
                    "timestamp": te[i], "direction": d, "confidence": conf[i],
                    "probability": yp[i], "signal_strength": float(ss[i]),
                    "asset": asset_name,
                }

                # Baseline: T+4
                signals_baseline.append({**base, "fwd_return": rets[3]})

                # decay_T3: close at T+3 if losing, else T+4
                exit_bar = 3 if rets[2] < 0 else 4
                signals_decay3.append({**base, "fwd_return": rets[exit_bar - 1]})

                # decay_T2: close at T+2 if losing, else T+4
                exit_bar2 = 2 if rets[1] < 0 else 4
                signals_decay2.append({**base, "fwd_return": rets[exit_bar2 - 1]})

                # fixed T+8
                signals_t8.append({**base, "fwd_return": rets[7]})

                month_count += 1

        print(f"  {str(month)}: {len(selected)} assets, {month_count} signals", flush=True)

    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.04, max_signal_mult=2.0,
    )

    print(f"\n{'='*80}")
    print("RESULTS — DYNAMIC TOP-30 + NCO PORTFOLIO")
    print(f"{'='*80}")

    for label, sigs_list in [
        ("BASELINE (fixed T+4)", signals_baseline),
        ("DECAY_T3 (close@45m if losing, else T+4)", signals_decay3),
        ("DECAY_T2 (close@30m if losing, else T+4)", signals_decay2),
        ("FIXED T+8 (hold 2 hours)", signals_t8),
    ]:
        df = pd.DataFrame(sigs_list).sort_values("timestamp").reset_index(drop=True)
        r = run_portfolio(df, pcfg)
        print_result(label, r)

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
