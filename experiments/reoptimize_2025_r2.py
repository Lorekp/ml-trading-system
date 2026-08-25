"""Re-optimization Round 2+3+4: Lookback/VolQ/Close sweep + 2024 robustness.

Uses the top 3 (C, threshold) pairs from Round 1:
  1. C=0.005, thr=0.14 (Sharpe 5.49, N=549)
  2. C=0.005, thr=0.12 (Sharpe 5.33, N=961)
  3. C=0.01,  thr=0.14 (Sharpe 5.25, N=1054) — current production

For each, sweeps lookback and vol_q. Then tests best on BOTH 2024 and 2025.
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

from signal_engine.config import load_config, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
TOP_N = 30
MAX_BARS = 8


def make_logreg(C):
    if NEW_SKLEARN:
        return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def flat_sharpe(signals):
    if len(signals) < 20:
        return {"sharpe": 0, "n": len(signals), "win": 0, "bps": 0, "n_days": 0}
    pnls = signals["direction"].values * signals["fwd_return"].values - 0.0001
    ts = pd.to_datetime(signals["timestamp"], utc=True)
    daily = pd.Series(5.0 * pnls, index=ts.values).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5:
        return {"sharpe": 0, "n": len(signals), "win": 0, "bps": 0, "n_days": 0}
    std = daily.std()
    sr = daily.mean() / std * np.sqrt(365.25) if std > 0 else 0
    return {"sharpe": sr, "n": len(signals), "win": (pnls > 0).mean() * 100,
            "bps": pnls.mean() * 10000, "n_days": len(daily)}


def generate_signals(asset_data, asset_volumes, C, lookback, test_start,
                      test_months, sample_idx):
    """Generate raw signals (all bars, no filtering) for a C+lookback combo."""
    all_signals = []
    cfg = load_config(Path("configs/default.yaml"))

    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)

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

        for symbol in selected:
            if symbol not in asset_data: continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]; n = len(close_arr)
            fc = ad["fc"]; y_all = ad["y_all"]; valid = ad["valid"]
            vol = ad["vol"]; timestamps_idx = ad["timestamps"]

            ms_local = ms; me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None); me_local = me_local.tz_localize(None)

            te_all = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vol_quantiles = {q: vol_train.quantile(q) for q in [0.25, 0.40, 0.50, 0.60, 0.75]}
            test_vol = vol.reindex(te_all)

            train_start = ms_local - pd.DateOffset(months=lookback)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200: continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg(C)
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te_all])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception: continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te_all)):
                bar_idx = timestamps_idx.get_loc(te_all[i])
                if bar_idx + MAX_BARS >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]
                rets = {k: d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, MAX_BARS + 1)}
                vol_val = float(test_vol.get(te_all[i], np.nan))

                all_signals.append({
                    "timestamp": te_all[i], "direction": d,
                    "confidence": conf[i], "probability": yp[i],
                    "signal_strength": float(ss[i]),
                    "asset": asset_name, "vol_val": vol_val,
                    **{f"ret_{k}": rets[k] for k in range(1, MAX_BARS + 1)},
                    **{f"vol_q_{q}": vol_quantiles[q] for q in vol_quantiles},
                })

    return pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True)


def apply_filters(df, threshold, vol_q, fwd_bars, close_rule):
    filtered = df[df["confidence"] > threshold].copy()
    if vol_q > 0:
        q_col = f"vol_q_{vol_q}"
        if q_col in filtered.columns:
            filtered = filtered[filtered["vol_val"] >= filtered[q_col]]
    if len(filtered) < 20:
        return filtered
    if close_rule == "fixed":
        filtered["fwd_return"] = filtered[f"ret_{fwd_bars}"]
    elif close_rule == "decay_T3":
        filtered["fwd_return"] = filtered.apply(
            lambda r: r["ret_3"] if r["ret_3"] < 0 else r[f"ret_{fwd_bars}"], axis=1)
    elif close_rule == "decay_T2":
        filtered["fwd_return"] = filtered.apply(
            lambda r: r["ret_2"] if r["ret_2"] < 0 else r[f"ret_{fwd_bars}"], axis=1)
    return filtered


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))

    print("=" * 95)
    print("RE-OPTIMIZATION R2: LOOKBACK + VOL_Q + CLOSE + 2024 ROBUSTNESS")
    print("=" * 95)

    # Load data
    print("\nLoading data...", flush=True)
    asset_data = {}; asset_volumes = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"): continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000: continue
            close = ohlcv_15m["close"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1","ret_3","ret_7","ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 3000: continue
            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
            asset_data[symbol] = {
                "close": close, "close_arr": ohlcv_15m["close"].values.astype(np.float64),
                "fc": fc, "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "timestamps": ohlcv_15m.index,
                "start_date": ohlcv_15m.index.min(), "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception: pass
    print(f"  Loaded {len(asset_data)} assets in {time.time()-t0:.0f}s")

    sample_idx = next(iter(asset_volumes.values())).index
    end_date = max(d["end_date"] for d in asset_data.values())

    # Top 3 configs from Round 1
    configs = [
        {"C": 0.005, "thr": 0.14, "label": "C005_t14"},
        {"C": 0.005, "thr": 0.12, "label": "C005_t12"},
        {"C": 0.01,  "thr": 0.14, "label": "C010_t14 (production)"},
    ]

    lookbacks = [6, 8, 10, 12, 14]
    vol_qs = [0.25, 0.40, 0.50]
    close_rules = ["fixed", "decay_T2", "decay_T3"]
    fwd_bars_list = [4, 6, 8]

    for cfg_entry in configs:
        C = cfg_entry["C"]; thr = cfg_entry["thr"]; label = cfg_entry["label"]

        # ===== 2025+ sweep =====
        print(f"\n{'='*95}")
        print(f"CONFIG: {label} (C={C}, thr={thr})")
        print(f"{'='*95}")

        print(f"\n  --- 2025+ LOOKBACK x VOL_Q ---")
        print(f"  {'LB':>4s} {'VQ':>5s} {'Sharpe':>7s} {'N':>6s} {'Win%':>5s} {'bps':>6s}")
        print("  " + "-" * 38)

        test_2025 = pd.period_range(start="2025-01", end=end_date.to_period("M"), freq="M")
        best_lb_vq = None; best_sr = -999

        for lb in lookbacks:
            raw = generate_signals(asset_data, asset_volumes, C, lb, "2025-01",
                                    test_2025, sample_idx)
            for vq in vol_qs:
                f = apply_filters(raw, thr, vq, 4, "decay_T3")
                s = flat_sharpe(f)
                flag = " *" if s["sharpe"] > best_sr and s["n"] > 200 else ""
                print(f"  {lb:>4d} {vq:>5.2f} {s['sharpe']:>7.3f} {s['n']:>6d} "
                      f"{s['win']:>4.1f}% {s['bps']:>5.1f}{flag}", flush=True)
                if s["sharpe"] > best_sr and s["n"] > 200:
                    best_sr = s["sharpe"]; best_lb_vq = (lb, vq)

        if best_lb_vq:
            print(f"  Best: lb={best_lb_vq[0]}, vq={best_lb_vq[1]}, Sharpe={best_sr:.3f}")

        lb_use = best_lb_vq[0] if best_lb_vq else 14
        vq_use = best_lb_vq[1] if best_lb_vq else 0.50

        # Close rule + fwd_bars sweep
        print(f"\n  --- CLOSE x FWD_BARS (lb={lb_use}, vq={vq_use}) ---")
        print(f"  {'Close':>10s} {'Fwd':>4s} {'Sharpe':>7s} {'N':>6s} {'Win%':>5s} {'bps':>6s}")
        print("  " + "-" * 42)

        raw = generate_signals(asset_data, asset_volumes, C, lb_use, "2025-01",
                                test_2025, sample_idx)
        best_close_fwd = None; best_sr2 = -999

        for cr in close_rules:
            for fb in fwd_bars_list:
                f = apply_filters(raw, thr, vq_use, fb, cr)
                s = flat_sharpe(f)
                flag = " *" if s["sharpe"] > best_sr2 and s["n"] > 200 else ""
                print(f"  {cr:>10s} {fb:>4d} {s['sharpe']:>7.3f} {s['n']:>6d} "
                      f"{s['win']:>4.1f}% {s['bps']:>5.1f}{flag}", flush=True)
                if s["sharpe"] > best_sr2 and s["n"] > 200:
                    best_sr2 = s["sharpe"]; best_close_fwd = (cr, fb)

        cr_use = best_close_fwd[0] if best_close_fwd else "decay_T3"
        fb_use = best_close_fwd[1] if best_close_fwd else 4

        # ===== ROBUSTNESS: Test on 2024 =====
        print(f"\n  --- ROBUSTNESS: 2024 OOS (lb={lb_use}, vq={vq_use}, {cr_use}, fwd={fb_use}) ---")
        test_2024 = pd.period_range(start="2024-01", end="2024-12", freq="M")
        raw_2024 = generate_signals(asset_data, asset_volumes, C, lb_use, "2024-01",
                                     test_2024, sample_idx)
        f_2024 = apply_filters(raw_2024, thr, vq_use, fb_use, cr_use)
        s_2024 = flat_sharpe(f_2024)

        # Also test same config on 2025
        f_2025 = apply_filters(raw, thr, vq_use, fb_use, cr_use)
        s_2025 = flat_sharpe(f_2025)

        print(f"  {'Period':>8s} {'Sharpe':>7s} {'N':>6s} {'Win%':>5s} {'bps':>6s} {'Days':>5s}")
        print("  " + "-" * 40)
        print(f"  {'2024':>8s} {s_2024['sharpe']:>7.3f} {s_2024['n']:>6d} "
              f"{s_2024['win']:>4.1f}% {s_2024['bps']:>5.1f} {s_2024['n_days']:>5d}")
        print(f"  {'2025+':>8s} {s_2025['sharpe']:>7.3f} {s_2025['n']:>6d} "
              f"{s_2025['win']:>4.1f}% {s_2025['bps']:>5.1f} {s_2025['n_days']:>5d}")

        robust = "ROBUST" if s_2024["sharpe"] > 2.0 and s_2025["sharpe"] > 2.0 else "NOT ROBUST"
        print(f"  Verdict: {robust}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
