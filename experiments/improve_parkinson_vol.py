"""Experiment 3: Parkinson volatility filter (replace EWM std).

Tests whether using Parkinson vol (from H/L, 26% more accurate) instead
of close-close EWM std for the vol filter improves signal quality.

Can't filter cached signals post-hoc (need H/L data per signal).
Instead: load raw OHLCV, compute Parkinson, apply filter, compare.
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.features import parkinson_vol
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
cfg = load_config(Path("configs/default.yaml"))
DATA_DIR = Path("data/binance-1m")
TOP_N = 30; MAX_BARS = 8
C = 0.02; LB = 2; THR = 0.10; FWD = 6; LEV = 5.0; DLV = 0.05; SIZING = 0.06

WINDOWS = [
    {"label": "W1", "test_start": "2024-07", "test_end": "2025-02", "cal_days": 245},
    {"label": "W2", "test_start": "2025-03", "test_end": "2025-10", "cal_days": 244},
    {"label": "W3", "test_start": "2025-11", "test_end": "2026-02", "cal_days": 122},
]


def generate_signals_with_vol_type(asset_data, asset_volumes, vol_type, vol_quantile, window):
    """Generate signals using either EWM std or Parkinson vol filter."""
    sample_idx = next(iter(asset_volumes.values())).index
    test_start = pd.Timestamp(window["test_start"] + "-01")
    end_date = max(d["end_date"] for d in asset_data.values())
    test_end = pd.Timestamp(window["test_end"] + "-28")
    if sample_idx.tz is not None:
        test_start = test_start.tz_localize(sample_idx.tz)
        test_end = test_end.tz_localize(sample_idx.tz)
    test_end = min(test_end, end_date)
    test_months = pd.period_range(start=test_start.to_period("M"), end=test_end.to_period("M"), freq="M")

    all_signals = []
    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None: ms = ms.tz_localize(sample_idx.tz)
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
            ts_idx = ad["timestamps"]

            ms_l = ms; me_l = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_l = ms.tz_localize(None); me_l = me_l.tz_localize(None)

            te = valid[(valid >= ms_l) & (valid <= me_l)]

            # Vol filter: EWM std or Parkinson
            if vol_type == "ewm":
                vol = ad["vol_ewm"]
            else:
                vol = ad["vol_park"]

            vol_train_start = ms_l - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_l)].dropna()
            if len(vol_train) < 50: continue
            vt = vol_train.quantile(vol_quantile)
            test_vol = vol.reindex(te)
            hv = (test_vol > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5: continue

            train_start = ms_l - pd.DateOffset(months=LB)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_l - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 100: continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                if NEW_SKLEARN:
                    model = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
                else:
                    model = LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception: continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            mask = conf > THR
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te)):
                if not mask[i]: continue
                bi = ts_idx.get_loc(te[i])
                if bi + MAX_BARS >= n: continue
                ep = close_arr[bi]
                if ep <= 0: continue
                d = dirs[i]
                fr = d * (close_arr[bi + FWD] - ep) / ep
                all_signals.append({
                    "timestamp": te[i], "direction": d, "confidence": conf[i],
                    "probability": yp[i], "signal_strength": float(ss[i]),
                    "asset": asset_name, "fwd_return": fr,
                })

    return pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True) if all_signals else pd.DataFrame()


def run_variant(asset_data, asset_volumes, vol_type, vol_q):
    tp = tc = tt = ttd = 0; mdd = mwd = 0.0
    per_window = []

    for w in WINDOWS:
        fi = generate_signals_with_vol_type(asset_data, asset_volumes, vol_type, vol_q, w)
        if len(fi) < 10:
            per_window.append({"wl": w["label"], "pnl_cal": 0})
            continue

        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=DLV,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=SIZING * LEV, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5:
            per_window.append({"wl": w["label"], "pnl_cal": 0})
            continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1:
            per_window.append({"wl": w["label"], "pnl_cal": 0})
            continue
        eq = pd.Series(sim["equity"][~sim["skipped"]], index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / 10000.0
        pnl = sim["equity"][-1] - 10000
        per_window.append({"wl": w["label"], "pnl_cal": pnl / w["cal_days"],
                          "dd": abs(dd.min()), "wd": abs(dr.min())})
        tp += pnl; tc += w["cal_days"]; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min())); mwd = max(mwd, abs(dr.min()))

    if tc == 0: return None
    return {"pnl_cal": tp / tc, "d2t": int(1000 / (tp / tc)) if tp > 0 else 9999,
            "dd": mdd, "wd": mwd, "trades": tt, "per_window": per_window}


def main():
    t0 = time.time()
    print("EXPERIMENT 3: PARKINSON VOL FILTER vs EWM STD")
    print("=" * 95)

    # Load data with both vol types
    print("Loading data...", flush=True)
    asset_data = {}; asset_volumes = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"): continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 3000: continue
            close = ohlcv_15m["close"]
            high = ohlcv_15m["high"]
            low = ohlcv_15m["low"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 2000: continue

            vol_ewm = close.pct_change().dropna().ewm(span=50, min_periods=20).std()
            vol_park = parkinson_vol(high, low, window=20)

            asset_data[symbol] = {
                "close": close, "close_arr": ohlcv_15m["close"].values.astype(np.float64),
                "fc": fc, "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol_ewm": vol_ewm, "vol_park": vol_park,
                "timestamps": ohlcv_15m.index,
                "start_date": ohlcv_15m.index.min(), "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            asset_volumes[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception: pass
    print(f"Loaded {len(asset_data)} assets in {time.time()-t0:.0f}s\n")

    print(f"{'Variant':<30s} {'$/cal':>7s} {'~Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Trades':>6s}  W1$/d  W2$/d  W3$/d")
    print("-" * 95)

    for label, vtype, vq in [
        ("EWM std q0.50 (baseline)", "ewm", 0.50),
        ("Parkinson q0.50", "park", 0.50),
        ("Parkinson q0.25", "park", 0.25),
        ("Parkinson q0.75", "park", 0.75),
        ("EWM std q0.25", "ewm", 0.25),
        ("EWM std q0.75", "ewm", 0.75),
    ]:
        r = run_variant(asset_data, asset_volumes, vtype, vq)
        if r is None: print(f"{label:<30s} NO DATA"); continue
        pw = {p["wl"]: p for p in r["per_window"]}
        w1 = pw.get("W1", {}).get("pnl_cal", 0)
        w2 = pw.get("W2", {}).get("pnl_cal", 0)
        w3 = pw.get("W3", {}).get("pnl_cal", 0)
        ok = r["dd"] < 0.10 and r["wd"] < 0.05
        print(f"{label:<30s} ${r['pnl_cal']:>6.2f} {r['d2t']:>5d} {r['dd']*100:>5.1f}% "
              f"{r['wd']*100:>5.2f}% {r['trades']:>6d}  "
              f"${w1:>5.2f} ${w2:>5.2f} ${w3:>5.2f}  "
              f"{'OK' if ok else 'FAIL'}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
