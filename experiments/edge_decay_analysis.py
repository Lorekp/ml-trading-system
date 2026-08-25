"""Edge decay analysis — T1 signal quality across all available history.

Tests the same signal (C=0.02, lb=2, thr=0.10, fixed T+6) on every month
from 2023-03 to 2026-02 to see if the edge is decaying or regime-dependent.
"""

import sys, os, time, warnings
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
from signal_engine.asset_selection import compute_monthly_dollar_volume

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
cfg = load_config(Path("configs/default.yaml"))
DATA_DIR = Path("data/binance-1m")
C = 0.02; LB = 2; THR = 0.10; TOP_N = 30; FWD = 6


def main():
    t0 = time.time()
    print("EDGE DECAY ANALYSIS - T1 (C=0.02, lb=2, thr=0.10, fixed T+6)")
    print("=" * 95)

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
    print(f"Loaded {len(asset_data)} assets in {time.time()-t0:.0f}s")

    sample_idx = next(iter(asset_volumes.values())).index
    end_date = max(d["end_date"] for d in asset_data.values())
    all_months = pd.period_range(start="2023-03", end=end_date.to_period("M"), freq="M")

    print(f"\nAnalyzing {len(all_months)} months...\n", flush=True)
    header = f"{'Month':<10s} {'Sigs':>6s} {'Ast':>4s} {'Win%':>6s} {'bps':>7s} {'MaxC':>7s} {'AvgC':>7s} {'BTC':>8s} {'BTC_v':>7s}"
    print(header)
    print("-" * 70)

    monthly = []

    for month in all_months:
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

        sigs = []
        for symbol in selected:
            if symbol not in asset_data: continue
            ad = asset_data[symbol]
            close_arr = ad["close_arr"]; n = len(close_arr)
            fc = ad["fc"]; y_all = ad["y_all"]; valid = ad["valid"]
            vol = ad["vol"]; ts_idx = ad["timestamps"]

            ms_l = ms; me_l = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_l = ms.tz_localize(None); me_l = me_l.tz_localize(None)

            te = valid[(valid >= ms_l) & (valid <= me_l)]
            vts = ms_l - pd.DateOffset(months=3)
            vt = vol.loc[(vol.index >= vts) & (vol.index < ms_l)].dropna()
            if len(vt) < 50: continue
            vq = vt.quantile(0.50)
            tv = vol.reindex(te)
            hv = (tv > vq).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
            if len(te) < 5: continue

            train_start = ms_l - pd.DateOffset(months=LB)
            if train_start < ad["start_date"]: continue
            emb = ad["median_gap"] * 5
            tc = ms_l - emb
            tr = valid[(valid >= train_start) & (valid < tc)]
            if len(tr) < 100: continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                if NEW_SKLEARN:
                    mdl = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
                else:
                    mdl = LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)
                mdl.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te])
                yp = mdl.predict_proba(Xte)[:, 1]
            except Exception: continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            mask = conf > THR

            for i in range(len(te)):
                if not mask[i]: continue
                bi = ts_idx.get_loc(te[i])
                if bi + FWD >= n: continue
                ep = close_arr[bi]
                if ep <= 0: continue
                d = dirs[i]
                fr = d * (close_arr[bi + FWD] - ep) / ep
                sigs.append({"asset": symbol.replace("USDT", ""), "dir": d,
                             "conf": conf[i], "fwd_ret": fr})

        # BTC stats
        btc_c = asset_data.get("BTCUSDT", {}).get("close")
        btc_str = ""; btc_v_str = ""
        if btc_c is not None:
            ms_n = ms_l if ms_l is not None else ms
            me_n = me_l if me_l is not None else ms + pd.offsets.MonthEnd(0)
            bm = btc_c.loc[ms_n:me_n]
            if len(bm) > 10:
                btc_str = f"{(bm.iloc[-1]/bm.iloc[0]-1)*100:>+6.1f}%"
                btc_v_str = f"{bm.pct_change().std()*100:.2f}%"

        if sigs:
            df = pd.DataFrame(sigs)
            pnls = df["dir"] * df["fwd_ret"]
            win = (pnls > 0).mean() * 100
            bps = pnls.mean() * 10000
            na = df["asset"].nunique()
            mc = df["conf"].max(); ac = df["conf"].mean()
            monthly.append({"m": str(month), "n": len(df), "win": win, "bps": bps, "mc": mc, "ac": ac})
            print(f"{str(month):<10s} {len(df):>6d} {na:>4d} {win:>5.1f}% {bps:>+6.1f} {mc:>7.4f} {ac:>7.4f} {btc_str:>8s} {btc_v_str:>7s}", flush=True)
        else:
            monthly.append({"m": str(month), "n": 0, "win": 0, "bps": 0, "mc": 0, "ac": 0})
            print(f"{str(month):<10s} {0:>6d} {0:>4d}   N/A     N/A     N/A     N/A {btc_str:>8s} {btc_v_str:>7s}", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY BY PERIOD")
    print(f"{'='*70}")

    periods = [
        ("2023 (bear/recovery)", "2023-03", "2023-12"),
        ("2024 H1 (pre-bull)", "2024-01", "2024-06"),
        ("2024 H2 (bull)", "2024-07", "2024-12"),
        ("2025 H1 (post-peak)", "2025-01", "2025-06"),
        ("2025 H2 (decline)", "2025-07", "2025-12"),
        ("2026 YTD (crash)", "2026-01", "2026-02"),
    ]

    for label, s, e in periods:
        p = [r for r in monthly if s <= r["m"] <= e and r["n"] > 0]
        if not p: print(f"  {label:<28s} NO DATA"); continue
        tn = sum(r["n"] for r in p)
        aw = np.mean([r["win"] for r in p])
        ab = np.mean([r["bps"] for r in p])
        ac = np.mean([r["ac"] for r in p])
        pm = sum(1 for r in p if r["bps"] > 0)
        print(f"  {label:<28s} sigs={tn:>6d} win={aw:.1f}% bps={ab:>+6.1f} conf={ac:.4f} pos={pm}/{len(p)}")

    # Rolling 6-month trend
    print(f"\nROLLING 6-MONTH TREND:")
    active = [r for r in monthly if r["n"] > 0]
    for i in range(6, len(active)):
        w = active[i-6:i]
        ab = np.mean([r["bps"] for r in w])
        an = np.mean([r["n"] for r in w])
        print(f"  ending {w[-1]['m']}: avg_bps={ab:>+6.1f}, avg_sigs/mo={an:.0f}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
