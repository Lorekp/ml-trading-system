"""Experiment 1: Asset age filter on T1 signals.

Tests filtering young/speculative assets from T1's cached signals.
Compares multiple age thresholds against T1 baseline.
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.data import load_binance_1m, resample_ohlcv
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/optuna/signal_cache")
DATA_DIR = Path("data/binance-1m")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
C_STR = "0.02000"; LB = 2; THR = 0.10; FWD = 6; LEV = 5.0; DLV = 0.05; SIZING = 0.06


def get_asset_first_dates():
    dates = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"): continue
        asset = d.name.replace("USDT", "")
        try:
            ohlcv = load_binance_1m(d.name, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv, "15min")
            dates[asset] = ohlcv_15m.index.min()
        except Exception: pass
    return dates


def run_with_age_filter(first_dates, min_age_days):
    tp = tc = tt = ttd = 0; mdd = mwd = 0.0; n_filtered = 0
    per_window = []

    for wl, cd in W:
        with open(CACHE_DIR / f"C{C_STR}_lb{LB}_{wl}.pkl", "rb") as f:
            raw = pickle.load(f)
        fi = raw[raw["confidence"] > THR].copy()
        fi["fwd_return"] = fi[f"ret_{FWD}"].values

        if min_age_days > 0:
            keep = []
            for _, row in fi.iterrows():
                fd = first_dates.get(row["asset"])
                if fd is None: keep.append(False); continue
                st = pd.Timestamp(row["timestamp"])
                if hasattr(fd, "tz") and fd.tz is not None and st.tz is None:
                    st = st.tz_localize(fd.tz)
                keep.append((st - fd).days >= min_age_days)
            before = len(fi)
            fi = fi[keep]
            n_filtered += before - len(fi)

        if len(fi) < 10:
            per_window.append({"wl": wl, "pnl": 0, "dd": 0, "wd": 0, "trades": 0})
            continue

        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=DLV,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=SIZING * LEV, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5:
            per_window.append({"wl": wl, "pnl": 0, "dd": 0, "wd": 0, "trades": 0})
            continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1:
            per_window.append({"wl": wl, "pnl": 0, "dd": 0, "wd": 0, "trades": 0})
            continue

        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / 10000.0
        pnl = sim["equity"][-1] - 10000

        per_window.append({
            "wl": wl, "pnl": pnl, "dd": abs(dd.min()), "wd": abs(dr.min()),
            "trades": sim["n_trades"], "pnl_cal": pnl / cd,
        })
        tp += pnl; tc += cd; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min())); mwd = max(mwd, abs(dr.min()))

    if tc == 0: return None
    return {
        "pnl_cal": tp / tc, "d2t": int(1000 / (tp / tc)) if tp > 0 else 9999,
        "dd": mdd, "wd": mwd, "trades": tt, "act": ttd / tc * 100,
        "filtered": n_filtered, "per_window": per_window,
    }


def main():
    t0 = time.time()
    print("EXPERIMENT 1: ASSET AGE FILTER")
    print("=" * 90)

    first_dates = get_asset_first_dates()
    print(f"Loaded {len(first_dates)} asset first dates in {time.time()-t0:.0f}s\n")

    print(f"{'Filter':<25s} {'$/cal':>7s} {'~Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Act%':>5s} {'Filt':>6s}  W1$/d  W2$/d  W3$/d")
    print("-" * 95)

    for label, age in [
        ("Baseline (no filter)", 0),
        ("Age > 90 days", 90),
        ("Age > 180 days", 180),
        ("Age > 270 days", 270),
        ("Age > 365 days", 365),
        ("Age > 540 days", 540),
        ("Age > 730 days (2yr)", 730),
    ]:
        r = run_with_age_filter(first_dates, age)
        if r is None: print(f"{label:<25s} NO DATA"); continue

        pw = {p["wl"]: p for p in r["per_window"]}
        w1 = pw.get("W1", {}).get("pnl_cal", 0)
        w2 = pw.get("W2", {}).get("pnl_cal", 0)
        w3 = pw.get("W3", {}).get("pnl_cal", 0)

        ok = r["dd"] < 0.10 and r["wd"] < 0.05
        print(f"{label:<25s} ${r['pnl_cal']:>6.2f} {r['d2t']:>5d} {r['dd']*100:>5.1f}% "
              f"{r['wd']*100:>5.2f}% {r['act']:>4.0f}% {r['filtered']:>6d}  "
              f"${w1:>5.2f} ${w2:>5.2f} ${w3:>5.2f}  "
              f"{'OK' if ok else 'DD!'}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
