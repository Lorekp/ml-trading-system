"""Experiment 2: SADF regime gate on T1 signals.

Computes rolling SADF on BTC. Suppresses signals that go against
the regime (long during bubble, short during crash).
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
from signal_engine.features import rolling_sadf
from signal_engine.data import load_binance_1m, resample_ohlcv
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/optuna/signal_cache")
DATA_DIR = Path("data/binance-1m")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
C_STR = "0.02000"; LB = 2; THR = 0.10; FWD = 6; LEV = 5.0; DLV = 0.05; SIZING = 0.06


def compute_btc_sadf():
    """Compute rolling SADF on BTC 15-min close."""
    print("Computing BTC SADF (this takes ~1 min)...", flush=True)
    ohlcv_1m = load_binance_1m("BTCUSDT", [DATA_DIR])
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    # SADF on 4h resampled close (faster than 15m, still captures regime)
    close_4h = ohlcv_15m["close"].resample("4h").last().dropna()
    sadf = rolling_sadf(close_4h, min_window=50, step=5)
    # Forward-fill to 15m timestamps
    sadf_15m = sadf.reindex(ohlcv_15m.index, method="ffill")
    return sadf_15m


def run_with_sadf_gate(btc_sadf, high_pctile, low_pctile, mode="suppress"):
    """
    mode='suppress': drop signals against regime
    mode='reduce': halve signal_strength against regime
    """
    # Compute thresholds from historical SADF
    valid_sadf = btc_sadf.dropna()
    if len(valid_sadf) < 100:
        return None
    high_thresh = valid_sadf.quantile(high_pctile / 100.0)
    low_thresh = valid_sadf.quantile(low_pctile / 100.0)

    tp = tc = tt = ttd = 0; mdd = mwd = 0.0; n_gated = 0
    per_window = []

    for wl, cd in W:
        with open(CACHE_DIR / f"C{C_STR}_lb{LB}_{wl}.pkl", "rb") as f:
            raw = pickle.load(f)
        fi = raw[raw["confidence"] > THR].copy()
        fi["fwd_return"] = fi[f"ret_{FWD}"].values

        # Get SADF at each signal timestamp
        keep = []
        for _, row in fi.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if btc_sadf.index.tz is not None and ts.tz is None:
                ts = ts.tz_localize(btc_sadf.index.tz)
            elif btc_sadf.index.tz is None and ts.tz is not None:
                ts = ts.tz_localize(None)

            # Find nearest SADF value
            idx = btc_sadf.index.searchsorted(ts)
            if idx >= len(btc_sadf):
                idx = len(btc_sadf) - 1
            sadf_val = btc_sadf.iloc[idx]

            if np.isnan(sadf_val):
                keep.append(True)
                continue

            direction = row["direction"]

            # Gate logic
            if sadf_val > high_thresh and direction > 0:
                # Bubble regime + LONG → suppress
                if mode == "suppress":
                    keep.append(False); n_gated += 1
                else:
                    keep.append(True)  # reduce handled via signal_strength
            elif sadf_val < low_thresh and direction < 0:
                # Crash regime + SHORT → suppress
                if mode == "suppress":
                    keep.append(False); n_gated += 1
                else:
                    keep.append(True)
            else:
                keep.append(True)

        fi = fi[keep]
        if len(fi) < 10:
            per_window.append({"wl": wl, "pnl": 0, "pnl_cal": 0})
            continue

        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=DLV,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=SIZING * LEV, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5:
            per_window.append({"wl": wl, "pnl": 0, "pnl_cal": 0})
            continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1:
            per_window.append({"wl": wl, "pnl": 0, "pnl_cal": 0})
            continue

        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / 10000.0
        pnl = sim["equity"][-1] - 10000

        per_window.append({"wl": wl, "pnl": pnl, "pnl_cal": pnl / cd,
                          "dd": abs(dd.min()), "wd": abs(dr.min())})
        tp += pnl; tc += cd; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min())); mwd = max(mwd, abs(dr.min()))

    if tc == 0: return None
    return {
        "pnl_cal": tp / tc, "d2t": int(1000 / (tp / tc)) if tp > 0 else 9999,
        "dd": mdd, "wd": mwd, "trades": tt, "act": ttd / tc * 100,
        "gated": n_gated, "per_window": per_window,
    }


def main():
    t0 = time.time()
    print("EXPERIMENT 2: SADF REGIME GATE")
    print("=" * 95)

    btc_sadf = compute_btc_sadf()
    valid = btc_sadf.dropna()
    print(f"BTC SADF: {len(valid)} values, range [{valid.min():.2f}, {valid.max():.2f}]")
    print(f"  p5={valid.quantile(0.05):.2f}, p50={valid.quantile(0.50):.2f}, "
          f"p90={valid.quantile(0.90):.2f}, p95={valid.quantile(0.95):.2f}\n")

    print(f"{'Gate':<35s} {'$/cal':>7s} {'~Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Gated':>6s}  W1$/d  W2$/d  W3$/d")
    print("-" * 100)

    for label, high, low, mode in [
        ("Baseline (no gate)", 100, 0, "suppress"),
        ("Suppress p90/p10", 90, 10, "suppress"),
        ("Suppress p95/p5", 95, 5, "suppress"),
        ("Suppress p99/p1", 99, 1, "suppress"),
        ("Suppress p85/p15", 85, 15, "suppress"),
        ("Suppress p80/p20", 80, 20, "suppress"),
        ("Suppress longs p95 only", 95, 0, "suppress"),
        ("Suppress longs p90 only", 90, 0, "suppress"),
        ("Suppress shorts p5 only", 100, 5, "suppress"),
    ]:
        r = run_with_sadf_gate(btc_sadf, high, low, mode)
        if r is None: print(f"{label:<35s} NO DATA"); continue

        pw = {p["wl"]: p for p in r["per_window"]}
        w1 = pw.get("W1", {}).get("pnl_cal", 0)
        w2 = pw.get("W2", {}).get("pnl_cal", 0)
        w3 = pw.get("W3", {}).get("pnl_cal", 0)

        ok = r["dd"] < 0.10 and r["wd"] < 0.05
        print(f"{label:<35s} ${r['pnl_cal']:>6.2f} {r['d2t']:>5d} {r['dd']*100:>5.1f}% "
              f"{r['wd']*100:>5.2f}% {r['gated']:>6d}  "
              f"${w1:>5.2f} ${w2:>5.2f} ${w3:>5.2f}  "
              f"{'OK' if ok else 'FAIL'}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
