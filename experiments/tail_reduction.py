#!/usr/bin/env python
"""Test tail reduction strategies on the 15m+1h prop firm config."""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.data import load_binance_1m, resample_ohlcv

CAPITAL = 100_000.0
BYBIT_EXTRA = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

# Load cached signals
sigs_15m = pd.read_parquet("results/signals_cache/logreg_15m.parquet")
sigs_4h = pd.read_parquet("results/signals_cache/logreg_4h.parquet")
merged_raw = pd.concat([sigs_15m, sigs_4h], ignore_index=True)
for c in ["regime_exposure_mult", "sizing_mult"]:
    if c not in merged_raw.columns: merged_raw[c] = 1.0
    merged_raw[c] = merged_raw[c].fillna(1.0)
merged_raw["probability"] = merged_raw["probability"].astype(float)
merged_raw = merged_raw[merged_raw["confidence"] >= 0.08].reset_index(drop=True)
merged_raw = merged_raw.sort_values("timestamp").reset_index(drop=True)

# BTC daily for vol overlay
btc_daily = resample_ohlcv(load_binance_1m("BTCUSDT", DATA_DIRS), "1D")["close"]
if btc_daily.index.tz is not None: btc_daily.index = btc_daily.index.tz_localize(None)

# Per-asset vol: scale sizing_mult by inverse of asset vol rank
# Higher vol assets get smaller positions
def apply_per_asset_vol_sizing(signals):
    """Scale each signal's sizing by inverse of its asset's recent vol."""
    out = signals.copy()
    # Compute per-asset rolling vol from OHLCV
    asset_vol_rank = {}
    for sym in out["asset"].unique() if "asset" in out.columns else []:
        try:
            full_sym = sym + "USDT" if not sym.endswith("USDT") else sym
            ohlcv = resample_ohlcv(load_binance_1m(full_sym, DATA_DIRS), "1D")["close"]
            if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
            vol = ohlcv.pct_change().rolling(20).std()
            # Rank within all assets: higher vol = lower rank
            asset_vol_rank[sym] = vol
        except Exception:
            pass

    if not asset_vol_rank:
        return out

    # For each signal, scale sizing by (1 / vol_rank_percentile)
    # Low vol asset (pct=0.2) -> mult=1.25, High vol (pct=0.8) -> mult=0.5
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)

    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in asset_vol_rank:
            continue
        vol_series = asset_vol_rank[asset]
        t = ts.iloc[i]
        idx = vol_series.index.searchsorted(t)
        if 0 < idx < len(vol_series):
            v = vol_series.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                # Normalize: BTC vol ~0.02, altcoin vol ~0.05
                # Scale: vol=0.02 -> mult=1.25, vol=0.05 -> mult=0.50
                mult = np.clip(0.025 / v, 0.3, 1.5)
                out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
    return out


def evaluate(signals, label, port_cfg, vol_overlay="btc"):
    merged = signals.copy()
    if vol_overlay == "btc":
        vol_cfg = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
            low_pct=0.20, high_pct=0.80, low_mult=1.25, mid_mult=1.00, high_mult=0.50)
        merged = apply_vol_overlay(merged, btc_daily, vol_cfg)
    elif vol_overlay == "btc_aggressive":
        vol_cfg = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
            low_pct=0.20, high_pct=0.80, low_mult=1.25, mid_mult=1.00, high_mult=0.25)
        merged = apply_vol_overlay(merged, btc_daily, vol_cfg)
    elif vol_overlay == "per_asset":
        merged = apply_per_asset_vol_sizing(merged)

    sim = simulate_portfolio(merged, port_cfg)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    n_trades = sim["n_trades"]
    avg_sz = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
    total_extra = n_trades * avg_sz * BYBIT_EXTRA / 10000
    dpnl_adj = dpnl - total_extra / max(len(dpnl), 1)

    dr = dpnl_adj / CAPITAL
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    worst_day = float(dr.min() * 100)
    cvar1 = float(-dr[dr <= dr.quantile(0.01)].mean() * 10000) if len(dr) > 100 else 0
    cvar5 = float(-dr[dr <= dr.quantile(0.05)].mean() * 10000) if len(dr) > 20 else 0
    pnl = float(dpnl_adj.sum())

    print(f"  {label:<45} SR={sr:.3f} DD={dd:.1f}% dDD={worst_day:.2f}% "
          f"CVaR1={cvar1:.0f}bp CVaR5={cvar5:.0f}bp PnL=${pnl:,.0f}")
    return {"sr": sr, "dd": dd, "worst_day": worst_day, "cvar1": cvar1, "cvar5": cvar5, "pnl": pnl}


def main():
    print("=" * 100)
    print("  TAIL REDUCTION STRATEGIES")
    print("=" * 100)

    configs = [
        # (label, max_dd, delever, max_per_asset, max_exposure, sizing, vol_overlay)
        ("BASELINE (current prod config)",
         0.15, 0.075, 0.06, 0.90, 0.04, "btc"),

        # 1. Tighter per-asset limits
        ("Per-asset 3% (from 6%)",
         0.15, 0.075, 0.03, 0.90, 0.04, "btc"),
        ("Per-asset 4%",
         0.15, 0.075, 0.04, 0.90, 0.04, "btc"),

        # 2. Tighter DD halt
        ("DD halt at 8% (from 15%)",
         0.08, 0.04, 0.06, 0.90, 0.04, "btc"),
        ("DD halt at 10%, delever at 5%",
         0.10, 0.05, 0.06, 0.90, 0.04, "btc"),

        # 3. Lower max exposure
        ("Max exposure 60% (from 90%)",
         0.15, 0.075, 0.06, 0.60, 0.04, "btc"),
        ("Max exposure 70%",
         0.15, 0.075, 0.06, 0.70, 0.04, "btc"),

        # 4. Aggressive vol overlay
        ("Aggressive vol (high=0.25x)",
         0.15, 0.075, 0.06, 0.90, 0.04, "btc_aggressive"),

        # 5. Per-asset vol sizing
        ("Per-asset vol sizing",
         0.15, 0.075, 0.06, 0.90, 0.04, "per_asset"),

        # 6. Combined: prop firm safe
        ("PROP SAFE: DD=10%, pa=4%, exp=70%",
         0.10, 0.05, 0.04, 0.70, 0.04, "btc"),
        ("PROP SAFE + per-asset vol",
         0.10, 0.05, 0.04, 0.70, 0.04, "per_asset"),
        ("PROP SAFE + aggressive vol",
         0.10, 0.05, 0.04, 0.70, 0.04, "btc_aggressive"),

        # 7. Conservative sizing combos
        ("3% sizing, DD=10%, pa=4%",
         0.10, 0.05, 0.04, 0.90, 0.03, "btc"),
        ("3% sizing, DD=10%, pa=3%",
         0.10, 0.05, 0.03, 0.90, 0.03, "btc"),
    ]

    results = []
    for label, max_dd, delever, pa, exp, sz, vol in configs:
        port = PortfolioConfig(
            starting_capital=CAPITAL, max_dd=max_dd, delever_start=delever,
            slippage=0.0001, max_exposure=exp, max_per_asset=pa,
            max_per_strategy=0.50, base_sizing_pct=sz, max_signal_mult=2.0,
            sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
        r = evaluate(merged_raw, label, port, vol)
        if r:
            results.append({"label": label, **r})

    # Summary sorted by PnL with DD < 10% and worst_day > -5%
    df = pd.DataFrame(results)
    viable = df[(df["dd"] < 10) & (df["worst_day"] > -5)]
    print(f"\n{'='*100}")
    print(f"  VIABLE FOR PROP FIRM (DD<10%, daily DD<5%): {len(viable)}/{len(df)}")
    print(f"{'='*100}")
    for _, r in viable.sort_values("pnl", ascending=False).iterrows():
        print(f"  {r['label']:<45} PnL=${r['pnl']:>9,.0f} SR={r['sr']:.3f} "
              f"DD={r['dd']:.1f}% dDD={r['worst_day']:.2f}% CVaR5={r['cvar5']:.0f}bp")


if __name__ == "__main__":
    main()
