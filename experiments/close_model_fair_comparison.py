"""Fair comparison: Baseline vs ML Close model with SAME portfolio constraints."""
import sys, os, time, warnings
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.config import PortfolioConfig

PORT_CFG = PortfolioConfig(
    starting_capital=10000.0, max_dd=0.15, delever_start=0.075,
    slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.40, base_sizing_pct=0.02, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)


def run_through_portfolio_sim(logreg_15m_df, base_4h, base_cs, label):
    """Run variant through SAME portfolio sim with SAME constraints."""
    merged = pd.concat([logreg_15m_df, base_4h, base_cs], ignore_index=True)
    for col in ["regime_exposure_mult", "sizing_mult"]:
        if col not in merged.columns:
            merged[col] = 1.0
        merged[col] = merged[col].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    if "hold_bars" not in merged.columns:
        merged["hold_bars"] = 4
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_portfolio(merged, PORT_CFG)
    if sim is None:
        print(f"  {label}: FAILED (None)")
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print(f"  {label}: FAILED (no daily PnL)")
        return None

    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    stats = compute_full_stats(dpnl, 10000, sim["n_trades"],
                                sim["n_trades"] / max(len(dpnl), 1))
    return {
        "sr": stats.get("sharpe", 0), "dd": stats.get("max_dd_pct", 0),
        "n": sim["n_trades"], "eq": sim["final_equity"], "dpnl": dpnl,
    }


def bybit_sim(logreg_15m_df, base_4h, base_cs, label, capital=100000):
    """Bybit VIP0 maker sim."""
    merged = pd.concat([logreg_15m_df, base_4h, base_cs], ignore_index=True)
    for col in ["regime_exposure_mult", "sizing_mult"]:
        if col not in merged.columns:
            merged[col] = 1.0
        merged[col] = merged[col].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    gross = merged["direction"].values * merged["fwd_return"].values
    net = gross - 4.5 / 10000
    edge = np.where(np.isfinite(merged["probability"].values),
                    np.abs(2 * merged["probability"].values - 1), 0.1)
    edge = np.clip(edge, 0, 1)

    eq = float(capital)
    pnl = np.zeros(len(merged))
    for i in range(len(merged)):
        sz = min(eq * 0.02 * edge[i], eq * 0.04)
        pnl[i] = sz * net[i]
        eq += pnl[i]

    ts = pd.to_datetime(merged["timestamp"].values)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    daily = pd.Series(pnl, index=ts).resample("1D").sum()
    daily = daily[daily != 0]
    sr = float(daily.mean() / daily.std() * np.sqrt(365.25)) if len(daily) > 30 and daily.std() > 0 else 0
    return sr, float(pnl.sum()), float(net.mean() * 10000)


def main():
    t0 = time.time()
    print("=" * 100)
    print("  FAIR COMPARISON: Baseline vs ML Close (SAME portfolio constraints)")
    print("=" * 100)

    # Load data
    base_15m = pd.read_parquet("results/signals_cache/logreg_15m.parquet")
    base_4h = pd.read_parquet("results/signals_cache/logreg_4h.parquet")
    base_cs = pd.read_parquet("results/signals_cache/cs_momentum.parquet")
    examples = pd.read_parquet("results/close_cache/examples_logreg_15m.parquet")
    sel = pd.read_parquet("results/close_cache/selected_logreg_15m_parallel.parquet")

    # ============================================================
    # Variant A: Baseline (fixed 4-bar hold)
    # ============================================================
    variant_a = base_15m.copy()
    variant_a["strategy"] = "logreg_15m"

    # ============================================================
    # Variant B: DecayT3 (exit if losing at bar 3)
    # ============================================================
    # From close examples: for each trade, check bar 3 and bar 4
    bar3 = examples[examples["bars_since_entry"] == 3].copy()
    bar4 = examples[examples["bars_since_entry"] == 4].copy()

    # Each trade has a unique trade_id. Map to first bar timestamp + asset
    first_bar = examples.groupby("trade_id").first().reset_index()

    # For DecayT3: if exit_return at bar 3 is negative, use bar 3; else bar 4
    bar3_by_trade = bar3.set_index("trade_id")["exit_return"]
    bar4_by_trade = bar4.set_index("trade_id")["exit_return"]

    # Default: full hold return from bar 4
    all_trades = bar4_by_trade.index.union(bar3_by_trade.index)
    decay_returns = bar4_by_trade.reindex(all_trades).copy()
    # Trades losing at bar 3: early exit
    losing_at_3 = bar3_by_trade[bar3_by_trade < 0].index
    common = losing_at_3.intersection(decay_returns.index)
    decay_returns.loc[common] = bar3_by_trade.loc[common]
    decay_returns = decay_returns.dropna()

    # Convert direction-signed exit_return back to unsigned fwd_return
    dir_by_trade = bar4.set_index("trade_id")["direction"]
    decay_fwd = decay_returns / dir_by_trade

    # Map back to base_15m by matching timestamp + asset
    fb = first_bar.set_index("trade_id")[["timestamp", "asset"]]
    fb["decay_fwd"] = decay_fwd
    fb["timestamp"] = pd.to_datetime(fb["timestamp"])

    variant_b = base_15m.copy()
    variant_b["strategy"] = "logreg_15m"
    variant_b["timestamp"] = pd.to_datetime(variant_b["timestamp"])
    merged_decay = variant_b.merge(
        fb[["timestamp", "asset", "decay_fwd"]],
        on=["timestamp", "asset"], how="left",
    )
    # Where decay_fwd is available, use it
    mask = merged_decay["decay_fwd"].notna()
    variant_b.loc[mask.values, "fwd_return"] = merged_decay.loc[mask, "decay_fwd"].values
    n_decay = int(mask.sum())

    # ============================================================
    # Variant C: ML Close (full features)
    # ============================================================
    variant_c = sel.copy()
    # Convert exit_return (direction-signed) back to unsigned for simulate_portfolio
    variant_c["fwd_return"] = variant_c["exit_return"] / variant_c["direction"]
    variant_c["probability"] = variant_c["entry_probability"]
    variant_c["confidence"] = variant_c["entry_confidence"]
    variant_c["signal_strength"] = variant_c["entry_signal_strength"].abs()
    variant_c["strategy"] = "logreg_15m"
    variant_c["hold_bars"] = 4
    if "regime_exposure_mult" not in variant_c.columns:
        variant_c["regime_exposure_mult"] = 1.0
    variant_c["sizing_mult"] = 1.0

    print(f"\nVariants: A={len(variant_a):,}, B={len(variant_b):,} (decay={n_decay:,}), C={len(variant_c):,}")

    # ============================================================
    # Run through SAME portfolio sim
    # ============================================================
    print(f"\n{'='*100}")
    print("  IDEALIZED (same simulate_portfolio, same constraints)")
    print(f"{'='*100}\n")

    for name, df in [("A: Baseline (fixed hold)", variant_a),
                      ("B: DecayT3 (bar 3 exit)", variant_b),
                      ("C: ML Close (full)", variant_c)]:
        r = run_through_portfolio_sim(df, base_4h, base_cs, name)
        if r:
            print(f"  {name:<30}: SR={r['sr']:.3f}  DD={r['dd']:.1f}%  "
                  f"Eq=${r['eq']:,.0f}  Trades={r['n']:,}")

    # ============================================================
    # Bybit layer
    # ============================================================
    print(f"\n{'='*100}")
    print("  BYBIT VIP0 MAKER ($100K, 4.5 bps roundtrip)")
    print(f"{'='*100}\n")

    for name, df in [("A: Baseline", variant_a),
                      ("B: DecayT3", variant_b),
                      ("C: ML Close", variant_c)]:
        sr, pnl, net = bybit_sim(df, base_4h, base_cs, name)
        print(f"  {name:<30}: SR={sr:.3f}  PnL=${pnl:,.0f}  Net={net:.2f}bps")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
