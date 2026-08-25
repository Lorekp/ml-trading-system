#!/usr/bin/env python -u
"""Production-grade backtest: ML Close + Vol Overlay + hftbacktest execution.

Pipeline:
1. Load cached signals + ML close model adjusted returns
2. Merge all 3 strategies with proper column alignment
3. Apply vol overlay
4. simulate_portfolio() with full constraints ($100K capital)
5. Apply Bybit execution costs on realized PnL
6. Full stat report

This is the AUTHORITATIVE backtest — proper portfolio sim, not bybit_sim shortcut.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.config import PortfolioConfig, VolOverlayConfig
from signal_engine.pipeline import apply_vol_overlay
from signal_engine.data import load_binance_1m, resample_ohlcv
from pathlib import Path

CAPITAL = 100_000.0

# OLD config (for reference comparison)
PORT_OLD = PortfolioConfig(
    starting_capital=CAPITAL,
    max_dd=0.15, delever_start=0.075,
    slippage=0.0001,  # 1 bps idealized slippage
    max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.40,
    base_sizing_pct=0.02, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)

# OPTIMIZED config (from strategy_optimizer.py Tier 1 sweep)
# Changes: max_per_strategy 0.40 -> 0.50, base_sizing_pct 0.02 -> 0.04
PORT_OPT = PortfolioConfig(
    starting_capital=CAPITAL,
    max_dd=0.15, delever_start=0.075,
    slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06,
    max_per_strategy=0.50,
    base_sizing_pct=0.04, max_signal_mult=2.0,
    sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50,
)

# Confidence filter: only trade when |prob - 0.5| >= MIN_CONFIDENCE
# Found by strategy_optimizer.py: 0.08 gives SR 2.86 Bybit (vs 1.40 baseline)
MIN_CONFIDENCE = 0.08

# Bybit VIP0 maker: 2bps maker + ~0.5bps spread = 4.5bps roundtrip
# Already have 1bps slippage in PORT; Bybit adds 3.5bps on top
BYBIT_EXTRA_COST_BPS = 3.5
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

# Vol overlay configs
VOL_OLD = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                            low_pct=0.20, high_pct=0.80,
                            low_mult=0.50, mid_mult=1.00, high_mult=1.25)
VOL_INVERTED = VolOverlayConfig(enabled=True, vol_window=20, pct_window=60,
                                 low_pct=0.20, high_pct=0.80,
                                 low_mult=1.25, mid_mult=1.00, high_mult=0.50)


def build_ml_close_signals():
    """Build ML-close-adjusted logreg_15m signals compatible with simulate_portfolio."""
    sel = pd.read_parquet("results/close_cache/selected_logreg_15m_parallel.parquet")

    out = pd.DataFrame()
    out["timestamp"] = sel["timestamp"]
    out["asset"] = sel["asset"]
    out["direction"] = sel["direction"]
    # Convert direction-signed exit_return back to unsigned for portfolio sim
    # portfolio sim does: direction * fwd_return - slippage
    out["fwd_return"] = sel["exit_return"] / sel["direction"]
    out["probability"] = sel["entry_probability"]
    out["confidence"] = sel["entry_confidence"]
    out["signal_strength"] = sel["entry_signal_strength"].abs()
    out["strategy"] = "logreg_15m"
    out["hold_bars"] = sel["bars_since_entry"].fillna(4).astype(int) if "bars_since_entry" in sel.columns else 4
    out["regime_exposure_mult"] = sel["regime_exposure_mult"] if "regime_exposure_mult" in sel.columns else 1.0
    out["sizing_mult"] = 1.0
    return out


def run_portfolio_sim(signals, config, label):
    """Run through full simulate_portfolio with all constraints."""
    sim = simulate_portfolio(signals, config)
    if sim is None:
        print(f"  {label}: simulate_portfolio returned None")
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print(f"  {label}: no daily PnL")
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)
    return sim, dpnl


def full_stats(dpnl, sim, capital, label):
    """Compute and print full statistics."""
    stats = compute_full_stats(dpnl, capital, sim["n_trades"],
                                sim["n_trades"] / max(len(dpnl), 1))

    dr = dpnl / capital
    monthly = dpnl.resample("MS").sum()
    n_cal = (dpnl.index.max() - dpnl.index.min()).days

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bsr = np.empty(5000)
    drv = dr.values
    for b in range(5000):
        s = rng.choice(drv, size=len(drv), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    # Losing streak
    streak = mx = 0
    for v in dpnl.values:
        if v < 0: streak += 1; mx = max(mx, streak)
        else: streak = 0

    pos_m = int((monthly > 0).sum())
    neg_m = int((monthly < 0).sum())

    sr = stats.get("sharpe", 0)
    print(f"""
{'='*100}
  {label}
  Capital: ${capital:,.0f}  |  Slippage: 1bps
{'='*100}

  Period:              {n_cal:,} cal days ({n_cal/365.25:.1f} years)
  Active trading days: {len(dpnl):,}
  Final equity:        ${sim['final_equity']:,.0f}

  --- RETURNS ---
  Total return:        {(sim['final_equity']/capital-1)*100:.1f}%
  Annualized return:   {stats.get('cagr',0):.1f}%
  Total PnL:           ${dpnl.sum():,.0f}
  PnL / day:           ${dpnl.sum()/max(n_cal,1):.2f}
  PnL / month:         ${dpnl.sum()/max(n_cal/30.4,1):,.0f}

  --- RISK-ADJUSTED ---
  Sharpe Ratio:        {sr:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}] 95% CI
  Sortino Ratio:       {stats.get('sortino',0):.3f}
  Calmar Ratio:        {stats.get('calmar',0):.2f}
  Max Drawdown:        {stats.get('max_dd_pct',0):.2f}%  (${capital*stats.get('max_dd_pct',0)/100:,.0f})

  --- TRADES ---
  Total trades:        {sim['n_trades']:,}
  Trades / day:        {sim['n_trades']/max(n_cal,1):.1f}
  Win rate:            {stats.get('win_rate',0):.1f}%
  Profit factor:       {stats.get('profit_factor',0):.2f}
  Halted:              {sim.get('halted', False)}

  --- WORST CASE ---
  Worst day:           ${float(dpnl.min()):,.0f}
  Worst month:         ${float(monthly.min()):,.0f}
  Best month:          ${float(monthly.max()):,.0f}
  Max losing streak:   {mx} days
  Positive months:     {pos_m}/{pos_m+neg_m} ({pos_m/(pos_m+neg_m)*100:.0f}%)
""")
    return stats, ci, monthly


def apply_confidence_filter(signals, min_conf):
    """Filter signals by minimum confidence (|prob - 0.5|)."""
    if "confidence" not in signals.columns:
        return signals
    mask = signals["confidence"] >= min_conf
    filtered = signals[mask].reset_index(drop=True)
    return filtered


def main():
    t0 = time.time()

    print("=" * 100)
    print("  PRODUCTION BACKTEST — OLD vs OPTIMIZED (Confidence Filter + Inverted Vol)")
    print("=" * 100)

    # Load signals
    base_15m = pd.read_parquet("results/signals_cache/logreg_15m.parquet")
    base_4h = pd.read_parquet("results/signals_cache/logreg_4h.parquet")
    base_cs = pd.read_parquet("results/signals_cache/cs_momentum.parquet")

    print(f"Signals: 15m={len(base_15m):,}, 4h={len(base_4h):,}, cs={len(base_cs):,}")

    # ================================================================
    # Build merged signals
    # ================================================================

    def merge(parts_list):
        parts = []
        for p in parts_list:
            d = p.copy()
            for col in ["regime_exposure_mult", "sizing_mult", "strategy", "hold_bars"]:
                if col not in d.columns:
                    if col == "strategy":
                        d[col] = "unknown"
                    elif col == "hold_bars":
                        d[col] = 4
                    else:
                        d[col] = 1.0
                d[col] = d[col].fillna(1.0) if col not in ["strategy"] else d[col].fillna("unknown")
            d["probability"] = d["probability"].astype(float)
            parts.append(d)
        merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        return merged

    all_sigs = merge([
        base_15m.assign(strategy="logreg_15m"),
        base_4h.copy(),
        base_cs.copy(),
    ])

    # Load BTC for vol overlay
    try:
        btc_1m = load_binance_1m("BTCUSDT", DATA_DIRS)
        btc_daily = resample_ohlcv(btc_1m, "1D")["close"]
        if btc_daily.index.tz is not None:
            btc_daily.index = btc_daily.index.tz_localize(None)
    except Exception as e:
        print(f"BTC load failed: {e}")
        btc_daily = None

    # ================================================================
    # [1] OLD BASELINE: original vol overlay, no confidence filter
    # ================================================================
    print(f"\n[1] OLD BASELINE (original config)...")
    old_merged = apply_vol_overlay(all_sigs, btc_daily, VOL_OLD) if btc_daily is not None else all_sigs.copy()
    r_old = run_portfolio_sim(old_merged, PORT_OLD, "Old baseline")
    if r_old:
        sim_old, dpnl_old = r_old
        s_old, ci_old, m_old = full_stats(dpnl_old, sim_old, CAPITAL,
                                           "OLD BASELINE (original vol overlay, no conf filter)")

    # ================================================================
    # [2] OPTIMIZED: inverted vol + confidence filter + better sizing
    # ================================================================
    print(f"\n[2] OPTIMIZED (inverted vol + conf>={MIN_CONFIDENCE} + sizing 4%)...")
    opt_merged = apply_vol_overlay(all_sigs, btc_daily, VOL_INVERTED) if btc_daily is not None else all_sigs.copy()
    opt_filtered = apply_confidence_filter(opt_merged, MIN_CONFIDENCE)
    print(f"  Confidence filter: {len(all_sigs):,} -> {len(opt_filtered):,} signals "
          f"({len(opt_filtered)/len(all_sigs)*100:.0f}% kept)")

    r_opt = run_portfolio_sim(opt_filtered, PORT_OPT, "Optimized")
    if r_opt:
        sim_opt, dpnl_opt = r_opt
        s_opt, ci_opt, m_opt = full_stats(dpnl_opt, sim_opt, CAPITAL,
                                           f"OPTIMIZED (inverted vol + conf>={MIN_CONFIDENCE} + 4% sizing)")

    # ================================================================
    # Bybit execution cost adjustment
    # ================================================================
    variants = []
    if r_old:
        variants.append(("Old baseline", sim_old, dpnl_old))
    if r_opt:
        variants.append(("Optimized", sim_opt, dpnl_opt))

    if variants:
        print(f"\n{'='*100}")
        print("  BYBIT VIP0 MAKER EXECUTION COST ADJUSTMENT")
        print(f"{'='*100}")

        bybit_results = {}
        for label, sim, dpnl in variants:
            n_trades = sim["n_trades"]
            avg_size = float(sim["sizes"][~sim["skipped"]].mean()) if (~sim["skipped"]).any() else 0
            extra_cost_per_trade = avg_size * BYBIT_EXTRA_COST_BPS / 10000
            total_extra_cost = n_trades * extra_cost_per_trade
            n_active_days = len(dpnl)
            daily_cost = total_extra_cost / max(n_active_days, 1)

            adjusted = dpnl - daily_cost
            adjusted_nz = adjusted[adjusted != 0]
            if len(adjusted_nz) > 30:
                dr = adjusted_nz / CAPITAL
                sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
                cum_eq = CAPITAL + adjusted_nz.cumsum()
                dd = float(((cum_eq.cummax() - cum_eq) / cum_eq.cummax()).max() * 100)
                pnl = float(adjusted_nz.sum())

                # Bootstrap CI
                rng = np.random.RandomState(42)
                bsr = np.empty(5000)
                drv = dr.values
                for b in range(5000):
                    s = rng.choice(drv, size=len(drv), replace=True)
                    bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
                ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

                print(f"  {label:<20}: SR={sr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  DD={dd:.1f}%  "
                      f"PnL=${pnl:,.0f}  trades={n_trades:,}  "
                      f"(fees=${total_extra_cost:,.0f}, ${daily_cost:.0f}/day)")
                bybit_results[label] = {"sr": sr, "ci": ci, "dd": dd, "pnl": pnl,
                                         "trades": n_trades, "adjusted": adjusted_nz}

        # Per-strategy breakdown (optimized config)
        if r_opt:
            print(f"\n  Per-Strategy (Optimized, Bybit costs):")
            for strat in ["logreg_15m", "logreg_4h", "cs_momentum"]:
                s = opt_filtered[opt_filtered["strategy"] == strat]
                if len(s) > 10:
                    n = len(s)
                    gross = s["direction"].values * s["fwd_return"].values
                    net = gross - BYBIT_EXTRA_COST_BPS / 10000 - 0.0001
                    print(f"    {strat:<15}: {n:>7,} signals  "
                          f"gross={gross.mean()*10000:.1f}bps  net={net.mean()*10000:.1f}bps")

    # ================================================================
    # Year-by-year comparison
    # ================================================================
    if r_opt and "Optimized" in bybit_results:
        print(f"\n{'='*100}")
        print("  YEAR-BY-YEAR (Optimized, Bybit costs)")
        print(f"{'='*100}")
        adj = bybit_results["Optimized"]["adjusted"]
        for year in sorted(adj.index.year.unique()):
            y = adj[adj.index.year == year]
            if len(y) > 20:
                sr_y = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 else 0
                cum = CAPITAL + y.cumsum()
                dd_y = float(((cum.cummax() - cum) / cum.cummax()).max() * 100) if len(cum) > 0 else 0
                print(f"  {year}: SR={sr_y:.3f}  PnL=${y.sum():,.0f}  DD={dd_y:.1f}%  days={len(y)}")

    # ================================================================
    # Monthly comparison
    # ================================================================
    if r_old and r_opt:
        print(f"\n{'='*100}")
        print("  MONTHLY PnL COMPARISON (last 14 months, idealized sim)")
        print(f"{'='*100}")
        print(f"{'Month':<10} {'Old':>12} {'Optimized':>12} {'Delta':>12}")
        print("-" * 48)
        recent = m_opt.iloc[-14:]
        for idx in recent.index:
            m = idx.strftime("%Y-%m")
            ov = m_old.get(idx, 0)
            nv = m_opt.get(idx, 0)
            print(f"{m:<10} ${ov:>11,.0f} ${nv:>11,.0f} ${nv-ov:>+11,.0f}")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
