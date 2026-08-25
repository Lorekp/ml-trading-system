"""How many assets should we trade? More = better? Where's the sweet spot?"""
import sys
sys.path.insert(0, "src")
import copy, warnings, os
import numpy as np
warnings.filterwarnings("ignore")

from pathlib import Path
from signal_engine.config import load_config
from signal_engine.pipeline import run_backtest

W = max(1, os.cpu_count() - 1)


def main():
    cfg = load_config(Path("configs/default.yaml"))
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08

    ranked = [
        ("SOL", "SOLUSDT", 6.18), ("AVAX", "AVAXUSDT", 5.81),
        ("BNB", "BNBUSDT", 5.28), ("SUI", "SUIUSDT", 5.19),
        ("LINK", "LINKUSDT", 5.05), ("DOT", "DOTUSDT", 4.71),
        ("ADA", "ADAUSDT", 4.68), ("AAVE", "AAVEUSDT", 4.56),
        ("XRP", "XRPUSDT", 4.53), ("NEAR", "NEARUSDT", 4.00),
        ("UNI", "UNIUSDT", 3.55), ("DOGE", "DOGEUSDT", 2.94),
        ("BTC", "BTCUSDT", 2.78), ("ARB", "ARBUSDT", 2.37),
        ("OP", "OPUSDT", 2.11), ("ETH", "ETHUSDT", 1.18),
    ]

    print("=" * 85)
    print("ASSET COUNT OPTIMIZATION — adding assets by Sharpe rank (best first)")
    print("=" * 85, flush=True)
    print(f"{'N':>3s} {'Added':>6s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'Trades':>7s} {'Ret/DD':>7s} {'PnL/day':>8s}")
    print("-" * 60)

    for n in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
        sel = ranked[:n]
        assets = {name: sym for name, sym, _ in sel}
        c = copy.deepcopy(cfg)
        c.assets = assets
        c.portfolio.max_per_asset = min(0.20, 0.80 / n)
        c.portfolio.max_exposure = min(0.80, n * c.portfolio.max_per_asset)

        r = run_backtest(c, max_workers=W)
        s = r.get("stats", {})
        if "error" in s:
            continue

        sharpe = s.get("sharpe", 0)
        ret = s.get("total_return_pct", 0)
        dd = max(s.get("max_dd_pct", 0.01), 0.01)
        trades = s.get("n_trades", 0)
        n_days = max(s.get("n_days", 1), 1)
        pnl_day = (s.get("final_equity", 10000) - 10000) / n_days
        added = ranked[n - 1][0]

        print(f"{n:>3d} +{added:<5s} {sharpe:>7.3f} {ret:>8.2f} {dd:>7.2f} {trades:>7d} {ret/dd:>7.1f} ${pnl_day:>7.2f}", flush=True)

    print()
    print("=" * 85)
    print("SELECTION COMPARISON (6 assets, different groups)")
    print("=" * 85, flush=True)
    print(f"{'Selection':<45s} {'Sharpe':>7s} {'Ret%':>8s} {'MaxDD%':>7s} {'Trades':>7s} {'PnL/day':>8s}")
    print("-" * 82)

    groups = [
        ("Top 6 by Sharpe", ranked[:6]),
        ("Bottom 6 by Sharpe", ranked[10:]),
        ("BTC+ETH+top4 alt", [ranked[12], ranked[15]] + ranked[:4]),
        ("Large-cap (BTC/ETH/XRP/BNB/SOL/ADA)", [
            ("BTC","BTCUSDT",0),("ETH","ETHUSDT",0),("XRP","XRPUSDT",0),
            ("BNB","BNBUSDT",0),("SOL","SOLUSDT",0),("ADA","ADAUSDT",0)]),
        ("Mid-cap (SUI/LINK/DOT/AAVE/NEAR/UNI)", [
            ("SUI","SUIUSDT",0),("LINK","LINKUSDT",0),("DOT","DOTUSDT",0),
            ("AAVE","AAVEUSDT",0),("NEAR","NEARUSDT",0),("UNI","UNIUSDT",0)]),
    ]

    for label, sel in groups:
        assets = {name: sym for name, sym, _ in sel}
        c = copy.deepcopy(cfg)
        c.assets = assets
        c.portfolio.max_per_asset = 0.20
        c.portfolio.max_exposure = 0.60

        r = run_backtest(c, max_workers=W)
        s = r.get("stats", {})
        if "error" in s:
            continue

        sharpe = s.get("sharpe", 0)
        ret = s.get("total_return_pct", 0)
        dd = max(s.get("max_dd_pct", 0.01), 0.01)
        trades = s.get("n_trades", 0)
        n_days = max(s.get("n_days", 1), 1)
        pnl_day = (s.get("final_equity", 10000) - 10000) / n_days

        print(f"{label:<45s} {sharpe:>7.3f} {ret:>8.2f} {dd:>7.2f} {trades:>7d} ${pnl_day:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
