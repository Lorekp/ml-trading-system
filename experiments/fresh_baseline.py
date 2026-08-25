"""Fresh baseline matching EXACT homeserver production config."""
import sys
sys.path.insert(0, "src")

import copy, warnings, os, json, time
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

from signal_engine.config import load_config
from signal_engine.pipeline import run_backtest


def main():
    W = max(1, os.cpu_count() - 1)
    cfg = load_config(Path("configs/default.yaml"))

    # Match EXACT homeserver production config
    cfg.regime_filter.enabled = False
    cfg.confidence_filter.mode = "absolute"
    cfg.confidence_filter.threshold = 0.08
    cfg.portfolio.base_sizing_pct = 0.04
    cfg.portfolio.slippage = 0.0005  # 5 bps
    cfg.portfolio.max_exposure = 0.90
    cfg.portfolio.max_per_asset = 0.03

    # 30 production assets (no meme coins)
    cfg.assets = {
        "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
        "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
        "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
        "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
        "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
        "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
        "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
        "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
    }

    print("=" * 80)
    print("FRESH BASELINE - EXACT HOMESERVER PRODUCTION CONFIG")
    print("30 assets | regime off | abs conf 0.08 | 4% sizing | 5 bps slippage")
    print("=" * 80, flush=True)

    t0 = time.time()
    r = run_backtest(cfg, max_workers=W)
    elapsed = time.time() - t0
    s = r["stats"]

    if "error" in s:
        print(f"ERROR: {s}")
        return

    dpnl = r["daily_pnl"]
    SC = cfg.portfolio.starting_capital

    # Bootstrap CI
    daily_ret = (dpnl / SC).values
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(5000):
        sample = rng.choice(daily_ret, size=len(daily_ret), replace=True)
        std = sample.std()
        if std > 0:
            boot.append(sample.mean() / std * np.sqrt(365.25))
    ci_lo = round(np.percentile(boot, 2.5), 3)
    ci_hi = round(np.percentile(boot, 97.5), 3)

    pnl_day = (s["final_equity"] - SC) / max(s["n_days"], 1)

    print(f"\nSharpe:          {s['sharpe']:.3f}  (95% CI: [{ci_lo}, {ci_hi}])")
    print(f"Sortino:         {s['sortino']:.3f}")
    print(f"Return:          {s['total_return_pct']:.2f}%")
    print(f"CAGR:            {s['cagr_pct']:.2f}%")
    print(f"Max DD:          {s['max_dd_pct']:.2f}%")
    print(f"Win Rate:        {s['win_rate_days']*100:.1f}%")
    print(f"Profit Factor:   {s['profit_factor']:.3f}")
    print(f"Trades:          {s['n_trades']:,}")
    print(f"Trades/day:      {s['trades_per_day']:.1f}")
    print(f"Days:            {s['n_days']}")
    print(f"Final Equity:    ${s['final_equity']:,.2f}")
    print(f"PnL/day:         ${pnl_day:.2f}")
    print(f"PSR:             {s['psr_pct']:.1f}%")
    print(f"\nPipeline: {r['n_total_raw']:,} raw -> {r['n_after_confidence']:,} filtered")
    print(f"Elapsed: {elapsed:.0f}s")

    # Save
    out = {
        "config": "homeserver_production_30assets",
        "stats": s,
        "bootstrap_ci": [ci_lo, ci_hi],
        "n_assets": 30,
        "slippage_bps": 5,
        "base_sizing_pct": 0.04,
        "elapsed": round(elapsed, 1),
    }
    with open("v4-results/baseline_before_tier1.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved to v4-results/baseline_before_tier1.json")


if __name__ == "__main__":
    main()
