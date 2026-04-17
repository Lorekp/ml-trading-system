"""CME SPAN margin approximations per instrument.

All values are calibrated to published CME margin requirements as of Jan 2026.
For live trading, use IBKR's real-time margin calculator.

Sources:
  - CME outright margins: cmegroup.com/clearing/margins/
  - COMEX metals: Jan 2026 shift to %-of-notional (8% initial, 8.8% heightened)
  - NYMEX energy: ~10% of notional historically
  - CME equity: flat $ SPAN ~$14K for ES
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FuturesMargin:
    instrument: str
    futures_im_pct: float         # fraction-of-notional for futures
    short_atm_im_per_contract: float   # $ for short ATM straddle
    futures_initial_margin: float      # typical $ at current prices (fallback)
    contract_size: float

    def long_straddle_margin(self, premium_usd: float) -> float:
        return premium_usd

    def short_straddle_margin(self, n_contracts: int = 1) -> float:
        return self.short_atm_im_per_contract * n_contracts

    def futures_hedge_margin(self, n_contracts: int = 1) -> float:
        return self.futures_initial_margin * n_contracts


# Metals (8% of notional)
MARGIN_GC = FuturesMargin("GC", 0.08, 26_400.0, 26_400.0, 100)        # $3300 × 100
MARGIN_SI = FuturesMargin("SI", 0.08, 12_000.0, 12_000.0, 5000)       # $30 × 5000

# Energy (10% of notional — NYMEX CL)
MARGIN_CL = FuturesMargin("CL", 0.10,  8_000.0,  8_000.0, 1000)       # $80 × 1000

# Equity index (flat SPAN)
MARGIN_ES = FuturesMargin("ES", 0.056, 14_000.0, 14_000.0, 50)        # ~$14K SPAN at ES 5000

INSTRUMENT_MARGIN = {"GC": MARGIN_GC, "SI": MARGIN_SI, "CL": MARGIN_CL, "ES": MARGIN_ES}


if __name__ == "__main__":
    print("=== CME Futures Margins (approx, Jan 2026) ===")
    for inst, m in INSTRUMENT_MARGIN.items():
        print(f"  {inst}: IM ${m.futures_initial_margin:,.0f}  "
              f"({m.futures_im_pct*100:.1f}% notional)  "
              f"short-ATM ${m.short_atm_im_per_contract:,.0f}")
