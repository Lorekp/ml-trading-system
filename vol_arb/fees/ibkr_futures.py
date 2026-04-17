"""IBKR fee model for CME futures and options — parameterized per instrument.

Fees are all-in per side (commission + exchange + NFA).

Sources:
  - IBKR Tiered pricing (<1000 contracts/mo): interactivebrokers.com/en/pricing/commissions-futures-options.php
  - CME fee schedule: cmegroup.com/company/clearing-fees.html
  - NFA assessment: $0.02/contract
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IBKRFuturesFees:
    """Fee model for CME futures and options on IBKR (tiered, <1000/mo).

    Attributes:
        instrument: CME root symbol (GC, SI, CL, ES)
        option_total_per_side: all-in $/leg for options
        futures_total_per_side: all-in $/side for futures (delta hedge)
        contract_size: units per contract (oz, bbl, or $-multiplier × 1)
    """
    instrument: str
    option_total_per_side: float
    futures_total_per_side: float
    contract_size: float
    # Optional breakdowns (for documentation / audit)
    option_commission: float = 0.85
    option_exchange_fee: float = 1.65
    option_nfa_fee: float = 0.02
    futures_commission: float = 0.85
    futures_exchange_fee: float = 1.65
    futures_nfa_fee: float = 0.02

    def straddle_round_trip(self, n_contracts: int = 1) -> float:
        """Cost for full straddle RT (4 legs: enter+exit call+put)."""
        return 4 * self.option_total_per_side * n_contracts

    def hedge_round_trip(self, n_futures: int = 1) -> float:
        """Cost per futures hedge round-trip."""
        return 2 * self.futures_total_per_side * n_futures

    def fee_per_notional(self, spot_price: float) -> float:
        notional = spot_price * self.contract_size
        if notional <= 0:
            return 0.0
        return self.straddle_round_trip(1) / notional


# ── Pre-built per-instrument instances ────────────────────────
# Metals (COMEX): $0.85 IBKR + $1.65 CME + $0.02 NFA = $2.52/leg
IBKR_GC = IBKRFuturesFees("GC", 2.52, 2.52, 100,
                          option_exchange_fee=1.65, futures_exchange_fee=1.65)
IBKR_SI = IBKRFuturesFees("SI", 2.52, 2.52, 5000,
                          option_exchange_fee=1.65, futures_exchange_fee=1.65)

# Energy (NYMEX): same CME structure
IBKR_CL = IBKRFuturesFees("CL", 2.52, 2.52, 1000,
                          option_exchange_fee=1.65, futures_exchange_fee=1.65)

# Equity index (CME): lower exchange fee ~$0.68
#   IBKR $0.85 + CME equity $0.68 + NFA $0.02 = $1.55/leg
IBKR_ES = IBKRFuturesFees("ES", 1.55, 1.55, 50,
                          option_commission=0.85, option_exchange_fee=0.68,
                          futures_commission=0.85, futures_exchange_fee=0.68)

INSTRUMENT_FEES = {"GC": IBKR_GC, "SI": IBKR_SI, "CL": IBKR_CL, "ES": IBKR_ES}


def print_comparison():
    print("=== IBKR Futures Options Fees (straddle RT, 1 contract) ===")
    for inst, f in INSTRUMENT_FEES.items():
        print(f"  {inst}: ${f.straddle_round_trip():.2f}/straddle  "
              f"(${f.option_total_per_side:.2f}/leg × 4)  "
              f"contract_size={f.contract_size}")


if __name__ == "__main__":
    print_comparison()
