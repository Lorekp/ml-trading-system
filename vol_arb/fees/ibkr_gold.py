"""IBKR fee model for GC gold futures options.

All fees in USD. Based on IBKR tiered pricing for US futures options.
Conservative estimates — actual may be slightly lower at higher volume.

Sources:
  - IBKR US Futures Options Commissions: interactivebrokers.com/en/pricing/commissions-futures-options.php
  - CME Group fee schedule: cmegroup.com/company/clearing-fees.html
  - NFA assessment: $0.02/contract (all futures)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IBKRGoldFees:
    """Fee model for GC (gold) futures options and futures on IBKR.

    Default values are IBKR Tiered pricing for monthly volume < 1,000 contracts.
    """

    # ── GC Options (OG) — IBKR Tiered Pricing ───────────────
    # Source: IBKR pricing page + CME fee schedule (Feb 2025)
    option_commission: float = 0.85    # $/contract/side (tiered, <1000/mo)
    option_exchange_fee: float = 1.65  # CME COMEX metals options (Feb 2025)
    option_nfa_fee: float = 0.02       # NFA assessment
    # Total per side
    option_total_per_side: float = 2.52  # 0.85 + 1.65 + 0.02

    # ── GC Futures (for delta hedging) ────────────────────────
    futures_commission: float = 0.85   # $/contract/side (tiered, <1000/mo)
    futures_exchange_fee: float = 1.65 # CME COMEX (approx, same as options)
    futures_nfa_fee: float = 0.02
    futures_total_per_side: float = 2.52

    # ── Contract specs ────────────────────────────────────────
    contract_size_oz: int = 100        # 1 GC option/future = 100 troy oz
    tick_size_oz: float = 0.10         # $0.10/oz minimum price increment
    tick_value: float = 10.0           # $10 per tick (100 oz × $0.10)

    # ── Exercise / Assignment ─────────────────────────────────
    # No separate fee — standard futures commission applies to resulting position
    exercise_fee: float = 2.52         # = normal futures commission + exchange
    assignment_fee: float = 2.52       # same as exercise

    def straddle_entry_cost(self, n_contracts: int = 1) -> float:
        """Cost to ENTER a straddle (buy call + buy put = 2 legs)."""
        return 2 * self.option_total_per_side * n_contracts

    def straddle_exit_cost(self, n_contracts: int = 1) -> float:
        """Cost to EXIT a straddle (sell call + sell put = 2 legs)."""
        return 2 * self.option_total_per_side * n_contracts

    def straddle_round_trip(self, n_contracts: int = 1) -> float:
        """Total cost for entry + exit of 1 straddle (4 legs)."""
        return 4 * self.option_total_per_side * n_contracts

    def hedge_round_trip(self, n_futures: int = 1) -> float:
        """Cost per futures hedge round-trip."""
        return 2 * self.futures_total_per_side * n_futures

    def fee_per_notional(self, spot_price: float = 2000.0) -> float:
        """Fee as fraction of notional for a straddle RT (for comparison)."""
        notional = spot_price * self.contract_size_oz
        return self.straddle_round_trip(1) / notional


# Pre-built instances for common scenarios
IBKR_TIERED = IBKRGoldFees()  # default tiered pricing

IBKR_FIXED = IBKRGoldFees(
    option_commission=1.50,    # IBKR fixed pricing (higher commission)
    option_total_per_side=1.92,
    futures_commission=1.50,
    futures_total_per_side=3.02,
)


def print_fee_comparison():
    """Print fee comparison vs Deribit BTC options."""
    fees = IBKR_TIERED
    gc_notional = 2000 * 100  # $200K
    gc_straddle_rt = fees.straddle_round_trip(1)
    gc_fee_pct = gc_straddle_rt / gc_notional * 100

    deribit_btc_notional = 85_000  # 1 BTC
    deribit_straddle_rt = 4 * 0.0003 * deribit_btc_notional  # 4 legs × 0.03%
    deribit_fee_pct = deribit_straddle_rt / deribit_btc_notional * 100

    print("=== Fee Comparison: IBKR Gold vs Deribit BTC ===")
    print(f"  IBKR GC straddle RT:     ${gc_straddle_rt:.2f} ({gc_fee_pct:.4f}% of notional)")
    print(f"  Deribit BTC straddle RT: ${deribit_straddle_rt:.2f} ({deribit_fee_pct:.4f}% of notional)")
    print(f"  Ratio: Deribit is {deribit_straddle_rt / gc_straddle_rt:.0f}× more expensive")
    print(f"  IBKR hedge (futures) RT: ${fees.hedge_round_trip(1):.2f}")


if __name__ == "__main__":
    print_fee_comparison()
