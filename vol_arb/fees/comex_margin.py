"""COMEX margin model for GC gold futures and options.

CME uses SPAN (Standard Portfolio ANalysis of risk) for margin.
This is a simplified approximation suitable for backtesting.

For live trading, use IBKR's real-time margin calculator.

Sources:
  - CME margin requirements: cmegroup.com/clearing/margins/outright-vol-scans.html
  - GC initial margin: ~$11,000 per contract (varies daily)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class COMEXMargin:
    """Simplified COMEX SPAN margin for GC gold.

    SPAN computes margin as the maximum expected loss over a 1-day
    holding period across multiple price/vol scenarios. We approximate
    with fixed percentages calibrated to recent CME published margins.
    """

    # ── Futures margin ────────────────────────────────────────
    # CME switched to %-of-notional in Jan 2026: 8% initial, 8.8% heightened
    # At gold $3300/oz × 100oz = $330K notional → ~$26,400 initial
    # At gold $2000/oz × 100oz = $200K notional → ~$16,000 initial
    futures_im_pct: float = 0.08       # 8% of notional (CME Jan 2026)
    futures_maint_pct: float = 0.072   # ~90% of initial
    # Fixed fallback for simpler calculations
    futures_initial_margin: float = 26_400.0   # at $3300/oz
    futures_maintenance_margin: float = 23_760.0

    # ── Option margin (long) ──────────────────────────────────
    # Long option: margin = premium paid (fully funded, no additional margin)
    # This is the simplest case — you can never lose more than premium.

    # ── Option margin (short) ─────────────────────────────────
    # Short option: SPAN scenario-based. Approximate as:
    #   IM ≈ max(futures_IM - OTM_amount, min_margin) + premium_received
    # For ATM straddle: IM ≈ futures_IM (both legs are ATM, max loss = futures move)
    # Short straddle: SPAN ≈ $15K-$25K depending on IV and DTE
    # Conservative: use futures IM as proxy
    short_atm_im_per_contract: float = 26_400.0  # ~= futures IM for ATM at $3300 gold

    # ── Contract specs ────────────────────────────────────────
    contract_size_oz: int = 100
    contract_notional_at_2000: float = 200_000.0  # 100 oz × $2000

    def long_straddle_margin(self, premium_usd: float) -> float:
        """Margin for long ATM straddle = total premium paid."""
        return premium_usd

    def short_straddle_margin(self, n_contracts: int = 1) -> float:
        """Margin for short ATM straddle ≈ futures IM per contract.

        SPAN computes the max loss across 16 risk scenarios. For an
        ATM straddle, the max scenario loss ≈ the futures scan range,
        which is what futures_initial_margin covers.
        """
        return self.short_atm_im_per_contract * n_contracts

    def futures_hedge_margin(self, n_contracts: int = 1) -> float:
        """Margin for delta-hedging futures position.

        Note: if you hold options + futures in the same portfolio,
        IBKR portfolio margin offsets them (lower total margin).
        We conservatively assume NO offset for backtesting.
        """
        return self.futures_initial_margin * n_contracts

    def max_contracts(
        self,
        equity: float,
        margin_utilization: float = 0.30,
        is_long: bool = True,
        premium_per_contract: float = 5000.0,
    ) -> int:
        """Max contracts given equity and target margin utilization.

        Args:
            equity: account equity in USD
            margin_utilization: fraction of equity to use as margin (conservative: 30%)
            is_long: True = long straddle (margin = premium), False = short
            premium_per_contract: estimated straddle premium in USD

        Returns:
            Maximum number of contracts (integer)
        """
        available = equity * margin_utilization
        if is_long:
            margin_per = premium_per_contract
        else:
            margin_per = self.short_atm_im_per_contract
        if margin_per <= 0:
            return 0
        return int(available / margin_per)

    def margin_utilization(
        self,
        n_contracts: int,
        equity: float,
        is_long: bool = True,
        premium_per_contract: float = 5000.0,
    ) -> float:
        """Compute margin utilization ratio."""
        if equity <= 0:
            return 1.0
        if is_long:
            total_margin = n_contracts * premium_per_contract
        else:
            total_margin = n_contracts * self.short_atm_im_per_contract
        return total_margin / equity


# Default instance
COMEX_GC = COMEXMargin()


if __name__ == "__main__":
    m = COMEX_GC
    print("=== COMEX GC Margin Model ===")
    print(f"  Futures IM: ${m.futures_initial_margin:,.0f}/contract")
    print(f"  Short ATM straddle IM: ${m.short_atm_im_per_contract:,.0f}/contract")
    print()
    for equity in [50_000, 100_000, 200_000]:
        n_long = m.max_contracts(equity, 0.30, is_long=True, premium_per_contract=5000)
        n_short = m.max_contracts(equity, 0.30, is_long=False)
        print(f"  ${equity:>10,} equity -> max {n_long} long contracts (30% util) "
              f"or {n_short} short contracts")
