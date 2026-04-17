"""Deribit amount-convention helpers.

Deribit has different amount conventions per instrument type:

- **BTC options** (``BTC-*``): amount is in BTC (base units).  Minimum
  contract size is 0.1 BTC, and amounts must be multiples of 0.1.
- **BTC perpetual** (``BTC-PERPETUAL``): amount is in USD notional,
  NOT in BTC.  Minimum is 10 USD, rounded to 10-USD increments.
  To hedge a ``delta_btc`` BTC delta, you must place an order for
  ``delta_btc * current_spot_usd`` USD.

These helpers convert strategy-level "BTC delta" numbers into the
correct Deribit amount format + round to the minimum increment.
"""

from __future__ import annotations

import math


# Minimum option contract size on Deribit BTC options (1 contract = 0.1 BTC)
OPTION_MIN_CONTRACT_BTC: float = 0.1
OPTION_AMOUNT_INCREMENT_BTC: float = 0.1

# Minimum perp order size on Deribit BTC-PERPETUAL (10 USD)
PERP_MIN_USD: float = 10.0
PERP_AMOUNT_INCREMENT_USD: float = 10.0


def round_to_increment(value: float, increment: float) -> float:
    """Round ``value`` to the nearest multiple of ``increment``.

    Half-up rounding.  Preserves sign.
    """
    if increment <= 0:
        return value
    return round(value / increment) * increment


def option_amount_btc(
    btc_amount: float,
    *,
    min_contract_btc: float = OPTION_MIN_CONTRACT_BTC,
    increment: float = OPTION_AMOUNT_INCREMENT_BTC,
) -> float:
    """Convert a strategy-level BTC amount to a valid Deribit option amount.

    Rounds to the minimum contract increment.  Returns 0.0 if the
    absolute value is below the minimum contract size.

    Returns a positive amount regardless of direction (direction is
    passed to Deribit via ``side`` = ``'buy'`` or ``'sell'``).
    """
    abs_amount = abs(btc_amount)
    if abs_amount < min_contract_btc:
        return 0.0
    return round_to_increment(abs_amount, increment)


def perp_amount_usd(
    btc_delta: float,
    spot_usd: float,
    *,
    min_usd: float = PERP_MIN_USD,
    increment_usd: float = PERP_AMOUNT_INCREMENT_USD,
) -> float:
    """Convert a BTC delta to the Deribit perp USD notional amount.

    Deribit's BTC-PERPETUAL is USD-quoted: to hedge ``btc_delta`` BTC
    of exposure, you trade ``btc_delta * spot_usd`` USD of the perp.
    Rounded to the minimum 10-USD increment and floored at the
    minimum notional.  Returns 0.0 if the absolute notional is below
    the minimum.

    Always returns a positive amount — direction is encoded in the
    order's ``side`` parameter.
    """
    if spot_usd <= 0:
        return 0.0
    usd_notional = abs(btc_delta) * spot_usd
    if usd_notional < min_usd:
        return 0.0
    return round_to_increment(usd_notional, increment_usd)


def side_from_signed(signed_amount: float) -> str:
    """Return 'buy' if the signed amount is positive, 'sell' otherwise.

    Convention: positive = buy (long), negative = sell (short).
    Zero raises because a zero-amount order is meaningless.
    """
    if signed_amount > 0:
        return "buy"
    if signed_amount < 0:
        return "sell"
    raise ValueError("side_from_signed called with zero amount")
