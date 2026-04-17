"""ATM straddle strike picker.

Queries Deribit's ``public/get_instruments`` endpoint for active BTC
options, filters by DTE, and selects the ATM call/put pair closest
to the current index price.  The result is a :class:`StraddleSelection`
that holds the two instrument names + their strike + the selected
expiry, ready to pass to the order manager.

Design:

- **Cacheable**: Deribit's instrument list changes at most once per
  day (when a new expiry becomes active), so we cache the instruments
  for ``instrument_refresh_hours`` and only re-query when stale.
- **Expiry selection**: we prefer the front (nearest) expiry within
  the DTE band.  More sophisticated routing (e.g., pick the expiry
  with highest open interest) is a follow-up.
- **Strike granularity**: Deribit uses 1000-USD strike increments for
  BTC.  We pick the single strike closest to index price; the "call"
  and "put" legs share that same strike (canonical ATM straddle).

The picker does NOT hit the WebSocket.  It uses the REST
``get_instruments`` endpoint via the :class:`DeribitRESTClient`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from deribit_paper.client.rest import DeribitRESTClient

logger = logging.getLogger(__name__)


# Deribit active option instruments look like "BTC-28NOV25-60000-C"
_ACTIVE_OPTION_RE = re.compile(
    r"^(?P<asset>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<type>[CP])$",
)


@dataclass
class StraddleSelection:
    """Result of :meth:`StrikePicker.pick_atm_straddle`."""
    call_instrument: str
    put_instrument: str
    strike: float
    expiry_dt: pd.Timestamp
    index_price: float
    moneyness: float  # strike / index_price (~1.0 for true ATM)
    dte_days: float

    @property
    def is_atm(self) -> bool:
        return abs(self.moneyness - 1.0) < 0.005  # within 0.5% of spot


class StrikePicker:
    """Picks the ATM straddle from Deribit's live instrument list.

    Caches the instrument list for up to ``instrument_refresh_hours``
    to avoid hammering the REST endpoint.  Call :meth:`pick_atm_straddle`
    at each bar where you want to open a new position.
    """

    def __init__(
        self,
        *,
        rest: DeribitRESTClient,
        currency: str = "BTC",
        dte_min_days: float = 1.0,
        dte_max_days: float = 14.0,
        instrument_refresh_hours: float = 1.0,
    ):
        self._rest = rest
        self.currency = currency.upper()
        self.dte_min_days = dte_min_days
        self.dte_max_days = dte_max_days
        self.instrument_refresh_hours = instrument_refresh_hours
        self._instruments_cache: list[dict[str, Any]] = []
        self._instruments_cached_at: pd.Timestamp | None = None

    async def refresh_instruments(
        self,
        *,
        now: pd.Timestamp | None = None,
    ) -> int:
        """Force-refresh the cached instrument list.  Returns the count.

        ``now`` is injectable so tests can drive cache staleness
        deterministically.  In production, callers pass the current
        wall-clock UTC time (or leave it None to default).
        """
        instruments = await self._rest.get_instruments(
            currency=self.currency, kind="option", expired=False,
        )
        self._instruments_cache = instruments
        self._instruments_cached_at = now if now is not None else pd.Timestamp.now(tz="UTC")
        logger.info(
            "StrikePicker refreshed %s instrument list: %d active options",
            self.currency, len(instruments),
        )
        return len(instruments)

    async def _get_instruments(self, now: pd.Timestamp) -> list[dict[str, Any]]:
        """Return the cached instruments, refreshing if stale."""
        stale = (
            self._instruments_cached_at is None
            or (now - self._instruments_cached_at)
                >= pd.Timedelta(hours=self.instrument_refresh_hours)
        )
        if stale:
            await self.refresh_instruments(now=now)
        return self._instruments_cache

    async def pick_atm_straddle(
        self,
        *,
        index_price: float,
        now: pd.Timestamp | None = None,
    ) -> StraddleSelection | None:
        """Pick the ATM call/put pair for the front expiry within the DTE band.

        Returns ``None`` if no suitable strike can be found (e.g., no
        active expiry in the DTE band, or the index has no strike
        within a reasonable distance).
        """
        now = now or pd.Timestamp.now(tz="UTC")
        instruments = await self._get_instruments(now)
        if not instruments:
            logger.warning("No %s option instruments available", self.currency)
            return None

        # Parse + filter by DTE and build a DataFrame for selection
        rows = []
        for inst in instruments:
            name = inst.get("instrument_name", "")
            parsed = _parse_instrument(name)
            if parsed is None:
                continue
            expiry_dt, strike, option_type = parsed
            dte_days = (expiry_dt - now).total_seconds() / 86_400.0
            if dte_days < self.dte_min_days or dte_days > self.dte_max_days:
                continue
            rows.append({
                "instrument_name": name,
                "expiry_dt": expiry_dt,
                "strike": strike,
                "option_type": option_type,
                "dte_days": dte_days,
            })

        if not rows:
            logger.warning(
                "No %s options in DTE band [%.1f, %.1f] from %d active instruments",
                self.currency, self.dte_min_days, self.dte_max_days,
                len(instruments),
            )
            return None

        df = pd.DataFrame(rows)

        # Front expiry: smallest dte_days
        front_expiry = df.sort_values("dte_days").iloc[0]["expiry_dt"]
        front_df = df[df["expiry_dt"] == front_expiry]
        calls = front_df[front_df["option_type"] == "C"]
        puts = front_df[front_df["option_type"] == "P"]
        if calls.empty or puts.empty:
            logger.warning(
                "Front expiry %s missing %s legs",
                front_expiry, "puts" if puts.empty else "calls",
            )
            return None

        # Intersect strikes on both sides so we get a matched pair
        common_strikes = sorted(set(calls["strike"]) & set(puts["strike"]))
        if not common_strikes:
            logger.warning(
                "Front expiry %s has no strikes with BOTH call and put",
                front_expiry,
            )
            return None

        # Pick the strike closest to the index price
        closest = min(common_strikes, key=lambda k: abs(k - index_price))
        call_row = calls[calls["strike"] == closest].iloc[0]
        put_row = puts[puts["strike"] == closest].iloc[0]

        return StraddleSelection(
            call_instrument=call_row["instrument_name"],
            put_instrument=put_row["instrument_name"],
            strike=float(closest),
            expiry_dt=pd.Timestamp(front_expiry),
            index_price=float(index_price),
            moneyness=float(closest) / float(index_price) if index_price > 0 else float("nan"),
            dte_days=float(call_row["dte_days"]),
        )


def _parse_instrument(name: str) -> tuple[pd.Timestamp, float, str] | None:
    """Parse a Deribit option instrument name into (expiry_dt, strike, type)."""
    match = _ACTIVE_OPTION_RE.match(name)
    if not match:
        return None
    expiry_str = match.group("expiry")
    strike = float(match.group("strike"))
    option_type = match.group("type")
    try:
        expiry_dt = pd.Timestamp(
            pd.to_datetime(expiry_str, format="%d%b%y"),
            tz="UTC",
        ) + pd.Timedelta(hours=8)
    except ValueError:
        return None
    return expiry_dt, strike, option_type
